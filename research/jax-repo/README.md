# JAX 仓库源码结构与组件分析

**分析对象**：`upstream/jax` = <https://github.com/jax-ml/jax> @ `2d66622450e2c8633cda2307688ef7aa294bd6eb`（tag `jax-v0.11.1`）
**配套图**：`figures/jax-repo-structure.svg`（源码结构树）、`figures/jax-repo-pipeline.svg`（上下游链路）
**图表生成脚本**：`make_diagrams.py`

---

## 0. 结论速览

JAX 仓库不是一个"纯 Python 库"，而是**四层合一**：

| 层 | 起点路径 | 一句话职责 |
|---|---|---|
| Python 前端 | `jax/` | 用户 API、追踪器与变换、Jaxpr、域库；真正实现几乎全在 `jax/_src/` |
| C++ 扩展 | `jaxlib/` | 调用缓存、编译入口、可执行对象与设备数组；与 XLA 仓库**共同构成**一个 jaxlib |
| PJRT 插件 | `jax_plugins/` | CUDA / ROCm / oneAPI 运行时插件包，`initialize()` 注册后端 |
| 工程外壳 | `build/ ci/ tests/ docs/ benchmarks/ examples/ third_party/` + `MODULE.bazel` | Bazel 构建、CI、测试、文档、外部依赖与补丁 |

三条最值得记住的结构事实：

1. **`jax/*.py` 大多是薄壳，`jax/_src/` 才是实现**。`jax/numpy`、`jax/scipy`、`jax/nn`、`jax/ops`、`jax/image`、`jax/random.py`、`jax/dlpack.py` 等基本都是 `from jax._src... import ...` 的转出；`jax/interpreters/` 是旧路径兼容层。真正在 `jax/` 顶层就地实现的是 `jax/experimental/` 的一部分（`jax2tf`、`sparse`、`jet`、`ode`、`rnn`、`multihost_utils`、`array_serialization`、`colocated_python`、`roofline` 等）。
2. **jaxlib 的 C++ 代码分布在两个仓库**。`jax/_src/lib/__init__.py` 里有一行注释直接写明：*"Jaxlib code is split between the Jax and the XLA repositories."* JAX 特有的部分（`jax.cc` 里的 `NB_MODULE(_jax)`、`jax_jit.cc`、`pjit.cc`、`traceback.cc`、`py_executable.cc`、`pytree.cc`、`mosaic/gpu/` 与后端 kernel）在 `jax/jaxlib/`；通用的 XLA/PJRT 绑定（`xla_builder`、`hlo`、`ops`、`types`、`ifrt`、`pjrt_ifrt`、`ifrt_proxy`、`transfer`）在 `xla/xla/python/`，由 `jaxlib/BUILD` 以 `@xla//xla/python:...` 引用（`jaxlib/` 目录中 `@xla//` 出现 645 次，其中 385 次在 `jaxlib/BUILD`；149 次指向 `@xla//xla/python`）。`jaxlib/*.cc` 与 `xla/xla/python/*.cc` 的**文件名集合完全不相交**，JAX 不编译任何 `xla/python` 的 `.cc`；XLA 侧不存在编译出的 `xla_extension`，`xla/python/BUILD:60-68` 只是覆盖 `xla_extension.py` 的 `pytype_strict_library`，而该文件实际 `from jax.jaxlib import _jax`。**Mosaic TPU 方言本体在 XLA**（`xla/xla/mosaic/dialect/tpu/`），`jaxlib/mosaic/dialect/tpu/` 下 9 个文件都是 16–21 行的 `#include` 转发壳；只有 `jaxlib/mosaic/gpu/` 与 `mosaic/dialect/gpu/` 是 JAX 自有实现。
3. **整条执行链是"追踪 → Jaxpr → MLIR(StableHLO+sdy) → XLA → PJRT → 设备"**，每一段都有明确的、可定位的 Python 或 C++ 交接点（见 §7.2）。

---

## 1. 版本锚点与证据边界

| 对象 | 版本 / revision | 锚点来源 |
|---|---|---|
| JAX | `jax-v0.11.1` = `2d66622450e2c8633cda2307688ef7aa294bd6eb` | `upstream-sources.lock`（`jax-v0.11.1 release tag`） |
| jaxlib（Python 侧要求） | `_minimum_jaxlib_version = '0.11.1'` | `upstream/jax/jax/version.py:160` |
| XLA | `dcf304bc5dca1932b99f740b911dbd73631a1a69` | `upstream/jax/MODULE.bazel`（archive_override）+ `upstream-sources.lock` |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` | `xla/third_party/stablehlo/workspace.bzl` |
| Shardy | `2832731619ffb4bcc718faa4aa8054214a68c969` | `xla/third_party/shardy/workspace.bzl` |
| LLVM | `75a45c373407c13a44c7abb28a78d891a97fe665` | `xla/third_party/llvm/workspace.bzl` |
| Triton | `96bc7e783a19958182794f477d5f72f9a77d5924` | `xla/third_party/triton/workspace.bzl` |

规模实测（固定检出，`find` 计数）：

| 区域 | 文件数 | 说明 |
|---|---|---|
| `jax/` 下 Python | 668（其中 `jax/_src` 373） | 前端与实现 |
| `jaxlib/` 全部 | 322（其中 `.cc`/`.h` 209） | C++ 扩展 |
| `tests/` Python 测试 | 261 | — |
| `docs/` `.rst`/`.md` | 242 | 含 autodidax 与 internals |

**证据边界**：本文结论全部来自固定检出的**源码阅读**（`SOURCE-ONLY`）。本轮没有重新构建 jaxlib，也没有在 CPU/TPU 上重新执行；因此本文不声明任何运行时或性能结论。涉及"当前 wheel 实际行为"的内容一律不写。行号对应该 commit，跨 revision 会漂移。

---

## 2. 仓库顶层结构

```
upstream/jax/
├── jax/                # Python 包：公共 API + 全部实现（668 个 .py）
├── jaxlib/             # C++ 扩展源码与打包（322 个文件，209 个 .cc/.h）
├── jax_plugins/        # PJRT 插件包：cuda / rocm / oneapi
├── build/              # Bazel 构建入口 build.py、requirements 锁、numpy.json
├── ci/                 # CPU/CUDA/ROCm/TPU 的 Bazel 与 pytest 流水线脚本
├── tests/              # Python 测试（261 个）
├── benchmarks/         # api / linalg / math / random / tracing / mosaic 基准
├── docs/               # 文档、教程、notebook（含 autodidax）
├── examples/           # FFI、C++、k8s、ONNX、MNIST、SPMD 示例
├── cloud_tpu_colabs/   # Cloud TPU Colab 配套
├── images/             # README 用图
├── third_party/        # 补丁与外部依赖封装（xla、absl、protobuf、grpc、rocm_wheels）
├── MODULE.bazel        # Bazel 模块依赖（xla、stablehlo、shardy、triton、llvm…）
├── WORKSPACE           # 传统 WORKSPACE 入口
├── BUILD.bazel         # 根 BUILD
├── setup.py            # `jax` wheel 打包（依赖 numpy/ml_dtypes/scipy/opt_einsum）
├── build_wheel.py      # 本地 wheel 构建辅助
└── pyproject.toml      # pytest/pyrefly/ruff 配置与构建后端
```

`jax/` 包内部结构：

```
jax/
├── __init__.py         # 公共 API 汇总导出（jax/__init__.py:94 起大量 from jax._src... import）
├── _src/               # ★ 真正的实现层（373 个 .py）
├── numpy/ lax/ nn/ ops/ scipy/ random.py                    # 域 API 薄壳（顶层无 fft.py/linalg.py）
├── core.py tree_util.py dtypes.py errors.py typing.py       # 公共类型 / 兼容层
├── stages.py export.py sharding.py distributed.py ffi.py    # 公共入口
├── extend/             # 无跨版本兼容保证的扩展 API（JEP 15856）
├── experimental/       # 实验特性（194 个文件）
├── interpreters/       # 兼容 shim → jax/_src/interpreters
├── lib/ image/ tools/ example_libraries/
└── version.py          # 版本与最低 jaxlib 版本
```

---

## 3. Python 实现层 `jax/_src/`

### 3.1 内核：`core.py`（4428 行）

整个 JAX 的"语言运行时"都在这一个文件里：

| 符号 | 位置 | 职责 |
|---|---|---|
| `Trace` | `jax/_src/core.py:844` | 一次变换的主对象，管理 `Tracer` 与常量提升 |
| `Tracer` | `jax/_src/core.py:970` | 变换期值的替身；`aval` 携带抽象类型 |
| `TraceTag` | `jax/_src/core.py:1318` | 嵌套变换的层级标签 |
| `AbstractValue` | `jax/_src/core.py:1822` | 抽象值基类 |
| `ShapedArray` | `jax/_src/core.py:2439` | 形状 + dtype + 弱类型标志 |
| `Token` | `jax/_src/core.py:3065` | 有序效应令牌 |
| `Jaxpr` | `jax/_src/core.py:99` | 平台无关 IR：`invars/outvars/eqns/consts/effects` |
| `JaxprEqn` | `jax/_src/core.py:449` | 单条等式：primitive + invars + params + effects |
| `Var` / `Literal` / `DropVar` | `:528` / `:560` / `:551` | Jaxpr 原子 |
| `Primitive` | `jax/_src/core.py:659` | 算子基类，`bind()` 是追踪分派入口 |

**架构模式**：JAX 的每个算子都是一个 `Primitive`，每种变换（自动微分、批处理、降级、抽象求值…）都通过**注册规则**扩展它。这就是 `def_*` 系列的来源，也是 JAX 可扩展性的根基。

### 3.2 变换：`jax/_src/interpreters/`

| 模块 | 行数 | 关键符号 | 职责 |
|---|---|---|---|
| `partial_eval.py` | 2428 | `JaxprTrace:114`、`PartialVal:67`、`trace_to_jaxpr_nounits:351`、`partial_eval_jaxpr_nounits:655` | 把 Python 函数追踪成 Jaxpr；区分"已知常量"与"未知值"。这是 `jit` 的前端 |
| `ad.py` | 1162 | `jvp:50`、`linearize:226`、`JVPTrace:515`、`backward_pass3:299` | 前向/反向自动微分；`linearize` 是 `vjp` 与 `jacfwd` 的公共底座 |
| `batching.py` | 789 | `BatchTrace:232`、`batch:317`、`batch_jaxpr:416` | `vmap`：把追踪值带上批次维度并为每个算子做广播/规约规则 |
| `mlir.py` | 3560 | `lower_jaxpr_to_module:1327`、`lower_jaxpr_to_fun:1652`、`ModuleContext:804`、`register_lowering:1003` | **jaxpr → MLIR**。注册 StableHLO（`dialects.hlo` 是 `stablehlo` 的别名）、`sdy`、`mpmd`、`mhlo`、`chlo` 方言（`:615` 起） |
| `pxla.py` | 2106 | `lower_sharding_computation:976`、`MeshComputation:1202`、`shard_args:86`、`ExecuteReplicated:338` | 网格/分片语义：把分片后的计算降级、处理输入分片与结果还原 |
| `remat.py` | 221 | `remat`、`checkpoint` 相关 | 重计算，用于降低反向传播的峰值内存 |

注意 `jax/interpreters/xla.py` **没有** `_src` 对应物：它是旧路径兼容模块，内容只有 `apply_primitive`（转出自 `jax/_src/dispatch.py`）、`canonicalize_dtype_handlers`（转出自 `jax/_src/dtypes.py`）和 `Backend = _jax.Client`。`jax/interpreters/{ad,batching,mlir,partial_eval,pxla}.py` 则都是 `jax/_src/interpreters/*` 的纯转出。

**变换如何叠加**：`jax.jit(jax.grad(jax.vmap(f)))` 从外到内依次建立 `JaxprTrace` → `JVPTrace` → `BatchTrace` 三层 `Trace`；最内层的 `Primitive.bind` 逐层向上"穿透"每层的规则，直到抵达一个能给出抽象值的层。这是 JAX 组合性的机制来源。

### 3.3 用户 API 与 jit 实现

| 文件 | 关键位置 | 职责 |
|---|---|---|
| `jax/_src/api.py` | `jit:204`、`grad:426`、`vmap:1003`、`jvp:1405`、`linearize:1486`、`vjp:1639`、`jacfwd:706`、`jacrev:791` | 用户变换公共实现；`jit` 本身只是参数解析后转交 `pjit.make_jit` |
| `jax/_src/pjit.py` | `make_jit:446`、`JitWrapped:681`、`pjit:694`、`_cpp_pjit:254`、`_pjit_lower:1223`、`_pjit_call_impl:1182` | jit 的真实实现：分片解析、位置参数类型推断、C++ 快速路径缓存、降级与调用 |
| `jax/_src/stages.py` | `Traced:406`、`Lowering:223`、`Compiled`、`Traced.lower:532`、`Lowering.compile:244` | `jax.stages` 的对象模型：`traced → lowered → compiled` 三段式 |
| `jax/_src/compiler.py` | `compile_or_get_cached:425`、`backend_compile_and_load:331`、`get_compile_options:180` | 编译编排：缓存键、持久编译缓存、编译选项、把 MLIR 模块交给后端 |
| `jax/_src/xla_bridge.py` | `discover_pjrt_plugins:433`、`register_plugin:583`、`register_plugin_callbacks:765`、`backends:786`、`get_backend:955` | 后端发现与 PJRT 插件注册：扫描 `jax_plugins` 命名空间与 entry-point，加载 `.so` |
| `jax/_src/config.py` | 约 111 个配置项 | 全局开关，例如影响降级路径的 `use_shardy_partitioner` |

**`jit` 的两条路径**（`pjit.py`）：

- **C++ 快速路径**：`_cpp_pjit:254` 把参数展平后交给 jaxlib 的调用缓存，避免重复 Python 追踪；这是常被引用的"jax_jit 调用缓存"在 Python 侧的入口。
- **Python 追踪路径**：缓存未命中时进入 `_trace_for_jit:484`，经 `partial_eval` 得到 `Jaxpr`，再由 `stages.Traced.lower` → `_resolve_and_lower` → `_pjit_lower` → `pxla.lower_sharding_computation`。

### 3.4 分片与设备网格

| 文件 | 关键符号 | 职责 |
|---|---|---|
| `jax/_src/sharding_impls.py` | `SingleDeviceSharding:106`、`GSPMDSharding:198`、`ShardingContext:365` | 分片对象与 SPMD 轴上下文 |
| `jax/_src/mesh.py` | `Mesh:217`、`AbstractMesh:455` | 物理/抽象设备网格 |
| `jax/_src/layout.py`、`op_shardings.py`、`partition_spec.py`、`named_sharding.py` | — | 布局、算子分片规则、`P(...)` 分区规格、命名分片 |
| `jax/_src/shard_map.py`、`custom_partitioning.py` | — | 显式分片与自定义分区 |
| `jax/_src/shard_alike.py`、`mesh_utils.py` | — | 分片对齐与网格构造工具 |

分片信息在降级时写入 MLIR：`use_shardy_partitioner` 打开后，`mlir.py:1872/1946` 会写 `sdy.sharding` 属性，`:1483` 触发 `sdy-lift-inlined-meshes`（`jax/_src/interpreters/mlir.py`）。

### 3.5 域库（算子与数值）

| 目录 | 内容 |
|---|---|
| `jax/_src/lax/` | 核心 primitives 及其规则：`lax.py`（逐元素/reduction/dot/conv）、`linalg.py`（Cholesky/QR/SVD/eigh，含 `register_cpu_gpu_lowering:748`）、`parallel.py`（集合通信）、`control_flow/`、`fft.py`、`convolution.py`、`slicing.py`、`scaled_dot.py`、`special.py`、`windowed_reductions.py`、`pallas_lowerings/` |
| `jax/_src/numpy/` | NumPy 兼容前端，24 个模块：`lax_numpy.py`、`ufuncs.py`、`ufunc_api.py`、`linalg.py`、`einsum.py`、`reductions.py`、`indexing.py`、`array_methods.py`、`vectorize.py`、`polynomial.py`… |
| `jax/_src/scipy/` | SciPy 子集，47 个模块：`sparse/`、`optimize/`、`signal.py`、`stats/`、`special.py`、`ndimage.py`、`cluster/`、`integrate.py`、`fft.py`、`interpolate.py` |
| `jax/_src/nn/`、`ops/`、`image/` | 激活与注意力、`segment_*` 散点算子、图像缩放 |
| `jax/_src/random/` | PRNG：`core.py`、`prng.py`、`threefry2x32.py`、`philox2x32.py`、`rbg.py`、`stateful_rng.py` |
| `jax/_src/cudnn/` | CuDNN 融合：`fused_attention_stablehlo.py`、`scaled_matmul_stablehlo.py`、`fusion.py` |
| `jax/_src/tpu/` | TPU 专有线性代数 |
| `jax/_src/third_party/scipy/` | 从 SciPy vendor 的补丁版本（`interpolate`、`special`、`linalg`） |

### 3.6 Pallas（自定义 kernel）

`jax/_src/pallas/`（62 个文件）是 JAX 面向 kernel 作者的门面，按后端分目录：

| 路径 | 职责 |
|---|---|
| `core.py`、`pallas_call.py:1136`、`helpers.py`、`primitives.py`、`utils.py`、`cost_estimate.py` | `BlockSpec` / `GridSpec` / `MemoryRef` / `MemorySpace` 等抽象，`pallas_call` 入口，`program_id`、`load/store`、`loop`、`kernel` |
| `mosaic/` | TPU 后端：`lowering.py`、`pipeline.py`、`primitives.py`、`sc_core.py`、`sc_lowering.py`（SparseCore）、`interpret/`（CPU 上的 Mosaic 解释执行）、`tpu_info.py` |
| `mosaic_gpu/` | GPU 后端：`lowering.py`、`pipeline.py`、`primitives.py`、`pallas_call_registration.py`、`torch.py` |
| `triton/` | Triton 后端：`lowering.py`、`primitives.py`、`gpu_info.py` |
| `fuser/` | 自定义 fusion 基础设施：`fusible.py`、`jaxpr_fusion.py`、`fusion.py`、`block_spec.py`、`custom_fusion_lib.py` |
| `hlo_interpreter.py` | 在 HLO 层面解释执行（用于调试与数值对照） |

Pallas kernel 最终以 **custom call** 的形式嵌回 StableHLO；TPU 路径落到 Mosaic TPU 方言（`jax/_src/pallas/mosaic/` 只负责生成，方言本体定义在 XLA 的 `xla/xla/mosaic/dialect/tpu/`，`jaxlib/mosaic/dialect/tpu/` 是转发壳），GPU 路径落到 Mosaic GPU 方言或 Triton。

### 3.7 导出、序列化与状态

| 路径 | 职责 |
|---|---|
| `jax/_src/export/` | `_export.py`（`export`/`exported`）、`serialization.py`（+ `serialization.fbs` flatbuffer schema）、`shape_poly.py`、`shape_poly_decision.py` |
| `jax/experimental/serialize_executable.py` | `jax.stages.Compiled` 的 PJRT 可执行序列化（**不可用于不可信输入**） |
| `jax/_src/state/`、`jax/_src/ref.py` | 可变状态与内存引用：`primitives.py`、`discharge.py`、`indexing.py`、`types.py` |
| `jax/_src/checkify.py` | 函数式错误检查 |
| `jax/experimental/array_serialization/` | 多主机异步 checkpoint（TensorStore），**在 `experimental/` 内就地实现，无 `_src` 对应物** |

### 3.8 调试、剖析、集群

| 路径 | 职责 |
|---|---|
| `jax/_src/debugger/` | `cli_debugger.py`、`colab_debugger.py`、`web_debugger.py`、`core.py` |
| `jax/_src/profiler.py`、`monitoring.py`、`xla_metadata.py` | profiling、事件/标量监听、XLA metadata 注入 |
| `jax/_src/clusters/` | `slurm_cluster.py`、`k8s_cluster.py`、`mpi4py_cluster.py`、`ompi_cluster.py`、`cloud_tpu_cluster.py` |
| `jax/_src/internal_test_util/` | 测试内部工具、导出向后兼容数据（58 个文件） |

---

## 4. jaxlib：C++ 扩展层

### 4.1 与 XLA 仓库的分工

`jax/_src/lib/__init__.py` 中的原话：*"Jaxlib code is split between the Jax and the XLA repositories."* 实际边界如下：

| 归属 | 内容 | 证据 |
|---|---|---|
| `jax/jaxlib/` | JAX 特有的 C++：`jax.cc`（`NB_MODULE(_jax)`，即 `jaxlib._jax` 这个扩展模块本体）、`jax_jit`（调用缓存）、`pjit`、`traceback`、`guard_lib`、`pytree`、`py_executable`/`py_array`/`py_client`、`sharding`/`partition_spec`、`ffi`、`callback`、`config`、`cpu/`、`gpu/`、`mosaic/gpu/`、wheel 打包 | `jaxlib/BUILD` 中的目标定义 |
| `xla/xla/python/` | 通用 XLA 绑定与 IFRT：`xla_builder`、`hlo`（`_hlo`）、`ops`、`types`、`version`、`nb_numpy`/`nb_helpers`/`nb_status`/`nb_absl_*` casters、`_profiler`+`_profile_data`、`refine_polymorphic_shapes`、`ifrt`、`pjrt_ifrt`、`ifrt_proxy`、`transfer/`、`compile_only_ifrt`，以及面向 JAX 的 SPMD custom-call 分区器（`inspect_sharding`、`custom_partition_callback`，命名空间为 `jax`） | `jaxlib/BUILD:100-102, 471-481` 等处的 `@xla//xla/python:...` 依赖 |
| `xla/xla/mosaic/dialect/tpu/` | **Mosaic TPU 方言的真实实现**（`tpu_dialect.cc` 1027 行、`tpu_ops.cc` 2576 行） | `jaxlib/mosaic/dialect/tpu/*` 仅有 `#include` 转发壳 |
| 第三方 | `mlir/`（MLIR Python 绑定）、`triton/`（Triton dialect） | 目录与 `.td`/`.cc` 文件 |

三条补充判据：

1. **归属看谁定义 `NB_MODULE`，而不是看 `.so` 装在哪**。XLA 的 `pywrap.bzl:87-89` 声明 `common_lib_packages = ["jaxlib"]`，于是 XLA 编译出的模块被物理链接进 JAX 的 `libjax_common` 并随 **jaxlib wheel** 一起发布——`jaxlib._hlo`、`jaxlib._profiler`、`jaxlib._profile_data` 就是"XLA 拥有、但住在 jaxlib 包里"的例子。
2. **JAX 只做重新实现或包装，从不复制**：`jaxlib/*.{cc,h}` 与 `xla/xla/python/*.{cc,h}` 的 **basename 集合完全不相交**；两边唯一重复的是构建胶水 `pywrap.bzl` 与 `pyinit_stub.c`。反向依赖（XLA→JAX）只有 `if_google([...])` 守卫项与标记为 `# @unused` 的 `pytype_deps`，OSS 构建中**不存在环**，运行期依赖方向是单向的 jaxlib → XLA。
3. **引用规模**：`jaxlib/` 的 BUILD/bzl 中有 645 处 `@xla//...`（`jaxlib/BUILD` 385、`cuda/` 97、`mosaic/gpu/` 46、`rocm/` 35、`gpu/` 27、`oneapi/` 16…），其中 `@xla//xla/python` 149、`@xla//xla/tsl` 137、`@xla//xla/pjrt` 104、`@xla//xla/ffi` 64。另有 315 处 `@llvm-project//`、8 处 StableHLO、15 处 Shardy、3 处 Triton 边。注意 `@xla//xla/python:NAME` 这种简写标签没有结尾斜杠，用 `@xla//xla/python/` 去 grep 会漏掉约一半。

jaxlib 里也**不存在**"LLO"这一名称（`jaxlib/mosaic` 全目录无匹配）。

### 4.2 组件清单

| 组件 | 关键文件 | 职责 |
|---|---|---|
| 调用缓存 / 快速分派 | `jax_jit.cc/.h`、`pjit.cc/.h`、`reentrant_hash_map.h`、`strong_lru_cache.cc` | 缓存 Python 调用签名到可执行对象的映射，避免重复追踪 |
| 运行时客户端 | `py_client.cc/.h`、`py_executable.cc/.h`、`py_array.cc/.h`、`py_values.cc`、`py_device.cc`、`py_memory_space.cc` | `PyClient` 包装 `xla::ifrt::Client` / `xla::PjRtClient`；`Compile` / `CompileAndLoad` 接收 `mlir::ModuleOp` |
| 编译入口 | `jaxlib/xla_compiler.cc/.h`（`BuildXlaCompilerSubmodule:86`）；HLO 工具 `hlo.cc` 在 **XLA 侧**（`@xla//xla/python:_hlo`） | MLIR 模块 → XLA 编译 |
| 分片 | `sharding.cc/.h`、`partition_spec.cc/.h`、`to_ifrt_sharding.cc/.h`、`custom_call_sharding.cc` | 分片对象的 C++ 表示与 IFRT 转换 |
| PyTree | `pytree.cc/.h`、`pytree.proto` | C++ 侧 PyTree 展平/还原 |
| 防护与配置 | `guard_lib.cc`、`config.cc`、`traceback.cc` | 参数防护、配置读写、Python traceback 注入 MLIR location |
| 回调与 FFI | `callback.cc`、`ffi.cc`、`py_host_callback.cc`、`custom_call_sharding.cc` | 主机回调、外部函数接口 |
| CPU | `cpu/lapack.cc`、`cpu/sparse.cc`、`cpu/lapack_kernels*.cc`、`cpu/tridiagonal_solve_kernels.cc`、`cpu/cpu_kernels.cc` | `_lapack` 包装 SciPy 的 Cython LAPACK/BLAS capsule（60 个 LAPACK 注册项）；`_sparse` 做 Eigen `csr @ dense`；`cpu_kernels.cc` 注册平台 `"Host"`，源码注释说明它"not used by JAX itself" |
| GPU 公共层 | `gpu/vendor.h`（956 行可移植性垫片：`JAX_GPU_CUDA`/`JAX_GPU_HIP`/`JAX_GPU_ONEAPI` 三选一，其余 `#error`）、`gpu/gpu_plugin_extension.cc`、`gpu/{ffi_wrapper,handle_pool,gpu_kernel_helpers}.h`、`gpu/*_kernels.cu.cc` | 同一份 `gpu/*.cc` 被 `cuda/`、`rocm/`、`oneapi/` 三套 BUILD 用不同宏编译；`gpu/BUILD` 本身只 `exports_files` 这些共享源码 |
| CUDA / ROCm / oneAPI | `cuda/{cuda_plugin_extension.cc,versions.cc}`、`rocm/rocm_plugin_extension.cc`、`oneapi/{oneapi_gpu_runtime.cc,oneapi_plugin_extension.cc}` | CUDA 有 `_versions`（构建/运行期版本探测）与 `cuPointerGetAttribute`；ROCm 无 `_versions`，用 `hipPointerGetAttribute`，且模块名 `rocm_plugin_extension` **没有前导下划线**；oneAPI 用 SYCL 包装，`Getrf` 是 `kUnimplemented` 占位 |
| Mosaic | `mosaic/dialect/gpu/`（`mosaic_gpu.td` 1307 行）、`mosaic/gpu/`（约 4.4k LOC，5 个 pass）、`mosaic/python/{tpu,mosaic_gpu}.py`；`mosaic/dialect/tpu/` 下 9 个文件全是 16–21 行的 `#include` 转发壳 | Mosaic **GPU** 方言与降级由 JAX 自有；TPU 方言本体与 serde 在 `xla/xla/mosaic/dialect/tpu/`（`tpu_ops.td` 1761 行、`transforms/serde.cc` 711 行） |
| MLIR 绑定 | `mlir/_mlir_libs/{jax_mlir_ext,tpu_ext,mosaic_gpu_ext,triton_ext}.cc`；`mlir.cc`（`BuildMlirSubmodule`）提供 `hlo_to_stablehlo`、`xla_computation_to_mlir_module`、`mlir_module_to_xla_computation`、`serialize/deserialize_portable_artifact` | `mlir/BUILD.bazel` 本身不编译代码，全部 dialect 是到 `@llvm-project//mlir/python:*`、`@stablehlo//`、`@shardy//` 的符号链接；只有上述 4 个是 JAX 自己写的 |
| Triton | `triton/triton.td`（**只有一行 `include "triton/Dialect/Triton/IR/TritonOps.td"`**）、`triton/triton_dialect_capi.cc`、`gpu/triton{,_kernels,_support}.cc` | dialect 本体属于上游 Triton；jaxlib 的 `triton_kernels.cc` 是**运行时**，执行 XLA 降级出来的 `triton_kernel_call_ffi` custom call。**jaxlib 里没有 GEMM emitter**，那在 XLA 侧 |
| 工具 | `tools/build_wheel.py`、`tools/build_gpu_plugin_wheel.py`、`tools/build_mosaic_wheel.py`、`jax.bzl`、`pywrap.bzl` | 各 wheel 的构建与依赖规则 |

### 4.3 命名与打包要点

- **可导入名 = `NB_MODULE` 字符串 = Bazel 目标名**。OSS 的 pybind 宏里 `module_name` 参数标记为 `@unused`（`xla/xla/tsl/tsl.bzl:593`，`.so` 名取自目标名 `:624`，导出符号为 `PyInit_<target>` `:628`）。本树中二者恰好一致，但并不是因果关系。
- **两套链接方式**：`nanobind_pywrap_extension`（`jaxlib/pywrap.bzl:29-85`）把实现放进 `cc_library`，可见 `.so` 用 `pyinit_stub.c` 重新导出 `Wrapped_PyInit_<M>`，并把所有此类模块合并进唯一的 `jaxlib/jax_common`（`pywrap_library(name="jax")`，`jax_common.json` 导出 `Wrapped_PyInit_*`、本地化 `*`）；`nanobind_extension`（`jax.bzl:180-199`）则是独立 `.so`，用于各后端插件 wheel 与 `cpu_feature_guard`。
- **打包时会搬目录**：`tools/build_wheel.py:355-392` 把所有 MLIR 扩展 `.so` 移入 `jaxlib/mlir/_mlir_libs/`，所以 Bazel 目标 `//jaxlib/mlir/_mlir_libs:_jax_mlir_ext` 最终以 `jaxlib.mlir._mlir_libs._jax_mlir_ext` 被导入。各目录的 `__init__.py` 是打包时合成的（源码里 `jaxlib/mosaic/python/__init__.py`、`jaxlib/mlir/*/__init__.py` 并不存在）。
- **没有任何 wheel 捆绑 `xla_extension.so` 或 `libtpu.so`**；`_tpu_ext.so` 会随 jaxlib wheel 发布。wheel 体积预算写在 `tools/wheel_size_test.py`：jaxlib 110 MiB、CUDA 插件 20 MiB、CUDA PJRT 180 MiB、Mosaic 40 MiB。

源码中可见的三处不一致（只作记录，未做修复，也未执行构建验证）：

- `cpu/cpu_kernels.cc` 缺少 `cpu/lapack.cc:223-226` 注册的四个 `lapack_{s,d,c,z}{ormqr,unmqr}_ffi` handler。
- `gpu/linalg.cc:32-33` 的 `…_cholesky_update_ffi` 用的是 `EncapsulateFunction`（旧式 capsule）而不是 `EncapsulateFfiHandler`，与 `_ffi` 后缀不一致。
- `jaxlib/cuda/_sparse` 在源码树中没有对应的 `.pyi`（另外七个 CUDA 扩展都有）。

### 4.4 编译出的扩展模块（nanobind）

`jaxlib/` 里每个 `NB_MODULE(...)` 就是一个会进入 wheel 的扩展模块。完整清单（`grep -rn "NB_MODULE" jaxlib --include=*.cc`）：

| 模块 | 定义处 | 内容 |
|---|---|---|
| `jaxlib._jax` | `jaxlib/jax.cc:284` | **主模块**：调用缓存、PyTree、`PyClient`/`PyArray`/`PyDevice`/`PyExecutable`、sharding、FFI、guard、config、traceback |
| `jaxlib._xla` | `jaxlib/xla.cc:171` | JAX 仓库内的 XLA 相关绑定。注意 `xla_client.py:28` 内部写的是 `from jaxlib import _jax as _xla`——它是 Python 门面里的局部别名，与这个 `_xla` 扩展模块不是同一个对象 |
| `jaxlib._ifrt_proxy` / `jaxlib._pathways` / `jaxlib._sdy_mpmd` | `ifrt_proxy.cc:137` / `pathways.cc:630` / `sdy_mpmd.cc:318` | IFRT proxy 客户端、Pathways 后端、sdy/MPMD 支持 |
| `jaxlib._pretty_printer`、`jaxlib.utils`、`jaxlib.weakref_lru_cache` | `_pretty_printer.cc:928`、`utils.cc:450`、`weakref_lru_cache_module.cc:22` | IR 美化、工具、缓存 |
| `jaxlib.cpu._lapack`、`jaxlib.cpu._sparse` | `cpu/lapack.cc:270`、`cpu/sparse.cc:34` | CPU kernel |
| `jaxlib.gpu.{_hybrid,_linalg,_prng,_rnn,_solver,_sparse,_triton}`、`jaxlib.cuda._versions` | `gpu/*.cc`、`cuda/versions.cc:25` | GPU kernel 与版本探测 |
| `jax_plugins.{cuda,rocm,oneapi}.*_plugin_extension` | `cuda/cuda_plugin_extension.cc:79` 等 | 三套 PJRT 插件扩展 |
| `jaxlib.mlir._mlir_libs.{_jax_mlir_ext,_tpu_ext,_mosaic_gpu_ext,_triton_ext}` | `mlir/_mlir_libs/*.cc` | MLIR 方言扩展模块 |
| `jaxlib.mosaic.gpu._mosaic_gpu_ext` | `mosaic/gpu/mosaic_gpu_ext.cc:113` | Mosaic GPU 绑定 |

Python 侧的消费入口是 `jax/_src/lib/__init__.py`：它做版本校验（`check_jaxlib_version`）、CPU 指令集守卫（`jaxlib.cpu_feature_guard`），再按上面的清单导入；MLIR 方言则由 `jax/_src/lib/mlir/dialects/__init__.py` 装载（`sdy`、`mpmd`、`stablehlo as hlo` 等）。

---

## 5. 插件层 `jax_plugins/`

| 插件 | 关键文件 | 职责 |
|---|---|---|
| CUDA | `cuda/__init__.py`、`cuda/plugin_setup.py`、`cuda/BUILD.bazel` | `initialize():340` 定位并加载 `xla_cuda_plugin.so`（`__init__.py:56`），失败时回退到 `pjrt_c_api_gpu_plugin.so`；以 `priority=500` 注册 `'cuda'` 后端；`_check_cuda_versions` 校验 CUDA/cuDNN/NCCL 版本。该 `.so` 的构建目标在 `jax_plugins/cuda/BUILD.bazel:36-50`（目标名 `pjrt_c_api_gpu_plugin.so`），由 `tools/build_gpu_plugin_wheel.py:142-144` 改名为 `xla_cuda_plugin.so` |
| ROCm | `rocm/__init__.py` | `initialize():67` 加载 `xla_rocm_plugin.so`，注册 `'rocm'` |
| oneAPI | `oneapi/__init__.py` | `initialize():184` 加载 oneAPI 插件，注册 `'oneapi'` |

发现机制在 `jax/_src/xla_bridge.py:433 discover_pjrt_plugins()`：优先扫描 `jax_plugins` 命名空间包，其次读取 `pyproject.toml`/`setup.py` 里 `jax_plugins` 组的 entry-point；也支持环境变量 `PJRT_NAMES_AND_LIBRARY_PATHS` 与 JSON 配置。注册后由 `_init_backend`（`:899`）创建客户端。

**TPU 不在本仓库的 `jax_plugins/` 里**：TPU 通过独立的 `libtpu` wheel 提供 PJRT 插件（`jax/_src/tpu_info.py`、`cloud_tpu_init.py` 与之交互），因此 TPU 支持在源码树中体现为"调用方 + Mosaic 方言"，而非插件包本身。

---

## 6. 构建、测试、文档与 CI

| 路径 | 职责 |
|---|---|
| `MODULE.bazel` | Bazel 模块依赖：`bazel_dep("xla")` 用 `archive_override` 固定到 XLA commit；从 XLA 的 module extension 取 `llvm-project`、`stablehlo`、`shardy`、`triton`、`compute_library`、`dlpack`、`nanobind`、`gloo`；对 `grpc`/`protobuf`/`abseil-cpp`/`rules_python` 打补丁 |
| `third_party/xla/revision.bzl`、`workspace.bzl` | XLA revision 与 WORKSPACE 兼容入口 |
| `build/build.py:62-72` | wheel 目标清单：`jaxlib`、`jax-cuda{12,13}-plugin`、`jax-cuda-pjrt`、`jax-rocm-plugin`、`jax-rocm-pjrt`、`jax-oneapi-plugin`、`mosaic-gpu-cuda` |
| `build/requirements_lock_3_1x.txt`、`requirements.in` | 各 Python 版本的锁定依赖 |
| `jaxlib/tools/build_wheel.py` 等 | jaxlib / 插件 / Mosaic wheel 的实际打包逻辑 |
| `ci/` | `run_bazel_test_{cpu,cuda,rocm,tpu,oneapi}*.sh`、`run_pytest_*.sh`、`envs/`、`k8s/`、`postprocess/` |
| `tests/` | 261 个测试；`tests/BUILD` 定义 Bazel 测试目标 |
| `benchmarks/` | `api_benchmark.py`、`linalg_benchmark.py`、`tracing_benchmark.py`、`mosaic/` 等 |
| `docs/` | 官方文档源（`docs/internals/` 说明 jaxpr、常数、分片等内部机制），`autodidax*.py` 是自底向上实现 JAX 的教程 |
| `pyproject.toml` | pytest 标记与 warning 过滤、pyrefly 类型检查、ruff 配置；`setup.py:64` 声明运行时依赖 |

---

## 7. 上下游链路

### 7.1 上游依赖

| 依赖 | 角色 | 进入 JAX 的位置 |
|---|---|---|
| **XLA** (`openxla/xla`) | 编译器（HLO pass、Shardy partitioner、布局、fusion）+ PJRT/IFRT 运行时 + Python 绑定 | `MODULE.bazel:31-36` archive_override；`jax/third_party/xla/revision.bzl:24`；本地 XLA 检出 HEAD 与该 pin 一致（已核对）。JAX 自身**不直接 pin** LLVM/StableHLO/Shardy/Triton（`MODULE.bazel:56-72` 由 XLA 的 module extension 提供） |
| **StableHLO** | 可移植算子集，JAX 的 MLIR 出口方言 | `xla/third_party/stablehlo`；`jax/_src/lib/mlir/dialects` 把 `stablehlo` 别名为 `hlo` |
| **Shardy** | 张量分片传播与 `sdy` 方言 | `xla/third_party/shardy`；`mlir.py:637` 注册 `sdy` 方言，`config.use_shardy_partitioner` 控制 |
| **LLVM** | CPU/GPU 代码生成（LLVM IR → 机器码），以及 MLIR 基础设施 | `xla/third_party/llvm` |
| **Triton** | GPU kernel DSL；Pallas-Triton 后端 | `xla/third_party/triton`；`jaxlib/triton/`、`jax/_src/pallas/triton/` |
| **Mosaic** | TPU/GPU kernel 方言与降级。**TPU 方言本体在 XLA**（`xla/xla/mosaic/dialect/tpu/`），GPU 方言在 JAX（`jaxlib/mosaic/gpu/`） | `jax/_src/pallas/{mosaic,mosaic_gpu}/`、`jaxlib/mosaic/` |
| **cuDNN / cuBLAS / cuSolver / cuSPARSE / NCCL / ROCm / oneDNN / XNNPACK / Eigen / ComputeLibrary / KleidiAI / oneDNN** | 后端 kernel 与集合通信 | `jaxlib/gpu/`、`jaxlib/cpu/`、`jax_plugins/`；revision 见 `upstream-sources.lock` 的 `cpu|`、`gpu|`、`runtime|` 段 |
| **Python 运行时依赖** | `numpy>=2.1`、`ml_dtypes>=0.5.0`、`scipy>=1.15`、`opt_einsum` | `setup.py:64-70`、`jax.egg-info/requires.txt` |
| **absl / protobuf / grpc / nanobind / dlpack / gloo / flatbuffers / zlib** | C++ 基础设施 | `MODULE.bazel` bazel_dep + `third_party/*` 补丁 |

### 7.2 运行时调用链（带精确交接点）

以 `y = jax.jit(f)(x)` 为例，正常（缓存未命中）路径：

```
[用户]  jax.jit(f)(x)
  │
  ├─1. jax/_src/api.py:204            jit() 解析参数 → pjit.make_jit()
  │
  ├─2. jax/_src/pjit.py:254           _cpp_pjit() 先走 jaxlib 的 C++ 调用缓存
  │       └─ 未命中 → jax/_src/pjit.py:484  _trace_for_jit()
  │
  ├─3. jax/_src/interpreters/partial_eval.py:114   JaxprTrace
  │       └─ 每个 Primitive.bind 被追踪成 JaxprEqn；产出 core.Jaxpr
  │
  ├─4. jax/_src/pjit.py:1223          _pjit_lower()
  │       └─ jax/_src/interpreters/pxla.py:976  lower_sharding_computation()
  │             ├─ 解析 mesh / in_shardings / out_shardings / layouts
  │             └─ jax/_src/interpreters/mlir.py:1327  lower_jaxpr_to_module()
  │                    ├─ 逐算子调用注册的 lowering 规则（register_lowering:1003）
  │                    ├─ 发射 StableHLO（dialects.hlo 别名）
  │                    └─ use_shardy_partitioner 时写 sdy.sharding（:1872/:1946）
  │       产出 jax/_src/stages.py:223  Lowering / pxla.MeshComputation:1202
  │
  ├─5. stages.Lowering.compile()  jax/_src/stages.py:244
  │       └─ jax/_src/compiler.py:425  compile_or_get_cached()
  │             └─ jax/_src/compiler.py:331  backend_compile_and_load()
  │                   └─ backend.compile_and_load(module, ...)      ← Python/C++ 边界
  │
  ├─6. jaxlib/py_client.h:179         PyClient::CompileAndLoad(mlir::ModuleOp, ...)
  │       └─ 内部持有 xla::ifrt::Client / xla::PjRtClient（py_client.h:60/78）
  │
  ├─7. XLA：MLIR → HLO → HLO pass pipeline（优化/布局/fusion/Shardy 分区）
  │       └─ xla/service、xla/hlo/pass、xla/pjrt、xla/backends/{cpu,gpu,interpreter}、xla/tpu
  │
  ├─8. PJRT 插件：libtpu / xla_cuda_plugin / xla_rocm_plugin（经 xla_bridge.register_plugin:583）
  │
  └─9. 设备执行 → 结果缓冲区 → jaxlib PyArray（py_array.cc）
          └─ 交给 pxla 的 ResultsHandler（pxla.py:315）还原成 jax.Array
```

其它入口的差异：

- `jax.grad(f)`：`api.py:426` → `interpreters/ad.py:226 linearize` / `backward_pass3:299`，在 Jaxpr 层完成转置，之后可与 `jit` 叠加。
- `jax.vmap(f)`：`api.py:1003` → `interpreters/batching.py:317 batch()`，为每个 primitive 应用 batch 规则（`defbroadcasting`/`defreducer`/`defvectorized`）。
- `jax.experimental.pallas.pallas_call`：`_src/pallas/pallas_call.py:1136` → 按平台选择 `mosaic` / `mosaic_gpu` / `triton` 的 `lowering.py`，以 custom call 形式回到 StableHLO，再进入同一编译链。

### 7.3 下游消费者

JAX 仓库本身不包含这些项目，但它们定义了 JAX 的公共 API 契约（`jax/_src/api.py`、`jax/_src/numpy/`、`jax/_src/lax/` 的稳定性要求主要服务于它们）：

| 类别 | 代表项目 | 用到的 JAX 面 |
|---|---|---|
| 建模框架 | Flax (linen/nnx)、Equinox、Haiku、Keras 3 | `jax.jit`/`grad`/`vmap`、`jax.numpy`、PyTree、`jax.random` |
| 训练与优化 | Optax、Orbax、Levanter、MaxText、T5X | `jax.grad`、checkpoint（`jax/_src/ad_checkpoint.py`）、`jax.sharding` |
| 推理与服务 | SGLang-JAX、vLLM-JAX、Pathways、jax2tf/TensorFlow | `pjit` 分片、`experimental/pallas`、`experimental/{transfer,colocated_python}` |
| 生态工具 | jaxtyping、numpyro、blackjax、jaxlie | `jax.extend`、`jax.stages`、`jax.export` |
| 自定义硬件/算子 | PJRT 插件作者、Pallas kernel 作者 | `jax_plugins/`、`jax/_src/pallas/`、`jax/experimental/pallas/ops/` |

---

## 8. 组件—职责速查表

| 组件 | 路径 | 职责 | 关键接口 |
|---|---|---|---|
| 追踪内核 | `jax/_src/core.py` | Tracer/Primitive/抽象值/Jaxpr/效应 | `Primitive.bind`、`Jaxpr` |
| 部分求值 | `jax/_src/interpreters/partial_eval.py` | 函数 → Jaxpr | `trace_to_jaxpr_nounits` |
| 自动微分 | `jax/_src/interpreters/ad.py` | JVP/VJP/linearize | `jvp`、`linearize`、`backward_pass3` |
| 批处理 | `jax/_src/interpreters/batching.py` | vmap | `batch`、`BatchTrace` |
| 降级 | `jax/_src/interpreters/mlir.py` | jaxpr → StableHLO(+sdy) | `lower_jaxpr_to_module` |
| 分片执行 | `jax/_src/interpreters/pxla.py` | mesh/分片计算 | `lower_sharding_computation` |
| 公共变换 | `jax/_src/api.py` | jit/grad/vmap/jvp/vjp/jac* | — |
| jit 实现 | `jax/_src/pjit.py` | 分片推断、C++ 快速路径 | `make_jit`、`_pjit_lower` |
| 阶段对象 | `jax/_src/stages.py` | traced/lowered/compiled | `Traced`、`Lowering`、`Compiled` |
| 编译编排 | `jax/_src/compiler.py` | 编译缓存与选项 | `compile_or_get_cached` |
| 后端桥接 | `jax/_src/xla_bridge.py` | 后端/插件注册与发现 | `register_plugin`、`get_backend` |
| 配置 | `jax/_src/config.py` | 约 111 个全局开关 | `jax.config` |
| 域库 | `jax/_src/{lax,numpy,scipy,nn,ops,random}/` | 算子与数值前端 | `jax.numpy.*`、`jax.lax.*` |
| Pallas | `jax/_src/pallas/` | 自定义 kernel（TPU/GPU/Triton） | `pallas_call`、`BlockSpec` |
| 导出 | `jax/_src/export/` | StableHLO 导出与序列化 | `jax.export.export` |
| 状态 | `jax/_src/state/`、`ref.py` | 可变状态与内存引用 | `ref`、`discharge` |
| C++ 调用缓存 | `jaxlib/jax_jit.cc`、`pjit.cc` | 缓存 Python 调用签名 | `PyClient` 前的分派 |
| C++ 运行时 | `jaxlib/py_client.cc`、`py_executable.cc` | PJRT/IFRT 客户端与可执行 | `Compile`、`CompileAndLoad` |
| C++ 数组 | `jaxlib/py_array.cc`、`py_device.cc` | 设备数组与设备对象 | `jax.Array` 的 C++ 侧 |
| CPU kernel | `jaxlib/cpu/` | LAPACK/稀疏/三对角 | custom call |
| GPU kernel | `jaxlib/gpu/`、`cuda/`、`rocm/`、`oneapi/` | 厂商库绑定与 kernel | 插件扩展 |
| Mosaic 方言 | `jaxlib/mosaic/` + `xla/xla/mosaic/dialect/tpu/` | GPU 方言在 jaxlib；TPU 方言在 XLA | `.td` + pass |
| MLIR 绑定 | `jaxlib/mlir/` | 方言装载与扩展模块 | `_jax_mlir_ext` |
| 插件包 | `jax_plugins/{cuda,rocm,oneapi}/` | PJRT 插件入口 | `initialize()` |
| 构建 | `build/`、`MODULE.bazel`、`*.bzl` | wheel 与依赖 | `build.py` |
| 测试/文档 | `tests/`、`docs/`、`benchmarks/` | 验证与说明 | — |

---

## 9. 图表说明

- **`figures/jax-repo-structure.svg`**：源码结构树。每个节点都是固定检出中真实存在的路径；右侧为该路径的职责；左侧色条区分 Python 实现 / C++ / 目录 / 构建测试 / 外部依赖。
- **`figures/jax-repo-pipeline.svg`**：上下游链路。自上而下为「上游依赖 → JAX Python API → 追踪与变换 → IR 与降级 → jaxlib C++ → XLA → 硬件 → 下游消费者」，框内标注关键源码位置。

两张图由 `make_diagrams.py` 生成，可复现：

```bash
python3 -B research/jax-repo/make_diagrams.py
```

## 10. 如何复核本文结论

```bash
# 1) 确认固定 revision 与依赖锚点
git -C upstream/jax rev-parse HEAD
python3 -B tools/sync-environment.py check

# 2) 结构树里的每个路径 token 都在固定检出中真实存在
python3 -B research/jax-repo/verify_structure.py
# -> checked 148 path tokens under upstream/jax
# -> all tokens resolved

# 3) 关键交接点仍在该位置
grep -n "def lower_jaxpr_to_module" upstream/jax/jax/_src/interpreters/mlir.py
grep -n "def compile_or_get_cached"  upstream/jax/jax/_src/compiler.py
grep -n "CompileAndLoad"             upstream/jax/jaxlib/py_client.h
grep -rn "@xla//xla/python"          upstream/jax/jaxlib/BUILD | head

# 4) 两个仓库的 C++ 文件名集合不相交
comm -12 <(ls upstream/jax/jaxlib/*.cc | xargs -n1 basename | sort) \
         <(ls upstream/xla/xla/python/*.cc | xargs -n1 basename | sort)
```

**已知不确定项**（保留为待验证，不写成结论）：

- `jax/experimental/array_serialization/` 在 `jax/_src/` 无对应物，是"就地实现"还是迁移中间态，未从 git 历史确认。
- `jax/scipy/spatial/` 与 `jax/_src/scipy/spatial/` 的 `__init__.py` 均为空，**只确认"为空"，未确认是否有意为之**。
- `jax/experimental/mosaic/gpu/` 的完整公开名单未逐一展开。
- CUDA/ROCm/TPU/xprof/tensorstore 相关分支依赖运行时环境，本轮未执行验证。
- `jaxlib` 与 XLA 的分工由 `BUILD` 依赖与目录对比得出；未做全量符号级追踪。
