# 普通 JAX/XLA 到 TPU 的编译路径

本文追踪由普通 JAX primitive 组成的函数。Pallas kernel 使用另一条 lowering，见
[Pallas/Mosaic TPU 路径](02-pallas-mosaic-tpu-path.md)。两条路径会在外层
StableHLO、PJRT 和 TPU runtime 处相遇。第一次遇到术语时可同时查看
[JAX→TPU 术语表](../index/glossary.md)。

## 边界与证据

公开源码中的控制流和类型标记为 `SOURCE-ONLY`。libtpu 内部的候选阶段必须用固定
target 的编译产物升级为 `COMPILE-TPU`；设备执行、通信和性能只能用 `RUN-TPU`
确认。当前 CPU wheel 与固定源码存在 `VERSION-SKEW`，所以本文不把本机导入 JAX
当作固定 XLA commit 的运行证据。证据等级与来源限定见
[证据约定](../contributing/evidence-conventions.md)。

## 冷编译调用链

```mermaid
sequenceDiagram
  participant U as Python caller
  participant J as JAX transformations
  participant M as JAX MLIR lowering
  participant C as JAX compiler/cache
  participant P as jaxlib / IFRT / PJRT C API
  participant T as libtpu compiler
  participant D as TPU runtime/device

  U->>J: call jitted function with pytrees
  J->>J: flatten, abstractify, trace, build transformed Jaxpr
  J->>M: lower Jaxpr for platform tpu
  M-->>C: StableHLO module + sharding metadata
  C->>C: build CompileOptions and query caches
  C->>P: compile_and_load(module, devices, options)
  P->>T: PJRT_Client_Compile / plugin ABI
  Note over T: internal stages require COMPILE-TPU evidence
  T-->>P: loaded executable metadata/handle
  P-->>C: LoadedExecutable
  C-->>J: cached executable and call handlers
  J->>P: execute with device arrays/buffers
  P->>D: enqueue program and transfers
  D-->>P: events / result buffers
  P-->>U: jax.Array; readiness may remain asynchronous
```

这是代表性的 cache miss 路径，不表示所有 API 调用都立即完成全部阶段：
`jax.jit(f)` 创建包装 callable；显式 `lower()` 可以只完成 tracing/lowering；
`compile()` 可以提前构建 executable；缓存命中可以跳过若干阶段；读取 host value 或
`block_until_ready()` 才会形成明确同步点。相应公开入口位于
[api.py](../../upstream/jax/jax/_src/api.py)、
[pjit.py](../../upstream/jax/jax/_src/pjit.py)、
[stages.py](../../upstream/jax/jax/_src/stages.py)和
[compiler.py](../../upstream/jax/jax/_src/compiler.py)。
**证据：SOURCE-ONLY。** 精确 cache key 和异步行为仍需匹配二进制 probe 才能升级为
`RUN-CPU` 或 `RUN-TPU`。

## 1. Python callable 到 transformed Jaxpr

`jit`、`grad` 和 `vmap` 都返回 callable transformation。它们能够嵌套，不是因为
三个工具按固定顺序直接改写 Python AST，而是因为 primitive operation 可由当前
`Trace` 解释，且 AD、batching、partial evaluation/lowering 等层各自提供规则。这个
组合性有契约条件：路径上的 primitive 必须具备相应 transformation/lowering rule，
并满足 shape、effect、sharding 等合法性约束；缺少规则或违反约束会报错。嵌套顺序也
会改变哪个 trace 先解释 primitive，以及产生的 Jaxpr、staging 和 cache 行为。

```mermaid
flowchart LR
  PY["Python function"] --> WRAP["jit / grad / vmap wrappers"]
  WRAP --> TRACE["Tracer values execute Python function"]
  TRACE --> BIND["Primitive.bind"]
  BIND --> RULES["active Trace and primitive rules"]
  RULES --> JAXPR["typed transformed Jaxpr"]
```

固定源码锚点：

- `Primitive`、`Trace`、`Tracer` 和 `Jaxpr`：
  [core.py](../../upstream/jax/jax/_src/core.py)；
- tracing/staging 与 partial evaluation：
  [partial_eval.py](../../upstream/jax/jax/_src/interpreters/partial_eval.py)；
- AD rules： [ad.py](../../upstream/jax/jax/_src/interpreters/ad.py)；
- batching rules： [batching.py](../../upstream/jax/jax/_src/interpreters/batching.py)。

这些文件证明解释器与 rule registry 的结构。具体组合产生的 trace stack、Jaxpr 和
cache 行为仍要由后续实验记录。**证据：SOURCE-ONLY。**

## 2. Jaxpr 到 StableHLO

[mlir.py](../../upstream/jax/jax/_src/interpreters/mlir.py)中的
`lower_jaxpr_to_module` 建立 module context，并针对 equation 的 primitive 查找
lowering rule。普通数值 primitive 的 rules 分布在 `jax/_src/lax` 等模块；它们
构造 StableHLO 或其他受支持 dialect 的 operations。
**证据：SOURCE-ONLY。**

这一边界的输入输出应记录为：

| 输入 | 输出 | 必须保留的上下文 |
|---|---|---|
| `Jaxpr`（`ClosedJaxpr` 在本版本只是兼容别名）、avals、effects | MLIR module，主要 payload 为 StableHLO | consts、lowering platforms、axis context、source locations、donation/alias 信息 |
| logical sharding | `sdy` 或兼容 sharding attributes/ops | mesh、manual/auto axes、global/local shape |
| tokens/effects | token values、ordered/unordered effect metadata | effect ordering 和 host callback 信息 |

StableHLO 的 operation/type 定义位于
[stablehlo/dialect](../../upstream/stablehlo/stablehlo/dialect/)。StableHLO 是这一
公开边界上的可移植 IR，不等于 XLA 内部的 `HloModule`，也不等于 TPU LLO。
**证据：SOURCE-ONLY。**

### Sharding 支路

`Mesh`、`PartitionSpec`、`NamedSharding`、`jit`/`pjit` sharding、`pmap` 和
`shard_map` 的 Python 入口并不代表六套完全独立的设备后端。它们建立不同的用户
语义和 axis context，随后把分片信息带入 lowering 与 compiler pipeline。
源码入口包括
[sharding_impls.py](../../upstream/jax/jax/_src/sharding_impls.py)、
[pxla.py](../../upstream/jax/jax/_src/interpreters/pxla.py)、
[shard_map.py](../../upstream/jax/jax/_src/shard_map.py)和
[pmap.py](../../upstream/jax/jax/_src/pmap.py)。
**证据：SOURCE-ONLY。**

启用 Shardy 的路径使用 `sdy` dialect 表达 mesh、tensor sharding、constraint 和
data-flow edge，再通过 import、propagation、export 等 passes 进入后续编译。
定义和 passes 分别位于
[sdy IR](../../upstream/shardy/shardy/dialect/sdy/ir/)与
[sdy transforms](../../upstream/shardy/shardy/dialect/sdy/transforms/)。XLA 的公开
集成代码在
[xla/service/spmd/shardy](../../upstream/xla/xla/service/spmd/shardy/)。
**证据：SOURCE-ONLY。**

公开源码不能证明目标 libtpu 对这一固定 Shardy 版本采用哪条 import/export、SPMD
或 fallback 路径。必须归档 pre/post partition HLO 才能确认。
**待验证：COMPILE-TPU。** collective 的实际 route、重叠和 ICI 时序只能由 profile
确认。**待验证：RUN-TPU。**

## 3. JAX 编译入口到 PJRT ABI

[compiler.py](../../upstream/jax/jax/_src/compiler.py)的
`backend_compile_and_load` 把 MLIR module、executable devices、compile options 和
host callbacks 交给 backend。[xla_bridge.py](../../upstream/jax/jax/_src/xla_bridge.py)
的 `make_tpu_client` 在 `tpu` plugin 尚未加载时动态加载 `libtpu.so`，随后按需初始化
plugin，再创建 C API client。**证据：SOURCE-ONLY。**

公开/私有切口可以进一步定位为：`make_tpu_client` 调用 jaxlib binding；公开
`pjrt_api.cc` 负责 `dlopen` 并解析 plugin 导出的 `GetPjrtApi`；公开 generic C API
client 序列化 `PJRT_Program`/compile options 并调用函数表中的
`PJRT_Client_Compile`。函数表背后的 TPU compiler/runtime 实现由匹配的
`libtpu.so` 提供，当前不在公开源码中。

公开 ABI 的关键对象和调用在：

- [PJRT C API](../../upstream/xla/xla/pjrt/c/pjrt_c_api.h)：`PJRT_Program`、
  `PJRT_Client_Compile`、loaded executable、buffer 与 event；
- [PJRT C API client](../../upstream/xla/xla/pjrt/c_api_client/)：C++ client 到 C ABI
  argument struct 的适配；
- [PJRT plugin loader](../../upstream/xla/xla/pjrt/pjrt_api.cc)：动态库加载和
  `GetPjrtApi` 符号解析；
- [公开 TPU C++ adapter](../../upstream/xla/xla/pjrt/plugin/xla_tpu/)：另一条构造
  TPU C API client 的公开入口，并非 JAX Python 路径的必经节点；
- [TPU C ABI glue](../../upstream/xla/xla/tpu/)：公开声明、初始化和类型转换边界；
- [jaxlib py_client.cc](../../upstream/jax/jaxlib/py_client.cc)与
  [py_executable.cc](../../upstream/jax/jaxlib/py_executable.cc)：Python object 与
  IFRT executable/array 的绑定。

PJRT 定义编译和执行的跨 plugin 接口，并不公开 TPU compiler 的内部 pass
pipeline。`PJRT_Client_Compile` 被调用只能证明程序跨过 ABI；不能据此命名 libtpu
内部类或推断 HLO-to-LLO 顺序。**证据：SOURCE-ONLY。**

## 4. libtpu 内部：待验证的编译问题

下面是要拿匹配 libtpu 源码和 dumps 回答的问题，不是本文已经证明的实现：

```mermaid
flowchart LR
  IN["PJRT compile payload"] --> IMPORT["StableHLO/HLO import?"]
  IMPORT --> OPT["HLO optimization and sharding?"]
  OPT --> LAYOUT["layout / fusion / scheduling / memory?"]
  LAYOUT --> LOWER["TPU lowering?"]
  LOWER --> LLO["LLO schema and passes?"]
  LLO --> PACK["bundle packer / target encoding?"]
```

每个问号都要求以下 `COMPILE-TPU` 证据：固定 libtpu build ID/source commit、target
generation/topology、完整 flags、输入 payload、pass trace、阶段性 IR、输出 executable
fingerprint。没有这些产物时，只能说“JAX 通过 PJRT 请求 TPU 编译”。

## 5. 用 pass 修改“模型拓扑”的公开入口

本基线包含
`jax.extend.xla.register_hlo_module_transformation`。callback 接收序列化的
`HloModuleProto`，返回修改后的 proto 或 `None`；当前 API 暴露 pre-scheduler 与
post-scheduler 两个位置。

调用链源码为：

```mermaid
flowchart LR
  API["jax.extend.xla"] --> PY["jax._src.xla_transform"]
  PY --> BIND["jaxlib/xla.cc"]
  BIND --> EXT["PJRT XlaTransform extension"]
  EXT -->|"if extension supported"| TPU["TPU plugin callback"]
```

- Python API： [jax/extend/xla.py](../../upstream/jax/jax/extend/xla.py)；
- 注册逻辑： [xla_transform.py](../../upstream/jax/jax/_src/xla_transform.py)；
- nanobind 与 proto round-trip： [jaxlib/xla.cc](../../upstream/jax/jaxlib/xla.cc)；
- PJRT extension ABI：
  [pjrt_c_api_xla_transform_extension.h](../../upstream/xla/xla/pjrt/c/pjrt_c_api_xla_transform_extension.h)；
- CPU/TPU 示例、`sin → cos` 与 async scheduler 示例：
  [xla_transform_test.py](../../upstream/jax/tests/xla_transform_test.py)。

接口和测试存在于固定源码。**证据：SOURCE-ONLY。** 当前 wheel 的运行 probe 应标为
`RUN-CPU` 并附 `VERSION-SKEW` qualifier；完全匹配的 binding 需要 source-built wheel
验证。TPU plugin 是否支持该 extension、callback 插入其内部 pipeline 的准确位置及
修改后的最终 HLO 必须由目标 libtpu 验证。**待验证：COMPILE-TPU。**

在这个语境中，“修改模型拓扑”指改变 `HloModule` 的 computation/instruction graph，
或在 post-scheduler 修改合法 schedule。它不表示改变物理 TPU pod 拓扑。pass 至少要
维护 shape、dtype、layout、schedule、sharding、alias/donation、effects 和数值契约。

本版本的测试还明确规避 persistent compilation cache，因为 transformation 不属于
其 cache key。pass 实验应注册 transformation 后强制新编译，并同时保存 cache 状态与
before/after HLO。**证据：SOURCE-ONLY。** 这是当前 commit 的行为，不能外推到其他
JAX/libtpu 版本。

## 6. executable 与设备执行

编译成功后，JAX 持有 loaded executable 和输入/输出 handlers；调用时通过 jaxlib/
IFRT/PJRT 传递 arrays/buffers 并取得 events/results。公开 Python/C++ 生命周期可从
[stages.py](../../upstream/jax/jax/_src/stages.py)、
[py_executable.cc](../../upstream/jax/jaxlib/py_executable.cc)和
[PJRT C API](../../upstream/xla/xla/pjrt/c/pjrt_c_api.h)追踪。
**证据：SOURCE-ONLY。**

下列结论不能离线取得：真实 launch 顺序、host/device overlap、HBM/VMEM placement、
DMA stalls、collective route、数值结果和 kernel 性能。**待验证：RUN-TPU。**

## 离线验证矩阵

| 问题 | 无 TPU 可做的工作 | 不能据此声称 | 目标证据 |
|---|---|---|---|
| transformation 组合 | 运行/检查 Jaxpr、avals、effects | TPU compilation 行为相同 | `RUN-CPU`，之后 `RUN-TPU` |
| TPU 定向前端 IR | 使用 export/lowering probe 生成平台定向 StableHLO | libtpu 接受或生成某种 LLO | `SOURCE-ONLY`/未来 probe |
| 分片语义 | 多 CPU device、AbstractMesh、sdy IR/pass 单测 | 真实 TPU collective 路径 | `RUN-CPU`，之后 `RUN-TPU` |
| HLO graph pass | source-built CPU jaxlib 上修改并比较 HLO/数值 | TPU plugin 支持同一 hook | `RUN-CPU`，之后 `COMPILE-TPU` |
| PJRT 生命周期 | CPU plugin 或 sample plugin 追踪 ABI | libtpu 内部线程、缓存和执行模型 | `RUN-CPU` |
| TPU HLO pipeline | 阅读共享 XLA passes、定义 capture schema | 私有 pass 顺序和 target decisions | `COMPILE-TPU` |
| LLO/bundle | 预先定义所需 manifest 与 lineage 字段 | 任何具体 opcode/encoding | `COMPILE-TPU` |
| 性能和通信 | 建立 workload/golden/profile 脚本 | TPU latency、bandwidth 或 overlap | `RUN-TPU` |

下一步做任何结构性 pass 时，应先在 CPU 上闭合“匹配 → rewrite → verifier → dump →
数值 → cache → rollback”，再把同一输入和断言迁移到固定 TPU target。
