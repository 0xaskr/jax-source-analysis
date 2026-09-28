# JAX 内部软件栈：概念、lowering 与下游接口

本文与[整合软件栈图](jax-internal-stack.svg)一起，从 JAX Python 源码解释值、变换、追踪、Jaxpr、lowering，以及 JAX 调用 jaxlib 和下游 XLA 的边界。它属于[软件栈开放探索](../index.md)。源码基线为 `jax-v0.11.1`（`2d66622450e2`）；本地 [`upstream/jax`](../../../upstream/jax) 检出的是基于该 tag 的源码注释提交 `3d1d3e8963b4`，以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。文中的行号指向当前检出，不能直接用于原始 tag。

## 一条阅读主线

```text
Python 函数 + jax.Array / PyTree
  → jit、grad、vmap 等变换影响函数如何被处理
  → Trace 解释操作；Tracer 代表追踪中的值；aval 描述值的抽象类型
  → Primitive.bind 触发操作；abstract_eval 推导输出 aval 与 effects
  → Jaxpr 用 Var / Literal / JaxprEqn 记录程序
  → 按 lowering 上下文、平台与规则表生成目标 IR
  → jaxlib / IFRT / PJRT / XLA 等继续编译、加载和执行
```

这条线用于定位概念，不表示每次 `jit`、`grad`、`vmap` 调用都立刻生成一个独立 Jaxpr，也不表示一个 Python API 恰好对应一个 primitive。例如 `jnp.matmul` 会处理输入类型、维度和批次，再调用 `lax.dot_general`。[`matmul` 源码](../../../upstream/jax/jax/_src/numpy/tensor_contractions.py#L138)

## 核心概念

| 概念 | 它负责什么 | 源码入口 |
|---|---|---|
| `jax.Array` | 面向用户的数组值，具有 shape、dtype 和 sharding，并与设备数据关联。追踪时的 Tracer 可以代表数组值，但不是实际数组对象。 | [`Array` / `ArrayImpl`](../../../upstream/jax/jax/_src/array.py#L185) |
| PyTree | 展开、重建列表、字典、元组等嵌套输入输出；它组织数据结构，不是计算 IR。 | [`tree_flatten` / `tree_unflatten`](../../../upstream/jax/jax/_src/tree_util.py#L88) |
| Sharding / Mesh | 描述逻辑数组如何分布到设备；`Mesh` 给设备轴命名，`NamedSharding` 等把分片与 mesh 关联。 | [`Mesh`](../../../upstream/jax/jax/_src/mesh.py#L222)、[`NamedSharding`](../../../upstream/jax/jax/_src/named_sharding.py#L78) |
| `jit` / `grad` / `vmap` | 分别组织暂存编译、自动微分与批量化。对 primitive 的微分、批量化处理另有规则。 | [`api.py`](../../../upstream/jax/jax/_src/api.py#L208) |
| Trace | 处理 primitive 调用的解释器对象。不同的 Trace 可以对同一操作作不同解释；它不是输入值。 | [`Trace`](../../../upstream/jax/jax/_src/core.py#L848) |
| Tracer | 追踪期间代表一个值，关联 Trace 与 aval。在动态 Jaxpr 追踪中，输入 Tracer 的 `.val` 对应一个 `Var`。 | [`Tracer`](../../../upstream/jax/jax/_src/core.py#L974)、[`DynamicJaxprTracer`](../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1365) |
| aval（`AbstractValue`） | 描述值的抽象类型。常见的 `ShapedArray` 包含 shape、dtype 等信息，不能只把 aval 理解成 shape。 | [`AbstractValue`](../../../upstream/jax/jax/_src/core.py#L1825)、[`ShapedArray`](../../../upstream/jax/jax/_src/core.py#L2437) |
| Primitive | 基础操作，例如 `add`、`dot_general`、`pallas_call`。`abstract_eval` 从输入 aval 推导输出 aval；执行、微分、批量化和 lowering 的实现围绕 primitive 组织。 | [`Primitive.bind`](../../../upstream/jax/jax/_src/core.py#L663) |
| Jaxpr | 追踪得到的程序表示，保存输入输出、常量、方程及 effects。一个方程的参数可以包含内层 Jaxpr。最终 Jaxpr 不需要保留 Trace 或 Tracer 实例。 | [`Jaxpr`](../../../upstream/jax/jax/_src/core.py#L105) |
| `Var` / `Literal` / `JaxprEqn` | `Var` 标识程序中的值并携带 aval；`Literal` 可保存已知常量的值；方程记录输入、输出、primitive、参数和 effects。 | [`Var`](../../../upstream/jax/jax/_src/core.py#L532)、[`Literal`](../../../upstream/jax/jax/_src/core.py#L564)、[`JaxprEqn`](../../../upstream/jax/jax/_src/core.py#L453) |
| `Traced` / `Lowered` / `Compiled` | JIT 流程暴露的阶段对象：已追踪、已 lowering、已编译。它们与设备已经执行是不同的事实。 | [`stages.py`](../../../upstream/jax/jax/_src/stages.py#L412) |

在 `x + y` 的追踪中，`x`、`y` 是 Tracer；它们的 aval 提供抽象类型。Trace 处理 primitive 调用，利用该 primitive 的抽象求值规则得到输出 aval，并构造 Jaxpr 方程。最终 Jaxpr 中代表输入与结果的是 `Var`，而 `add` 是方程引用的 primitive。[`DynamicJaxprTrace.default_process_primitive`](../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1751)

**抽象求值与 lowering 是两类规则。**前者回答输出的抽象类型和 effects；后者回答在当前平台及上下文中生成什么 IR。普通 MLIR lowering 有公共规则与平台专用规则。[`register_lowering`](../../../upstream/jax/jax/_src/interpreters/mlir.py#L1003)

本版本还使用 **HiJAX / LoJAX** 区分部分高层与低层的 JAX 类型、primitive：当 Trace 要求低层表示时，高层 primitive 可通过 `to_lojax` 展开。这是 JAX 内部的表示转换，不能把它直接理解成“选择 StableHLO 或 Mosaic”。`pallas_call_p` 在当前源码中就是 `HiPrimitive`。[`Primitive.bind_with_trace`](../../../upstream/jax/jax/_src/core.py#L732)、[`HiPrimitive`](../../../upstream/jax/jax/_src/hijax.py#L75)、[`pallas_call_p`](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L72)

追踪与编译还会用到**部分求值**：`PartialVal` 区分已知常量与只有 aval 的未知值，帮助决定哪些计算可立即处理、哪些需要留在暂存程序中。这与“Tracer 是否持有实际数组元素”不是同一个问题。[`PartialVal`](../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L67)

## Pallas 增加的抽象

Pallas 的 kernel 首先是用户编写的 Python 函数。`pallas_call` 使用 Ref 类型的参数追踪它，得到**内层 kernel Jaxpr**；外层 Jaxpr 有一条 `pallas_call` 方程，参数包含内层 Jaxpr 和 `grid_mapping`。内外两层使用同一种 Jaxpr 结构，区别在方程、参数、输入类型和 effects。[kernel 追踪](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L788)、[外层绑定](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L1112)

| 概念 | 作用 |
|---|---|
| `Ref` / `AbstractRef` | 表示 kernel 内可读写的位置。`x_ref[...]` 读值，`out_ref[...] = value` 写值；Ref 与普通数组值的语义不同。[`AbstractRef`](../../../upstream/jax/jax/_src/state/types.py#L434) |
| `get` / `swap` / effects | 读写仍由 Jaxpr primitive 表达。`swap` 有 `WriteEffect`；即使它返回的旧值无人使用，也不能只按“输出 Var 未使用”删除写回。[`swap`](../../../upstream/jax/jax/_src/state/primitives.py#L181)、[DCE 规则](../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1241) |
| `GridSpec` / `BlockSpec` / `GridMapping` | 描述调用网格、每次调用看到的块以及映射。`BlockSpec` 可指定 `memory_space`；单看一条 `swap` 方程不能确定最终写 HBM 还是 VMEM。[`BlockSpec`](../../../upstream/jax/jax/_src/pallas/core.py#L558)、[`GridMapping`](../../../upstream/jax/jax/_src/pallas/core.py#L946) |

“kernel”在不同语境下可能指 Python 函数、内层 kernel Jaxpr、Mosaic TPU MLIR，或后端生成的设备代码。普通 `jnp.matmul` 没有 Pallas 的内层 kernel Jaxpr，后端仍可能为它生成设备代码或选择库调用。

## 两个 Jaxpr 示例

下面两个代码块是按固定版本源码整理的**精简结构示例**：保留输入/输出类型、关键方程和 Pallas 的内外层关系；`...` 表示省略的静态参数或方程，并非逐字的 `jax.make_jaxpr` 输出。具体变量名、参数及附加方程取决于追踪上下文。仓库另有一个[实际保存的 batched matmul Jaxpr](../../call-to-llo/matmul/matmul-lowering.ipynb)；它的输入形状和精度选项与此处的二维教学示例不同。

普通计算取 `jnp.matmul(a, b)`，其中 `a: f32[2,3]`、`b: f32[3,4]`。核心方程如下；`dimension_numbers` 的收缩轴是左输入的第 1 维和右输入的第 0 维：

```text
{ lambda ; a:f32[2,3] b:f32[3,4]. let
    c:f32[2,4] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      ...
    ] a b
  in (c,) }
```

Pallas 取一个逐元素加法 kernel，其 Python 形式是 `out_ref[...] = x_ref[...] + y_ref[...]`，输入和输出均为 `f32[8,128]`，`grid=()`，使用默认的整数组 `BlockSpec`。外层 Jaxpr 的 `pallas_call` 方程把内层 kernel Jaxpr 作为参数；为便于并排阅读，下面把嵌套的 `jaxpr=` 展开成命名的 `kernel`：

```text
外层：{ lambda ; a:f32[8,128] b:f32[8,128]. let
          c:f32[8,128] = pallas_call[
            jaxpr=kernel  grid_mapping=...  ...
          ] a b
        in (c,) }

kernel：{ lambda ; x:Ref{f32[8,128]} y:Ref{f32[8,128]}
                   o:Ref{f32[8,128]}. let
            u:f32[8,128] <- x[]
            v:f32[8,128] <- y[]
            w:f32[8,128] = add u v
            o:Ref{f32[8,128]}[] <- w
          in () }
```

内层的 `x[]`、`y[]` 是 `get` 读，`o[] <- w` 对应有写入 effect 的 `swap`；输出通过 Ref 写入，kernel 函数本身不返回结果值。[`get` / `swap` 的 Jaxpr 打印形式](../../../upstream/jax/jax/_src/state/primitives.py#L65)、[`pallas_call_p.bind` 的 `jaxpr` 与 `grid_mapping` 参数](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L1112)、[`pallas_call` 默认 grid/BlockSpec](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L1137)

## 普通 JAX 与 Pallas TPU 的表示路径

| 调用 | Jaxpr 中是什么 | 本地源码能确认的后续表示 |
|---|---|---|
| 二维 `jnp.matmul(a, b)` | 普通 Jaxpr 的核心方程为 `dot_general`；没有 Pallas kernel Jaxpr。 | 常规 lowering 为外层 `stablehlo.dot_general`。[`matmul`](../../../upstream/jax/jax/_src/numpy/tensor_contractions.py#L273)、[规则注册](../../../upstream/jax/jax/_src/lax/lax.py#L6294) |
| TPU、`interpret=False` 的 `pallas_call` | 外层 `pallas_call[jaxpr=内层 kernel Jaxpr, grid_mapping=...]`；内层可含 `get`、`add`、`swap`。 | 内层按 Mosaic TPU 规则 lowering 成 Mosaic TPU MLIR；序列化后放入外层 `stablehlo.custom_call` 的配置。[平台分支](../../../upstream/jax/jax/_src/pallas/pallas_call.py#L845)、[kernel lowering](../../../upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L435)、[配置序列化](../../../upstream/jax/jax/_src/tpu_custom_call.py#L249) |

`pallas_call` 在 `interpret=True`、CPU、GPU 与 TPU 上有不同分支，并非一律生成 Mosaic TPU MLIR。对 TPU 路径，**Mosaic TPU MLIR 不是 LLO，也不是最终设备可执行代码**。仅有 Jaxpr、StableHLO 或 Mosaic TPU MLIR，不能断定对应的 TPU 编译或执行已发生；每个阶段需要自己的产物与版本证据。固定调用的逐层材料见 [`call-to-llo`](../../call-to-llo/README.md)。

## JAX 与 jaxlib 的分界及下游调用

**边界是 Python 包与 native/绑定的所有权边界，不是 Jaxpr 之后唯一的一条时间线。**`jax` 的源码通过 [`jax._src.lib`](../../../upstream/jax/jax/_src/lib/__init__.py#L86) 导入 `jaxlib`；本仓库的 [`upstream/jax/jaxlib`](../../../upstream/jax/jaxlib) 有部分 jaxlib 源码，相关 XLA 实现还位于单独锁定的 [`upstream/xla`](../../../upstream/xla)。JAX 至少在构造 MLIR、提交编译以及执行/数组包装时跨过这条边界。

| 流程 | JAX 侧 API | 下游组件与 API | 输入 → 输出 | 源码 |
|---|---|---|---|---|
| 获得设备后端 | `xla_bridge.get_backend(platform)` | 经 `jaxlib.xla_client` 得到 `Client`；TPU 可以注册 PJRT C API plugin。 | 平台/配置 → 后端 `Client`。 | [`xla_bridge.py`](../../../upstream/jax/jax/_src/xla_bridge.py#L933)、[TPU plugin](../../../upstream/jax/jax/_src/xla_bridge.py#L205) |
| 构造普通 MLIR | `mlir.lower_jaxpr_to_module(...)` 调用 `jax._src.lib.mlir`。 | `jaxlib.mlir.ir` 及 `hlo`（StableHLO）、`sdy`（Shardy）方言绑定；如 `hlo.dot_general`。 | Jaxpr、输入输出 aval、平台、sharding、layout 等 → `LoweringResult.module`（MLIR `ir.Module`）。 | [JAX lowering](../../../upstream/jax/jax/_src/interpreters/mlir.py#L1327)、[方言导入](../../../upstream/jax/jax/_src/lib/mlir/dialects/__init__.py#L62) |
| 构造 Pallas TPU payload | `pallas_call_tpu_lowering_rule(...)`、`lower_jaxpr_to_pipelined_module(...)`、`tpu_custom_call.lower_module_to_custom_call(...)`。 | 借用 jaxlib MLIR/Mosaic TPU 绑定来构造内层 module 和外层 custom call。 | kernel Jaxpr、GridMapping、参数 → Mosaic TPU MLIR module → 带 bytecode/Base64 body 的外层 StableHLO `custom_call`。 | [Pallas lowering](../../../upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L435)、[配置序列化](../../../upstream/jax/jax/_src/tpu_custom_call.py#L249) |
| 提交编译与加载 | `compiler.backend_compile_and_load(backend, module, executable_devices, options, host_callbacks)`。 | 常规后端调用 jaxlib `Client.compile_and_load` / `PyClient::CompileAndLoad`；compile-only 客户端有 `compile` 分支。 | MLIR module、设备列表、`CompileOptions`、回调 → `PyLoadedExecutable` / `LoadedExecutable`。 | [JAX 调用](../../../upstream/jax/jax/_src/compiler.py#L330)、[jaxlib 入口](../../../upstream/jax/jaxlib/py_client.cc#L475) |
| jaxlib 转交 IFRT | `PyClient` 持有 `ifrt::Client`。 | `GetDefaultCompiler()->CompileAndLoad(...)`；将 MLIR 包装成 `ifrt::HloProgram`。 | `HloProgram(MLIR)`、`XlaCompileOptions`、设备 → `ifrt::LoadedExecutable`。 | [`PyClient`](../../../upstream/jax/jaxlib/py_client.cc#L372)、[IFRT 调用](../../../upstream/jax/jaxlib/py_client.cc#L412) |
| 常见的 PJRT 兼容实现 | JAX 不直接操作这层 C++ 对象。 | `PjRtCompiler::CompileAndLoad` → `pjrt_client()->CompileAndLoad(...)`；具体 provider 实现后端。 | MLIR module、编译选项 → `PjRtLoadedExecutable`。 | [IFRT/PJRT 桥](../../../upstream/xla/xla/python/pjrt_ifrt/pjrt_compiler.cc#L91)、[PJRT 调用](../../../upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L763) |
| 提交执行与包装结果 | `pxla` 调用 `xla_executable.execute_sharded(input_bufs)`，用 `out_handler` 包装结果。 | jaxlib `PyLoadedExecutable::ExecuteSharded` → IFRT `LoadedExecutable::Execute` → PJRT provider。 | JAX 数组所持的 `PyArray`/IFRT arrays → `PyExecuteResults`/IFRT arrays、状态或 token → 用户可见结果数组。 | [JAX 执行入口](../../../upstream/jax/jax/_src/interpreters/pxla.py#L413)、[jaxlib 执行](../../../upstream/jax/jaxlib/py_executable.cc#L424) |

StableHLO 是 MLIR 程序的方言与兼容性契约；Shardy 的 `sdy` 用来表示和传播分片，是否启用相应分片器还受编译选项控制。JAX 构造这些 IR 时使用 jaxlib 提供的 Python 绑定，随后由 jaxlib/IFRT/PJRT 把整个 module 交给后端；它们不是两个独立的网络服务。[`hlo` / `sdy` 绑定](../../../upstream/jax/jax/_src/lib/mlir/dialects/__init__.py#L58)、[Shardy 选项](../../../upstream/jax/jax/_src/compiler.py#L211)

LLVM 处在更深的代码生成路径中，不是 JAX 直接调用的 Python API。锁定的 XLA GPU 编译器包含 LLVM IR 的验证和后端处理；具体 CPU/GPU 路径也可以选择其他 emitter 或库调用，不能把所有运算概括成同一条 LLVM 路径。[XLA GPU 的 LLVM 后端](../../../upstream/xla/xla/service/gpu/gpu_compiler.cc#L2706)

图的 XLA 一栏表示**常见 PJRT 兼容路径**，不能推出每个后端的内部实现完全相同。CPU/GPU 的代码生成、TPU plugin/libtpu 的内部编译以及 LLO 都在更下游；单靠 JAX 源码与前述 IR 不能替代目标后端产物。运行时二进制若与这里引用的源码不同，应记录 `VERSION-SKEW`。
