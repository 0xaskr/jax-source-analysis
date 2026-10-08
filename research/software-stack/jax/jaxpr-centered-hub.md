# JAX：Jaxpr 的定义、产生、变换与消费

JAX 以 **Jaxpr** 表达程序，再围绕它组织自动微分、批量化、程序拆分和编译。普通 JAX 函数与 Pallas kernel 都可以形成 Jaxpr；区别在于输入类型、使用的原语以及后续处理规则。

[组件图](jaxpr-centered-hub.svg) · [软件栈总览](../overview/overview-software-stack-components-layered.md) · [下一层：jaxlib / MLIR Module](../jaxlib/mlir-module-centered-hub.md)

本文依据 [`upstream-sources.lock`](../../../upstream-sources.lock) 中的 JAX 提交 `361c43e072cce92b7d3e9bdaf4dd16db26c49043`。源码链接均固定到该提交；以下示例用于说明结构，未执行 JAX 或设备计算。

<a id="jaxpr-definition"></a>

## 1. 定义：Jaxpr 表示什么

**Jaxpr 是用于表达 JAX / Pallas 代码的中间表示，描述输入如何通过原语及其组合产生输出，以及计算中发生的可观察效果。**它把程序保存为显式对象，使不同解释器能够读取同一段计算，并应用各自的规则。[`core.Jaxpr`][src-jaxpr] 定义程序边界和主体，[`JaxprEqn`][src-eqn] 定义其中的一次原语应用。

| 结构 | 表示的内容 | 约束 |
|---|---|---|
| `all_invars / invars / outvars` | 全部输入变量、调用输入与有序输出引用。 | `invars` 排除已附常量的前缀；输出可以引用 Var 或 Literal。 |
| `constvars / consts` | 已附常量对应的变量及其值。 | 两者一一对应；`constvars` 取自 `all_invars` 的前缀。 |
| `eqns` | 按解释顺序排列的原语方程。 | 每个方程记录输入、输出、primitive、params 和 effects；params 可以携带子 Jaxpr。 |
| `Var / Literal / aval` | 值的身份、内联字面量与抽象类型。 | Var 通过身份连接依赖；aval 可表达 shape、dtype，也可表达 Ref 等类型。 |
| `effects` | 读写等需要保留的可观察效果。 | 程序变换须遵守效果规则，不能只依据数值输出判断方程是否有用。 |
| `DebugInfo / source_info / ctx` | 函数来源、方程来源与解释上下文。 | 变换和错误报告需要相应的来源信息。 |

这一固定版本令 [`ClosedJaxpr = Jaxpr`][src-closed]，`.jaxpr` 兼容访问器返回自身；常量直接由同一对象保存。检查类型关系时应以这个实现为准。

[`check_jaxpr`][src-check] 检查变量先定义后使用、单次绑定、输入输出类型及原语约束。这些规则使方程构成一段可以被解释的程序。是否支持某个平台的 lowering，则取决于对应规则。

<a id="jax-concepts"></a>

### 1.1 原语、抽象类型与追踪器

[`Primitive`][src-bind] 表达操作；Jaxpr 保存它的应用。抽象求值根据输入 aval 推导输出 aval 和 effects，微分规则定义切向或余切计算，batching 规则处理批维，lowering 规则生成目标 IR。同一个原语可以接受多种解释。

[`Trace`][src-trace-class] 负责解释当前 primitive 调用，[`Tracer`][src-tracer] 代表追踪期间的值。它们参与 Jaxpr 的构造和变换；构造完成的 Jaxpr 用变量和方程保存程序。[`aval`][src-abstract] 描述值的抽象类型，例如 [`ShapedArray`][src-shaped] 中的 shape 和 dtype。

这也说明 JAX 保留 Jaxpr 这一层的作用：自动微分、批量化和部分求值可以在 JAX 原语语义上组合，Pallas 也能在同一套追踪机制中表达 Ref 读写。之后才由 lowering 选择 StableHLO、Mosaic 等表示。若直接产生 StableHLO，上述规则、类型和效果处理仍需要在新的表示上重新建立。

<a id="pallas"></a>
<a id="pallas-definition"></a>

### 1.2 Pallas：同一种表示中的两段程序

外层 Jaxpr 表达调用者的数组计算，其中一条 `pallas_call` 方程携带内层 kernel Jaxpr。内层程序通常从输入 Ref 读取、向输出 Ref 写入；kernel 的 Python 返回值应为 `None`。两者使用同一个 `core.Jaxpr` 类，但程序边界不同。见 [`pallas_call_p.bind`][src-p-bind] 与 [`_trace_kernel_to_jaxpr`][src-p-trace]。

[`GridMapping`][src-p-grid] 则保存 grid、block mappings、输入输出与 scratch 等调用约定。它和 kernel Jaxpr 是分别传递的对象：Jaxpr 说明执行什么计算，GridMapping 说明每次 kernel 调用看到哪些块、如何映射到调用网格。图中可分别查看[内层程序](jaxpr-centered-hub.svg#inner_jaxpr)、[外层程序](jaxpr-centered-hub.svg#outer_jaxpr)和[调用映射](jaxpr-centered-hub.svg#mapping)。

<a id="jaxpr-format"></a>

### 1.3 如何读打印出的 Jaxpr

[`make_jaxpr` 的源码文档][src-make-jaxpr] 用 `sin(cos(x))` 展示下面的形式：

```text
{ lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
```

分号前列出常量变量，此例为空；分号后是输入 `a`，`f32[]` 表示 float32 标量。`let` 后依次列出方程，`b` 和 `c` 是结果绑定，`in (c,)` 指定程序输出。变量名由打印器分配，不要求与 Python 局部变量名一致。

带参数的方程通常写作 `y = primitive[param=value] x`。方括号保存静态参数，操作数位于其后；参数也可以包含子 Jaxpr。附带数组常量的完整值保存在 `.consts` 中，打印文本通常只显示变量及类型。相关实现见 [`_pp_eqn`][src-pp-eqn]、[`pp_kv_pair`][src-pp-param] 和 [`JaxprPpContext`][src-pp-context]。

<a id="jaxpr-production"></a>

## 2. 产生：函数如何成为 Jaxpr

普通 JAX 路径以 **Python 函数和输入的抽象类型**为起点。PyTree 的嵌套结构由调用边界保存，追踪器处理展平后的输入输出。以 JIT 为例，[`_trace_for_jit`][src-trace-jit] 组织函数、avals 和来源信息，再进入 Jaxpr 追踪。

| 处理者 | 输入与操作 | 输出 |
|---|---|---|
| 函数展平与 JIT 入口 | 整理动态参数、输入 avals 和函数的输入输出约定。 | 可追踪函数及其抽象输入。 |
| [`trace_to_jaxpr_nocache`][src-trace] | 建立 `DynamicJaxprTrace`，用代表输入的 Tracer 调用函数。 | 携带常量的 Jaxpr 与输出抽象类型 FlatTree。 |
| [`DynamicJaxprTrace` 的原语处理][src-process] | 读取输入的抽象类型，调用抽象求值规则，并用 `make_eqn` 记录方程。 | frame 中的方程和代表结果的 Tracer。 |
| [`JaxprStackFrame.to_jaxpr`][src-frame] | 汇总 frame 的输入、输出、常量、方程和 effects。 | `Jaxpr` 对象及按调用约定传递的常量值。 |

追踪执行的是 Python 函数中的程序构造过程。能够由静态信息决定的 Python 逻辑在这时处理；依赖动态数据的操作需由相应原语表达。追踪结果可以直接被消费，也可以继续变换。

<a id="pallas-production"></a>

### 2.1 Pallas 的内层追踪与外层绑定

[`pallas_call`][src-p-call] 先用 [`get_grid_mapping`][src-p-grid-build] 整理 kernel 调用约定，得到 GridMapping 及抽象 Ref 输入。随后 [`_trace_kernel_to_jaxpr`][src-p-trace] 在对应网格环境中追踪 kernel，运行 DCE，并检查返回结构和捕获常量的限制，返回 kernel Jaxpr 与允许捕获的常量。

接着 [`pallas_call_p.bind`][src-p-bind] 将 kernel Jaxpr、GridMapping、数组实参和编译参数交给当前 Trace。在外层 Jaxpr 追踪场景中，该调用被记录为一条方程，和前后的普通数组方程一起组成调用者程序。这样，内层计算保存在参数中，外层值依赖仍由调用方程的输入输出连接。

<a id="jaxpr-transformation"></a>

## 3. 变换：程序如何被改写

Jaxpr 变换既包括消除无用计算，也包括构造具有新含义的计算。JVP 产生原值与切向计算，`vmap` 对应的 batching 规则引入批量语义；它们与 DCE 这类保持所需计算语义的优化用途不同。

| 方向与入口 | 输入及规则 | 输出与下一步 |
|---|---|---|
| 前向微分：[`jvp_jaxpr`][src-jvp] | Jaxpr、输入切向非零标记和零值实例化选项；按原语 JVP 规则解释并追踪。 | 同时表达原值和切向计算的新 Jaxpr，以及输出切向非零标记。 |
| 线性化：[`linearize_jaxpr`][src-linearize] | Jaxpr、切向非零标记、转发和实例化选项；把前向计算与线性计算分开。 | 前向 Jaxpr、线性 Jaxpr、residual 的类型结构及转发信息，供后续线性计算或反向传播使用。 |
| 批量化：[`batch_jaxpr2`][src-batch] | Jaxpr、轴信息和 `in_axes`；由 batching 规则传播批维并重新追踪。 | 批量化后的 Jaxpr 与 `out_axes`，供调用方整理输出轴。 |
| 部分求值：[`partial_eval_jaxpr_nounits`][src-pe] | Jaxpr、已知/未知输入标记和实例化选项；按依赖拆分计算。 | 已知与未知两部分 Jaxpr、输出未知标记和 residual 类型，使剩余计算能接收所需中间值。 |
| 死代码消除：[`dce_jaxpr`][src-dce] | Jaxpr 与需要保留的输出标记；沿依赖和效果规则保留必要方程。 | 精简后的 Jaxpr 与 `used_inputs`，调用方据此调整输入。 |
| 高层表示展开：[`pe.lower_jaxpr`][src-low] | 高层 Jaxpr 和低层 avals；按高层类型、原语协议展开。 | 低层 Jaxpr 及输出类型结构，继续交给 JAX 的后续处理。 |

反向模式还需要转置规则。固定版本的 [`backward_pass3`][src-transpose] 处理线性化程序，逆序遍历需要转置的方程，按原语规则累积输入余切。它可以在追踪环境下构造反向计算，也可以参与求值；不能把所有自动微分都归结为一次 `jvp_jaxpr` 调用。

这些方向可以组合，具体顺序由调用方决定。`jit / grad / vmap` 是函数层入口，不与表中每个内部函数一一对应。变换实现可能重新解释并追踪，也可能直接重建方程；[`Jaxpr.replace`][src-jaxpr] 是重建对象的工具，具体语义由使用它的变换规则决定。

<a id="pallas-transformation"></a>

### 3.1 Pallas 同时变换程序和调用约定

[`_pallas_call_jvp_rule`][src-p-jvp] 调用 `ad.jvp_jaxpr` 构造 kernel 的微分程序，随后重排原值与切向 Ref，更新 GridMapping 和输出类型，再次绑定 `pallas_call`。当前实现对动态 grid bounds、索引操作数、输入输出 alias 和 mesh 等情况设有明确限制。

[`_pallas_call_batching_rule`][src-p-batch] 根据批维和网格条件选择处理分支，可能复用 kernel，也可能调整程序或映射。两类规则都需要维持 kernel 签名与调用参数一致；只修改内层方程不足以完成调用的变换。

<a id="jaxpr-consumption"></a>

## 4. 消费：谁使用 Jaxpr，得到什么

| 消费者 | 如何使用 Jaxpr | 结果 |
|---|---|---|
| [`eval_jaxpr`][src-eval] | 用常量与实参建立值环境，逐方程调用 `primitive.bind`，最后读取输出引用。 | 当前解释环境中的输出值；在其他 Trace 下也可参与程序变换。 |
| [`check_jaxpr`][src-check] | 检查变量、类型和原语约束。 | 成功返回 `None`，失败报告 `JaxprTypeError`。 |
| [`pretty_print`][src-print] | 读取程序结构、类型、effects 和来源信息。 | 用于阅读与诊断的文本。 |
| [`lower_jaxpr_to_module`][src-module] | 结合平台、分片等上下文，逐方程生成 MLIR。 | 含 MLIR Module 的 `LoweringResult`，继续进入编译路径。 |

编译方向由 [`jaxpr_subcomp`][src-subcomp] 完成方程级转换：从变量环境读取输入 IR 值，查找原语的 lowering 规则，调用规则并将返回值绑定到输出变量。规则通过 [`register_lowering`][src-register] 等入口注册，通常生成 StableHLO 等操作。

<a id="pallas-consumption"></a>
<a id="representation-paths"></a>

### 4.1 Pallas kernel 如何接入外层 MLIR

外层 lowering 遇到 `pallas_call` 时，进入 [`_pallas_call_lowering`][src-p-dispatch]，按解释模式、平台和所选后端分派。下面展开 `interpret=False` 的 Mosaic TPU 路径：

1. [`pallas_call_tpu_lowering_rule`][src-p-tpu] 接收 kernel Jaxpr、GridMapping 和 lowering 上下文。
2. [`lower_jaxpr_to_pipelined_module`][src-pallas] 根据 kernel 方程、Ref 与网格映射构造 Mosaic TPU MLIR Module。
3. [`_lower_mosaic_module_to_asm`][src-p-serde] 克隆模块，运行 `mosaic-serde` 并写出 MLIR 字节码；函数名中的 `asm` 在这里指模块序列化。
4. [`custom_call` 构造代码][src-p-emit] 将序列化结果放入 `tpu_custom_call` 的配置，将操作数和结果 IR 值接回外层程序。

最终，完整外层 Module 同时包含普通计算和 kernel 调用。Mosaic TPU MLIR 是 kernel lowering 的结果，不能据此把它称为 LLO。GPU 的专用 lowering 与解释模式由各自分支处理。

<a id="jaxlib-boundary"></a>
<a id="downstream"></a>

### 4.2 Module 交给谁

JAX 将完整外层 Module 交给 [`compile_or_get_cached`][src-compile-cache] 等编译入口。常规编译路径经 [`backend_compile_and_load`][src-backend-compile] 调到 jaxlib 的 [`PyClient::CompileAndLoad`][src-py-compile]，由后者克隆 Module、包装为 IFRT HloProgram 并提交编译。这里开始研究的是 [MLIR Module 的生命周期](../jaxlib/mlir-module-centered-hub.md)。

编译返回的 executable 之后与实际数组一起参与执行。程序表示的转换、可执行对象的建立和数组数据的执行分别由相应接口完成；完整衔接见[总览中的编译与执行路径](../overview/overview-software-stack-components-layered.md)。

<a id="rebuild"></a>

图的重建命令与校验说明见[组件索引](../index.md#图文维护)。

[src-abstract]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L1832
[src-backend-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/compiler.py#L330
[src-batch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/batching.py#L383
[src-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L690
[src-check]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L3604
[src-closed]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L349
[src-compile-cache]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/compiler.py#L424
[src-dce]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1204
[src-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L460
[src-eval]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L807
[src-frame]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1517
[src-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L105
[src-jvp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L1044
[src-linearize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L130
[src-low]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2271
[src-make-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/api.py#L2100
[src-module]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327
[src-p-batch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L508
[src-p-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1358
[src-p-call]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1136
[src-p-dispatch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L845
[src-p-emit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L461
[src-p-grid]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L946
[src-p-grid-build]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L1268
[src-p-jvp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L245
[src-p-serde]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L491
[src-p-tpu]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[src-p-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L788
[src-pallas]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008
[src-pe]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L655
[src-pp-context]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4124
[src-pp-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4223
[src-pp-param]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4190
[src-print]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L248
[src-process]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1750
[src-py-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475
[src-register]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1003
[src-shaped]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L2444
[src-subcomp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L2129
[src-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2048
[src-trace-class]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L855
[src-trace-jit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pjit.py#L491
[src-tracer]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L981
[src-transpose]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L299
