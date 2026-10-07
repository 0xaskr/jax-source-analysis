# JAX：Jaxpr 的定义、产生、变换与消费

本文与 [JAX 内部架构 SVG](jaxpr-centered-hub.svg) 共同说明唯一核心概念 [Jaxpr](jaxpr-centered-hub.svg#core)，承接 [software-stack overview](../overview/overview-software-stack-components-layered.md)。研究对象始终是 Jaxpr：程序语义与结构是什么，如何构造，如何改写，以及如何被解释或转换。

定义区独立说明 Jaxpr 的静态语义、结构与约束；产生、变换、消费三个区域用真实的程序传递和调用关系连接。构造出的 Jaxpr 可以按需变换后交给消费者，也可以直接被检查、打印、解释或 lower。四个视角仍是研究同一对象的不同问题，**不是必须依次经过的四个阶段**；定义区的结构关系不表示运行步骤，消费者彼此并列。编译缓存、MLIR 序列化和数组执行放在消费产物的后续去向中。

| 阅读问题 | 本文入口 | 对应图区 |
|---|---|---|
| Jaxpr 代表什么、由什么组成、有哪些约束？ | [定义](#jaxpr-definition) | [定义区](jaxpr-centered-hub.svg#view_definition) |
| 谁根据什么输入构造出 Jaxpr？ | [产生](#jaxpr-production) | [产生区](jaxpr-centered-hub.svg#view_production) |
| 谁按什么规则改变 Jaxpr，返回什么？ | [变换](#jaxpr-transformation) | [变换区](jaxpr-centered-hub.svg#view_transformation) |
| 谁如何使用 Jaxpr，生成什么结果？ | [消费](#jaxpr-consumption) | [消费区](jaxpr-centered-hub.svg#view_consumption) |

本页统一使用 JAX 提交 `361c43e072cce92b7d3e9bdaf4dd16db26c49043`，版本以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。所有源码链接指向该固定提交；示例区分结构示意、源码文档示例与可供用户运行的检查代码，本次不执行其中的 JAX 代码。

专题入口：[相关概念](#jax-concepts) · [打印语法](#jaxpr-format) · [Pallas 的四个视角](#pallas) · [表示路径](#representation-paths) · [jaxlib 边界](#jaxlib-boundary) · [图文维护](#rebuild)。

<a id="pallas"></a>

**Pallas 阅读入口。**[内层 kernel Jaxpr](jaxpr-centered-hub.svg#inner_jaxpr) 与 [外层调用者 Jaxpr](jaxpr-centered-hub.svg#outer_jaxpr) 是同一抽象的两个程序实例，主图仍只突出一个核心概念。它们分别在 [定义](#pallas-definition)、[产生](#pallas-production)、[变换](#pallas-transformation) 和 [消费](#pallas-consumption) 中展开；Mosaic Module 之后的封装另见 [消费产物的后续去向](#downstream)。平台示例限定为 `interpret=False` 的 Mosaic TPU 路径，其他模式与平台由分派规则选择。

<a id="jaxpr-definition"></a>

## 1. 定义：程序语义、结构与约束

### 1.1 Jaxpr 对象与良构性

[`core.Jaxpr`][src-jaxpr] 表达一段可被解释的程序。普通数组实参由调用方提供；已经附带的常量值则随 Jaxpr 保存。

| 结构 | 在程序中的作用 | 约束 |
|---|---|---|
| `all_invars / invars / outvars` | 保存全部输入变量、调用输入和有序输出引用。 | `invars` 排除已附值的常量前缀；`outvars` 可以是 Var 或 Literal。 |
| `constvars / consts` | 将前导变量与已附常量值配对。 | `constvars` 由 `all_invars` 的对应前缀取得；未附值的前导变量仍按普通输入处理。 |
| `eqns` | 按解释顺序保存方程。 | `JaxprEqn` 记录 primitive、params、输入输出和 effects。 |
| `Var / Literal / aval` | 连接数据依赖并描述抽象类型或内联字面值。 | Var 的身份参与值环境索引；Literal 不采用附带常量的调用约定。 |
| `effects / is_high / DebugInfo` | 记录程序效果、高层表示标记和函数来源。 | 方程另带 `source_info / ctx`；变换需保持程序与元数据一致。 |

当前固定版本明确令 [`ClosedJaxpr = Jaxpr`][src-closed]，兼容访问器 `.jaxpr` 返回自身。阅读旧资料时，应核对是否仍采用“ClosedJaxpr 包装另一个 Jaxpr”的模型；本页按当前单类实现解释 `constvars / consts`。

图中 [JaxprEqn](jaxpr-centered-hub.svg#eqn) 与 [Var / Literal](jaxpr-centered-hub.svg#var)、[Primitive](jaxpr-centered-hub.svg#primitive) 的连线是持有和引用关系。primitive 的实现、Trace 和 Tracer 属于解释或构造机制，不作为 Jaxpr 的额外核心对象。

[`check_jaxpr`][src-check] 输入 Jaxpr，检查变量先定义后使用、单次绑定、类型以及相应原语约束；通过时返回 `None`，失败时报告 `JaxprTypeError`。程序通过这些检查与已经完成后端编译是两个不同事实。

<a id="jax-concepts"></a>

### 1.2 与 Jaxpr 配合的值、类型和解释机制

下表用于辨认 Jaxpr 周围的对象和机制；它们分别提供输入、类型或处理规则，不因此增加本图的核心概念。

| 概念 | 它负责什么 | 固定源码入口 |
|---|---|---|
| `jax.Array` / `ArrayImpl` | 用户可见的数组值及其实现，具有 shape、dtype、sharding，并与运行时数据关联。 | [`Array`][src-array]、[`ArrayImpl`][src-array-impl]。 |
| PyTree | 把嵌套输入输出展平为叶子，再按树结构重建；它组织调用结构。 | [`tree_flatten / tree_unflatten`][src-pytree]。 |
| Mesh / Sharding | 给设备集合命名，并描述逻辑数组的分片与放置。 | [`Mesh`][src-mesh]、[`NamedSharding`][src-sharding]。 |
| `jit / grad / vmap` | 分别组织暂存编译、微分和批量化；变换规则围绕 primitive 及解释器组织。 | [`jit`][src-jit]、[`grad`][src-grad]、[`vmap`][src-vmap]。 |
| Trace | 处理 primitive 调用的解释器对象；不同 Trace 可以给同一操作不同解释。 | [`Trace`][src-trace-class]。 |
| Tracer | 追踪期间代表一个值，关联 Trace 与 aval；动态 Jaxpr 追踪还把它关联到 Var 或 Literal。 | [`Tracer`][src-tracer]、[`DynamicJaxprTracer`][src-dynamic-tracer]。 |
| aval | 值的抽象类型。`ShapedArray` 包含 shape、dtype 等信息，aval 还可以表达 Ref、高层类型等。 | [`AbstractValue`][src-abstract]、[`ShapedArray`][src-shaped]。 |
| Primitive | 操作定义；抽象求值、求值实现、微分、batching 和 lowering 采用不同规则。 | [`Primitive.bind`][src-bind]。 |
| Jaxpr | 保存程序输入输出、常量、方程及 effects；方程参数还可携带子 Jaxpr。 | [`Jaxpr`][src-jaxpr]。 |
| `Var / Literal / JaxprEqn` | 分别表达值身份、内联已知值与 primitive 应用。 | [`Var`][src-var]、[`Literal`][src-literal]、[`JaxprEqn`][src-eqn]。 |

**抽象求值与 lowering 回答不同问题。**抽象求值确定输出类型和 effects；lowering 根据平台、类型及上下文生成 IR 操作和值。一个 Python API 也不一定只调用一个 primitive，例如 [`jnp.matmul`][src-matmul] 会先处理维度和类型，再组织 `lax.dot_general`。

<a id="pallas-definition"></a>

### 1.3 Pallas 实例：计算程序与调用映射分开定义

| 对象 | 定义与约束 | 具体位置 |
|---|---|---|
| kernel Jaxpr | 表达块内方程及 effects；常规 kernel 从输入 Ref 读取，向输出 Ref 写入。 | [`_trace_kernel_to_jaxpr`][src-p-trace]、[`_pallas_call_jvp_rule`][src-p-jvp]。 |
| 外层 `pallas_call` 方程 | invars/outvars 连接外层值，`params.jaxpr` 携带内层程序。 | [`pallas_call_p.bind`][src-p-bind]。 |
| `GridSpec / BlockSpec` | 描述调用网格及一次 kernel 调用所见的块；BlockSpec 可带 memory space。 | [`GridSpec`][src-grid-spec]、[`BlockSpec`][src-block-spec]。 |
| `GridMapping` | 整理 grid、block mappings、输入输出和 scratch 等约定，与 kernel Jaxpr 一起被 lowering 消费。 | [`GridMapping`][src-p-grid]。 |

[GridMapping](jaxpr-centered-hub.svg#mapping) 是独立配套对象，不是 Jaxpr 字段。仅看一条 `swap` 或 Ref 的普通类型简写，不能确定最终写入 HBM 还是 VMEM；需要结合映射、内存空间和后续 lowering。

<a id="jaxpr-format"></a>

### 1.4 打印语法如何映射到对象结构

打印文本用于观察前述定义，不构成独立的稳定交换格式。下列内容说明文本与字段的对应；产生文本的 `pretty_print` 接口归入[消费视角](#jaxpr-consumption)。

#### 从一段源码文档示例开始

固定版本 [`make_jaxpr` 文档字符串][src-make-jaxpr] 使用 `sin(cos(x))` 展示下面的打印形式。这里引用它解释格式，没有运行该函数，也不把它作为本次采集的输出。

```text
{ lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
```

| 片段 | 如何对应对象 |
|---|---|
| `{ lambda` | 一段 Jaxpr 的开始，不表示运行时创建 Python lambda。 |
| `; a:f32[]` | 分号前是 constvars，此例为空；之后是普通 invars。`f32[]` 表示 float32 标量的抽象类型。 |
| `. let b = ...; c = ...` | 顺序排列的方程；后面的方程可以使用前面定义的结果。 |
| `in (c,)` | 有序输出原子值列表；一个输出有尾随逗号，零个输出是 `()`。 |

`a/b/c` 来自 [`JaxprPpContext`][src-pp-context] 的变量命名，不承诺与 Python 局部变量名相同。一次调用中的 `3.0` 不会仅因为成为动态输入，就作为输入 Var 的具体值保存在方程中。

#### 头部、常量与类型

默认打印器的阅读提纲如下；它不是用于覆盖所有自定义打印规则的完整形式文法。

```text
{ lambda <const-binders> ; <input-binders>. let
    <output-binders> = <primitive>[<name>=<parameter> ...] <input-atoms>
    ...
  in (<output-atoms>) }
```

下面是附带数组常量的**结构示意**：

```text
{ lambda a:f32[2] ; b:f32[2]. let
    c:f32[2] = add a b
  in (c,) }
```

`a` 的完整值需查看 `.consts`，打印头部只显示变量和类型；标量也可能直接作为 `Literal` 内联，例如 `1.0:f32[]`。本版本 constvars、consts 和普通输入的划分仍以 [Jaxpr 定义](#jaxpr-definition)为准。

类型简写来自 aval 的 `str_short()`；`f32[2,3]` 中的数字是维度，不是下标，也不足以确定设备存储布局。Python 的 tuple/dict 等 PyTree 外壳由调用边界保存，Jaxpr 的输入输出列表按扁平叶子组织。见 [`pp_aval / pp_vars`][src-pp-aval] 和 [`tree_flatten`][src-pytree]。

#### 方程、静态参数与多输出

以二维 `f32[2,3] @ f32[3,4]` 为例，以下仅是关键方程的**缩略结构示意**，`...` 省略其余参数：

```text
c:f32[2,4] = dot_general[
  dimension_numbers=(([1], [0]), ([], []))
  ...
] a b
```

`a/b` 是方程输入，`c` 是输出绑定；`dimension_numbers` 在 `eqn.params` 中，左/右收缩维度分别为 1 和 0。打印参数并非追加的运行时数组实参。普通打印器按参数名排序，参数值可递归包含 Jaxpr，见 [`_pp_eqn`][src-pp-eqn] 和 [`pp_kv_pair`][src-pp-param]。

方程可以有多个输出，也可以没有左侧绑定。`_` 表示未继续使用的结果；是否可删除方程仍由依赖和 effects 决定。`in (...)` 可以引用输入、中间结果或 Literal。程序的单次定义约束由 [`check_jaxpr`][src-check] 检查，文本中的 `=` 表示 IR 结果绑定。

#### 嵌套程序、Ref 与 effects

Pallas 方程可以在 `jaxpr=` 参数中保存 kernel Jaxpr。下面是**缩略结构示意**，未展示其他参数、索引、scratch 和完整效果签名：

```text
{ lambda ; a:f32[8,128] b:f32[8,128]. let
    c:f32[8,128] = pallas_call[
      jaxpr={ lambda ; x:Ref{f32[8,128]}
                       y:Ref{f32[8,128]}
                       o:Ref{f32[8,128]}. let
                u:f32[8,128] <- x[]
                v:f32[8,128] <- y[]
                w:f32[8,128] = add u v
                o:Ref{f32[8,128]}[] <- w
              in () }
      grid_mapping=...
      ...
    ] a b
  in (c,) }
```

`AbstractRef` 表示可读写的位置，区别于普通值 aval；打印还可携带内存空间。`u <- x[]` 是读操作的自定义显示，`o[] <- w` 是忽略旧值时的写入显示。state 与 Pallas 的不同 primitive 都可能采用类似符号，辨认时应读取 `eqn.primitive.name`。见 [`AbstractRef`][src-ref]、[state 打印规则][src-state-pp] 和 [Pallas 打印规则][src-pallas-pp]。

[`ReadEffect / WriteEffect`][src-ref-effects] 标记 Ref 读写。默认文本不展示完整效果签名；设置 `print_effects=True` 后，Jaxpr 打印器会在输出列表后附加 `: { ... }`。效果集合及名称由程序决定，不能把 `in ()` 理解为程序没有可观察行为。[`pp_jaxpr_skeleton`][src-pp-skeleton]

#### 检查真实对象的代码

以下代码供在匹配版本的环境中检查对象，本次没有运行。它追踪并读取 Jaxpr，不包含显式后端编译或设备执行调用。

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(jnp.cos(x))

program = jax.make_jaxpr(f)(3.0)
print(program)
print(program.pretty_print(print_effects=True, source_info=True))
print(program.pretty_print(custom_pp_eqn_rules=False))
print("constvars:", program.constvars)
print("consts:", program.consts)
print("invars:", program.invars, "outvars:", program.outvars)
for eqn in program.eqns:
    print(eqn.primitive.name, eqn.invars, eqn.outvars)
    print("params:", eqn.params, "effects:", eqn.effects)
```

`pretty_print(custom_pp_eqn_rules=False)` 关闭 Ref 等自定义方程显示；`print_shapes=False`、`name_stack=True` 等选项仅改变阅读形式。自动分析应访问对象字段，避免依赖字母变量名、换行或 `<-` 字符。[`Jaxpr.pretty_print`][src-print]

<a id="jaxpr-production"></a>

## 2. 产生：输入、追踪与构造位置

### 2.1 函数、抽象输入与解释环境

以追踪中的 `x + y` 为例：输入 Tracer 提供 aval，`Primitive.bind` 把操作交给 Trace；动态追踪器调用抽象求值规则，得到输出 aval 和 effects，再记录方程并返回输出 Tracer。最终 Jaxpr 中的值依赖由 Var/Literal 表达，不需要把构造它的 Trace 实例保存为字段。[`default_process_primitive`][src-process] 还包含常量折叠和转发等分支，因此不能把每次 Python 操作都机械计为一条最终方程。

产生区的 [函数与抽象输入](jaxpr-centered-hub.svg#p0) → [动态追踪](jaxpr-centered-hub.svg#p1) → [frame 转换](jaxpr-centered-hub.svg#p2) → [构造出的 Jaxpr](jaxpr-centered-hub.svg#produced) 对应以下路径。该产物遵循[核心 Jaxpr 定义](jaxpr-centered-hub.svg#core)，并沿跨区连线进入按需变换或直接消费路径。

| 位置 | 输入 | 构造动作与输出 |
|---|---|---|
| [`_trace_for_jit`][src-trace-jit] | 函数、JIT 配置、mesh、DebugInfo、输入 avals 和调用参数。 | 分离静态参数，处理输入输出树、分片及 donation 约定，组织后续追踪。 |
| [`trace_to_jaxpr_nocache`][src-trace] | 待追踪函数、抽象输入与追踪选项。 | 建立动态追踪环境，取得输出及其程序表示。 |
| [`DynamicJaxprTrace.make_eqn`][src-make] | 输入 tracer、输出 avals、primitive、params 和 effects。 | 创建追踪记录和输出 tracer；frame 保存变量、常量映射及方程。 |
| [`JaxprStackFrame.to_jaxpr`][src-frame] | 输出 tracer、frame 中的输入与方程、来源信息。 | `get_eqns()` 转成 JaxprEqn，汇总 effects，调用 `Jaxpr(...)`；返回 Jaxpr 与 `constvals`。 |

`to_jaxpr` 先快照常量映射，再取得方程，以保持常量变量及其值在转换期间有效。返回的程序与 `constvals` 由调用方按约定组合或传递，不能仅把 tracing 的临时记录当成最终 Jaxpr。

用户入口 [`make_jaxpr`][src-make-jaxpr] 在内部使用 `jit(...).trace(...)`，并按需把捕获的常量附到返回的 Jaxpr 上。动态输入提供抽象信息，静态参数参与追踪边界；输出还可以按 `return_shape` 选项附带结果的类型树。

从构造结果继续阅读跨区关系，有两条主路：

| 路径 | 对应图中节点 | 含义 |
|---|---|---|
| 按需变换后消费 | [构造结果](jaxpr-centered-hub.svg#produced) → [变换输入](jaxpr-centered-hub.svg#transform_input) → 所选变换 → [变换结果](jaxpr-centered-hub.svg#after) → [消费输入](jaxpr-centered-hub.svg#consume_input)。 | 选择适用的变换，并携带新 Jaxpr 及附加调用信息继续处理。 |
| 直接消费 | [构造结果](jaxpr-centered-hub.svg#produced) → [消费输入](jaxpr-centered-hub.svg#consume_input)。 | 不要求先经过图中列出的变换；随后按目的选择解释、检查、打印或 lowering。 |

“变换输入”和“消费输入”用于标明跨区交接的程序与上下文，不表示源码中新增了两种 Jaxpr 类。

<a id="pallas-production"></a>

### 2.2 Pallas 实例：构造 kernel Jaxpr 与外层方程

[`pallas_call`][src-p-call] 组织 kernel、输入输出规格及配置，[`get_grid_mapping`][src-p-grid-build] 形成映射和 kernel 抽象参数。随后 [`_trace_kernel_to_jaxpr`][src-p-trace] 在相应 trace 环境中先追踪、再 DCE，返回 Jaxpr 与捕获的 consts。

正常 kernel 的 Python 返回应为 `None`，结果经 Ref 写入表达；追踪函数检查非 indexer kernel 的返回树，并限制不满足 Ref 约束的捕获常量。外层包装仍然返回数组结构。因而“kernel 返回 None”不等于“外层调用没有结果”。

最后 [`pallas_call_p.bind`][src-p-bind] 将数组实参、kernel Jaxpr、GridMapping、输出 avals 与编译参数交给当前 Trace。图中 [调用者](jaxpr-centered-hub.svg#caller)到 [kernel 输入](jaxpr-centered-hub.svg#kernel) 表示参数来源；[内层程序](jaxpr-centered-hub.svg#inner_jaxpr)进入 [Pallas 绑定](jaxpr-centered-hub.svg#pallas_bind)，在外层 Jaxpr 追踪场景中再交给[外层动态追踪](jaxpr-centered-hub.svg#p1)和 frame 构造过程。生成的[外层 Jaxpr](jaxpr-centered-hub.svg#outer_jaxpr) 中，方程参数分别引用内层 Jaxpr 与 GridMapping，两者保持独立。

<a id="jaxpr-transformation"></a>

## 3. 变换：规则、程序变化与共同重建机制

### 3.1 五类通用处理及其返回值

紫色分支分别消费 Jaxpr，实际组合由调用方与变换条件决定。图中的 [“变换结果：Jaxpr 与附加信息”](jaxpr-centered-hub.svg#after) 汇总结果类型，不代表所有变换共享一个程序实例；Pallas 的部分分支可复用 kernel，并更新调用参数。

| 变换与图中节点 | 输入及规则 | 输出或改变 |
|---|---|---|
| [`dce_jaxpr`][src-dce]（[t0](jaxpr-centered-hub.svg#t0)） | Jaxpr、`used_outputs / instantiate`；沿依赖和 DCE 规则保留所需方程。 | 新 Jaxpr 与 `used_inputs`。 |
| [`batch_jaxpr2`][src-batch]（[t1](jaxpr-centered-hub.svg#t1)） | Jaxpr、轴信息和 `in_axes`；通过 batching 规则重新解释并追踪。 | 批量化 Jaxpr 与 `out_axes`。 |
| [`partial_eval_jaxpr_nounits`][src-pe]（[t2](jaxpr-centered-hub.svg#t2)） | Jaxpr、已知/未知标记及 `instantiate`。 | 已知与未知两部分 Jaxpr、输出未知标记及 residual 类型。 |
| [`linearize_jaxpr`][src-linearize]（[t3](jaxpr-centered-hub.svg#t3)） | Jaxpr、切向非零标记、实例化与转发选项。 | 前向及线性 Jaxpr，以及 residual 结构、非零和转发信息。 |
| [`pe.lower_jaxpr`][src-low]（[t4](jaxpr-centered-hub.svg#t4)） | 高层 Jaxpr 和低层 avals；按高层类型与原语协议展开。 | 低层 Jaxpr 与输出类型结构。 |

### 3.2 依赖、效果和高低层表示的约束

DCE 不是简单删除“结果未使用”的所有方程。[`has_effects`][src-has-effects] 过滤允许消除的效果，再由规则决定保留情况；例如本版本的可消除效果集合包含 `ReadEffect`，不能说所有带 effect 的方程都必须保留。Pallas 中未被使用的 `swap` 旧值与写入效果需要分别判断。

部分求值和线性化会产生多段程序，residual 连接相应阶段；batching 的 `out_axes` 与 DCE 的 `used_inputs` 则描述调用结果或签名变化。接续这些变换时，附加信息与新 Jaxpr 一样重要。

[`PartialVal`][src-partialval] 用于部分求值，区分已知值与仅有 aval 的未知值。它帮助决定哪些计算可在当前阶段处理、哪些保留在暂存程序中；不能把“未知”简单理解为数组一定不存在于某个设备。

本版本还区分 HiJAX / LoJAX 的部分高层和低层类型、primitive。若原语是高层操作而当前 Trace 要求低层表示，[`bind_with_trace`][src-bind-trace] 调用 `to_lojax`。这是 JAX 内部的表示转换；StableHLO 或 Mosaic 的选择发生在相应 lowering 路径。`pallas_call_p` 在该版本中是 [`HiPrimitive`][src-pallas-primitive]。

### 3.3 共同机制：Jaxpr.replace 与 Jaxpr 构造

[`Jaxpr.replace / Jaxpr(...)`][src-jaxpr]（[图中 t5](jaxpr-centered-hub.svg#t5)）接收原程序与待替换字段，构造新的程序对象；调用方负责签名、方程、常量和元数据的一致性。它是上述规则可以采用的共同重建机制，**不是与 DCE、batching 等并列的第六类语义变换算法**。

因此，“调用构造函数得到了新 Jaxpr”只说明对象被重建，具体改变了什么计算、为何保持约束，仍需由所应用的变换规则解释。

<a id="pallas-transformation"></a>

### 3.4 Pallas 实例：同步改写程序、Ref 顺序和映射

| 规则 | 输入与处理 | 输出与限制 |
|---|---|---|
| [`_pallas_call_jvp_rule`][src-p-jvp] | primal/tangent、kernel Jaxpr 与映射；生成 JVP 程序，重排输入输出 Ref，更新 block mappings、数量及输出 avals。 | 重新 bind 并拆分 primal/tangent 输出；该版本检查动态 grid、index operands、alias、mesh 等不支持条件。 |
| [`_pallas_call_batching_rule`][src-p-batch] | 轴信息、批维、数组参数、kernel 与映射。 | 按分支处理批维及网格；无批维时可直接重绑原调用，返回结果与输出批维。 |

这两类规则按需应用，不能只改 kernel 方程而忽略调用约定。图中 [Pallas 专用变换](jaxpr-centered-hub.svg#transforms) 用普通变换节点表达，不再另设一个核心概念。

专用变换的结果经 [kernel 与调用约定](jaxpr-centered-hub.svg#kernel_after) 返回 [Pallas 绑定](jaxpr-centered-hub.svg#pallas_bind)，接续当前解释环境；在外层记录程序的场景中继续追踪。这里可能改写程序，也可能复用 kernel 并更新调用参数；不能把每条分支都理解为必然新建一份 kernel Jaxpr。

<a id="jaxpr-consumption"></a>

## 4. 消费：解释、检查、打印与 lowering

### 4.1 并列的消费者

下面四类消费者都以 Jaxpr 为对象，按调用目的选择；它们不组成 `eval → check → pretty_print → lowering` 的固定链。[消费输入](jaxpr-centered-hub.svg#consume_input) 承接构造出的程序或变换结果，再分别交给相应消费者。

| 消费接口 | 输入 | 使用方式与结果 |
|---|---|---|
| [`eval_jaxpr`][src-eval] | Jaxpr、常量与调用实参。 | 维护变量环境，逐方程调用 `primitive.bind`，最后读取 `outvars`；结果也可处在另一 Trace 的解释环境中。 |
| [`check_jaxpr`][src-check] | Jaxpr。 | 检查类型及良构性，返回 `None` 或抛出错误。 |
| [`pretty_print`][src-print] | Jaxpr 与显示选项。 | 读取结构、类型及来源，返回可读文本。 |
| lowering：普通模块与 Pallas 专用规则 | Jaxpr，加上平台、类型、分片、效果或 GridMapping 等上下文。 | 解释程序并生成 MLIR；普通入口为 [`lower_jaxpr_to_module`][src-module]，kernel 入口为 [`lower_jaxpr_to_pipelined_module`][src-pallas]。 |

### 4.2 普通 lowering：把 Jaxpr 转为外层 MLIR

普通编译方向从 [模块 lowering](jaxpr-centered-hub.svg#c3) 进入 [`jaxpr_subcomp`](jaxpr-centered-hub.svg#lower_rules)。后者逐方程读取输入 IR 值，查找 primitive 的 lowering 规则，并把返回的 IR 值写回输出变量环境。注册入口是 [`register_lowering`][src-register]，消费循环见 [`jaxpr_subcomp`][src-subcomp]。

[`lower_jaxpr_to_module`][src-module] 返回 `LoweringResult`，其中 `module` 是生成的外层 MLIR Module，还附带接续处理所需的信息。这一结果属于编译输入，尚不是已加载 executable 或运行时数组。

Jaxpr 可以直接交给 lowering，不要求先经历图中列出的所有变换。Pallas lowering 则处理 kernel Jaxpr 与专用上下文；普通 JAX 程序不必先生成 Pallas kernel。

<a id="pallas-consumption"></a>

### 4.3 Pallas lowering：解释外层方程与内层 Ref 程序

| 阶段 | 具体消费位置 | 输入 → 输出 |
|---|---|---|
| 外层规则分派 | [`jaxpr_subcomp`][src-subcomp] → [`_pallas_call_lowering`][src-p-dispatch]；由 [规则注册][src-p-register] 接入。 | 外层 IR 实参、上下文及 eqn.params → 所选平台 lowering。 |
| 内层 TPU lowering | [`pallas_call_tpu_lowering_rule`][src-p-tpu] → [`lower_jaxpr_to_pipelined_module`][src-pallas]。 | kernel Jaxpr、GridMapping 及 TPU 配置 → 独立 Mosaic TPU MLIR Module。 |

[外层 lowering](jaxpr-centered-hub.svg#outer_lowering) 根据方程选择 Pallas 规则；规则把 kernel Jaxpr 与 GridMapping 交给专用 lowering，得到 [Mosaic Module](jaxpr-centered-hub.svg#mosaic_module)。普通 lowering 和专用 kernel lowering 消费的都是 Jaxpr 实例，但上下文、类型及结果用途不同。

到 Module 产出为止，Jaxpr 已被相应规则消费；后续序列化的输入对象转为 MLIR Module。

<a id="representation-paths"></a>

### 4.4 表示分支与 Jaxpr 消费边界

| 调用 | Jaxpr 中的表示 | 固定源码可确认的后续表示 |
|---|---|---|
| 普通二维 `jnp.matmul(a, b)` | 普通方程包含 `dot_general`，无需 Pallas kernel Jaxpr。 | [`_dot_general_lower`][src-dot-lower] 构造 `hlo.dot_general`；`hlo` 是 StableHLO 绑定，规则也有 CPU/TPU 注册。 |
| TPU、非解释模式的 `pallas_call` | 外层方程持有 kernel Jaxpr 和 GridMapping，内层可含 Ref 读写。 | Mosaic TPU lowering 生成独立 Module，经序列化作为外层 custom call 的配置。 |
| 高层 primitive 的展开 | 高层 Jaxpr/类型按协议转为低层 Jaxpr/类型。 | `to_lojax` 与 `pe.lower_jaxpr` 仍处理 JAX 表示；之后再进入相应 MLIR lowering。 |

“kernel”可指 Python 函数、内层 Jaxpr、Mosaic 模块或后端设备代码，应随阶段说明具体对象。普通 JAX 程序虽然没有 Pallas 内层 Jaxpr，下游仍可以生成设备 kernel 或选择库调用。

本页不把所有设备的编译过程画成相同的 IR 序列。后续公开 CPU/GPU 编译分别见 [HLO](../xla/hlo-centered-hub.md)、[CPU LLVM IR](../xla/cpu-llvm-ir-centered-hub.md)和 [GPU LLVM IR](../xla/gpu-ir-centered-hub.md)。

<a id="downstream"></a>

### 4.5 消费产物的后续去向：Module 封装、编译与执行

下述过程对应图中 [后续去向辅助区](jaxpr-centered-hub.svg#downstream)。这里继续追踪的是 Jaxpr 消费后的 MLIR Module、payload、executable 和数组对象；不能把它们重新列作 Jaxpr 的定义或变换。

#### Mosaic Module 序列化与外层回接

| 阶段 | 具体消费位置 | 输入 → 输出 |
|---|---|---|
| 序列化 | [`_lower_mosaic_module_to_asm`][src-p-serde]。 | 克隆 Module，运行版本化 `mosaic-serde` pass，写出 MLIR bytecode 及相关标记。 |
| 配置封装 | [`CustomCallBackendConfig.to_json`][src-p-json]、[`lower_module_to_custom_call`][src-p-wrap]。 | 模块字节与选项 → `custom_call_config.body` 中的 base64 数据及配置。 |
| 外层操作构造 | [`mlir.custom_call` 的调用位置][src-p-emit]。 | operands、结果类型、配置及 `tpu_custom_call` 目标名 → 外层操作与 IR 结果值。 |

沿图阅读两条程序路径：[外层 lowering](jaxpr-centered-hub.svg#outer_lowering) 消费外层方程并选择规则；[Mosaic Module](jaxpr-centered-hub.svg#mosaic_module)经 [payload](jaxpr-centered-hub.svg#payload)封装，再接入 [外层 custom call](jaxpr-centered-hub.svg#outer_module)。lowering 返回的 IR 结果进入外层后续操作。

`lower_jaxpr_to_module` 组织完整外层 Module，其中可以同时存在普通 StableHLO 操作和 custom call。最终提交的是整个外层 Module，经过缓存与编译调用后进入 [原生编译边界](jaxpr-centered-hub.svg#compile)，而不是把内层 Module 当作完整调用者程序直接替换。

- **kernel Jaxpr / Mosaic Module** 是程序表示，分别被专用 lowering 和序列化流程消费。
- **payload / backend_config** 是编译期的程序字节与配置，和外层 operands 分别传递。
- **IR 结果值** 是外层操作的 SSA 结果；运行时数组及 Buffer 由后续执行流程产生。

`_lower_mosaic_module_to_asm` 虽含 `asm` 名称，当前可见实现写出的是 MLIR bytecode，不能据名称把它解释成 TPU 机器汇编。Mosaic TPU MLIR 不称为 LLO，也不是最终设备代码；私有编译阶段的范围见 [TPU 证据边界](../libtpu/llo-boundary-centered-hub.md)。

#### 显式阶段接口、缓存与执行包装

[`stages.py`][src-traced] 暴露了可观察的阶段对象。它们携带继续处理所需的信息，不能只凭对象名推断设备已执行。

| 阶段 | 入口与类 | 可观察内容 | 下一步 |
|---|---|---|---|
| 已追踪 | `jax.jit(f).trace(...)` → [`Traced`][src-traced] | `.jaxpr`，以及按需要提供的 `.lojax`、输入输出树和类型。 | `.lower(...)`。 |
| 已 lowering | [`Lowered`][src-lowered] | `.compiler_ir(...)`、`.as_text(...)` 及输出信息。 | `.compile(...)`。 |
| 已编译 | [`Compiled`][src-compiled] | 已关联可执行对象、参数约定和结果包装信息。 | 调用该对象提交实参。 |
| 执行提交 | [`ExecuteReplicated.__call__`][src-execute] | 输入数组经过 handler 后进入 `execute_sharded`，按 effects/callback 情况处理 token。 | 包装输出对象，并按需要观察就绪。 |

`Lowered.compiler_ir()` 和 `as_text()` 是调试观察接口，返回内容受后端与版本影响；相应文档明确不将它们保证为可靠的可移植序列化格式。`Compiled` 的创建与调用同样是两个动作。[`Lowered`][src-lowered]

缓存也需要按所处阶段区分：

- **追踪与参数处理缓存**：[`_infer_params`][src-infer] 解析实参签名与 avals，向 `_infer_params_cached` 查询；命中时复用程序参数，否则调用 `_trace_for_jit`。函数、静态参数、抽象类型及相关上下文影响这一路径，不能只用“是否首次调用 Python 函数”判断。
- **编译缓存**：[`compile_or_get_cached`][src-compile-cache] 接收已 lowering 的 Module、设备和编译选项，解析缓存策略。缓存键不可用时直接编译；持久缓存命中时恢复可执行对象，可涉及反序列化与加载；未命中时进入编译及缓存写入/共享分支。
- **缓存键与目标约束**：[`_get_cache_key`][src-cache-key] 把 computation、设备、编译选项与 backend 交给缓存键生成。它不同于 Python 层追踪缓存；不能假设相同 shape 在所有目标和配置下都共享同一机器代码。

辅助区中的 [编译缓存](jaxpr-centered-hub.svg#compile_cache) → [后端编译调用](jaxpr-centered-hub.svg#backend_compile) → [已加载程序](jaxpr-centered-hub.svg#loaded_exec) 聚合这些选择。缓存命中可以减少重复编译，不因此提供本次设备执行完成的证据。

[`ExecuteReplicated.__call__`][src-execute] 先按保留参数索引整理实参，通过 `in_handler` 取得输入 Buffer 包装，再调用已加载对象的 `execute_sharded`。返回结果经 `consume_with_handlers` 包装；有序效果、其他效果或 host callback 的路径还处理 token。

[输入数组](jaxpr-centered-hub.svg#input_arrays)与已加载程序在 [执行入口](jaxpr-centered-hub.svg#execute) 汇合，形成 [结果数组](jaxpr-centered-hub.svg#result_arrays)。结果对象返回和数据就绪分别由上层对象及 Future/等待机制表达，详见 [IFRT Array](../ifrt/array-centered-hub.md) 与 [PJRT Buffer](../pjrt/buffer-centered-hub.md)。

<a id="jaxlib-boundary"></a>

### 4.6 组件边界：JAX 与 jaxlib 的多处交互

JAX Python 代码在构造 IR、提交编译、调用执行对象和包装数组时都使用 jaxlib。它是一条组件与原生绑定的边界，不等于 Jaxpr 后面只跨越一次的时间分界。方言导入可见 [`stablehlo as hlo` 与 `sdy`][src-dialects]。

| 时机 | JAX 侧位置 | 交给下游的对象及返回 |
|---|---|---|
| 取得 backend | [`get_backend`][src-backend] | 平台与配置 → Client；具体后端注册及实现依平台选择。 |
| 构造普通 Module | [`lower_jaxpr_to_module`][src-module] | 借助 jaxlib MLIR/方言绑定构造 Module、操作、类型与属性。 |
| 构造 Pallas Module 与 payload | [`pallas_call_tpu_lowering_rule`][src-p-tpu]、[`lower_module_to_custom_call`][src-p-wrap] | 使用 MLIR/Mosaic 绑定，把内层程序封装进外层编译输入。 |
| 编译与加载 | [`backend_compile_and_load`][src-backend-compile] | Module、设备、选项与回调 → 常规 backend 的 `compile_and_load`；compile-only client 另有 `compile` 分支。 |
| 原生编译入口 | [`PyClient::CompileAndLoad`][src-py-compile] | 函数内部 clone Module，包装 `ifrt::HloProgram`，再交给 [IFRT Compiler][src-py-ifrt]。 |
| 执行与结果包装 | [`ExecuteReplicated.__call__`][src-execute] → [`PyLoadedExecutable::ExecuteSharded`][src-py-execute] | 输入数组包装 → IFRT 执行接口 → 结果包装与相应状态。 |

`HloProgram` 在这个构造位置承载 MLIR Module，其名称不能证明已经转换成 XLA `HloModule`。jaxlib 的 clone 发生在 `CompileAndLoad` 内部，随后下游才按具体编译路径处理。详见 [jaxlib / MLIR Module](../jaxlib/mlir-module-centered-hub.md)。

运行时输入沿 [PyArray][src-py-array]、IFRT Array 与设备 Buffer 的持有/映射关系传递，不能把每一层包装都解释成复制一次数组。编译返回可执行对象、执行返回输出对象、数据完成三者也分别发生，provider 的内部实现见相关组件文档。

<a id="rebuild"></a>

## 附录：图文维护与证据范围

### 生成与校验

本目录的选定产物是 `jaxpr-centered-hub.svg` 与本 Markdown。SVG 将静态定义独立成块，并连接产生、按需变换和消费的实际程序/调用链，同时保留直接消费旁路；消费结果的后续去向放在辅助区。图沿用 overview 的浅色分区、唯一核心概念和流程颜色：蓝色为编译程序，紫色为变换，灰色为结构/观察关系；运行时对象与完成状态沿相应数据和返回通路表达。

共享生成器是 [`render_software_stack_component_flows.py`](../../../tools/render_software_stack_component_flows.py)，图数据在 [`software_stack_component_flow_data.py`](../../../tools/software_stack_component_flow_data.py)。在仓库根目录运行：

```sh
python3 -B tools/diagram_environment.py run tools/render_software_stack_component_flows.py --component jax
```

先生成到临时目录以便审阅：

```sh
python3 -B tools/diagram_environment.py run tools/render_software_stack_component_flows.py --component jax --output-dir /tmp/jax-component-review
```

上述共享入口把 JAX SVG 写到输出目录的 `jax/` 子目录。旧 [`render_jaxpr_centered_hub.py`](../../../tools/render_jaxpr_centered_hub.py) 转交共享生成器，其 `--output-dir` 直接指定 SVG 所在目录；日常维护优先使用共享入口。

绘图环境由 [`diagram_environment.py`](../../../tools/diagram_environment.py) 和 [`env/diagrams.lock.json`](../../../env/diagrams.lock.json) 管理。首次需要依赖时使用 `diagram_environment.py sync`，已有环境用 `diagram_environment.py check` 验证；环境发生变更还需按仓库要求运行 `python3 -B tools/sync-environment.py check` 及相应自测。

源码检查由 [`component_diagram_sources.py`](../../../tools/component_diagram_sources.py) 根据锁文件、固定检出和源码锚点完成。生成检查覆盖唯一核心概念、必需路径、命名连线、文字边界及图内目标；文档修改还应核对节点链接、源码提交与输入输出描述。选定仓库产物保持 SVG，无需增加 PNG。

### 证据范围与后续阅读

本页统一按固定 JAX 源码核对概念、追踪、变换、打印、Pallas、编译缓存及调用边界。源码链接中的行号随该提交固定；图文后续升级版本时，应同时重新定位符号和检查调用关系。

这次工作只提供源码检查与静态图文证据，没有运行本文的检查代码，也没有执行 CPU/GPU 计算、TPU 模拟、离线 TPU 编译或真实 TPU 执行。图示生成仅验证产物和引用，不能代替程序运行证据；如未来使用的二进制与引用源码不匹配，应明确记录 `VERSION-SKEW`。

继续阅读：[软件栈索引](../index.md) · [overview](../overview/overview-software-stack-components-layered.md) · [jaxlib Module](../jaxlib/mlir-module-centered-hub.md) · [IFRT Array](../ifrt/array-centered-hub.md) · [PJRT Buffer](../pjrt/buffer-centered-hub.md) · [TPU 边界](../libtpu/llo-boundary-centered-hub.md)。

[src-abstract]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L1832
[src-array]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/basearray.py#L43
[src-array-impl]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/array.py#L185
[src-backend]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_bridge.py#L933
[src-backend-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/compiler.py#L330
[src-batch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/batching.py#L383
[src-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L690
[src-bind-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L739
[src-block-spec]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L558
[src-cache-key]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/compiler.py#L609
[src-check]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L3604
[src-closed]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L349
[src-compile-cache]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/compiler.py#L424
[src-compiled]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/stages.py#L710
[src-dce]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1204
[src-dialects]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/lib/mlir/dialects/__init__.py#L59
[src-dot-lower]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/lax/lax.py#L6266
[src-dynamic-tracer]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1365
[src-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L460
[src-eval]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L807
[src-execute]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/pxla.py#L400
[src-frame]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1517
[src-grad]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/api.py#L393
[src-grid-spec]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L1217
[src-has-effects]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1262
[src-infer]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pjit.py#L635
[src-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L105
[src-jit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/api.py#L208
[src-linearize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L130
[src-literal]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L571
[src-low]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2271
[src-lowered]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/stages.py#L590
[src-make]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1674
[src-make-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/api.py#L2100
[src-matmul]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/numpy/tensor_contractions.py#L138
[src-mesh]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/mesh.py#L222
[src-module]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327
[src-p-batch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L508
[src-p-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1358
[src-p-call]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1136
[src-p-dispatch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L845
[src-p-emit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L461
[src-p-grid]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L946
[src-p-grid-build]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L1268
[src-p-json]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L249
[src-p-jvp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L245
[src-p-register]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L997
[src-p-serde]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L491
[src-p-tpu]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[src-p-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L788
[src-p-wrap]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
[src-pallas]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008
[src-pallas-pp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/primitives.py#L167
[src-pallas-primitive]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L72
[src-partialval]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L67
[src-pe]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L655
[src-pp-aval]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4170
[src-pp-context]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4124
[src-pp-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4223
[src-pp-param]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4190
[src-pp-skeleton]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L4248
[src-print]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L248
[src-process]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1750
[src-py-array]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_array.h#L141
[src-py-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475
[src-py-execute]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L484
[src-py-ifrt]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L412
[src-pytree]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tree_util.py#L88
[src-ref]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/state/types.py#L434
[src-ref-effects]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/state/types.py#L70
[src-register]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1003
[src-shaped]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L2444
[src-sharding]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/named_sharding.py#L78
[src-state-pp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/state/primitives.py#L525
[src-subcomp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L2129
[src-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2048
[src-trace-class]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L855
[src-trace-jit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pjit.py#L491
[src-traced]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/stages.py#L412
[src-tracer]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L981
[src-var]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L539
[src-vmap]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/api.py#L949
