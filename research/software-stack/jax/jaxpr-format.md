# Jaxpr 的格式与打印语法

本文以本仓库锁定的 JAX v0.11.1 源码为准，解释 `print(jax.make_jaxpr(f)(...))` 中每一部分如何对应内存里的 Jaxpr 对象。`upstream/jax` 当前检出是基于 `jax-v0.11.1` 的源码注释提交 `3d1d3e8963b4`，具体修订见 [`upstream-sources.lock`](../../../upstream-sources.lock)。这里讨论的是 **Jaxpr 的人类可读打印表示**，不是一套供外部程序依赖的稳定文本解析格式，也不是 StableHLO 或 Mosaic TPU MLIR。

## 先读一段完整输出

固定版本的 [`jax.make_jaxpr` 文档字符串](../../../upstream/jax/jax/_src/api.py#L2144)给出以下函数和输出：

```python
import jax

def f(x):
    return jax.numpy.sin(jax.numpy.cos(x))

jax.make_jaxpr(f)(3.0)
```

```text
{ lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
```

它可以分成五段：

| 片段 | 含义 |
|---|---|
| `{ lambda` | 一段 Jaxpr 的开始。它表示一个带参数的计算图，并非运行时创建的 Python `lambda` 对象。 |
| `; a:f32[]` | 分号左侧是绑定常量的变量（此例为空），右侧是普通输入变量。`a` 的抽象类型为 `f32[]`，即标量 `float32`。 |
| `. let b:f32[] = cos a; c:f32[] = sin b` | 顺序排列的方程。`cos` 产生 `b`，后面的 `sin` 使用 `b` 并产生 `c`。一行显示时，方程之间可以用 `;` 分隔；换行显示时可能只看到换行。 |
| `in (c,)` | 输出原子值列表。尾随逗号说明这里恰有一个输出；零个输出显示为 `()`。 |
| `}` | 这段 Jaxpr 结束。 |

`a`、`b`、`c` 是打印器给 Jaxpr `Var` 分配的名字，不承诺等于 Python 源码的变量名，也不是实际输入数值。Jaxpr 记录的是值之间的依赖和抽象类型；上述 `3.0` 用于确定输入的抽象信息，不会因为打印出 `a:f32[]` 就作为 `a` 的数值保存在这条方程中。[`Var` 与 `Literal`](../../../upstream/jax/jax/_src/core.py#L539)、[打印变量名的上下文](../../../upstream/jax/jax/_src/core.py#L4124)

## 阅读用语法

下面是针对**默认打印器**的阅读提纲，不是完整的、可用于解析全部 Jaxpr 输出的形式文法。实际换行、缩进、参数值和某些 primitive 的写法由打印器及各 primitive 的自定义规则决定。[`pp_jaxpr_skeleton`](../../../upstream/jax/jax/_src/core.py#L4248)、[`_pp_eqn`](../../../upstream/jax/jax/_src/core.py#L4223)

```text
Jaxpr     ≈ { lambda <const-binder>* ; <input-binder>* . let
                <equation>*
              in ( <output-atom>* ) [ : { <effect>* } ] }

equation  ≈ [<output-binder>* =] <primitive> [<name>=<parameter-value> ...]
              <input-atom>*
binder    ≈ <name>:<aval>          例如 b:f32[2,3]
atom      ≈ <name> | <literal>    例如 b 或 1.0:f32[]
```

`*` 只表示数量可以为零或多个，多个变量/输出之间按打印器规则以空格或逗号分隔；方括号表示某部分可省略。这里的 `[...]` 同时也是 Jaxpr 打印 primitive 参数时采用的实际括号，需结合上下文区分。打印出的参数值可以是普通 Python 对象的字符串形式，也可以再次包含 Jaxpr；因此不要把这份提纲当作一门封闭的文本语言。普通方程的参数名在 [`_pp_eqn`](../../../upstream/jax/jax/_src/core.py#L4227) 中排序，再以 [`name=value`](../../../upstream/jax/jax/_src/core.py#L4190) 形式显示。

## 头部：常量、输入、类型

头部固定以 `lambda`、分号和句点组织两组绑定变量：

```text
{ lambda <constvars> ; <invars>. let ... in (...) }
```

分号前的 `constvars` 对应追踪时捕获、提升出来的常量；分号后的 `invars` 对应函数的动态输入。一个只有普通输入的 Jaxpr 因而写作 `lambda ; a:f32[]`，**前面的空白不是缺了参数**。如果有常量，打印文本只显示它的变量和类型，不显示其完整数组值；真实值由对象的 `consts` 保存，并与 `constvars` 逐一配对。例如下面仅是结构示意：

```text
{ lambda a:f32[2] ; b:f32[2]. let
    c:f32[2] = add a b
  in (c,) }
```

其中 `a` 是常量变量，`b` 是调用输入；要知道 `a` 的具体数组内容，需查看 `jaxpr.consts`。固定版本的 [`Jaxpr.constvars`、`Jaxpr.invars`、`Jaxpr.consts`](../../../upstream/jax/jax/_src/core.py#L105)给出了这三者的关系。标量常量也可能直接以内联 `Literal` 出现在方程中，例如 `1.0:f32[]`，无需占据一个 `constvar`。[`Literal.pretty_print`](../../../upstream/jax/jax/_src/core.py#L595)

特别注意版本差异：本地 [`core.py`](../../../upstream/jax/jax/_src/core.py#L349)明确令 `ClosedJaxpr = Jaxpr`，`Jaxpr` 自己携带可为空的 `consts`，而 `jaxpr` 属性是兼容旧代码的访问器。仓库里的较早版 [Jaxpr 教程](../../../upstream/jax/docs/jaxpr.md#L47)仍讲述“`ClosedJaxpr` 包装 `Jaxpr`”的双类模型；阅读 **v0.11.1 实现**时应以本地 `core.py` 为准。`jax.make_jaxpr` 也会在返回前将捕获常量附到 Jaxpr 上。[`make_jaxpr_f`](../../../upstream/jax/jax/_src/api.py#L2170)

`:<aval>` 是变量的抽象类型标注。常见的 `f32[2,3]` 表示二维 `float32` 数组，`i32[]` 表示标量 `int32`；方括号里的数字是形状维度，不是下标。`aval` 还可能携带分片、内存空间等信息，打印出来的简写由类型自己的 `str_short()` 决定；不能只根据 `f32[2,3]` 断言设备上的实际存储布局。[`pp_aval`](../../../upstream/jax/jax/_src/core.py#L4170)、[`ShapedArray.str_short`](../../../upstream/jax/jax/_src/core.py#L2533)

Jaxpr 头部按**扁平输入叶子**列出变量。Python 函数即使接受或返回 PyTree，打印结果也不会在这里照搬其 tuple/dict 外壳；调用边界的 PyTree 结构由另外的扁平化信息负责。[`Jaxpr.invars` / `outvars`](../../../upstream/jax/jax/_src/core.py#L158)、[固定版本教程的 PyTree 示例](../../../upstream/jax/docs/jaxpr.md#L128)

## 方程：输出、primitive、参数、输入

一般方程可读作“把右侧 primitive 作用于输入原子值，用左侧变量绑定结果”：

```text
c:f32[3,4,6] = dot_general[
  dimension_numbers=(([2], [0]), ([], []))
  precision=(Precision.HIGHEST, Precision.HIGHEST)
  preferred_element_type=float32
] a b
```

这里 `a` 和 `b` 是输入 **Var**，`c` 是新定义的输出 **Var**；`dot_general` 是 primitive 名，括号内的 `dimension_numbers`、`precision` 等是其静态 `params`，不属于方程末尾的运行时输入。上例的形状与参数取自仓库保存的 [batched matmul Jaxpr 捕获](../../call-to-llo/matmul/matmul-lowering.ipynb)：输入分别为 `f32[3,4,8]` 和 `f32[8,6]`、输出为 `f32[3,4,6]`。对于二维 `f32[2,3] @ f32[3,4]`，收缩轴则是 `(([1], [0]), ([], []))`，见[概念文档中的结构示例](jax-concepts.md#两个-jaxpr-示例)。

一个方程不保证只定义一个结果：`JaxprEqn.outvars` 是列表，primitive 可有多个输出；也允许不显示左侧绑定。输入 `JaxprEqn.invars` 则由 `Var` 或 `Literal` 组成。`JaxprEqn` 对象还持有 `effects`、`source_info` 和上下文 `ctx`，但默认打印通常不把这些全部展开。[`JaxprEqn` 字段](../../../upstream/jax/jax/_src/core.py#L460)、[`new_jaxpr_eqn`](../../../upstream/jax/jax/_src/core.py#L514)

`_` 表示一个结果值未在后面被读取；它并不说明产生它的方程可以删除。例如 `jax.make_jaxpr(jax.grad(f))(3.0)` 的固定源码示例中可见 `_:f32[] = sin b`。对于带 effect 的操作，写入仍可能必须保留。[`make_jaxpr` 示例](../../../upstream/jax/jax/_src/api.py#L2152)、[`_dropvars`](../../../upstream/jax/jax/_src/core.py#L3644)

## `in (...)` 与求值顺序

`let` 中的方程按顺序定义中间变量，后面的方程才能使用前面定义的变量；`in (...)` 指定整段 Jaxpr 的输出，可以引用输入变量、中间变量或字面量。`(c,)` 是一个输出，`(c, d)` 是两个，`()` 是零个。Jaxpr 对象分别保存 `eqns` 与 `outvars`；`eval_jaxpr` 顺序执行各方程，最后读取 `outvars`。[`Jaxpr` 属性](../../../upstream/jax/jax/_src/core.py#L105)、[`eval_jaxpr`](../../../upstream/jax/jax/_src/core.py#L807)

这种变量绑定具有单次定义的约束：`check_jaxpr` 会检查变量先定义后读取、同一变量不重复绑定，以及抽象类型是否匹配。因此打印里的 `=` 是 IR 中的结果绑定，不等同于 Python 变量可重复赋值的语义。[`check_jaxpr`](../../../upstream/jax/jax/_src/core.py#L3604)

## 嵌套 Jaxpr、Ref 与 effect

某些 primitive 的 `params` 可包含另一段 Jaxpr。例如 Pallas 的外层方程可把 kernel Jaxpr 放在 `jaxpr=` 参数中，同时携带 `grid_mapping`。下例是**缩略结构示意，不是原样捕获**；实际打印还可能包含更多参数、索引和 scratch Ref：

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
    ] a b
  in (c,) }
```

`Ref{f32[8,128]}` 是可读写位置的抽象类型，与普通值 `f32[8,128]` 不同；如果 Ref 带有显式内存空间，还可能打印为 `Ref<...>{...}`。[`AbstractRef.str_short`](../../../upstream/jax/jax/_src/state/types.py#L603) `u <- x[]` 是 `get`/load 的自定义漂亮打印，`o[] <- w` 是旧值被忽略时 `swap`/store 的打印方式。此处的 `<-` **不能按一般 Jaxpr 方程的 `=` 规则反推 primitive 名**；同一符号可能由 state 或 Pallas 的不同读写 primitive 打印出来。需要区分时，检查 `eqn.primitive.name`，或关闭自定义方程打印。[state `get`/`swap` 打印规则](../../../upstream/jax/jax/_src/state/primitives.py#L525)、[Pallas `load`/`swap` 打印规则](../../../upstream/jax/jax/_src/pallas/primitives.py#L167)

Ref 读写具有 effect。默认 Jaxpr 打印不展示 effect 签名；`jaxpr.pretty_print(print_effects=True)` 会在 `in (...)` 后附上 `: { ... }`。对是否可以删掉一个方程，不能只看其结果变量是否是 `_`，还要看该方程的 effect。[打印选项与 effect 后缀](../../../upstream/jax/jax/_src/core.py#L4258)、[`ReadEffect` / `WriteEffect`](../../../upstream/jax/jax/_src/state/types.py#L70)

## 如何检查真实对象

以下代码只展示**如何追踪并检查对象**；本文没有运行它，也没有据此声称完成 CPU 执行、TPU 编译或设备执行：

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(jnp.cos(x))

jaxpr = jax.make_jaxpr(f)(3.0)
print(jaxpr)
print(jaxpr.pretty_print(print_effects=True, source_info=True))
print(jaxpr.constvars, jaxpr.consts, jaxpr.invars, jaxpr.outvars)

for eqn in jaxpr.eqns:
    print(eqn.primitive.name, eqn.invars, eqn.outvars, eqn.params, eqn.effects)
```

`pretty_print(custom_pp_eqn_rules=False)` 可避开 Ref 等方程的自定义显示，便于辨认底层 primitive；`print_shapes=False` 会省略打印时的部分类型标注，`name_stack=True` 可显示名称上下文。它们只改变阅读表示，不改变 Jaxpr 对象。[`Jaxpr.pretty_print`](../../../upstream/jax/jax/_src/core.py#L248)、[`_pp_eqn`](../../../upstream/jax/jax/_src/core.py#L4223)

阅读复杂输出时，先划出 `lambda` 头部、`let` 方程和 `in` 输出；再按顺序追每个 Var 的定义与使用；最后展开 `params` 中的子 Jaxpr、Ref 和 effect。若要做自动化分析，直接访问 `Jaxpr` / `JaxprEqn` 字段，并用 `check_jaxpr` 检查良构性；避免依赖字母变量名、换行位置或 `<-` 等漂亮打印符号。这里只确认了本地固定源码的结构和打印逻辑，没有做运行时验证。[`check_jaxpr`](../../../upstream/jax/jax/_src/core.py#L3604)
