Q：jax 部分最核心的概念是什么？
A：jaxpr。 jax是个函数式的语言， 并且是个jit，核心概念就是各种IR， 那么单纯的在jax 这一层级中，jaxpr作为ir， 就是核心。

Q: jaxpr的格式是什么？
A：research/software-stack/jax/jaxpr-format.md
``` jaxpr
{
    lambda ; a:f32[]. let
      b:f32[] = cos a
      c:f32[] = sin b
    in (c,)
}
```

Q: jaxpr 主要由哪些组成?
A: 主要由 in array, out array, Primitive, consts array, effect
A: effect要单独拿出来, 是因为jaxpr没人是无副作用的纯函数
```jaxpr
  { lambda ; a:f32[]. let
      io_callback[
        callback=_FlatCallback(callback_func=<function record_on_host at 0xXXXXXXXXXXXX>, in_tree=PyTreeDef(((*,), {})))
        ordered=True
        result_avals=()
        sharding=None
      ] a
      b:f32[] = add a 1.0:f32[]
    in (b,) : { OrderedIO } }
```
A: 具体可见代码 core.py -> class jaxpr
```python
class Jaxpr:
  # 所有输入变量；前 len(_consts) 个已附常量值，其余由调用者传入。
  _all_invars: list[Var]
  # 按返回顺序排列的输出，可以是计算结果 Var 或已知值 Literal。
  _outvars: list[Atom]
  # 程序体中的方程，按求值顺序记录 primitive 调用。
  _eqns: list[JaxprEqn]
  # 整个 Jaxpr 的效果集合，由方程的 effects 汇总而来。
  _effects: Effects
  # 被追踪函数、输入名和结果路径等信息，用于诊断与报错。
  _debug_info: DebugInfo
  # 是否涉及 HiJAX 原语或抽象值；决定何时需要转成 LoJAX。
  _is_high: bool
  # 与 _all_invars 前缀中的变量一一对应的已附常量值。
  _consts: list[Any]
```

Q: jaxpr在jax中，具体表现为什么? 一个字符串？还是一个class? 还是某种特殊的数据格式？
A: 是一个类，upstream/jax/jax/_src/core.py -> class: jaxpr

Q：jaxpr中的 add, mul, cos, sin 这些是怎么定义的？在哪里可以看见？怎么确定语义集合？
A：这些对应的是 Primitive，并且和其一一对应

Q: Primitive是否可以自定义？
A：Primitive 是可以自定义的，需要补充shape推导，变换规则(grad,vmap等),lowering规则就可以

Q：jaxpr为什么还分成两种类型，一种是hijax，一种是lojax？
A：你可以理解hijax主要是使用lojax语义所组成的自定义的 jaxpr，具体目的是自定义数据类型以及自定义原语同时，又让他们可以被grad和vmap应用。
A：相比Primitive 可以做到更灵活，主要是自定义数据类型，比如说3bit量化，或者是2Bit量化等.
A: hijax定位有点像是更高抽象层级的 自定义Primitive, 不过hijax提供了高层接口.
A：具体可以参考 https://docs.jax.dev/en/latest/hijax_types.html
```text
{
    lambda ; a:f32[2,3]. let
      b:QArrayTy(shape=(2, 3)) = to_q a
      c:f32[2,3] = from_q b
    in (c,)
}
jax.config.update("jax_custom_vjp3", True)

@jax.custom_vjp
def cube(x):
    return x * x * x

def fwd(x):
    return x * x * x, x

def bwd(saved_x, g):
    return (3.0 * saved_x * saved_x * g,)

cube.defvjp(fwd, bwd)

本地追踪 cube 得到的 high Jaxpr（is_high=True）是：

{ lambda ; a:f32[]. let
    b:f32[] = call_hi_primitive[_prim=CustomVJPTraced] a
in (b,) }

展开后，low Jaxpr（is_high=False）保留一个内层程序，其中是两次 mul：

{ lambda ; a:f32[]. let
    b:f32[] = eval_jaxpr[
    call_jaxpr={ lambda ; c:f32[]. let
        d:f32[] = mul c c
        e:f32[] = mul d c
        in (e,) }
    ] a
in (b,)
}
```

Q: jaxpr 可以通过哪些方法产生出来? (jaxpr class)
A: jax.make_jaxpr(func) 可以快速拿到, 在源码中也可以通过.trace 拿到, jax.jit 这种情况 也会在内部生成

Q: jaxpr 这个class中的这些属性有什么用? 如何使用的?
A:

Q: 以jaxpr为中心, 整个jaxpr是由哪些class生成的? 中间会做哪些变换 ? 哪些地方会去消费 ?

Q: jaxpr中的debug info 有什么用?
A:
