# JAX 内部软件栈

这一目录从 JAX 自身的概念与实现出发，不固定某个算子或 kernel。先看唯一的 [JAX 内部软件栈图](jax-internal-stack.svg)：它把概念、tracing、Pallas lowering 及 JAX/jaxlib 下游边界连成一条路径。文字解释、各组件调用的 API 和输入输出见 [概念与下游接口](jax-concepts.md)；逐段阅读 Jaxpr 的头部、方程、参数、嵌套程序及 Ref 写法，见 [Jaxpr 的格式与打印语法](jaxpr-format.md)。图由 [`tools/render_jax_internal_stack.py`](../../../tools/render_jax_internal_stack.py) 生成。

源码基线为 JAX v0.11.1；本地检出是基于该 tag 的源码注释提交，修订记录见 [`upstream-sources.lock`](../../../upstream-sources.lock)。图文依据本地源码整理，未作为 CPU 或 TPU 执行证据。
