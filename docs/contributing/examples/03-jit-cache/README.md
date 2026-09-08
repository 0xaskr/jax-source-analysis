# 最小证据包：`jit` shape cache

问题：相同 callable、配置、shape 和 dtype 的第二次调用是否再次 tracing？

探针在 CPU 上连续调用相同 shape 两次，再改用另一个 shape。capture 显示 tracing 计数为 `1, 1, 2`。这支持“相同 shape 的第二次调用复用 tracing 结果，而 shape 改变会产生新的 trace”这一有限结论；它不证明 TPU 编译或执行行为。

源码索引把 cache miss 后进入的 `_trace_for_jit` 锚定到锁定 JAX revision。完整实验见 `labs/001-jit-cpu/`。
