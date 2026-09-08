# 最小证据包：`jit` shape cache

问题：相同 callable、配置、shape 和 dtype 的第二次调用是否再次 tracing？

探针在 CPU 上连续调用相同 shape 两次，再改用另一个 shape。探针会显式断言 shape
和 tracing 计数；不满足 `1, 1, 2` 时返回非零状态。capture 同时保存 stdout 和运行结束
时实际映射的 jaxlib native binaries 指纹。inventory 使用 package-relative path，关联
锁定 wheel 的 SHA-256、jaxlib build revision 和 JAX source component；校验器由后二者
推导 `VERSION-SKEW`。这支持“相同 shape 的第二次调用复用 tracing 结果，而 shape
改变会产生新的 trace”这一有限结论；它不证明 TPU 编译或执行行为。

`topic.json.reproduce` 与 capture 的 `producer` 保存完全相同的结构化 `argv`，并用
`stdout_path` 绑定 stdout artifact；manifest 中没有 shell 重定向字符串。当前 inventory
声明 `python-extension` 和 `cpu-backend` 两个 runtime role，因此能够支持 `RUN-CPU`。

源码索引把 cache miss 后进入的 `_trace_for_jit` 锚定到锁定 JAX revision。运行使用的
editable JAX 与 jaxlib wheel 不同版，因此 evidence level 为 `RUN-CPU`，并带有
`VERSION-SKEW` qualifier。完整实验见 `labs/001-jit-cpu/`。
