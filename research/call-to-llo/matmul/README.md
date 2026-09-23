# `jnp.matmul` 调用链

[源码走读](matmul-lowering-walkthrough.md)从 Python API、Jaxpr/`dot_general_p` 追到 StableHLO；[Notebook](matmul-lowering.ipynb)提供表示与旧 CPU 实验对照。[逐 pass 记录](matmul-pass-walkthrough.md)和[链路图](figures/matmul-lowering-chain.svg)描述已有的 CPU 编译材料；链路图由 [生成脚本](render_matmul_walkthrough.py)维护。

这些记录的 CPU `DotThunk`、LLVM 对象和 pass 结果不是 TPU LLO。继续这条链需要固定 `jnp.matmul` 输入、目标 TPU、JAX/jaxlib/libtpu 版本及同一次编译标识，然后保存 TPU HLO、后端阶段产物与 LLO。共用的 [capture 脚本](../../software-stack/tools/matmul_probe.py)、[软件栈总览](../../software-stack/source/overview.md)及[TPU 公开边界](../../software-stack/runtime/tpu-runtime-boundary.md)在开放探索目录。
