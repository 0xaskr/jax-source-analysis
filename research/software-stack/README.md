# JAX 软件栈：开放探索

本目录按组件和问题探索整个 JAX 软件栈，不以某个算子或 kernel 的 LLO 为唯一目标。可从源码结构、程序表示、编译器、运行时或观测工具任意进入；各份历史实验保留原有版本和证据范围。

| 方向 | 入口 |
|---|---|
| 仓库与源码归属 | [JAX 仓库结构](jax-repo/README.md)、[源码索引导读](source-index-digest.md)、[组件图](figures/overview-software-stack.svg) |
| 表示与编译 | [软件栈总览](overview.md)、[Shardy 往返](shardy-round-trip.md)、[LLVM 与对象文件](llvm-and-objects.md) |
| 运行时 | [TPU 公开运行边界](tpu-runtime-boundary.md)、[CPU executable 与 trace](cpu-executable-and-trace.md) |
| 观测与性能 | [profiling](profiling.md)、[Roofline](roofline.md)、[Fusion 与内存](fusion-and-memory.md)、[调度与重叠](overlap-and-scheduling.md) |
| 构建与复现 | [源码构建](source-build.md)、[运行环境](runtime-environment.md)、[构建输入](build-inputs.md) |
| 自定义 kernel 的通用机制 | [Pallas 对照](pallas-comparison.md)、[Pallas 两层表示](overview.md#pallas-的两层表示) |

原有的探针、验证器、Notebook、JSON 记录和历史计划仍存放在这里。其中 `matmul_probe.py` 也提供许多探针共用的 capture/指纹函数，因此留作实验基础设施；固定调用的纵向分析见 [调用到 LLO](../call-to-llo/README.md)。旧 [PLAN.md](PLAN.md)、[status.json](status.json) 和 [交接记录](AGENT-HANDOFF-2026-09-15.md) 只是历史资料，不构成本目录当前任务队列。

引用旧结果时核对锁定源码和运行时版本，分别标明源码检查、CPU 执行、TPU 模拟、离线 TPU 编译和真实 TPU 执行。Mosaic TPU MLIR 不等于 LLO。
