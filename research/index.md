# JAX 研究索引

`research/` 只有两个研究方向。阅读入口和材料归属按**研究问题**划分，不按曾经的 `jax-stack` 文件夹或某次实验计划划分。源码版本以 [`upstream-sources.lock`](../upstream-sources.lock) 和 [`source-archives.lock`](../source-archives.lock) 为准；历史运行记录里的版本、路径和证据等级按其原始记录解释。

| 方向 | 负责什么 | 目录入口 |
|---|---|---|
| JAX 软件栈的开放探索 | 不预设算子、kernel 或性能目标；从组件、表示、编译、运行与观测中发现问题 | [software-stack/README.md](software-stack/README.md) |
| 固定调用到 LLO 的纵向追踪 | 固定一次 `jnp.matmul` 和一次 Pallas kernel 调用，逐层关联编译输入、输出与目标 TPU 的 LLO | [call-to-llo/README.md](call-to-llo/README.md)（下分 [matmul](call-to-llo/matmul/README.md) 与 [pallas](call-to-llo/pallas/README.md)） |

## 一、软件栈开放探索

原 `research/jax-stack/` 的材料已整理到 [software-stack](software-stack/README.md)，与 [JAX 仓库结构](software-stack/jax-repo/README.md) 合并为同一个方向。可从任意组件进入，不把旧计划当作新的任务队列。

| 主题 | 阅读入口 | 范围 |
|---|---|---|
| 源码与构建 | [组件/路径总览](software-stack/source/overview.md)、[源码索引导读](software-stack/source/source-index-digest.md) | JAX、jaxlib、StableHLO、Shardy、XLA 和依赖的归属与连接 |
| 编译器 | [Shardy 往返](software-stack/compiler/shardy-round-trip.md)、[LLVM 与对象](software-stack/compiler/llvm-and-objects.md) | 表示转换、pass、CPU 代码生成；具体证据按后端区分 |
| 运行时 | [TPU 公开接口](software-stack/runtime/tpu-runtime-boundary.md)、[CPU executable](software-stack/runtime/cpu-executable-and-trace.md) | 提交、等待、可执行对象及公开/私有边界 |
| 观测与性能 | [profiling](software-stack/profiling/profiling.md)、[Roofline](software-stack/profiling/roofline.md)、[Fusion 与内存](software-stack/performance/fusion-and-memory.md) | host/编译/设备事件、成本与内存观测 |
| Pallas 与工作负载 | [Pallas 对照](software-stack/pallas/pallas-comparison.md)、[SGLang v7 工作负载](software-stack/workload/sglang-v7-workload.md) | 通用机制与既有工作负载材料 |
| 复用与历史 | [脚本及原始结果](software-stack/tools/README.md)、[历史记录](software-stack/history/README.md) | 代码、JSON 证据、旧计划和状态的边界 |

## 二、固定调用到 LLO

[call-to-llo](call-to-llo/README.md) 下的 [matmul](call-to-llo/matmul/README.md) 与 [pallas](call-to-llo/pallas/README.md) 各负责一条纵向链。每条链都应固定输入、源码/运行时版本、目标设备与同一次编译的标识；没有产物的阶段保持为缺口。

| 调用 | 已有材料 | 当前证据边界 |
|---|---|---|
| `jnp.matmul` | [源码走读](call-to-llo/matmul/matmul-lowering-walkthrough.md)、[Notebook](call-to-llo/matmul/matmul-lowering.ipynb)、[逐 pass 记录](call-to-llo/matmul/matmul-pass-walkthrough.md)、[链路图](call-to-llo/matmul/figures/matmul-lowering-chain.svg) | 已有 Python/Jaxpr/StableHLO 走读与 **CPU** 后端记录；尚无对应的 TPU LLO |
| `manual_double_buffer_add` Pallas kernel | [源码](call-to-llo/pallas/pallas_double_buffer_dma.py)、[外层 StableHLO](call-to-llo/pallas/pallas_double_buffer_dma.stablehlo.mlir)、[内层 Mosaic TPU MLIR](call-to-llo/pallas/pallas_double_buffer_dma.mosaic.mlir)、[结果](call-to-llo/pallas/pallas_double_buffer_dma.results.json) | CPU host 上的 TPU lowering 与模拟检查；记录明确为未 TPU 编译、未真机执行，尚无 LLO |

`jnp.matmul` 的目标路径是 Jaxpr/`dot_general_p` → StableHLO → TPU 编译入口 → 目标后端编译产物 → LLO。Pallas 路径在外层 `tpu_custom_call` 中携带 Mosaic TPU MLIR，再进入 TPU 后端。**Mosaic TPU MLIR 不是 LLO**；CPU 执行、TPU 模拟、离线 TPU 编译和真实 TPU 执行也不可互相代替。运行时二进制与引用源码不匹配时记录 `VERSION-SKEW`。
