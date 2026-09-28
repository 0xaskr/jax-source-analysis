# JAX 到硬件：各层级的核心抽象

本文按从 JAX 前端到硬件接口的层级关系，列出各软件层级的核心抽象。

## 1. 各软件层级的核心抽象

每层选取一个代表性核心抽象；编译后端按硬件及编译路径保留分叉。

| 顺序 | 软件层级 | 核心抽象 |
|---|---|---|
| 1 | JAX 前端层 | **Jaxpr** |
| 2 | jaxlib 原生绑定层 | **MLIR Module（StableHLO）** |
| 3 | IFRT 框架运行时层 | **Array** |
| 4 | PJRT 设备运行时接口层 | **Buffer** |
| 5 | XLA 优化层 | **HLO** |
| 6 | 编译后端层 | **LLO（TPU）/ LLVM IR（CPU）/ LLVM IR、Triton IR（GPU）** |
| 7 | 目标汇编层 | **ASM** |
| 8 | 设备运行时与驱动层 | **执行提交** |
| 9 | 软硬件指令接口层 | **ISA** |

JAX 利用 jaxlib 提供的 MLIR/StableHLO 绑定构造编译输入，再经 IFRT/PJRT 提交后端；编译结果返回运行时加载、执行。表中的 StableHLO 标在相应绑定接口处，其语义由 StableHLO 方言定义。运行时接口的详细说明见第 3 节。

## 2. 后端分叉

| 目标 | 常见的后端表示与代码生成路径 | 边界 |
|---|---|---|
| CPU | LLVM IR → 目标机器码 | 具体指令集取决于目标 CPU，例如 x86-64 或 AArch64。 |
| NVIDIA GPU | 原生 emitter 或 Triton 编译路径 → LLVM IR / PTX → cubin 中的 GPU 机器码 | PTX 是虚拟指令集表示；SASS 是目标 GPU 机器指令的汇编表示。 |
| TPU | TPU 专用后端内部表示 → TPU 设备程序与指令 | `libtpu` 内部阶段无法仅凭当前开源源码完整枚举；LLO 的内容和位置需要与具体版本、编译产物对应。 |

GPU 后端还会选择 cuBLAS、cuDNN、NCCL 等库调用，不是所有 HLO 操作都现场生成一个 kernel。多个操作可以融合，单个操作也可以被分解，不能把一个 Jaxpr primitive 与一个硬件指令一一对应。参见 [XLA GPU 架构](https://openxla.org/xla/gpu_architecture)。

PTX 到物理 GPU 二进制的关系见 [NVIDIA CUDA 平台说明](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)；SASS 与 cubin 的查看方式见 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)。

## 3. 运行时接口层

以下层级负责取得设备、提交编译、加载程序、管理数据以及提交执行。它们连接上述编译层级与设备执行。

| 层级 | 核心抽象 | 主要作用 |
|---|---|---|
| jaxlib 绑定层 | **MLIR Module（StableHLO）** | 为 JAX 提供 MLIR、编译器和运行时的原生绑定。 |
| IFRT | **Array** | 面向框架表示逻辑数组与已加载程序；数组可以分布到多个设备。 |
| PJRT | **Buffer** | 统一设备、数据缓冲区，以及编译、加载和执行接口。 |
| 设备运行时与驱动 | **执行提交** | 对接后端特有的执行与内存机制，并报告完成或错误。 |
| 硬件 | **执行单元** | 执行设备程序，完成读写和通信。 |

例如，PJRT 的编译接口接收程序并返回可执行对象，而执行接口接收输入数据并产生输出。PJRT 不是位于 HLO 和 LLVM IR 之间的一种 IR。参见 [PJRT C++ 接口概览](https://openxla.org/xla/pjrt/cpp_api_overview)。

JAX 侧的绑定、IFRT/PJRT 调用及执行入口，见 [JAX 内部软件栈文档](../jax/jax-concepts.md#jax-与-jaxlib-的分界及下游调用)。

## 4. MLIR、Shardy 与 Pallas/Mosaic 的位置

- **MLIR** 是编译器基础设施。StableHLO、Shardy 的 `sdy`、TritonIR 和 Mosaic 相关方言使用 MLIR 的操作、类型、区域与 pass 机制；MLIR 跨越多个表示层级。
- **Shardy** 关注分片表示、传播与分区。它在张量程序的编译过程中处理分片信息，不是所有程序在 HLO 之后必经的一种低层代码表示。
- **Pallas/Mosaic** 属于专用 kernel 编译分支。Pallas kernel 仍可追踪为 Jaxpr；TPU 路径生成 Mosaic TPU MLIR，并作为外层 custom call 的 payload 交给后端继续处理。
- **Mosaic TPU MLIR 不是 LLO，也不是最终 TPU 机器码。**普通 JAX 程序也不要求先经过 Pallas。

Pallas 与普通 JAX 的两条表示路径见 [本地 JAX 概念文档](../jax/jax-concepts.md#普通-jax-与-pallas-tpu-的表示路径)。
