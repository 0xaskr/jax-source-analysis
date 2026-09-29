# JAX 软件栈开放探索

这里从组件和问题出发探索 JAX 软件栈，不预设算子、kernel 或性能目标。当前目录保存概念文档与图示入口；原有专题文档、实验结果和计划已移除。源码以本地 [`upstream/`](../../upstream) 检出为准，修订版本与来源见 [`upstream-sources.lock`](../../upstream-sources.lock)、[`source-archives.lock`](../../source-archives.lock) 和 [`env/environment.lock.json`](../../env/environment.lock.json)。

`upstream/jax` 当前位于以 JAX v0.11.1 为基础的源码注释提交。XLA、StableHLO、Shardy 和 LLVM 的本地检出采用稀疏范围；目录中找不到某个文件时，应先确认它是否尚未落地。
note.md 是开发者的理解笔记，不要去修改。

## 图示的视角与深度

`overview/` 从整体视角描述整个软件栈：组件的职责与位置、跨组件调用、程序表示的转换，以及编译、加载和执行路径。这里的“概览”限定的是观察范围，不限定细节程度；为了讲清完整路径，可以展开关键 API、class 和源码边界，图示也不受固定画布尺寸限制。以内容完整、文字可读和连线清楚为准。

`overview/` 以外的一级目录直接用组件名命名。现有 [`jax/`](jax/index.md) 收录 JAX 概念文档与组件内部架构图；以后分析其他顶层组件时，再以该组件名建目录。组件目录聚焦其内部机制，细节应比全栈图更深入，进入具体 API、class、方法及其调用和数据流。组件图文可以标出与外部的接口，但主体应是当前组件内部；需要解释多个组件如何衔接时，放在 `overview/`。图示归属由要解释的范围决定，不由节点数量或细节多少决定。

原版全栈总图保留在 [`overview-software-stack.svg`](overview/overview-software-stack.svg)。总图中的 `pallas_add` 片段用于说明表示层次，原离线捕获已移除，不能据此认定当前环境完成了编译或设备执行。

各层级的核心抽象及其关系见 [JAX 到硬件：各层级的核心抽象](overview/jax-to-hardware-abstractions.md)。

当前选定的全栈图为[浅色分层架构版](overview/overview-software-stack-components-layered.svg)，将系统架构、跨组件流程和组件内部协作画在同一张图上，自上而下展开前端、运行时接口、编译后端和设备执行。每个组件用大框包裹，首要抽象用强调色突出，内部连线说明定义、产生、变换和消费关系；编译程序与运行时数据在执行提交处汇合。后续优化沿用这一格式。

`overview/` 仅保留原版全栈总图和浅色分层架构版这两张 SVG。分层图的独立生成脚本为 [`render_overview_software_stack_components.py`](../../tools/render_overview_software_stack_components.py)。运行 `python3 -B tools/render_overview_software_stack_components.py --preview-dir /tmp/jax-stack-components-preview` 仅重建选定 SVG 并输出预览；另加 `--output-dir /tmp/jax-stack-components-svg` 可先生成到临时目录。脚本核验固定源码、文字边界、文字重叠、连线穿越节点与组件内节点归属；仍需视觉审阅。ASM 标为表示层，GPU 库调用保留旁路，TPU 内部阶段继续标明证据边界。本图不构成设备执行或性能验证。

## 组件内部：核心抽象中心辐射图

组件图采用已选定的 Jaxpr 中心辐射格式：中央突出研究对象，上方说明定义，左侧说明产生，下方展示以该对象为输入和输出的变换，右侧列出消费者及结果。所有图均为完整 SVG，可放大阅读具体类、函数与固定提交源码链接；不使用固定模型示例。

| 组件 / 内部子系统 | 核心抽象与 SVG | 范围 |
|---|---|---|
| JAX | [Jaxpr](jax/jaxpr-centered-hub.svg) | 定义、追踪构造、可组合变换、解释与 lowering |
| jaxlib | [MLIR Module（StableHLO）](jaxlib/mlir-module-centered-hub.svg) | 绑定、模块结构、构造与 pass、编译提交与序列化 |
| IFRT | [Array](ifrt/array-centered-hub.svg) | 逻辑数组、分片、复制 / 重组、执行与就绪状态 |
| PJRT | [Buffer](pjrt/buffer-centered-hub.svg) | 设备存储、复制 / bitcast / donation、执行与生命周期 |
| XLA | [HLO](xla/hlo-centered-hub.svg) | 模块 / 计算 / 指令、导入与重写、后端消费 |
| XLA CPU 后端 | [LLVM IR](xla/cpu-llvm-ir-centered-hub.svg) | Module 构造、优化与拆分、机器码和 CPU 执行计划 |
| XLA GPU 后端 | [LLVM IR / Triton IR](xla/gpu-ir-centered-hub.svg) | 多条代码生成路径、设备库边界、目标代码与执行计划 |
| libtpu / TPU 后端 | [LLO 的证据边界](libtpu/llo-boundary-centered-hub.svg) | 公开插件输入输出与待确认的私有表示 / pass / 消费者 |
| StreamExecutor / CUDA | [执行提交](stream-executor/submission-centered-hub.svg) | kernel / 传输提交、依赖、命令记录、驱动调用与完成状态 |

新增八张图的画布均为 10300 × 6700。CPU / GPU 后端属于 XLA，图片保存在 `xla/` 下；ASM 和 ISA 在后端及执行图中作为表示与契约解释，不为它们虚构独立软件组件。设备运行时图以公开 CUDA 实现为具体路径，不代表 CPU / TPU 也使用相同机制。

除 JAX 外的八张图由 [`render_software_stack_component_hubs.py`](../../tools/render_software_stack_component_hubs.py) 生成，内容和锚点保存在 [`software_stack_component_hub_data.py`](../../tools/software_stack_component_hub_data.py)。运行 `python3 -B tools/render_software_stack_component_hubs.py --output-dir /tmp/component-hubs/svg --preview-dir /tmp/component-hubs/png` 可先重建到临时目录；去掉 `--output-dir` 更新仓库 SVG。使用 `--component jaxlib|ifrt|pjrt|xla|cpu|gpu|tpu|runtime` 中的一个值可单独生成一张。

脚本核对锁定的 JAX、XLA、LLVM 与 StableHLO 修订、源码文件内容和符号行号，再校验核心抽象关系端点、文字边界和连线。稀疏检出中未展开的文件从固定 Git 对象读取，不修改源码检出。图中灰虚线表示定义约束，棕虚线表示待确认关系。TPU 图未声称获得 LLO 实物或确认私有 pass；Mosaic TPU MLIR 不作为 LLO。全部图示证据仅来自源码与静态生成检查。

## 从源码进入

| 方向 | 本地入口 | 阅读入口 |
|---|---|---|
| JAX Python、tracing、Jaxpr 与 jaxlib | [JAX](../../upstream/jax) | [JAX 概念](jax/index.md)、[Jaxpr 中心辐射图](jax/jaxpr-centered-hub.svg) |
| 程序表示与分片 | [StableHLO](../../upstream/stablehlo)、[Shardy](../../upstream/shardy) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| 编译控制与后端路径 | [XLA](../../upstream/xla) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| LLVM/MLIR 与 CPU 代码生成 | [LLVM](../../upstream/llvm-project) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| Pallas 的通用 lowering 机制 | [JAX Pallas](../../upstream/jax/jax/_src/pallas)、[XLA Mosaic](../../upstream/xla/xla/mosaic) | [Pallas 分支](jax/jax-internal-stack.svg) |

这些图文帮助定位源码入口。图中的路径和行号属于制作时的记录，引用前需与当前锁定源码重新核对。源码检查、CPU 执行、TPU 模拟、离线 TPU 编译与真实 TPU 执行分别取证；运行时与所引源码不一致时标记 `VERSION-SKEW`。Mosaic TPU MLIR 不是 LLO。

固定 `jnp.matmul` 与双缓冲 Pallas kernel 的逐调用追踪见 [call-to-llo](../call-to-llo/README.md)。
