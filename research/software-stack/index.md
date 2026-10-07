# JAX 软件栈开放探索

这里从组件和问题出发探索 JAX 软件栈，不预设算子、kernel 或性能目标。当前目录保存概念文档与图示入口；原有专题文档、实验结果和计划已移除。源码以本地 [`upstream/`](../../upstream) 检出为准，修订版本与来源见 [`upstream-sources.lock`](../../upstream-sources.lock)、[`source-archives.lock`](../../source-archives.lock) 和 [`env/environment.lock.json`](../../env/environment.lock.json)。

`upstream/jax` 当前位于以 JAX v0.11.1 为基础的源码注释提交。XLA、StableHLO、Shardy 和 LLVM 的本地检出采用稀疏范围；目录中找不到某个文件时，应先确认它是否尚未落地。
note.md 是开发者的理解笔记，不要去修改。

## 图示的视角与深度

`overview/` 从整体视角描述整个软件栈：组件的职责与位置、跨组件调用、程序表示的转换，以及编译、加载和执行路径。这里的“概览”限定的是观察范围，不限定细节程度；为了讲清完整路径，可以展开关键 API、class 和源码边界，图示也不受固定画布尺寸限制。以内容完整、文字可读和连线清楚为准。

`overview/` 以外的一级目录直接用组件名命名。现有 [`jax/`](jax/jaxpr-centered-hub.md) 以唯一一组同名 SVG / Markdown 组织 JAX 概念、Jaxpr 格式、内部机制和 Pallas 分支；以后分析其他顶层组件时，再以该组件名建目录。组件目录聚焦其内部机制，细节应比全栈图更深入，进入具体 API、class、方法及其调用和数据流。组件图文可以标出与外部的接口，但主体应是当前组件内部；需要解释多个组件如何衔接时，放在 `overview/`。图示归属由要解释的范围决定，不由节点数量或细节多少决定。

各层级的核心抽象及其关系见 [JAX 到硬件：各层级的核心抽象](overview/jax-to-hardware-abstractions.md)。

当前选定的全栈图为[分层架构：编译与执行的完整往返](overview/overview-software-stack-components-layered.svg)，配套说明见 [Markdown 文档](overview/overview-software-stack-components-layered.md)。图中每个模块只突出一个核心概念，CPU / GPU / TPU 后端分栏各算一个模块；结构成员、接口、辅助表示和返回对象作为普通节点展开。文档按定义、产生、变换、消费解释核心概念，并对照图中端口说明调用和返回关系。

新增[扩展与观测接口版](overview/overview-software-stack-components-extended.svg)，在当前分层图上吸收旧总图的 A–E 扩展接口，并补充 F 运行观测入口。图内增加 HLO hook 的调用与返回关系；右侧接口卡片和主图通过字母按钮互相定位，核心概念仍为每个模块一个。两张分层图保留作对照；旧总图已移除，原引用迁到新图的对应接口。接口解释见[配套文档](overview/overview-software-stack-components-layered.md#extension-interfaces)。

扩展版由 [`render_overview_software_stack_extensions.py`](../../tools/render_overview_software_stack_extensions.py) 生成：`python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_extensions.py`。它仅输出 `overview-software-stack-components-extended.svg`，复用分层图布局并额外核验扩展接口的固定源码、HLO 回调关系和定位链接。

组件标题可进入详细图，编号端口和底部索引可定位图内对应节点。所有跨框流程均有沿线标签，拱桥表示交叉线互不连接；虚线关系结合颜色和文字区分按需分支、返回、观察与待确认。图和生成器统一使用“核心概念”措辞。

当前图采用浅色分层架构的组件分布：JAX / jaxlib 居中，IFRT / PJRT、XLA / 编译后端、ASM / 设备运行时左右并列，硬件位于底部。组件内沿用三列节点，大框角色标记区分实现、接口、可观察表示与硬件契约，侧栏说明 MLIR 的跨层作用；画布按内容和连线需要扩展。CPU/GPU 编译产物经 provider、PJRT、IFRT 和 Python 逐层包装，输出按 Buffer → Array → PyArray 返回，完成状态单独表达。Pallas 的 Mosaic payload 经外层 `stablehlo.custom_call` 接入 Module，Shardy 展示按条件运行的模块 pass，TPU provider 单列公开边界。API 与传递对象直接标在跨框连线旁。

核心概念依次为 JAX 的 Jaxpr、jaxlib 的 MLIR Module、IFRT 的 Array、PJRT 的 PjRtBuffer、XLA 的 HLO、CPU 和 GPU 后端各自的 LLVM IR、TPU 后端的 LLO 证据边界、ASM、运行时的执行提交，以及硬件层的 ISA。GPU 的 Triton 路径与设备库旁路保留在相应流程中。

当前图由 [`render_overview_software_stack_flows.py`](../../tools/render_overview_software_stack_flows.py) 生成，节点内容和唯一核心概念映射保存在 [`software_stack_overview_flow_data.py`](../../tools/software_stack_overview_flow_data.py)。命令为 `python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_flows.py`，仅更新正式文件 `overview-software-stack-components-layered.svg`；可用 `--output-dir /tmp/jax-stack-flows` 先生成到临时目录。脚本核验固定源码、每个模块恰好一个核心概念、编译 / 执行 / 输出 / Pallas 路径连通性、节点归属、跨框连线避让与排版。XLA 完整检出缺失时，使用 [`overview_flow_sources.json`](../../tools/overview_flow_sources.json) 中锁定提交及 SHA-256 的源码文件缓存；首次可加 `--fetch-sources` 读取固定提交，存入已忽略的 `artifacts/environment/overview-sources/`。后续可离线重建；缺失或被改动的源码会使校验失败。

[`render_overview_software_stack_components.py`](../../tools/render_overview_software_stack_components.py) 继续提供共享绘图能力，其命令入口转交当前生成器，接受相同参数并只输出 SVG。两个入口均重建当前版本。

绘图环境由 [`env/diagrams.lock.json`](../../env/diagrams.lock.json) 锁定 Ubuntu 24.04 amd64 的 PyGObject、Pango、Rsvg、cairo、原生依赖和 Noto CJK 字体包版本、下载地址与 SHA-256，使用 Python 3.12。首次运行 `python3 -B tools/diagram_environment.py sync` 下载并校验依赖，仅解压到已忽略的 `artifacts/environment/diagrams/`；不需要 sudo，也不修改 JAX 虚拟环境。`python3 -B tools/diagram_environment.py check` 核验已安装文件、中文常规 / 粗体字形和 SVG → PNG 渲染。

使用 `diagram_environment.py run` 可固定进程内的原生库、绑定和字体路径；直接执行绘图脚本也能发现仓库内依赖，`--help` 不加载绘图库。字体测量使用 Noto Sans CJK SC，SVG 保留系统字体后备。

绘图环境检查与 `tools/sync-environment.py check` 的 JAX 工作区检查分别进行；绘图成功不代表缺失的上游源码已恢复，也不构成设备执行或性能验证。

Pallas 的内外层程序整合在 JAX 图文的四个研究角度中，可从 [kernel Jaxpr](jax/jaxpr-centered-hub.svg#inner_jaxpr) 与 [Pallas 阅读入口](jax/jaxpr-centered-hub.md#pallas) 定位。图文说明外层 `pallas_call` 方程如何携带内层 kernel Jaxpr 与 GridMapping，以及 Mosaic TPU 模块如何经序列化成为外层 `stablehlo.custom_call` 的 payload。同一文档还包括 [JAX 概念](jax/jaxpr-centered-hub.md#jax-concepts)、[Jaxpr 格式](jax/jaxpr-centered-hub.md#jaxpr-format)、[jaxlib 分界](jax/jaxpr-centered-hub.md#jaxlib-boundary)、[表示路径](jax/jaxpr-centered-hub.md#representation-paths)及[重建命令](jax/jaxpr-centered-hub.md#rebuild)。

## 组件内部：核心概念与对象流转

九张组件主图采用当前 overview 的浅色分区、唯一核心概念、流程配色和命名连线。卡片标明实际归属，编号连线和底部索引可定位两端，标题导航可返回 overview 对应模块或进入相邻组件。原 `*-centered-hub.svg` 文件名保留，以维持已有引用。

九张组件图统一将定义独立放在左侧，集中说明核心抽象的语义、结构与约束；产生、变换、消费横向分成三块，以实际调用和对象传递连通。定义卡不充当执行阶段，操作区使用具体实例；可选变换与直接消费支路按源码分别表达。处理其他对象的后续流程位于灰色辅助区。对应 Markdown 按相同四个角度解释具体类、函数、调用位置与输入输出。

JAX 可从 [定义](jax/jaxpr-centered-hub.svg#view_definition)、[产生](jax/jaxpr-centered-hub.svg#view_production)、[变换](jax/jaxpr-centered-hub.svg#view_transformation)、[消费](jax/jaxpr-centered-hub.svg#view_consumption)进入；Pallas 的内外层程序与专用规则沿相应路径展开，Module 序列化、编译缓存和数组执行放在[消费结果的后续去向](jax/jaxpr-centered-hub.svg#downstream)。其他组件沿用这些分区锚点。TPU / libtpu 的四区明确标出 LLO 证据缺口，已核验的[公开接口调用链](libtpu/llo-boundary-centered-hub.svg#downstream)独立展开，不补造内部流转。

| 组件 / 内部子系统 | 核心抽象与 SVG | 配套 Markdown | 范围 |
|---|---|---|---|
| JAX | [Jaxpr](jax/jaxpr-centered-hub.svg) | [文档](jax/jaxpr-centered-hub.md) | 定义、产生、变换、消费四视角；Pallas 实例与消费结果的后续去向 |
| jaxlib | [MLIR Module（StableHLO）](jaxlib/mlir-module-centered-hub.svg) | [文档](jaxlib/mlir-module-centered-hub.md) | 绑定、模块结构、构造与 pass、编译提交与序列化 |
| IFRT | [Array](ifrt/array-centered-hub.svg) | [文档](ifrt/array-centered-hub.md) | 逻辑数组、分片、复制 / 重组、执行与就绪状态 |
| PJRT | [Buffer](pjrt/buffer-centered-hub.svg) | [文档](pjrt/buffer-centered-hub.md) | 设备存储、复制 / bitcast / donation、执行与生命周期 |
| XLA | [HLO](xla/hlo-centered-hub.svg) | [文档](xla/hlo-centered-hub.md) | 模块 / 计算 / 指令、导入与重写、后端消费 |
| XLA CPU 后端 | [LLVM IR](xla/cpu-llvm-ir-centered-hub.svg) | [文档](xla/cpu-llvm-ir-centered-hub.md) | Module 构造、优化与拆分、机器码和 CPU 执行计划 |
| XLA GPU 后端 | [GPU LLVM IR](xla/gpu-ir-centered-hub.svg) | [文档](xla/gpu-ir-centered-hub.md) | 原生 / Triton / 设备库分支、独立 kernel 编译、目标代码与执行计划 |
| libtpu / TPU 后端 | [LLO 的证据边界](libtpu/llo-boundary-centered-hub.svg) | [文档](libtpu/llo-boundary-centered-hub.md) | 公开插件输入输出与待确认的私有表示 / pass / 消费者 |
| StreamExecutor / CUDA | [执行提交](stream-executor/submission-centered-hub.svg) | [文档](stream-executor/submission-centered-hub.md) | kernel / 传输提交、依赖、命令记录、驱动调用与完成状态 |

每张组件图与同目录、同名 Markdown 一一对应。文档围绕唯一核心概念说明定义、产生、变换与消费，按图中的节点和关系解释主要路径，并列出具体接口的输入输出、可选分支和固定提交源码。先读文档理解语义，再通过节点链接在 SVG 中定位。

画布宽高与卡片高度按内容生成，不固定统一尺寸。正文使用 29px，主节点标题使用 36px，唯一核心概念使用填色粗框。CPU / GPU 后端属于 XLA，图片保存在 `xla/` 下；ASM 和 ISA 作为表示与契约解释。设备运行时图展开 kernel、传输和 CommandBuffer 三条提交分支，明确提交 Status 与异步完成的不同观察点；具体实现以 CUDA 为例。TPU 图只将公开 ABI 连接成编译与执行路径，LLO 核心概念及四视角证据缺口单独保留，未画出假定的私有 pass。

九张主图统一由 [`render_software_stack_component_flows.py`](../../tools/render_software_stack_component_flows.py) 生成；流程与组件归属保存在 [`software_stack_component_flow_data.py`](../../tools/software_stack_component_flow_data.py)，四视角基础内容复用 [`software_stack_component_hub_data.py`](../../tools/software_stack_component_hub_data.py) 和 Jaxpr 内容表。编译组件的四区编排在 [`software_stack_compiler_views.py`](../../tools/software_stack_compiler_views.py)，运行时组件在 [`software_stack_runtime_views.py`](../../tools/software_stack_runtime_views.py)，TPU 证据边界在 [`software_stack_tpu_views.py`](../../tools/software_stack_tpu_views.py)。运行 `python3 -B tools/diagram_environment.py run tools/render_software_stack_component_flows.py --output-dir /tmp/component-flows` 可先生成到临时目录；去掉 `--output-dir` 更新仓库 SVG。使用 `--component jax|jaxlib|ifrt|pjrt|xla|cpu|gpu|tpu|runtime` 可单独生成一张。生成器只输出 SVG；旧 `render_software_stack_component_hubs.py` 和 `render_jaxpr_centered_hub.py` 命令转交新生成器。

[`component_diagram_sources.py`](../../tools/component_diagram_sources.py) 核对源码锁、实际 Git 根目录与修订、文件内容和符号行号。完整检出缺失时，使用 [`component_diagram_sources.json`](../../tools/component_diagram_sources.json) 与 overview 清单中的固定提交及 SHA-256 文件缓存；首次加 `--fetch-sources` 可填充已忽略的 `artifacts/environment/overview-sources/`。通常重建离线完成，缺失或被改动的证据会使校验失败。生成时检查每图一个核心概念、必需流程、命名关系、正交连线、独立连线不共线、文字边界与图内跳转目标。Mosaic TPU MLIR 不作为 LLO。全部图示证据仅来自源码与静态生成检查。

## 从源码进入

| 方向 | 本地入口 | 阅读入口 |
|---|---|---|
| JAX Python、tracing、Jaxpr 与 jaxlib | [JAX](../../upstream/jax) | [JAX / Jaxpr 文档](jax/jaxpr-centered-hub.md)、[内部架构图](jax/jaxpr-centered-hub.svg) |
| 程序表示与分片 | [StableHLO](../../upstream/stablehlo)、[Shardy](../../upstream/shardy) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| 编译控制与后端路径 | [XLA](../../upstream/xla) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| LLVM/MLIR 与 CPU 代码生成 | [LLVM](../../upstream/llvm-project) | [软件栈全貌](overview/overview-software-stack-components-layered.svg) |
| Pallas 的通用 lowering 机制 | [JAX Pallas](../../upstream/jax/jax/_src/pallas) | [JAX 图中的 Pallas 分支](jax/jaxpr-centered-hub.svg#inner_jaxpr)、[Pallas 章节](jax/jaxpr-centered-hub.md#pallas) |

这些图文帮助定位源码入口。图中的路径和行号属于制作时的记录，引用前需与当前锁定源码重新核对。源码检查、CPU 执行、TPU 模拟、离线 TPU 编译与真实 TPU 执行分别取证；运行时与所引源码不一致时标记 `VERSION-SKEW`。Mosaic TPU MLIR 不是 LLO。

固定 `jnp.matmul` 与双缓冲 Pallas kernel 的逐调用追踪见 [call-to-llo](../call-to-llo/README.md)。
