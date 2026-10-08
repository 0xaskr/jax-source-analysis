# JAX 软件栈：总览与组件研究

[总览文档](overview/overview-software-stack-components-layered.md)与[总览图](overview/overview-software-stack-components-layered.svg)说明 JAX / Pallas 程序从追踪、编译到执行的整体路径。下面的组件图文进一步研究每一层的核心抽象。

## 组件入口

每篇正文和对应 SVG 都围绕四个问题展开：**定义**说明抽象表示什么、由什么组成及其约束；**产生**追踪输入与构造者；**变换**说明规则及改变的内容；**消费**说明谁使用它、如何使用以及得到什么。具体类、函数和调用位置链接到锁定版本源码。

| 组件 | 核心抽象 | 文档 | SVG |
|---|---|---|---|
| JAX | Jaxpr：由原语及其组合表达的程序 | [Jaxpr](jax/jaxpr-centered-hub.md) | [组件图](jax/jaxpr-centered-hub.svg) |
| jaxlib | MLIR Module：承载 StableHLO、Mosaic 等方言的程序模块 | [MLIR Module](jaxlib/mlir-module-centered-hub.md) | [组件图](jaxlib/mlir-module-centered-hub.svg) |
| IFRT | Array：带类型、形状和分片信息的逻辑数组 | [Array](ifrt/array-centered-hub.md) | [组件图](ifrt/array-centered-hub.svg) |
| PJRT | PjRtBuffer：设备存储及其所有权、就绪状态 | [PjRtBuffer](pjrt/buffer-centered-hub.md) | [组件图](pjrt/buffer-centered-hub.svg) |
| XLA | HloModule：计算及指令组成的编译程序 | [HLO](xla/hlo-centered-hub.md) | [组件图](xla/hlo-centered-hub.svg) |
| XLA CPU 后端 | LLVM IR：CPU 代码生成使用的低层程序表示 | [CPU LLVM IR](xla/cpu-llvm-ir-centered-hub.md) | [组件图](xla/cpu-llvm-ir-centered-hub.svg) |
| XLA GPU 后端 | GPU LLVM IR：GPU kernel 的低层程序表示 | [GPU LLVM IR](xla/gpu-ir-centered-hub.md) | [组件图](xla/gpu-ir-centered-hub.svg) |
| TPU provider / libtpu | LLO：按公开证据界定已知内容与内部缺口 | [LLO 的证据边界](libtpu/llo-boundary-centered-hub.md) | [组件图](libtpu/llo-boundary-centered-hub.svg) |
| 设备运行时与驱动 | 执行请求：程序、存储、依赖和提交参数 | [执行请求](stream-executor/submission-centered-hub.md) | [组件图](stream-executor/submission-centered-hub.svg) |

[Pallas 的内外层程序](jax/jaxpr-centered-hub.md#pallas)随 Jaxpr 的四个角度展开。TPU 示例说明 kernel Jaxpr 如何生成 Mosaic TPU MLIR，再接回外层 custom call；Mosaic TPU MLIR 不称为 LLO。

组件图沿用 overview 的配色和卡片样式，左侧说明定义，右侧三个区域展示产生、变换和消费。连线直接标明传递对象或调用动作，虚线结合颜色表达可选分支、返回或观察关系；交叉拱桥表示连线互不连接。各区可独立阅读，变换按具体调用选择。点击源码链接可以查看实现，顶部导航可返回 overview 或进入相邻组件。

## 图文维护

九张组件 SVG 由 [`render_software_stack_component_flows.py`](../../tools/render_software_stack_component_flows.py) 生成，Markdown 独立维护。内容修改后需要同时核对两者。

| 内容 | 维护位置 |
|---|---|
| JAX 与公共对象流定义 | [`software_stack_component_flow_data.py`](../../tools/software_stack_component_flow_data.py)、[`render_jaxpr_centered_hub.py`](../../tools/render_jaxpr_centered_hub.py) |
| jaxlib、XLA、CPU 与 GPU 后端 | [`software_stack_compiler_views.py`](../../tools/software_stack_compiler_views.py) |
| IFRT、PJRT、设备运行时 | [`software_stack_runtime_views.py`](../../tools/software_stack_runtime_views.py) |
| TPU 公开边界与证据缺口 | [`software_stack_tpu_views.py`](../../tools/software_stack_tpu_views.py) |
| 固定源码锚点与缓存校验 | [`component_diagram_sources.py`](../../tools/component_diagram_sources.py)、[`component_diagram_sources.json`](../../tools/component_diagram_sources.json) |

绘图环境由 [`env/diagrams.lock.json`](../../env/diagrams.lock.json) 锁定，使用 Ubuntu 24.04 amd64、Python 3.12、Pango、Rsvg、cairo 和 Noto CJK 字体。首次安装仅解压到工作区内已忽略的缓存目录：

```bash
python3 -B tools/diagram_environment.py sync
python3 -B tools/diagram_environment.py check
```

先生成到临时目录进行检查：

```bash
python3 -B tools/diagram_environment.py run tools/render_software_stack_component_flows.py --output-dir /tmp/component-flows
```

去掉 `--output-dir` 即更新九张组件 SVG。可用 `--component jax` 单独生成 JAX 图，其他取值为 `jaxlib`、`ifrt`、`pjrt`、`xla`、`cpu`、`gpu`、`tpu`、`runtime`。组件生成器不重建 overview。

生成时核对源码修订、文件内容和符号位置，并检查核心抽象数量、节点归属、必要路径、连线与文字边界、图内链接。固定源码检出缺失时，可使用清单中的 SHA-256 校验缓存；首次添加 `--fetch-sources` 下载锁定提交的文件。静态检查完成后还需查看 SVG 的实际渲染结果。

版本依据是 [`upstream-sources.lock`](../../upstream-sources.lock)、[`source-archives.lock`](../../source-archives.lock)及环境锁文件。源码检查、CPU 执行、TPU 模拟、离线 TPU 编译和真实 TPU 执行分别取证；运行时二进制与所引源码不一致时标记 `VERSION-SKEW`。

[`note.md`](note.md) 是开发者笔记，保持原样。
