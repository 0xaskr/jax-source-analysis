# JAX 研究索引

本目录按两种方式阅读：先横向探索整个 JAX 软件栈，或沿一个具体调用纵向追踪到目标 TPU 的 LLO。索引指向已有材料，也标出尚未取得证据的边界；各材料中的版本、运行环境和历史结果以其原始记录为准。

## 一、JAX 软件栈的开放探索

目录入口：[software-stack/README.md](software-stack/README.md)。这部分不预设算子、业务场景或性能目标。按组件和问题寻找源码入口，逐步建立表示、编译、运行与观测之间的关系。

| 探索方向 | 入口 | 关注点 |
|---|---|---|
| JAX 仓库结构 | [JAX 仓库导读](software-stack/jax-repo/README.md) | Python 实现、jaxlib、插件及其源码归属。 |
| 软件栈全貌 | [组件与路径总览](software-stack/overview.md) | Jaxpr、StableHLO、Shardy、XLA、IFRT/PJRT 与 Pallas 的位置；区分编译控制链和程序表示。 |
| 分片与代码生成 | [Shardy 往返](software-stack/shardy-round-trip.md)、[LLVM 与对象文件](software-stack/llvm-and-objects.md) | 分片传播、CPU 代码生成及其观察边界。 |
| 运行时与观测 | [TPU 公开运行接口](software-stack/tpu-runtime-boundary.md)、[profiling](software-stack/profiling.md)、[Roofline](software-stack/roofline.md) | 提交、等待、host/编译/设备事件，以及指标所需的证据。 |
| 自定义 kernel | [普通 JAX 与 Pallas 对照](software-stack/pallas-comparison.md)、[Pallas 两层表示](software-stack/overview.md#pallas-的两层表示) | 从 Pallas 表达到外层调用和内层 Mosaic TPU MLIR 的分叉。 |

这些入口用于发现问题和扩充地图，不作为固定的研究任务队列。引用旧实验或源码行号时，先核对其版本与当前锁文件；CPU 执行、TPU 模拟、离线 TPU 编译和真实 TPU 执行分别记录。

## 二、从两个具体调用下探到 LLO

目录入口：[call-to-llo/README.md](call-to-llo/README.md)。这部分固定调用、输入、JAX/XLA/libtpu 版本和目标 TPU，逐层保存**输入、输出表示与生成它的编译阶段**。LLO 必须来自匹配目标后端的编译产物；Mosaic TPU MLIR 是 Pallas 路径上的另一种表示，不能当作 LLO。

### `jnp.matmul(...)`：普通 JAX 路径

`jnp.matmul` → `dot_general_p` / Jaxpr → `stablehlo.dot_general` → TPU 编译入口 → 目标后端的 HLO/低层编译 → **LLO**。

1. **Python 到 StableHLO**：[matmul 源码走读](call-to-llo/matmul-lowering-walkthrough.md)定位 API、primitive、变换规则与 lowering；[交互 Notebook](call-to-llo/matmul-lowering.ipynb)提供表示对照。走读后半段和现有运行产物是 **CPU 路径**，不能据此认定 TPU 的 pass 或 LLO。
2. **TPU 编译交界**：[软件栈总览](software-stack/overview.md#默认-tpu-提交路径与开源-hlo-条件分支)区分默认 TPU 提交与 HLO 导出等条件分支；[公开运行接口](software-stack/tpu-runtime-boundary.md)定位 PJRT C API 与 libtpu 边界。
3. **到 LLO 的缺口**：尚需在固定目标上取得这次 `jnp.matmul` 编译对应的后端 HLO、pass/dump 与 LLO，并用版本和编译标识把它们接到同一次调用。现有 CPU `DotThunk` 材料不是该 TPU 链路的终点。

### 一个 Pallas kernel：`manual_double_buffer_add`

`pallas_call` / kernel Jaxpr → 外层 `stablehlo.custom_call @tpu_custom_call` ＋ payload 中的 **Mosaic TPU MLIR** → TPU plugin 编译 → **LLO**。

1. **固定调用与 kernel**：[双缓冲 DMA 源码](call-to-llo/examples/pallas_double_buffer_dma.py)包含单次 `grid=()` 的 `pallas_call`、VMEM 双缓冲及显式 DMA；[结果记录](call-to-llo/examples/pallas_double_buffer_dma.results.json)保存当时的 lowering 与解释器检查。
2. **两层表示**：[外层 StableHLO](call-to-llo/examples/pallas_double_buffer_dma.stablehlo.mlir)和[解码出的内层 Mosaic TPU MLIR](call-to-llo/examples/pallas_double_buffer_dma.mosaic.mlir)来自同一 custom call payload；[Pallas 降低层次图](software-stack/figures/overview-pallas-lowering.svg)用于定位两者的关系。
3. **到 LLO 的缺口**：现有记录明确标为 CPU host 上的 TPU lowering 与 `SIM-TPU` 检查，没有 TPU 编译、LLO 或硬件重叠测量。后续需用匹配的 libtpu/目标 TPU 编译该 kernel，取得 LLO，并与上述外层调用和内层 payload 对齐。

两条路径最终都应保留从调用到 LLO 的同一次编译证据。若后端没有提供可读取的 LLO，索引停在已验证的最后一层，并明确记录缺口，不推断私有编译阶段的具体实现。
