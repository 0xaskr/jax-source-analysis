# 从固定调用追踪到 LLO

本目录只处理两条具体调用链：[普通 `jnp.matmul`](#jnp-matmul) 和 [Pallas 双缓冲 kernel](#pallas-kernel)。每条链应固定输入、源码/运行时版本和目标 TPU，并保存同一次编译中各阶段的产物。当前材料尚未拿到匹配目标的 LLO，因此这里是可继续延伸的追踪记录，不宣称已经走通终点。

## `jnp.matmul`

固定调用的 [源码走读](matmul-lowering-walkthrough.md)、[交互 Notebook](matmul-lowering.ipynb) 与 [逐 pass 记录](matmul-pass-walkthrough.md)；[链路图](figures/matmul-lowering-chain.svg) 由 [生成脚本](render_matmul_walkthrough.py) 维护。旧 Notebook 和逐 pass 结果包含 CPU 运行及 CPU 后端证据，不能推导出 TPU 的后端 pass 或 LLO。共用的 [matmul capture 脚本](../software-stack/matmul_probe.py) 和 [软件栈总览](../software-stack/overview.md) 在全栈目录。

已定位 `jnp.matmul`、Jaxpr/`dot_general_p` 和 StableHLO lowering；下一层需要为同一次调用取得目标 TPU 的 HLO、后端 dump 和 LLO，并记录 JAX、jaxlib、libtpu 与目标设备的身份。

## Pallas kernel

固定例子是 `manual_double_buffer_add`：[kernel 源码](examples/pallas_double_buffer_dma.py)、[外层 StableHLO](examples/pallas_double_buffer_dma.stablehlo.mlir)、[payload 中的 Mosaic TPU MLIR](examples/pallas_double_buffer_dma.mosaic.mlir)、[结果记录](examples/pallas_double_buffer_dma.results.json)。它们展示 `pallas_call` 到 `tpu_custom_call` 的两层表示。

结果记录中的 `tpu_compiled`、`tpu_executed` 和 `hardware_overlap_measured` 都是 `false`。要继续下探，需要对该 kernel 在匹配的 TPU 环境中编译，并将所得 LLO 与这次调用的外层 custom call 和内层 payload 对齐；Mosaic TPU MLIR 本身不是 LLO。通用 Pallas 对照与 [层次图](../software-stack/figures/overview-pallas-lowering.svg) 在全栈目录。
