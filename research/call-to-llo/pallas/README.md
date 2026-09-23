# `manual_double_buffer_add` Pallas 调用链

[固定 kernel](pallas_double_buffer_dma.py)使用一个 `pallas_call`、VMEM 双缓冲和显式 DMA。已有 [外层 StableHLO](pallas_double_buffer_dma.stablehlo.mlir)、从 custom call payload 解码的 [Mosaic TPU MLIR](pallas_double_buffer_dma.mosaic.mlir)及[结果记录](pallas_double_buffer_dma.results.json)。这些文件应按同一例子的调用与 lowering 一起阅读。

结果记录中的 `tpu_compiled`、`tpu_executed`、`hardware_overlap_measured` 均为 `false`。这里尚无 LLO 或真机性能证据。继续时需在匹配的目标 TPU 环境编译该 kernel，保存编译身份、后端产物和 LLO，并与外层调用及内层 payload 对齐。通用机制见 [Pallas 对照](../../software-stack/pallas/pallas-comparison.md)与[表示图](../../software-stack/figures/overview-pallas-lowering.svg)；Mosaic TPU MLIR 不能当作 LLO。
