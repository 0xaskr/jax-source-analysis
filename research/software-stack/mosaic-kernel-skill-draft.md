# Mosaic ↔ Pallas Kernel：流水表示与转换方案（草稿）

> 本地润色稿，基于白桦的 [《mosaic <-> kernel skill（Draft）》](https://outline.infiscale-tech.com/doc/mosaic-kernel-skilldraft-PhutY0NlEA) revision 18，读取于 2026-09-16。本文聚焦 TPU 路径中的 Mosaic TPU MLIR，下文简称 Mosaic IR。

## 背景：上层如何把流水交给底层 Agent

本文对应两项需求：

- [基于流水的 Kernel 编写（INTERINFRA-103）](https://plane.infiscale-tech.com/infra/browse/INTERINFRA-103/)
- [基于 Kernel LLO 与子图流水进行理论与编译产物分析（INTERINFRA-104）](https://plane.infiscale-tech.com/infra/browse/INTERINFRA-104/)

按照 Otter 的设计，上层负责规划流水，底层 Agent 根据流水编写 kernel。要让这两个环节衔接起来，首先需要回答：**流水应该用什么形式表达，才能让底层 Agent 准确实现上层的设计？**

前期对 Google accelerator-agents / MaxKernel Agent 的分析，主要关注其围绕具体参数和代码组织输入、输出的方式。这些材料提供了 kernel 生成与优化流程的参考，但尚未直接解决这里的流水表示问题。相关内容见 [Google accelerator-agents / MaxKernel Agent 分析](https://outline.infiscale-tech.com/doc/google-accelerator-agentsmaxkernel-agent-oX9ZSiuyLW)。

## 流水需要表达什么

本文所说的流水，重点是**数据搬运与计算在时间上的重叠（overlap）**。例如，在计算当前数据块时，提前搬运后续数据块，以减少计算等待数据的时间。

因此，流水表示需要描述的不只是计算步骤，还包括这些步骤的执行条件：

| 内容 | 需要表达的信息 |
| --- | --- |
| 计算 | 每个阶段执行什么操作，处理哪个数据块 |
| 搬运 | 数据从哪里搬到哪里，何时发起搬运 |
| 缓冲区 | 数据存放的位置，以及缓冲区何时可以复用 |
| 依赖与同步 | 计算前需要等待什么，搬运完成后如何通知后续步骤 |
| 迭代与重叠 | 不同迭代中的搬运、计算如何交错安排 |

只有这些信息足够明确，底层 Agent 才能据此实现预期流水。IR 能否表达这些关系，以及编译后是否在设备上形成有效的 overlap，需要分别验证。

## 为什么考虑 Mosaic IR

StableHLO 能够表达张量计算语义，例如矩阵乘法、加法和控制流。但对于这里关注的 kernel 内部流水，仅有这些计算语义，还不足以明确约束数据搬运、缓冲区复用与同步安排。

在 StableHLO 层，可以尝试用自定义属性标注切分维度、循环维度或并行维度，例如 `split_dim`、`loop_dim`、`parallel_dim`。这些名称应当视为待定义的扩展约定；需要相应的编译 pass 或后端逻辑解释它们，才能将标注意图落实为执行安排。

StableHLO 还提供了 `custom_call` 扩展接口，用于承载由具体实现定义的操作。因此，更准确的问题是：**我们需要的流水信息，应该直接放在哪一层 IR 中，才能便于生成、检查和转换？**

Mosaic IR 是一个值得验证的候选。在 Pallas 的 TPU 编译路径中，kernel Jaxpr 会被 lowering 为 Mosaic TPU MLIR。相关 lowering 代码包含 DMA 发起、DMA 等待和信号量操作，能够显式承载与搬运、同步有关的信息。它因此比只描述张量计算的表示更接近本文所需的 kernel 流水语义。

基于这一点，我们拟探索以 **Mosaic IR 作为上层流水设计与底层 Pallas kernel 实现之间的交接表示**。落地时还需要固定具体的 IR 阶段、版本和支持范围，避免将不同 lowering 阶段的产物混为同一种接口。

## Pallas kernel 如何进入 StableHLO

理解这条路径，需要区分外层程序中的调用节点与 kernel 内部的 IR。

在本地固定版本的 JAX 源码中，Pallas TPU kernel 的主要处理过程如下：

```text
Pallas kernel Python
    ↓ tracing
kernel Jaxpr
    ↓ lowering
Mosaic TPU MLIR
    ↓ Mosaic 序列化处理、写出 MLIR bytecode、Base64 编码
backend_config.custom_call_config.body
    ↓ 作为调用配置嵌入外层程序
stablehlo.custom_call @tpu_custom_call
```

具体而言，`lower_jaxpr_to_pipelined_module(...)` 生成 Mosaic module；随后，序列化逻辑将 module 写为 MLIR bytecode，经 Base64 编码后放入 `backend_config` 中的 `custom_call_config.body`。外层程序通过目标名为 `tpu_custom_call` 的调用节点携带这份配置。

因此，在外层 StableHLO 文本中，kernel 的内部计算没有展开为普通的 StableHLO 算子，而是以序列化 IR 的形式随调用一起传递。直接阅读文本时，通常只能看到编码后的 body。

这种封装符合 `custom_call` 承载实现相关操作与配置的接口用途。仅凭 body 被编码，尚不能将其判断为临时性的 workaround。对分析工具而言，更直接的需求是：**提取内嵌的 kernel IR，并通过匹配版本的反序列化与打印流程，将其转换为可阅读、可检查的表示。** Base64 解码本身只会得到字节内容，并不等同于还原可读的 Mosaic 文本，更不等同于还原 Pallas 源码。

## Mosaic ↔ Pallas 的转换目标

围绕上述交接表示，我们希望建设 Mosaic 与 Pallas kernel 之间的转换工具。两个方向需要分别界定。

**Pallas → Mosaic：复用已有 lowering 路径，导出 kernel IR。**

这一方向已有编译路径作为基础。工具需要明确导出位置与 IR 阶段，使调用者能够直接检查计算、搬运和同步关系。对于已经封装在外层 StableHLO 中的 kernel，也可以探索从 `custom_call` 配置中提取其内嵌 IR。

**Mosaic → Pallas：在限定范围内，探索生成满足相同计算语义与流水约束的 kernel。**

这一方向仍需验证。已有的正向 lowering 路径，并不能直接证明任意 Mosaic IR 都可以反向转换为 Pallas，也不能保证恢复原始 Python 代码的结构。

一个可行性验证方向是：先限定 Mosaic IR 的阶段与操作子集，再检查这些操作是否能够映射到 Pallas 的表达能力。反向生成的目标可以是计算语义等价、保留约定流水关系的 Pallas 实现；具体等价标准需要在验证时明确。

因此，现阶段可以先打通 Pallas → Mosaic 的导出与检查，再选取包含搬运、计算和同步的最小流水样例，验证反向生成能力。往返验证应同时检查计算结果与约定的流水关系；实际 overlap 和性能收益则需要单独通过设备执行确认。

## 来源与核验范围

原文的四张配图分别涉及 StableHLO 属性扩展、`custom_call` 示例、Pallas lowering 路径，以及包含编码 body 的 StableHLO 产物。本稿将其中的关键说明整理为上述文字与流程，原始截图保留在 [Outline 原文](https://outline.infiscale-tech.com/doc/mosaic-kernel-skilldraft-PhutY0NlEA) 中。

本文关于 lowering、DMA／同步和序列化的说明，核对了本仓库 [upstream-sources.lock](../../upstream-sources.lock) 固定的源码：

- JAX：`2d66622450e2c8633cda2307688ef7aa294bd6eb`。
- StableHLO：`7b1b15781ccbd770f50c7eef4b0c3e03834649fd`。

| 说明 | 本地源码入口 |
| --- | --- |
| kernel Jaxpr → Mosaic module | [`pallas_call_registration.py`](../../upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L435) |
| DMA 发起与等待、信号量 lowering | [`lowering.py`](../../upstream/jax/jax/_src/pallas/mosaic/lowering.py#L4975) |
| Mosaic 序列化与 MLIR bytecode 写出 | [`tpu_custom_call.py`](../../upstream/jax/jax/_src/tpu_custom_call.py#L485) |
| body 的 Base64 编码与配置封装 | [`BackendConfig.to_json`](../../upstream/jax/jax/_src/tpu_custom_call.py#L243) |
| `tpu_custom_call` 节点构建 | [`tpu_custom_call.py`](../../upstream/jax/jax/_src/tpu_custom_call.py#L455) |
| StableHLO `custom_call` 接口定义 | [`StablehloOps.td`](../../upstream/stablehlo/stablehlo/dialect/StablehloOps.td#L2649) |

本次核验限于源码阅读，未执行转换脚本、TPU 编译或设备测试。Otter 与 MaxKernel Agent 的背景沿用原文及其关联分析；Mosaic → Pallas 的可行范围、语义保持能力和实际 overlap 仍是待验证项。本文中的 Mosaic TPU MLIR 不应称为 LLO。
