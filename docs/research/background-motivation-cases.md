# 背景与动机：公开案例索引

后续已补充[组织工程场景与上游需求](organization-engineering-cases.md)，作为计划
背景的主要依据。本文保留首轮公开案例，其中 C01、C06、C10、C12、C13、C14 的
提出者也在后续组织成员核对中得到确认；“公开案例”不等同于“外部人的案例”。

检索日期：2026-09-08。用途：为 [PLAN.md](../../PLAN.md) 的背景、研究问题和预期价值
提供工程案例。本文精选 14 个公开案例；补充的内部案例、组织仓库盘点和原始 API
快照保存在访问受限的本地材料中。

## 阅读和证据边界

- 来源为原始 issue、PR、维护者/报告者评论和相关代码差异。作者、创建/更新时间、
  API 状态、PR 是否合并、head revision 和快照指纹见
  [来源元数据](background-motivation-sources.json)。
- 这些是历史工程报告。本轮没有重跑其中的 CPU/TPU 实验，没有独立检查原始设备
  profile，也没有证明报告中的问题仍存在于本项目固定基线。它们不登记为本项目的
  `RUN-CPU`、`RUN-TPU` 或其他 capture，不改变 coverage。
- 区分报告者假设、维护者解释、代码变更和运行验证。自动评审机器人的结论不作为
  独立事实；源码 diff 只能支持“修改了什么”，不能代替测试。
- 状态以检索时快照为准。`closed` 不必然表示修复，`merged` 不必然表示全部场景
  验证通过。公开 fork 的合并不等于上游项目接收。
- CPU/GPU 案例只作为公共接口和工程方法对照。Mosaic TPU MLIR、TPU LLO、静态
  调度与真实执行时间线保持区分。

## C01：HLO 回调需求穿过 JAX、jaxlib、PJRT 和 IFRT

来源：[JAX #38829](https://github.com/jax-ml/jax/issues/38829)，作者 `hhhhsdxxxx`；
后续提案 [JAX #38943](https://github.com/jax-ml/jax/pull/38943)。

报告者在 JAX/jaxlib 0.10.2、TPU7x 的 topology-based AOT 路径上注册 Python HLO
transform，观察到编译挂起且回调入口没有执行。他给出 Python/C++ 调用链和 GIL
等待循环，并提出释放 GIL 的修改。这把“希望改写 HLO”的需求推进到了运行时接口。

后续状态尤其重要：#38943 **关闭但未合并**。作者在
[关闭说明](https://github.com/jax-ml/jax/issues/38829#issuecomment-5251087834) 中指出，
OpenXLA [329d892](https://github.com/openxla/xla/commit/329d8923474b880c43940b40ff8d99d29172fcea)
移除 PjRt-IFRT 异步编译线程池，间接消除了所报告的等待循环。本轮核对了该 commit
的差异；没有据此重做 TPU 验证。

**研究问题：** 如何找到改写入口、观察 callback、定位跨线程等待，并区分原提案和
最终使问题消失的变更？对应 P1、P5、P6、P10。

## C02：局部 layout 需求需要下游编译器能力

来源：[JAX #33543](https://github.com/jax-ml/jax/issues/33543)，作者 `yuwei-qin`。

报告者希望在启用全局优化时，为部分 kernel 保留指定 tiling，发现提供的
`Layout.tiling` 没有随 layout constraint 传到下游。维护者
[明确回复需要 XLA 支持](https://github.com/jax-ml/jax/issues/33543#issuecomment-3577584341)，
随后说明当时没有实现时间表。检索时 issue 仍开放。

**研究问题：** Python 配置、lowering 表达和 backend 支持之间如何核对能力，如何
提出能够让上下游共同验收的接口需求？对应 P4、P5、P9、P10。

## C03：分片封装使后续 pass 看不到 composite

来源：[JAX #29223](https://github.com/jax-ml/jax/issues/29223)，作者 `wenscarl`。

报告讨论 `lax.composite` 与 `custom_partitioning` 的组合：下游希望识别的 composite
结构在封装或分片后不再以所需形式出现，导致模式匹配受阻。讨论围绕 pass 位于分片
之前还是之后、结构何时进入图以及 XLA 是否应保留它展开。检索时仍开放；不能把
讨论中的候选修复位置写成已完成修复。

**研究问题：** 如何追踪语义标记跨表示的存续、消失和内联，选择正确的 pass stage？
对应 P4、P5、P7、P13。原始 scaled-matmul 场景不作为 TPU 性能证据。

## C04：Pallas、自动微分和分片组合后的额外通信

来源：[JAX #21855](https://github.com/jax-ml/jax/issues/21855)，作者 `southfreebird`。

报告者在 `shard_map`、`custom_vjp` 和 Pallas 的组合中观察到意外 AllReduce。
维护者给出当时的
[replication-rule 解释和内部 API 绕过](https://github.com/jax-ml/jax/issues/21855#issuecomment-2241203744)，
并指出公开注册接口缺口。最后一条维护者说明纠正了误关闭，检索时仍开放。

**研究问题：** kernel 的数值实现之外，哪些变换规则决定梯度、复制信息和通信？
对应 P3、P7、P8。原始复现包含 GPU/Triton 路径，不能证明 TPU 上同样存在额外通信。

## C05：外层内存放置与内层 DMA 的一致性

来源：[JAX #39744](https://github.com/jax-ml/jax/issues/39744)，作者 `m-braganca`。

报告者给出一个很小的 Pallas 复现：在 `jit` 内生成但 kernel 不读取的额外参数，
会使特定切片复制返回零。跟进者提供了 libtpu 版本对照及关闭 CMEM assignment 的
对照，提出外层 operand 放置与内层 DMA 路径不一致的解释；另有参与者无法在不同
TPU 代际复现。检索时仍开放。

**研究问题：** 如何同时保留外层 HLO、内层 Mosaic/LLO、输入位置和 target 条件，
避免只看 kernel 源码就断言错误来源？对应 P8–P12。CMEM 归因仍按跟进者解释引用。

## C06：下游缓存绕过与上游问题报告是两个交付物

来源：[JAX #38004](https://github.com/jax-ml/jax/issues/38004)，作者 `Rodrian7`；
关联 [sglang-jax #1225](https://github.com/sgl-project/sglang-jax/pull/1225)。

报告描述 TPU v6e 特定非零起始设备子集上的 cold-cache 成功、warm-cache 崩溃，
包含子集对照和独立 JAX 复现。下游 PR 通过跳过受影响子集的 persistent cache
恢复可用性，且已合并；上游 issue 检索时仍开放，报告者明确不知道 libtpu 内部根因。

**研究问题：** 如何从应用失败抽出最小复现，区分临时缓解与上游修复，并验证缓存
序列化、设备分配和重用条件？对应 P6、P7、P11、P14。

## C07：数值异常跨项目升级，根因解释仍可被反驳

来源：[JAX #35409](https://github.com/jax-ml/jax/issues/35409)，作者 `martin-marek`；
进一步报告 [OpenXLA #38597](https://github.com/openxla/xla/issues/38597)。

报告从 tied-embedding 梯度中的 NaN/Inf 出发，对照 StableHLO 和优化 HLO，提出
alias 分析与 buffer 复用假设。XLA 维护者指出其关于 transpose/in-place 的部分
推理不能直接成立，并
[转交 TPU 团队调查](https://github.com/openxla/xla/issues/38597#issuecomment-4080164687)。
两个 issue 检索时均开放。

**研究问题：** 如何把模型症状推进到源码假设，同时保留反证和争议，找到实际负责
的维护者？对应 P3、P5、P6、P14。本案例不把报告者的竞态解释视为已证实根因。

## C08：在 TPU lowering 报错，最终修改下游 kernel

来源：[JAX #36750](https://github.com/jax-ml/jax/issues/36750)，作者 `shungcp`；
关联 [tpu-inference #2138](https://github.com/vllm-project/tpu-inference/pull/2138)。

大型模型在特定 token 数触发 `tpu.reshape` 错误。维护者要求缩小复现，并纠正
reshape 与 transpose 的混淆。报告者随后
[修正归因](https://github.com/jax-ml/jax/issues/36750#issuecomment-4303921121)，
指向下游 gather kernel 的 layout、pack format 和非法 shape-cast 修改。下游 PR
已合并；JAX issue 已关闭。

**研究问题：** 如何从嵌在大模型中的 kernel 提取复现，避免把报错位置直接当作
缺陷归属？对应 P8、P9、P13、P14。下游 PR 也描述了编译器变化触发适配需求，
因此不能把本例扩大为“编译器与问题无关”。

## C09：硬件优化需求可能没有可承诺的上游支持

来源：[JAX #40093](https://github.com/jax-ml/jax/issues/40093)，作者 `ayaka14732`。

用户希望通过 Pallas 使用 TPU v4 BarnaCore 来支持自定义 kernel。维护者讨论了
旧架构、host 驱动方式和 TensorFlow 耦合，并
[明确以不计划实现或不可行为由关闭](https://github.com/jax-ml/jax/issues/40093#issuecomment-5399509469)。
API 的关闭原因字段为 `completed`，但评论语义不是功能完成，本文以原文说明区分。

**研究问题：** 如何区分硬件资源存在、软件接口暴露和团队愿意支持这三件事，给出
有依据的范围与替代路径？对应 P9–P12。

## C10：batch padding 引出的数值合同问题

来源：[JAX #34080](https://github.com/jax-ml/jax/issues/34080)，作者 `aolemila`。

报告者为减少推理重复编译而把 batch 补齐到预编译尺寸，随后发现 `dot_general`
的浮点结果随 batch 变化。维护者
[按预期浮点行为关闭](https://github.com/jax-ml/jax/issues/34080#issuecomment-3938828257)，
解释精度选项和逐位一致性要求的区别。原始环境为 JAX/jaxlib 0.8.1、TPU v6e。

**研究问题：** 如何分别定义数学等价、容差内一致和逐位一致，避免把未承诺的数值
要求当作已确认 bug？对应 P3、P4、P13、P14。下游若需要 batch-invariant rollout，
仍需单独提出并验证更强的数值合同。

## C11：能够导出 HLO，还需要能够检查优化结果

来源：[JAX #22270](https://github.com/jax-ml/jax/issues/22270)，作者 `balancap`。

报告者希望检查 FP8 matmul 的融合结果与 custom-call 配置，提出更便于阅读和
结构化提取的 HLO 接口需求。讨论反映其分析依赖临时文本解析；检索时 issue 仍开放。

**研究问题：** 如何建立 IR 查询、源码定位和 pass 结果检查工具，使导出产物能够
支持实际判断？对应 P4、P5、P14、索引 I001–I004。原场景为 GPU FP8 融合。

## C12：运行时事件窗口与静态内存计划回答不同问题

来源：公开 [primatrix/skills #104](https://github.com/primatrix/skills/pull/104)，
作者 `sii-xinglong`，已合并。

PR 描述只依赖 profile 窗口内 allocator 事件会遗漏采集开始前的分配，因此加入
HLO `BufferAssignmentProto` 解析以及静态/动态两种观察面的对照。本轮核对了
新增 loader、CLI 和 proto 改动；未重放原 profile。

**研究问题：** 内存数字的分母、时间窗口、静态计划和动态使用量如何定义并对照？
对应 P6、P12、P14。该 PR 的静态峰值不能直接提升为设备实际总内存峰值。

## C13：模型量化格式推动 kernel 提供方扩展支持

来源：公开 [primatrix/tokamax #2](https://github.com/primatrix/tokamax/pull/2)，
作者 `sii-xinglong`，已在该 fork 合并。

模型所需的 block-wise FP8 格式推动 `quant_block_spec` 的兼容性约束及参数化测试
发生修改。最终 diff 保留 scale/block 与 tile 的约束，不是无限放宽支持范围。
PR 报告了 TPU 测试；本轮未复跑，也没有确认 openxla/tokamax 接收该 fork 改动。

**研究问题：** 如何将 workload 的 dtype、scale 与 tiling 合同传递到 Pallas
入口及验证任务？对应 P8、P9、P13。Tokamax 在本项目中仍仅作为 workload 提供方。

## C14：应用并发设计依赖对 JAX dispatch 的理解

来源：公开 [primatrix/sglang-jax #291](https://github.com/primatrix/sglang-jax/pull/291)，
作者 `pengchengneo`，检索时开放。

PR 用受控 demo 比较双线程提交与单线程异步提交，借助 future 数据依赖、延后
`device_get` 和 profile 检查 host/device overlap。作者报告 demo 结果相当。
这是应用设计对运行时语义的验证需求，不构成删除真实 serving 线程的普遍依据。

**研究问题：** 如何解释同步点、future、数据依赖和 dispatch，并通过实验判断
实际 overlap？对应 P6、P11、P14；应用调度器的完整实现不纳入本项目。

## 对计划的归纳

这些案例支持六类研究价值：软件栈认识、修改位置选择、上下游需求协作、数值与
兼容性验证、性能证据解释、团队知识沉淀。具体到后续实验，应能够回答：

1. 症状先在哪个表示或运行阶段出现，之前的观察面是否正常？
2. 当前层拥有什么信息，所需语义在哪一层丢失或尚未支持？
3. 修改需要保持什么合同，如何得到最小的正例、反例和回滚实验？
4. 问题由谁处理，当前状态是提案、缓解、修复、预期行为还是不计划支持？

案例数量不是覆盖指标。内部与公开案例之间的相似性也不能证明它们共享根因。
后续只有在固定基线完成独立源码核对和相应 capture 后，才能按项目证据约定提升深度。
