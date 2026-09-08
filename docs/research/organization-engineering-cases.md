# 组织工程场景与上游需求

检索日期：2026-09-08。本专题按提出问题的人及其工程场景组织，作为
[PLAN.md](../../PLAN.md) 背景与动机的主要案例依据。

先以获授权的 GitHub 组织成员记录核对贡献者，再检索其在 JAX、XLA、Tokamax 和
XProf 的公开记录。组织归属在本轮仅指检索时的 GitHub 成员身份，不推断雇佣关系
或历史时点身份。原始成员记录保留在受限归档中；公开来源、作者、状态和快照指纹
见 [来源索引](organization-engineering-sources.json)。

本轮检查了 30 名相关仓库活跃贡献者在 `jax-ml`、`openxla` 的 60 个作者限定查询，
得到 37 条 issue/PR 候选，其中包括不属于本项目主线的模型接入或文档请求。下文
选择 16 组软件栈场景，不把同一问题的多条跟进重复计为独立缺陷。另有 16 组新增
内部 PR 场景保存在本地受限材料中，与首轮 12 组内部材料相互补充。

所有运行、性能与根因内容均按历史报告引用，本轮没有重新运行设备实验或提升
coverage。`closed`、`merged`、代码落地、绕过和验证完成分别记录。第一轮
[外部公开对照案例](background-motivation-cases.md) 保留作补充。

## O01：HLO hook 已有，但编辑接口仍不够

`Iamleos`：[JAX #39023](https://github.com/jax-ml/jax/issues/39023)，开放。

需求十分直接：能够注册 HLO transformation 并反序列化 module，但希望进一步
增加 instruction、复制或编辑图，现有 Python 接口不足，因此向 JAX 请求更多
可调用能力。该请求没有在本轮快照中得到已完成实现的证据。

对应研究问题：在哪一层拿到什么表示，hook、序列化格式和可编辑对象接口之间
是什么关系；图改写实验需要哪些构建、绑定和不变量。对应 P1、P4、P5。

## O02：图改写遇到跨线程编译调用问题

`hhhhsdxxxx`：[JAX #38829](https://github.com/jax-ml/jax/issues/38829) 与
[#38943](https://github.com/jax-ml/jax/pull/38943)，均关闭，后者未合并。

TPU AOT 路径上的 Python transform 回调挂起，报告沿 jaxlib、PJRT/IFRT 和 GIL
等待链提出修改。作者后来说明 OpenXLA 执行模型变化间接消除了原死锁。
详见首轮 C01 的关闭原因和固定 commit。对应 P1、P5、P6、P10。

## O03：恒等变换也暴露多 slice verifier 合同差异

`hhhhsdxxxx`：[OpenXLA #47778](https://github.com/openxla/xla/issues/47778)，关闭。

报告使用一个原样返回 HLO proto 的 transform：未安装 hook 时编译成功，安装后
generic verifier 对 per-slice device assignment 与 global sharding 的设备数量
产生冲突。维护者讨论了给 transform verifier 配置 metadata 的需要。
本轮没有核实关闭对应的具体修复 commit，不能仅凭关闭状态宣称某个版本已修复。

对应研究问题：pass 正确性还依赖它所在阶段的模块合同与 topology 上下文。
对应 P5、P7、P10。

## O04：修改 HLO 后 alias 丢失与 OOM

`pathfinder-pf`：[JAX #40280](https://github.com/jax-ml/jax/issues/40280)，关闭。

报告称 HLO transform 后 `input_output_alias` 消失并出现 HBM OOM。维护者要求
可运行复现；作者随后称升级 jaxlib 至 0.11.1 后恢复。没有足够材料把作者的 XLA
bug 判断定位到具体提交。

对应研究问题：如何检查变换前后的 alias、donation、buffer lifetime 和实际版本，
而不仅仅比较 operation 数量。对应 P1、P5、P6、P14。

## O05：通信需求推动 JAX mesh 构造改动

`wangfakang`：[JAX #39340](https://github.com/jax-ml/jax/pull/39340)，已合并。

PR 从 TPU v7 训练的通信映射需求出发，修改物理轴候选选择的优先级，在等规模
候选中考虑带宽差异。已核对 `mesh_utils.py` 与对应测试差异；PR 中的性能收益
没有在本轮重新测量，也不作为所有 workload 的保证。

对应研究问题：逻辑 Mesh、物理 topology 和通信强度怎样影响源码中的选择规则，
何时需要推动上游改动。对应 P7、P11、P14。

## O06：BF16 gather 碰到特定 SparseCore lowering 限制

`pathfinder-pf`：[JAX #39577](https://github.com/jax-ml/jax/issues/39577)，开放。

报告希望原生 gather BF16 数据，给出 Pallas SparseCore 复现及特定 indirect-transfer
operation 只支持 32-bit 元素的报错。环境、kernel 和具体报错层已给出，但没有
已验证的解决方案。不能把该 operation 的限制概括为 SparseCore 全面不支持 BF16。

对应研究问题：如何区分 dtype、primitive、lowering、内存传输与硬件能力边界。
对应 P8、P9、P10。

## O07：同一个 primitive 在不同 kernel 类型中支持不同

`Iamleos`：[JAX #32991](https://github.com/jax-ml/jax/issues/32991)，开放。

TPU v6e Pallas kernel 中的 `cumsum` 无法按 TensorCore 路径 lowering。回复指出
当时 SparseCore kernel 有相应支持，给出使用方式，并表示会向 compiler 团队
了解 TensorCore 支持难度。其跨核通信说明绑定当时版本，不能当作当前通用限制。

对应研究问题：如何从 JAX primitive 找到 backend-specific lowering 和负责团队，
判断修改 kernel 还是补编译器支持。对应 P2、P8、P9。

## O08：量化精度要求与 tile、流水线和 VMEM 取舍

`pengchengneo`：[Tokamax #839](https://github.com/openxla/tokamax/issues/839)，开放。

作者围绕 block-wise FP8 GMM 提出量化块大小、实际性能、循环展开及 VMEM 使用的
取舍，并询问跨 K tile 的流水线技巧与相关上游能力。其对 MXU 周期和硬件开销的
解释属于作者分析，本轮没有确认其因果分解或证明某种实现是唯一可行路径。

对应研究问题：如何把 workload 的量化合同与实际编译产物、内存和性能联系起来，
形成具体的 kernel/compiler 需求。对应 P8–P10、P12–P14。

## O09：模型量化格式形成上游支持提案

`sii-xinglong`：[Tokamax #794](https://github.com/openxla/tokamax/pull/794)，关闭未合并。

该上游 PR 提议调整量化 scale 与 tile 的兼容性约束，支持 block-wise FP8 GMM。
首轮 C13 中记录的组织 fork 改动与之相关，但 fork 合并和上游接受是两个状态。
这里补上实际存在的上游提案，不能再只凭 fork 记录推断是否曾提交上游。

对应研究问题：如何把内部 workload 需求整理为有边界、有测试的上游改动。
Tokamax 仍只提供 workload 与接口合同；对应 P8、P9、P13。

## O10：缺少 runtime counter，需要核对采集链版本

`pathfinder-pf`：[XProf #2807](https://github.com/openxla/xprof/issues/2807)，开放。

按官方 kernel demo 操作后仍看不到 runtime counter。回复把排查指向 libtpu 的
采集支持和 XProf 前端版本。作者后续还询问图中 MXU 轨道数量的含义。

对应研究问题：数据由 runtime、编译器还是 UI 提供，版本和采集配置如何验证，
显示轨道能否映射为硬件资源。对应 P1、P11、P12、P14。

## O11：LLO 与 kernel 内 DMA 的可见性缺口

`Prayer3th`、`elbertwang`、`pathfinder-pf`：
[XProf #2441](https://github.com/openxla/xprof/issues/2441)、
[#2416](https://github.com/openxla/xprof/issues/2416)、
[#2446](https://github.com/openxla/xprof/issues/2446)。前两项关闭，后一项开放。

场景包括训练 profile 的 LLO 区域为空，以及只看到 Pallas call 前的 DMA、看不到
kernel 内 DMA。讨论涉及采集版本、显示范围和视图解释。原始请求在不同版本与
场景发生，不能据此认定它们共享一个 bug，也没有统一的修复结论。

对应研究问题：如何检查 dump、profile、解析器和 UI 的责任边界。对应 P9–P12、P14。

## O12：有计数器仍不清楚怎样指导优化

`pathfinder-pf`：[XProf #2481](https://github.com/openxla/xprof/issues/2481)、
[#2482](https://github.com/openxla/xprof/issues/2482)、
[#2483](https://github.com/openxla/xprof/issues/2483)，均开放。

这些请求分别询问 Scalar ALU 忙碌时怎样定位指令、采样为何不等间隔，以及柱高
是否采用相对尺度。它们证明组织有解释观测口径的实际需求；图像与请求本身不能
证明硬件执行机制，也不直接证明 UI 实现错误。

对应研究问题：统计量的时间窗口、归一化、采样方式及其到代码的映射。
对应 P12、P14 和症状/计数器索引。

## O13：升级完整软件栈后出现模型精度差异

`pathfinder-pf`：[JAX #33111](https://github.com/jax-ml/jax/issues/33111)，开放。

推理评估在升级 JAX、jaxlib、Flax 和 libtpu 后变差，并观察到 hidden states 差异。
这是多组件共同变化的报告，没有最小的单变量归因；不能把结果只归因于 JAX。

对应研究问题：怎样固定版本、做差分和二分，区分浮点预期差异与实际错误。
对应 P1、P13、P14。

## O14：减少重编译的 padding 策略提出更强数值要求

`aolemila`：[JAX #34080](https://github.com/jax-ml/jax/issues/34080)，关闭。

为了重用预编译尺寸而 padding batch，随后发现浮点结果变化。维护者按预期浮点
行为关闭；见首轮 C10。下游需要 batch-invariant rollout 时，应单独明确其数值
合同和成本。对应 P3、P4、P13、P14。

## O15：服务设备子集与 persistent cache 的组合问题

`Rodrian7`：[JAX #38004](https://github.com/jax-ml/jax/issues/38004)，开放。

从下游多 engine 场景缩小为独立 JAX cold/warm-cache 和设备子集复现，并另行
提交下游绕过；见首轮 C06。该作者在本轮组织成员记录中得到确认，所以公开上游
记录不能因仓库不属于组织就被归为“外部人的场景”。对应 P6、P7、P11、P14。

## O16：内部 kernel 向上游交付时还需要 API、参考实现和可见的验证

`0xaskr`、`Fred33146`：
[Tokamax #1001](https://github.com/openxla/tokamax/pull/1001)、
[#1103](https://github.com/openxla/tokamax/pull/1103)、
[#1327](https://github.com/openxla/tokamax/pull/1327)。

#1001 已关闭未合并，作者明确先回组织 fork 评审。#1327 的 chunked XLA KDA
提案仍开放。#1103 的 API 状态也仍开放，但维护者和贡献者说明代码已经进入 main，
并指出同步系统造成 PR 状态未更新；本轮进一步读取了对应历史提交
[84c7a4fc3b0d](https://github.com/openxla/tokamax/commit/84c7a4fc3b0d)。
因此这里分别记录“PR 状态”和“代码落地说明”，不把开放状态等同于代码未落地。

交付需求涵盖 forward/VJP、packed 输入、fallback、benchmark，以及外部贡献者
无法自行访问的内部 presubmit 诊断。对应 P3、P8、P13、P15；具体模型框架内部
仍不纳入源码分析边界。

## 内部私有仓库中的补充场景

受限索引另外保存 16 组新增场景及作者、PR、head revision 和讨论依据。它们包括
残差标签与上层 remat 策略配套、优化器图大小与内存、gather/gradient 生命周期、
AOT/JIT 配置一致性、无意清缓存、packed reference 的 tracing 限制、scalar prefetch
容量，以及 import 时提前初始化 backend 等问题。

其中跨仓库 residual/remat 场景存在明确的配套 PR 链接和讨论，不仅是主题相似。
完整 locator 和具体工程数据保留在受限档案，公开摘要仅用于组织研究问题。

## 对背景与动机的调整

以组织自身反复提出的工程问题选择实验顺序：先能说明“哪里改、信息如何保留、
哪些版本和合同需要验证、该找谁协作”，再用其他组织的案例校验方法能否泛化。
问题归属与公开/私有是两个维度，不能再用仓库可见性代替提出者的归属。
