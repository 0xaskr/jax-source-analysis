# Kickoff 覆盖复查：已完成子项与剩余验收

复查基于 kickoff revision 51、当前 [PLAN](PLAN.md)、[结构化状态](status.json) 与下列
已保存证据。**CPU lowering 参考与编译器诊断 Hack 已有完整的构建/加载/运行/回滚材料；
实际推理 fusion/Pallas 注入、split 和 TPU 设备验收仍未完成。** 三组交付物都有可读产物，
不据文件数量宣称整个 kickoff 已验收。

这是文档与证据覆盖复查，未新增设备运行。XSpace 里程碑的运行/解析验证见
[xspace-contexts.md](xspace-contexts.md)；历史与源码 wheel 的结果继续分别保留。

## 三组交付物

| 交付组 | 已有材料 | 仍需补齐 |
|---|---|---|
| 软件栈总览 | [组件与编译/Pallas 图](overview.md)，固定依赖、公开/私有边界；[LLVM/ORC](llvm-and-objects.md) 与 [CPU runtime](cpu-executable-and-trace.md) | [普通 TPU 公开提交/完成/等待链](tpu-runtime-boundary.md) 已补 SOURCE-ONLY；目标 libtpu/LLO/运行证据仍需 U03 |
| matmul 源码与 API 索引 | [198 个入口、77 条关系](source-index.json)；四种程序变换；[逐 pass 导读](matmul-pass-walkthrough.md)、IR/LLVM/object/thunk；[10 单元 Notebook](matmul-lowering.ipynb) | 普通 TPU 与 Pallas 的目标编译产物和运行映射；分片传播入口目前只有简要索引，不能声称完整的分片图研究 |
| 简单 Hack 实验集 | [编译事件补丁及验收](pass-event-acceptance.md)、[5 单元 Notebook](compiler-pass-hack.ipynb)、构建/加载/回滚；属性、fusion/memory、调度、profiling 的脚本与 Notebook | 指定推理业务中“无需修改源码”的 fusion/Pallas 注入与 split；精度、目标内存峰值及设备事件对照 |

Kickoff 要求关键流程和关键 pass，未要求穷举每个内部函数、所有后端指令或完整 Bazel
缓存闭包。已有依赖审计不等于完整 action 闭包；这项限制应保留，但不应自动成为每个
Hack 的新增验收前提。相同原则适用于未研究的 Eigen/YNN microkernel 和 RTL。

## R01–R12 的子项覆盖

| ID | 已回答或验证的子项 | 尚未满足的部分及依赖 |
|---|---|---|
| R01 组件/API | [overview](overview.md) 覆盖主要组件、依赖和 CPU/TPU 编译分支；CPU executable 与对象接口已有证据 | [公开提交/完成/等待链](tpu-runtime-boundary.md) 已补；Shardy round trip 关键接口继续研究，私有实现与真机需 U03 |
| R02 lowering | [源码 003](source-runtime-baseline.md) 四组 matmul 数值与重载；[640 个 pass 边界](matmul-pass-walkthrough.md)、22 组叶子改写；LLVM/object/thunk/runtime 对应 | TPU 目标后端产物、LLO 与设备映射需 U03；不把 CPU emitter 解释为 TPU 实现 |
| R03 普通/Pallas | [相同 matmul 对比](pallas-comparison.md)、两种 interpret、内层 Mosaic 与外层 custom-call、生产 lowering 标记 | 匹配 libtpu 接受/编译保留、LLO 和真实 TPU 数值/设备行为需 U03 |
| R04 Hack 能力 | [成功构建与加载](source-runtime-baseline.md)、[pass 补丁](pass-event-acceptance.md)、25 个 C++ 测试、cold/warm/filter、源码和 runtime 回滚 | 在指定推理业务/目标环境复用该流程仍需 U01–U03；编译诊断子项已完成 |
| R05 fusion | [CPU 合法性/盈利入口](fusion-and-memory.md)，普通融合、单算子 wrapper 与 YNN 库改写的真实 pass 对照 | 指定业务的算子对、允许修改方式与 Pallas 注入需 U01/U02；目标 TPU 结果需 U03 |
| R06 内存 | [liveness/allocation/offset、alias/donation](fusion-and-memory.md) 与实际 pointer/invalidation；源码 wheel 复验通过 | 目标 TPU allocator/HBM 峰值需 U03；logical peak 诊断差异已记录，未修复，不作为实测峰值 |
| R07 split | 已建立 peak 指标和 alias/liveness 的解释边界；尚无业务 split 结果 | 模型入口、并行维度、精度合同和修改边界需 U01/U02；改写前后运行峰值需 U03 |
| R08 overlap | [双 CPU 对照](overlap-and-scheduling.md)、async→sync、控制边与预期错误；[latency model 入口](latency-model.md) | CPU 两个逻辑设备不能证明 TPU overlap；拓扑、设备 trace、同步与时间对照需 U03 |
| R09 属性/编辑 | [源码 metadata](source-metadata-baseline.md) 验证字符串/布尔/typed 属性差异、HLO setter、add→subtract 实际编译执行；直接问题已有 CPU 答案 | TPU 阶段的保留/消费不能由 CPU 推导；若业务方案依赖特定属性，还需其语义与 U03 验证 |
| R10 roofline | [已有模块、输入/单位/缺失值与修正位置](roofline.md) 已定位；说明为何单张 StableHLO 不足以给实测点 | 目标上限、trace 时间、算量来源与内存分解需 U03；未构建或执行 XProf native converter |
| R11 自定义 cost | [标签与默认成本分离](attributes-and-cost.md)，GPU 消费者 SOURCE-ONLY；[原生 parser](latency-parser-native.md) 15 个 C++ 测试 | 自定义“可并行维度”的实际业务含义与 cost provider 需 U01/U02；目标调度消费/性能需 U03 |
| R12 三类事件 | Host 生命周期及 [49 对 context](xspace-contexts.md)；CPU [编译 pass Hack](pass-event-acceptance.md)；Pallas lowering 中配对 trace_start/stop | Host/CPU 编译子项已验证；设备事件、LLO、保留/嵌套/开销需 U03；派生 flow 未计为 XProf 原生或 UI 验收 |

“已回答直接问题”与“整个跨后端目标已完成”分别记录。所有整体 requirement 仍保持
partial，R07 为 pending；这不是说已完成的 CPU 子项需要再次无条件重跑。

## 下一步如何推进

1. **可以现在做：Shardy import/export 与 HLO round trip。** 普通 TPU 公开运行接口
   已由 [本轮源码导读](tpu-runtime-boundary.md) 补齐，区分提交路径和三种完成范围。
   当前 Shardy 仅有 propagation pipeline 的简要索引；下一步沿固定 `MlirToXlaComputation`
   的 GSPMD fallback、sdy round trip 与 propagation/export 接口核对关键阶段，区分源码路径
   与现有 matmul 实际捕获。未定义的分片业务或目标硬件不自动纳入实验假设。
2. **需要 U01/U02：业务 Hack 定义。** 确定实际模型仓库/入口、固定输入、精度标准，
   以及“不修改源码”约束哪些层。选定 fusion 算子对和 split 并行维度后，再写具体 pass
   改写与 Pallas 注入方案。当前教学图不自动升级为真实业务。
3. **需要 U03：TPU 编译与执行。** 明确型号、访问方式、libtpu/profiler 版本与验收范围。
   先验证运行身份和最小普通/Pallas 数值，再检查设备事件/LLO，最后测内存、overlap 和
   roofline。每种证据单独标记。

U01–U03 仍未回答，U04 已确认三类事件全研究；本次复查不擅自代填这些输入。
源码树和用户原暂存清理改动均保留。恢复队列与下一条命令见 [status.json](status.json)。
