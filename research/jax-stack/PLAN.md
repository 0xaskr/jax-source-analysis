# JAX 软件栈索引研究

依据：Outline kickoff revision 51，更新时间 `2026-09-14T11:26:38.696Z`。
来源与正文指纹见 [kickoff-source.json](kickoff-source.json)。这是清理后新建的研究，
不继承旧分析的完成状态。原有暂存区和五个上游源码树保持原样。

## 完整范围与验收

| ID | 研究目标 | 能证明完成的材料 |
|---|---|---|
| R01 | 软件栈组件、职责、依赖与 API | 固定源码引用；普通 CPU/TPU、Pallas/Mosaic、运行时调用图；公开与私有实现边界 |
| R02 | matmul 与 jit/grad/vmap 的 lowering 全过程 | 固定输入及数值对照；Jaxpr、StableHLO、导出 HLO、实际后端 pass dump、代码生成产物；阶段与源码对应 |
| R03 | 普通 JAX 与 Pallas/Mosaic 差异 | 同一 matmul 的表达、分块、内存访问和两层 IR 对照；解释器、TPU 编译和执行分别取证 |
| R04 | 编译诊断与源码 Hack | 定位修改点；可逆补丁、编译/安装命令、实际加载身份、日志与数值对照、回滚 |
| R05 | fusion 的条件与实现 | 指定 HLO 的合法性与盈利条件；相关 pass 的输入输出；实际推理图的 fusion 和 Pallas 注入实验 |
| R06 | 显存分配与复用 | HLO schedule、liveness、alias/donation、buffer assignment 与实际 buffer 对照；对应分配源码 |
| R07 | split 降低显存峰值 | 指定推理图的峰值节点/链路；沿并行维度循环分块；改写前后精度、IR 和内存测量 |
| R08 | 通信与计算 overlap、调度 | 调度策略和 pass；控制入口；有明确设备/拓扑的执行证据与优化对照 |
| R09 | StableHLO/HLO 差异与自定义属性 | 同一程序转换前后信息逐项对照；属性传递/丢失实验；HLO 编辑约束与修改入口 |
| R10 | roofline 自动计算 | 检查已有模块；区分静态算量/字节估计、硬件上限与实际运行；确认缺失输入和修正点 |
| R11 | 自定义属性与 cost model | 检查属性和分析 API 能否衔接；最小实验；不支持时列出所需组件修改 |
| R12 | XProf 自定义事件（host、编译 pass、设备执行） | 分别定位起止入口；生命周期与 trace 统计；设备 IR 标记、编译保留和真机事件分别验证 |

最终交付保持 kickoff 的三组：软件栈总览（Markdown/Mermaid）；源码与 API 索引
（Markdown/JSON/Python/IR/日志）；Hack 实验集（Notebook/说明/补丁/构建与验证脚本/记录）。
以上 R01–R12 的结果全部归入这三组。`TBD` 不计为已定义的研究要求。

## 本轮进度与恢复入口

CPU matmul、两种 Pallas 解释路径、host/编译事件和开放源码 Mosaic 标记已有可复查材料；
完整 kickoff 尚未完成。源码索引扩展至 120 个入口、38 条关系，包括固定 XProf 源码；Pallas/profiling
Notebook 的 7 个代码单元已真实执行。
原始 capture 继续留在忽略目录；验证器校验清单、哈希与关键语义。

固定源码构建已在唯一容器 `jax-kickoff-cpu-source-002` 中运行，原始上游树保持干净。
首个尝试为提高 jobs 从 2 到 4 而正常中断，缓存复用；不要重启旧容器。
恢复时先执行 `docker inspect --format '{{json .State}}' jax-kickoff-cpu-source-002`，再读
`artifacts/builds/kickoff-cpu-source-002/attempts/0001/build.log`；不要依据宿主机同号 PID
重新启动构建。当前尚未完成 wheel 及加载验收，细节见 [source-build.md](source-build.md)。

属性/cost 对照已完成 CPU capture：六种 metadata 情形、直接 MLIR 属性丢失边界、native
HLO 属性 setter、HLO add→subtract 的编译执行、opaque custom-call 未知成本。结果见
[attributes-and-cost.md](attributes-and-cost.md)。已有 roofline 模块与修正位置见
[roofline.md](roofline.md)，其中 XProf native converter 与目标 TPU 运行尚未验证。

属性/cost 里程碑已在隔离 worktree 合并并正常推送（merge `50bd4ce`），主工作区和原暂存
修改保留。Fusion/memory 已新增 11 组 CPU 对照及 2028 个产物：普通 fusion、单算子 wrapper、
donation 的静态/运行时差异、临时量与存储复用，以及逻辑 peak 诊断差异，见
[fusion-and-memory.md](fusion-and-memory.md)。这些结果不计为真实业务 split 或 TPU 内存验收。

核心依赖审计已完成：三份基础归档与 pin 一致，43 个补丁目标文件的独立复放与实际
Bazel 源码相等，1,085 个缓存 payload 的哈希通过；这不等于完整离线闭包或 wheel 验收。
通信/调度 CPU 参考已记录 3 个数值通过样本、4 个可复现失败，见
[overlap-and-scheduling.md](overlap-and-scheduling.md)。实际 async pair 被 CPU 转回同步，
同步 `control_dep` 变成 dot→all-reduce 控制边；异步控制路径的失败需匹配 wheel 复验。
[调度 Notebook](overlap-scheduling.ipynb) 的 4 个单元已真实执行，包含新的双 CPU capture。

LHS `latency_metadata` 的解析、模型选择与调度调用链已核对，见
[latency-model.md](latency-model.md)。新增 8 个 CPU 标签/成本对照与 45 个产物；
源码中的 GPU 模型消费不计为 GPU/TPU 执行。属性 Notebook 现有 6 个代码单元，已全部
真实执行，包含新进程重新捕获。

CPU LLVM/对象代码/ORC 的选定接口已对应到运行产物，见 [LLVM 与对象](llvm-and-objects.md)。
三个 ELF 对象与 HLO/LLVM 函数及序列化包中的原始字节唯一对应；这是离线审计，不新增
runtime load 证据。matmul Notebook 扩展为 8 个单元，已真实执行。

[编译事件补丁](pass-event-patch.md) 已在独立副本应用、反向应用并恢复原始字节；按构建
wrapper 要求生成规范 Git diff。未修改当前运行中的 clone/config，也未编译加载该补丁。
构建期间已完成 [cold/warm/filter 验收脚本](pass-event-acceptance.md)：当前 wheel 的两组
负对照、运行中构建拒绝、事件语义反例和 Notebook 新进程重跑均通过；两组各三次输出
的最大绝对误差低于 5.56e-9，自定义事件计数均为 0。正对照要求成功的指定补丁构建及
wheel/native payload 字节绑定，目前仍未执行。完成构建后先做无补丁 wheel 加载基线，
再应用补丁、重编译和回滚。下一命令仍是 inspect 当前构建容器。目标 TPU overlap 仍需 U03。
完整恢复队列、已确认 U04 和未回答 U01–U03 保存在 [status.json](status.json)。

## 执行次序

1. 核对五个固定源码版本，记录当前 Python/JAX/jaxlib 与实际加载二进制。
2. 建立基础 matmul、vmap、grad、jit 组合的 CPU 对照，记录 tracing 与编译观测，
   分开保存导出 HLO 和实际后端的 pass dump。
3. 沿真实产物建立源码/API 索引，补齐组件总览和普通/Pallas 两条路径。
4. 逐项研究 R05–R12，将能独立验证的机制做成最小实验，保留原始产物。
5. 手动下载/构建固定源码；确认加载自建二进制，再做诊断改动、重新编译和回滚。
6. 在指定的实际推理业务上完成 fusion/Pallas 注入及 split 实验，验证数值与内存。
7. 对照 R01–R12 和三组交付物逐项验收。CPU 样例和源码索引不能替代实际业务或 TPU 验收。

## 等待用户明确的输入

- U01：实际推理模型、仓库、入口、参考输入与精度标准。
- U02：fusion/split 的“不修改源码”是否同时约束模型代码和 JAX/XLA 源码。
- U03：可用 TPU 型号、访问条件、匹配 libtpu 资料及本轮设备验收范围。

这些输入未确认前，先推进固定源码、组件索引、matmul CPU 对照和通用机制研究。
不把任意教学模型指定为实际推理业务，不把 CPU 结果计为 TPU 结果。

## 证据边界

U04 已确认：“三类都研究：host、编译 pass、设备执行”。当前结果见
[profiling.md](profiling.md)：host 生命周期、CPU 冷编译 pass 事件、Pallas 生产 lowering
生成的配对标记已验证。后者只是 CPU 主机上的开放源码 TPU lowering，不计为 TPU backend
编译或执行。两种 Pallas 解释路径对照见 [pallas-comparison.md](pallas-comparison.md)。

每个源码索引条目记录 component、revision、path、symbol、行号、输入输出、调用关系、
约束、关联实验和未验证项。每个运行记录保存命令、输入、生产脚本、运行环境与二进制
身份、产物 SHA-256、数值断言及失败信息。

证据使用 `SOURCE-ONLY`、`RUN-CPU`、`SIM-TPU`、`COMPILE-TPU`、`RUN-TPU`、
`REPLAY-OFFLINE`；运行二进制与固定源码不一致时增加 `VERSION-SKEW`。
Mosaic TPU MLIR 不称为 LLO。自建二进制完成之前，当前 wheel 的 native pass dump
仅证明该 wheel 的行为，不能证明固定 XLA C++ revision 已被执行。

原始 capture 与 Outline 正文放在 Git 忽略的 `artifacts/jax-stack/`；研究文档、
脚本、Notebook、索引和较小的选定证据放在 `research/jax-stack/`。
