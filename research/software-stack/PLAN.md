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
完整 kickoff 尚未完成。源码索引扩展至 219 个入口、92 条关系，包括固定 XProf 源码；Pallas/profiling
Notebook 的 7 个代码单元已真实执行。
原始 capture 继续留在忽略目录；验证器校验清单、哈希与关键语义。

固定源码构建 003 已完成，独立环境的 Git identity、wheel/native 字节绑定、pass 负对照和
完整 **18 组 CPU 数值 / 3,898 个产物**均通过，见 [源码运行基线](source-runtime-baseline.md)。
该组证据没有 `VERSION-SKEW`；旧 wheel 和失败记录单独保留。原始上游树保持干净，
宿主机环境未覆盖。自定义 pass 补丁已完成 25 个 C++ 测试、实际构建/加载与回滚。

属性/cost 对照已在源码 wheel 003 完成复验：六种 metadata 情形、直接 MLIR 属性丢失边界、native
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
同步 `control_dep` 变成 dot→all-reduce 控制边。匹配源码复验后，显式 layout 的异步控制
错误变为 Host 未注册 `all-reduce-start`；其余三个 Python/MLIR 失败仍在。
[调度 Notebook](overlap-scheduling.ipynb) 的 4 个单元已真实执行，包含新的双 CPU capture。

LHS `latency_metadata` 的解析、模型选择与调度调用链已核对，见
[latency-model.md](latency-model.md)。8 个 CPU 标签/成本样本已在源码 wheel 复验。
两类实验共 14 组 metadata、2 组 HLO 改写执行、117 个产物；生产与复查进程均绑定
源码 native payload，旧结论在本例保持一致，见 [源码 metadata 基线](source-metadata-baseline.md)。
实际检查了 2 次 running build 拒绝、2 次旧 wheel 生产拒绝、2 次旧 native reader 拒绝。
源码中的 GPU 模型消费仍不计为 GPU/TPU 执行。属性 Notebook 的 8 个代码单元已全部
真实执行，包含新进程旧 wheel 对照、源码结果和原生 parser 结果的归档复查。

固定源码的 latency parser 原测试先以未修改源码运行通过；随后只在隔离 clone 的
测试文件加入 14 个边界样本，共 15 个 C++ 测试全部通过。零/负整数返回数值；
int64 越界、小数、空串、非法文本与科学计数法返回 nullopt；配置的单位缩放通过。
测试源码已恢复原字节，二进制、XML、完整日志和独立补丁复放已核对，见
[原生 parser 结果](latency-parser-native.md)。这不增加 GPU consumer 或 TPU 执行证据。

CPU LLVM/对象代码/ORC 的选定接口已对应到运行产物，见 [LLVM 与对象](llvm-and-objects.md)。
三个 ELF 对象与 HLO/LLVM 函数及序列化包中的原始字节唯一对应；这是离线审计，不新增
runtime load 证据。匹配源码 003 的三个对象也已单独复查；matmul Notebook 扩展为
10 个单元，已真实执行，包含旧 CPU 实算、源码 pass/对象归档及 NumPy 对照。

[编译事件补丁](pass-event-patch.md) 已完成规范 Git diff、应用/反向应用及原始字节恢复。
无补丁源码基线里，默认和禁用 algsimp 的 generic 事件均为 3，与固定源码的 filter 前
TraceMe 位置一致；自定义事件为 0，warm 范围没有新编译事件。
[cold/warm/filter 验收脚本](pass-event-acceptance.md) 要求补丁构建成功、实际 native 字节
匹配，且过滤后消失的是自定义 leaf 事件，不能用 generic 计数替代。

C++ 测试的 Googletest 宏缺失已通过独立测试依赖副本修复；生产构建没有该 override。
实际加载首版补丁后发现 `program_id` 被导出器隐藏，最终补丁保留原字段并增加
`research_program_id`。25 个 C++ 测试再次通过；默认/过滤自定义事件为 127/124，
其中 algsimp 为 3/0；generic algsimp 仍为 3/3。warm 和数值对照通过。
三个源文件已反向恢复为原始字节，独立 clone 干净；新环境重新加载无补丁 003 后，
自定义事件回到 0/0。统一审计与真实失败回归反例通过，见
[pass-hack-results.json](pass-hack-results.json)。这是 CPU 编译器 Hack 验收，目标 TPU
overlap、编译和设备事件仍需 U03。

构建终态追加依赖审计：三份归档、43 个补丁目标与 1,093 个缓存 payload 的完整性通过；
50 个产物及六个反例由验证器复查。它仍不证明完整 action 输入闭包或离线可重放。
另已保存 199 个缓存 repository 的 451 份描述文件，核对配置的 Python 3.12.13
归档及解释器/共享库/头文件，467 个产物通过语义与反例验证，见 [构建输入记录](build-inputs.md)。
该记录不把缓存条目当作实际目标依赖，不把构建后本机探测当作构建期进程采样。
匹配源码 matmul 的逐 pass 导读已完成当前样本审计，见 [matmul-pass-walkthrough.md](../call-to-llo/matmul-pass-walkthrough.md)。
640 个 pipeline 边界形成 584 组配对；22 组叶子文本变化与 3 组嵌套汇总分开记录。
59 个派生产物、形状/operand/layout/fusion 语义和六个反例复查通过；匹配源码的
LLVM/object 审计另有 20 个产物。两个新验证器都实际拒绝了旧 wheel reader。
CPU executable 与运行时分支已进一步取证，见 [cpu-executable-and-trace.md](cpu-executable-and-trace.md)。
两个来源各有 11 个 thunk；两处剩余 dot 均保存为 DotThunk，对应固定源码的 Eigen contraction。
新进程的 12 次带 trace 数值调用通过，观察到 33 个 producer 和 33 个完成事件。
原型环境相等检查因新增映射 _mlirHlo.so 失败；正式采集只允许新增且绑定到源码 wheel 的库，
原有库/源码/其余环境保持一致。39 个运行、94 个 schema、62 个审计产物已复查，
19 个反例及 5 个真实 Notebook 单元通过；旧 wheel 的 producer 和 verifier 均实际被拒绝。
原始 XSpace 关联已验证，见 [xspace-contexts.md](xspace-contexts.md)：49 组一对一关联、
13 组跨线程；新增四个 source-bound CPU 任务，覆盖超大 uint64 ID、逆序完成与异常终点。
98 个原始事件与 JSON 唯一对应，18 个零 duration 导出为 1 ps；重复 _src 的覆盖顺序已核对。
17 个新增运行、45 个审计产物、20 个拒绝样本和六个正向边界通过；5 个真实 Notebook 单元通过归档复查。
派生 flow 不是 XProf 原生预处理或 UI 验收。
[覆盖复查](coverage-review.md) 已逐项对应三组交付及 R01–R12，修正总览过期的构建/标记状态；
CPU 参考和编译诊断子项保留完成结果，不将穷举内部调用或完整缓存闭包新增为验收要求。
普通 TPU 的 [公开执行接口](tpu-runtime-boundary.md) 已完成源码子项：新增 32 个入口、
24 条关系，分开 Python/C++ 提交分支、buffer/执行 status/effect 等待及 callback 生存期。
12 项结论、15 个源码检索条件和九个结构反例通过来源检查；没有新增设备运行。
[Shardy 往返与分区](shardy-round-trip.md) 已补 21 个入口、15 条关系及双 CPU 实验。
Shardy/GSPMD 各三次数值通过；35+30 个新产物显示传播补属性仍为 `[8,12]`，后续 XLA SPMD
才产生 `[4,12]`。五个内部 MLIR 保存点、四对旧单 CPU 相同边界、九个反例及五个真实
Notebook 单元通过；旧 native reader 在 IR 解析前被实际拒绝。两次验证器接口适配失败保留，
没有重跑生产任务。当前公开接口和 CPU 参考队列已完成；下一阶段业务定义需 U01/U02，
目标 TPU 编译/运行/设备事件将按用户指定的 Falcon v7 四芯片继续。

[独立环境](runtime-environment.md) 的旧 wheel 复验与新增源码 wheel 复验分别保留，
不改写旧证据。源码基线仍观察到 logical peak 诊断差异，不能据此推导物理内存峰值。
**JAX 版本判定已定案（2026-09-15）**：PyPI 与 GitHub releases 的最新 stable JAX 均为
`0.11.1`（2026-08-17），`0.11.2` 未发布。用户决定**跟踪 stable 最新版**，pin 已从
0.11.2 开发快照 `5832e866` 换成 released tag **`jax-v0.11.1` (`2d66622450e2`)**，
配套换 XLA/Shardy/LLVM/Triton；**StableHLO 不变**。commit `c3a88db` + `35dc506`。
换 pin 后 `sync-environment.py check` 的结论由 `VERSION-SKEW` 改善为 **`ALIGNED`**。
libtpu 仍由 pinned `setup.py:27` 规定为 `0.0.46.*`，不采用更新的 0.0.47。
**换 pin 使旧 CPU wheel、build fingerprints 与 `source-index.json` 的 revision 全部失效**，
需重建/重新核对，逐项见 [交接文档 4.2](AGENT-HANDOFF-2026-09-15.md)。
完整恢复队列、已确认 U01–U04 和选定业务合同 保存在 [status.json](status.json)。

## 执行次序

1. 核对五个固定源码版本，记录当前 Python/JAX/jaxlib 与实际加载二进制。
2. 建立基础 matmul、vmap、grad、jit 组合的 CPU 对照，记录 tracing 与编译观测，
   分开保存导出 HLO 和实际后端的 pass dump。
3. 沿真实产物建立源码/API 索引，补齐组件总览和普通/Pallas 两条路径。
4. 逐项研究 R05–R12，将能独立验证的机制做成最小实验，保留原始产物。
5. 手动下载/构建固定源码；确认加载自建二进制，再做诊断改动、重新编译和回滚。
6. 在指定的实际推理业务上完成 fusion/Pallas 注入及 split 实验，验证数值与内存。
7. 对照 R01–R12 和三组交付物逐项验收。CPU 样例和源码索引不能替代实际业务或 TPU 验收。

## 已确认的业务与设备阶段

用户指定 SGLang-JAX、TPU v7 四芯片、Falcon；限制业务模型代码，允许 JAX/XLA 可逆修改。
U01 的具体场景选择由研究执行：首选 Qwen3-8B 长输入 prefill 的 MLP，见
[业务合同与验收](sglang-v7-workload.md)。原等待输入状态保留为历史，不继续重复询问。

预检已通过并完成 raw XSpace 复查。用户现要求本地交接，完整恢复入口为 [详细交接上下文](AGENT-HANDOFF-2026-09-15.md)。最新 baseline-005 因旧 enabled 产物目录缺失在模型前失败，租约为空；先解除回收步骤对业务的阻断，再选择入口继续，不能直接重复提交当前 YAML。
SGLang 固定 JAX 0.8.1 / libtpu 0.0.30，与本仓库源码构建分开记录 `VERSION-SKEW`。
先验证设备、普通/Pallas 实算与未修改 MLP 兼容性，再执行真实权重基线、编译器 fusion/split
和设备事件审计。随机权重 smoke 不计为业务验收，目标仍未全部完成。

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
Mosaic TPU MLIR 不称为 LLO。历史旧 wheel 的 native pass dump 只证明该 wheel 的行为；
源码基线 003 的新 capture 另有构建、Git identity 和实际加载 native 字节绑定。

原始 capture 与 Outline 正文放在 Git 忽略的 `artifacts/jax-stack/`；研究文档、
脚本、Notebook、索引和较小的选定证据放在 `research/software-stack/`。
