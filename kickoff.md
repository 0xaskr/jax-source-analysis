# JAX → TPU 白盒源码分析项目 Kickoff

本文用于项目启动与团队对齐，说明研究问题、交付目标、执行方法和验收方式。
详细阶段计划以 [PLAN.md](PLAN.md) 为准；进度、版本和覆盖深度分别以
[status](manifests/status.json)、[baseline](manifests/baseline.json) 和
[coverage](manifests/coverage.json) 为准，本文不维护第二套执行状态。

## 1. 背景与动机

本项目面向开展 JAX/TPU 模型性能与 kernel 优化的 AI Infra 工程师，也为刚进入
软件栈的开发者提供学习入口。实际工作中，一个模型或 kernel 问题可能需要沿着
JAX 变换、编译器、运行时和设备观测逐层定位；修改位置、上下游接口和验证条件
需要一起确定。

仓库已整理的[组织工程场景](docs/research/organization-engineering-cases.md)
包括 HLO 编辑能力需求、图改写后的 alias 与内存问题、分片与通信映射、Pallas
lowering 支持，以及 profile 和 LLO 的可见性问题。这些历史记录用于提出研究问题；
其中报告的根因和性能结果不构成本项目固定基线上的运行证据。

研究围绕四项工程需求展开：

- **时间成本**：减少重复查源码、准备环境和无效试跑，记录从接到问题到完成验收的
  总投入，包括人工、构建、设备、返工和资料维护成本。
- **方向性**：把症状与源码职责、接口约束、可修改位置连接起来，说明每次实验要
  排除什么假设，以及何时需要调整方向或推动上下游配套改动。
- **可控性**：固定版本、输入、缓存条件和资源预算，保留执行状态与回滚入口，使
  多次试验和跨会话工作能够接续。
- **可验证性**：核对修改是否进入实际加载的二进制，并分别验证语义、编译产物、
  内存与运行表现，使结论可以复核。

工程师与 AI 共同使用这些资料进行检索、实现和实验。预期收益需要通过代表性任务
的对照验证；资料建设本身也计入成本，不预先声称能够达到固定的效率或性能提升。

## 2. 目标

建立一套版本固定、可检索、可执行、可修改和可回滚的 JAX → TPU 白盒资料库，
使使用者能够从工程问题找到干预位置，并用可复现的证据完成验收。

具体交付目标如下：

1. **解释程序与变换机制**：用固定源码、实验和 IR 解释 `jit`、`grad`、`vmap`
   的组合，以及 tracing、lowering、编译、缓存和执行的区别。
2. **建立跨层地图**：记录 API、Jaxpr、StableHLO/Shardy、XLA、IFRT/PJRT 的
   输入输出与调用关系；取得匹配输入后，继续补齐 libtpu compiler/runtime、
   TPU-specific LLO 和硬件映射。
3. **完成源码修改闭环**：按组件完成 patch、构建、安装、实际加载证明、定向测试
   和回滚，覆盖 JAX、jaxlib、StableHLO、Shardy、XLA，以及后续取得的 libtpu/LLO。
4. **验证代表性图改写**：先完成 HLO transformation 机制实验与模式检测，再在
   语义、runtime 和 sharding 合同建立后，用真实 workload 验证安全的计算图改写。
5. **沉淀可复用材料**：交付源码导读、双向索引、命令行 probe、IR/日志 capture、
   可逆补丁、分析文章和故障定位案例，支持团队学习与上游贡献。

分析从公开 JAX API 或纯 JAX/Pallas 入口开始。Tokamax 和自研框架只提供固定版本的
workload、输入契约和数值参考；它们在 JAX 之上的框架设计不进入研究范围。

按[现有架构划分](docs/architecture/00-whole-stack.md)，分别研究普通 JAX/XLA TPU
路径和 Pallas/Mosaic TPU 路径，并将 host 控制调用链与编译载荷链分开记录。
Mosaic TPU MLIR 不称为 LLO。libtpu 内部阶段、LLO 与具体硬件的对应关系，需要
匹配源码、目标配置和编译或设备产物支持后才能形成结论。

## 3. 研究方法

每个研究单元以明确问题开始，最终形成可以审阅、复现和恢复的证据包。

1. **定义问题与实验合同**。记录症状、预期行为、候选解释、允许修改的范围、输入
   shape/dtype、数值容差、资源预算和停止条件。真实 workload 接入后，先枚举其
   可达 feature，再据此决定专题优先级。
2. **固定源码与运行来源**。以 [upstream-sources.lock](upstream-sources.lock)
   和 manifests 锁定 revision、工具链、二进制 SHA-256、实际加载路径及编译配置。
   优先检查固定本地源码，使用 symbol、caller/callee 和数据结构建立索引。
3. **构造最小探针并留存观察**。从小规模 synthetic workload 开始，采集实际生成的
   Jaxpr、IR、日志与结构化断言；每次 capture 绑定 producer argv、输入、环境、
   revision 和产物 hash，保留失败与拒绝样本。
4. **用对照检验假设**。一次改变一个关键条件，区分首次 tracing、缓存复用、编译
   和设备执行。图改写先做模式检测、输出拒绝原因，再验证合法阶段和必要不变量。
   数值一致、IR 改变和性能改善分别取证。
5. **验证修改生效并回滚**。保存可逆 patch，在隔离环境构建和安装，核对实际加载
   的产物，运行相称的正例与负例，再恢复基线并检查源码树状态。
6. **整理证据与工程评价**。把 claim 关联到 source index、probe 和 capture，运行
   对应语义校验器与 selftest。使用代表性任务比较常规资料与新增研究产物的使用
   效果，记录总成本、方向调整依据、恢复能力和验收可靠性。

证据严格使用[证据规范](docs/contributing/evidence-conventions.md)中的等级：

| 等级 | 可记录的验证 |
|---|---|
| `SOURCE-ONLY` | 在固定 revision 上检查源码、接口或构建定义 |
| `RUN-CPU` | 在真实 CPU backend 上执行并完成断言 |
| `SIM-TPU` | 使用 TPU/Pallas interpreter 完成模拟验证 |
| `COMPILE-TPU` | 针对固定 TPU target 编译成功，尚未在设备执行 |
| `RUN-TPU` | 在记录了代际和 topology 的真实 TPU 上执行并完成断言 |
| `REPLAY-OFFLINE` | 用锁定工具重放或分析已有 capture |

运行二进制与引用源码 revision 不一致时，额外标记 `VERSION-SKEW`。CPU 结果不
用于推断 TPU runtime、LLO、硬件、通信、内存布局或性能；模拟与编译成功也不能
替代真机验证。

每个主题逐步形成 `README.md`、`topic.json`、`source-index.json`、probe 和 capture；
修改类主题增加 patch、构建与回滚记录。大体积或私有产物保留在 Git 外，提交脱敏
locator 和 hash。每个独立里程碑更新计划与状态，按范围运行检查，再提交并推送。

## 4. 验收标准

验收以可复核产物和适用证据为依据。以下是目标门槛，当前完成情况从 coverage
清单读取；其中 `covered` 只表示达到已记录深度。

| 验收项 | 通过条件 | 主要交付物 |
|---|---|---|
| 版本与环境 | 固定源码和工具链可恢复；运行证据绑定实际二进制；错位如实标记 | baseline、环境锁、构建记录、native inventory |
| 变换机制 | 对代表函数解释有效的 `jit/grad/vmap` 组合，以源码规则、IR 和自动断言验证 | 变换 Lab、source index、capture |
| 源码修改 | 按组件完成修改、构建、安装、加载证明、测试和回滚；回滚后恢复基线 | 可逆 patch、构建 manifest、运行与回滚记录 |
| Pass 机制 | `sin → cos` 探针产生预期语义变化；验证 callback、合法 stage、缓存隔离及清理；`dot` matcher 有命中和拒绝样本 | before/after HLO、probe、断言 |
| 安全图改写 | 在真实 workload 上验证 forward、gradient、shape/dtype、effects、sharding、alias/donation、schedule、metadata 和缓存合同 | pass contract、对照 capture、负例 |
| TPU 纵向链路 | 固定 libtpu、target 与 topology；分别取得编译和真机证据，将关键源码与 IR、LLO、内存、通信和 profile 关联 | target manifest、编译 dump、真机 capture |
| 检索与交接 | 从 API、symbol、IR op、pass 或故障症状找到源码与证据；新会话能从状态文件恢复下一动作 | 查询工具、源码导读、实验/回放入口、文章 |
| 工程效果 | 在相同任务约束下记录成本、决策、失败与恢复过程；收益结论有对照并说明适用范围 | 代表性任务评价记录 |

最近的阶段验收是：在 source-built CPU jaxlib 上解释固定 synthetic workload 的
`jit(grad(vmap(f)))`，保存 Jaxpr/StableHLO 和断言，并完成隔离缓存的 `sin → cos`
机制探针及 `dot` detection-only matcher。语义保持的真实 workload 改写在 intake、
runtime 与 sharding 基础完成后验收。

长期完成时，workload census 中每个适用项达到审查后的目标深度；关键纵向路径达到
L5，其余可达分支至少达到 L1，且满足各自更高的目标要求。适用项不再保留未收口的
`unobserved` 或 `blocked`，`not-applicable` 有依据。L4/L5 的机器校验合同由
Q016/I004 补齐，在此之前验证器继续拒绝相应深度声明，不通过修改文档提升覆盖。

## 5. 任务拆解

任务对应 [PLAN.md](PLAN.md) 的既有阶段。表中的依赖表示进入该任务验收所需的
基础，实际恢复顺序服从 `status.next_actions`。

| 工作包 | 对应阶段/队列 | 主要任务与产物 | 依赖与启动条件 |
|---|---|---|---|
| 项目基础与恢复 | P0、Q001–Q004 | 维护计划、baseline、证据规范、coverage、架构与状态工具 | 已有基础上持续维护；实验前恢复本机环境 |
| 运行来源与源码构建 | P1、Q005/Q006 | loader resolution、Bazel 依赖闭包、正式 jaxlib build、隔离验证、native 诊断与回滚 | 核心源码、CPU 环境和固定构建工具链就绪 |
| 真实 workload 接入 | Q007/Q008 | 固定 Tokamax/自研框架入口、调用契约、数值参考和 feature census | 取得对应源码及可复现输入；先于真实专题取舍 |
| 程序模型与变换 | P2/P3、Q009–Q014 | Lab 模板、Primitive/Tracer/Jaxpr、AD、batching、组合实验和微型 JAX | 匹配 runtime 与可执行 Lab 基础 |
| Lowering 与 Pass 机制 | P4/P5、Q015 | primitive 到 IR 索引、pass runner、`sin → cos`、`dot` matcher、原生 pass 构建入口 | 变换基础及 source-built jaxlib；先验证机制 |
| Runtime 与 Sharding | P6/P7 | IFRT/PJRT 生命周期、异步执行、alias/donation、多 CPU 分片与 collective 语义 | 匹配 runtime；为安全改写建立合同 |
| Pallas 与 Mosaic | P8/P9 | 内外层 Jaxpr、interpreter、bounds/race、Mosaic MLIR 与 serde fixture | Pallas 可执行环境；TPU 编译行为另行验证 |
| libtpu 与 TPU runtime | P10/P11、Q017 | 固定源码/ABI/target、两条 compiler 路径、load/launch/buffer/event、修改与回滚 | 匹配 libtpu 与工具链；运行部分需要真实 TPU |
| LLO 与硬件映射 | P12 | target-bound schema、pass、source lineage、bundle 与 profile/counter 对照 | libtpu、固定 target 和对应编译/设备证据 |
| 纵向案例与故障诊断 | P13/P14 | 真实训练/推理 dossier、安全 topology rewrite、内存/通信/性能与故障案例 | workload census、runtime/sharding 合同；TPU 结论依赖设备证据 |
| 索引、文章与迁移 | Q016/I001–I004、P15 | 全局索引与查询、IR parser gate、L4/L5 合同、文章、漂移检测和基线迁移 | 随主题积累交付；完整索引覆盖依赖真实 census |

当前 P1 队列依次为：

1. `q005-native-dependency-capture`：补齐当前 `VERSION-SKEW` runtime 的动态依赖取证。
2. `p1-bazel-dependency-closure`：归档并校验 Bazel 外部依赖闭包。
3. `p1-official-build`：完成 wrapper 管理的正式构建，记录 wheel 与持久日志。
4. `p1-isolated-validation`：隔离安装并验证 provenance 和 Lab 001。
5. `p1-native-diagnostic-patch`：重编译诊断改动，证明实际加载并回滚。
6. `q009-lab-template`：建立通用 Lab 与无界面验证入口。

2026-09-13 本机启动检查发现 `.venv` 不存在，JAX、XLA、StableHLO、Shardy 和 LLVM
五个核心 submodule 尚未初始化。指定状态命令无法启动，系统 Python 下的状态校验
也因缺少源码和运行库失败。此项是本机恢复前置条件，历史环境通过记录保留原意。
恢复入口如下；检查通过后继续工具打印的 `first-ready-action`：

```bash
python3 -B tools/sync-environment.py sync
python3 -B tools/sync-environment.py check
.venv/bin/python -B tools/project-status.py --check
.venv/bin/python -B tools/project-status.py
```

正式源码构建另需通过[固定 Docker 工具链入口](docs/building/source-built-jaxlib.md)
的严格 preflight。Tokamax、自研框架、libtpu 和 TPU target/device 的取得情况继续
由状态清单记录；相关输入到位后按依赖启动对应工作包。
