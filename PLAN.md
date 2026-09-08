# JAX → TPU 白盒源码分析执行计划

> 状态：Active
> 启动日期：2026-09-07
> 当前阶段：P1 runtime provenance 与 source-built jaxlib
> 当前工作重点：先补齐当前 wheel 的动态依赖/loader resolution，再完成可留证的源码构建、隔离安装和运行时 provenance 验证

## 1. 背景、动机与任务定义

### 1.1 工程背景

本项目面向基于 JAX/TPU 软件栈开展模型性能与 kernel 优化的 AI Infra 工程师，
源于实际工作中反复需要跨层定位、修改和验证的需求。当优化涉及整个模型的计算图
或 HLO 时，工程师首先需要知道相关表示在哪里生成、经过哪些变换、应该在哪个
位置介入，以及修改需要保持哪些约束。对于刚进入团队的开发者，困难还包括认识
软件栈有哪些组成部分，以及这些部分如何共同完成一次编译和执行。

组织贡献者的工程记录表明，需求可能从模型或 kernel 仓库提出，却需要调用方策略、
JAX 表达与变换规则、XLA pass、Pallas/Mosaic lowering、libtpu 或分析工具配合。
例如，kernel 暴露的残差信息需要上层重计算策略消费；HLO 变换需要保持分片与内存
约束；解释性能变化需要核对最终编译产物、运行版本和观测口径。一个仓库中的局部
改动因此可能形成对上下游接口、编译能力或验证工具的新需求。

这些场景要求把症状、源码位置、接口约束和运行证据联系起来。工程成本贯穿问题
定位、环境准备、修改、编译试跑、结果复查和上下游协作。判断应改哪一层、优先
验证哪个假设、何时停止当前方向，以及怎样证明改动满足原目标，都是性能工程的
必要工作；定位过程中得到的认识也需要沉淀为团队能够复用的材料。

在 AI 辅助的工作方式下，AI 可以参与源码检索、解释、实现和实验执行，而交付仍需
满足具体 workload、版本、资源预算与验收条件。候选方案的生成速度不能单独代表
任务完成的速度或可靠性。本项目关注这些工作所需的软件栈知识与实验条件，支持
工程师和 AI 在可接受的成本内选准方向、控制试错，并得到可复核的结果。

背景首先取材于组织内部贡献者实际提出的工程场景：既包括内部仓库的配套改动，
也包括这些贡献者提交到 JAX、XLA、Tokamax、XProf 的上游需求。成员归属与仓库
可见性分别核对，不能把发生在公开上游的请求自动算作组织外部案例。
主要场景、作者与处理状态见
[`docs/research/organization-engineering-cases.md`](docs/research/organization-engineering-cases.md)，
来源元数据见
[`docs/research/organization-engineering-sources.json`](docs/research/organization-engineering-sources.json)。
首轮[其他公开案例](docs/research/background-motivation-cases.md)保留作对照。
它们证明这些工程问题与需求确实被提出过；其中的历史测试、性能报告和根因假设不
构成本项目固定基线上的执行证据，也不提高 coverage depth。

### 1.2 组织内部提出的工程场景

| 工程问题 | 代表案例 | 对本项目的要求 |
|---|---|---|
| 能拿到 HLO，但缺少完整编辑能力，且改写可能破坏内存合同 | `Iamleos` 请求更多编辑接口：[JAX #39023](https://github.com/jax-ml/jax/issues/39023)；`pathfinder-pf` 报告 transform 后 alias 消失与 OOM：[#40280](https://github.com/jax-ml/jax/issues/40280) | 找到真正的修改接口，检查 alias/donation、buffer lifetime 和版本差异 |
| hook 本身也有跨线程与阶段合同 | `hhhhsdxxxx` 报告 [JAX #38829](https://github.com/jax-ml/jax/issues/38829) 的回调死锁，以及 [XLA #47778](https://github.com/openxla/xla/issues/47778) 的多 slice 恒等变换 verifier 错误 | 同时追踪控制调用链和编译载荷，理解 pass stage 与 topology 上下文 |
| 训练通信需求需要修改框架的映射规则 | `wangfakang` 提交 [JAX #39340](https://github.com/jax-ml/jax/pull/39340)，调整 TPU v7 mesh 物理轴候选优先级；已合并 | 连接逻辑 sharding、物理 topology、通信强度与源码选择规则 |
| kernel 希望使用的操作不一定有对应 lowering | `Iamleos` 的 TensorCore `cumsum` 请求：[#32991](https://github.com/jax-ml/jax/issues/32991)；`pathfinder-pf` 的 SparseCore BF16 gather 请求：[#39577](https://github.com/jax-ml/jax/issues/39577) | 建立 primitive、kernel 类型、dtype、传输 operation 和 backend 支持的映射 |
| 模型量化合同对 kernel 和 compiler 提出额外需求 | `pengchengneo` 提出 FP8 block size、VMEM 和流水线取舍：[Tokamax #839](https://github.com/openxla/tokamax/issues/839)；`sii-xinglong` 提交 block-wise FP8 支持提案：[#794](https://github.com/openxla/tokamax/pull/794) | 将量化精度要求、tiling、内存和编译产物纳入同一个验证任务 |
| 获得 profile 后仍缺少可见性或解释依据 | `pathfinder-pf` 的 counter、DMA 与采样请求：[XProf #2807](https://github.com/openxla/xprof/issues/2807)、[#2446](https://github.com/openxla/xprof/issues/2446)、[#2482](https://github.com/openxla/xprof/issues/2482)；`Prayer3th` 的 LLO 空白报告：[#2441](https://github.com/openxla/xprof/issues/2441) | 核对采集、版本、解析和 UI 各层，解释统计口径及其到指令/源码的映射 |
| 软件栈升级、batch padding 和缓存重用会改变观察结果 | `pathfinder-pf` 的版本升级精度报告：[#33111](https://github.com/jax-ml/jax/issues/33111)；`aolemila` 的 batch 数值请求：[#34080](https://github.com/jax-ml/jax/issues/34080)；`Rodrian7` 的设备子集 warm-cache 报告：[#38004](https://github.com/jax-ml/jax/issues/38004) | 建立版本、数值合同、缓存和设备分配的对照矩阵，避免直接把相关性当成根因 |
| 内部 kernel 向上游交付需要公共接口和可复核验证 | `0xaskr` 的 KDA reference 提案：[Tokamax #1001](https://github.com/openxla/tokamax/pull/1001)；`Fred33146` 的 Pallas 与 chunked XLA 提案：[#1103](https://github.com/openxla/tokamax/pull/1103)、[#1327](https://github.com/openxla/tokamax/pull/1327) | 保留 forward/VJP、输入合同、fallback 和测试边界，区分 PR 状态与实际代码落地 |

内部私有 PR 还提供了更直接的配套场景：kernel 暴露残差标签，上层 remat 策略
消费该标签；以及优化器图大小、gather residual 生命周期、AOT/JIT flags 一致性、
import 时提前初始化 backend 等问题。其精确 PR、作者和版本保存于受限索引，
公开材料只引用抽象研究问题。它们要求本项目能够沿调用边界解释改动如何传递。

以上案例包含已合并改动、未合并提案、仍开放的问题、预期行为和不计划实现的请求。
不能把 PR 合并、issue 关闭、临时绕过和上游根因修复视为同一状态。与 CPU/GPU 有关
的案例只提供公共接口或排查方法的对照，不外推为 TPU 行为。

### 1.3 研究动机与预期价值

本项目要建立从工程问题到源码入口、候选修改、实验和验收的可复用链路。源码导读、
索引、探针与编译产物共同服务于以下四项工程能力：

1. **时间成本：降低从接到问题到完成验收的总成本。** 通过可导航的源码关系、固定版本的
   复现入口、已有编译产物和排查记录，减少重复搜索、环境准备与无效试跑。评价同时
   记录日历时间、人工介入、构建和设备成本，以及返工与资料维护成本；保留未完成
   尝试，并区分首次探索和后续复用的成本。
2. **方向性：提高目标、修改位置与实验顺序的选择质量。** 从实际性能目标出发，连接模型
   调用、kernel、JAX 变换、lowering、compiler pass、runtime 和工具的职责与约束。
   能够说明为何选择当前修改位置、哪些上下游需要配套，以及下一项观察可以排除
   什么假设；遇到反证时有依据地调整方向。对性能比较，先检查 workload、最终操作
   和测量口径是否可比，再解释差异。
3. **可控性：让修改与试错过程有明确边界，并能恢复。** 明确输入、语义、允许改动的范围和资源预算，
   固定或记录实际生效的源码、二进制、编译配置、设备及缓存条件。用与任务规模相称
   的脚本和记录管理执行、结果、停止与回滚，使中断后的工程师或 AI 能接续已有工作。
   跨仓库配套改动、临时绕过的适用范围及撤销条件也需要能够追溯。
4. **可验证性：让正确性、实际生效与工程收益有据可查。** 在任务开始前明确验收，按修改范围
   检查数值、gradient、sharding、alias/donation、effects、缓存与内存等约束，核对
   源码改动是否进入实际加载的二进制及目标编译路径。将静态编译计划、动态事件、
   kernel 指标与端到端效果分别判断，并用独立对照和必要反例支持结论，保留版本、
   target 和适用范围，支持后续复核、升级与回滚。

这些能力服务于团队学习和工程协作：新人能够沿代表性问题理解每层职责并走通一次
定位、修改和验收；已有经验的工程师能够复用证据、减少重复工作，并把局部需求
整理为带有最小复现、接口合同和验收条件的上游 issue、RFC 或 patch。AI 作为这些
材料的使用者与执行协作者，研究主线仍是 JAX 到 TPU 的软件栈及其工程行为。

上述收益是需要检验的研究目标。历史案例证明问题与需求存在，不证明 AI 无法解决，
也不代表本项目已经降低成本。通过代表性任务比较现有源码和常规工具与新增研究
产物的使用效果，在相同任务约束下检查总成本、方向修正依据、过程恢复能力和验收
可靠性；允许不同实现满足同一合同。资料建设与维护的投入也纳入评价，并据此调整
研究优先级与产物复杂度。具体分析范围服从下述 JAX/Pallas 边界。

### 1.4 任务定义与分析边界

本项目面向从事 JAX/TPU 模型性能与 kernel 优化的 AI Infra 工程师，同时为刚进入该
软件栈的开发者提供学习路径，建设一套版本固定、源码可跳转、实验可执行、编译产物
可回放、源码修改可验证的 JAX → TPU 白盒资料库，支持工程师与 AI 从实际问题出发
选择修改方向，在约定范围和预算内推进工作，并依据证据验收与复核结果。

分析边界从 JAX API 开始。控制调用链与编译载荷链分开记录，避免把 host API
调用顺序和 backend 内部 pass 顺序画成一条线。

控制调用链：

```text
JAX API
→ dispatch / cache / jaxlib binding
→ IFRT / PJRT client
→ PJRT plugin loader / C API
→ libtpu compiler 与 runtime
→ executable load / launch / buffer / event
→ TPU driver、firmware 边界与硬件资源
```

普通 JAX 编译载荷链：

```text
Python function
→ primitive / tracing / transformed Jaxpr
→ JAX MLIR lowering
→ StableHLO（按需携带 Shardy/sdy）
→ PJRT compile request
→ libtpu 中的 HLO import、TPU passes、layout 与 scheduling
→ TPU-specific LLO
→ bundle / machine code
```

Pallas 是条件分支：kernel Jaxpr lowering 为 Mosaic TPU MLIR，并序列化进外层
custom call 后进入 PJRT/libtpu。FFI、其他 custom call 和 export/serialization 等
旁路同样由真实 workload census 决定是否展开，不预设每个入口都会经过 Shardy 或
Mosaic。

Tokamax 接入后作为固定 commit 的真实 workload。Tokamax 和自研框架只负责提供纯 JAX/Pallas 入口、输入契约和数值 golden；其上层框架设计不属于分析范围。

CPU 是当前可执行验证平台，用于公共 JAX 前端、StableHLO/HLO、pass、PJRT 和分片语义实验。TPU 编译器、LLO、真实通信与性能结论分别在取得匹配的 libtpu 源码/二进制和 TPU 环境后补齐。

## 2. 完成标准

项目达到长期目标时，应能做到：

1. 对一个 Tokamax 训练或推理入口解释 `jit`、`grad`、`vmap` 为什么能够组合，并给出对应的源码规则。
2. 从 Python 调用逐层得到该入口实际经过的 Jaxpr、StableHLO、可选 Shardy/HLO、
   可选 Mosaic TPU MLIR、LLO 和硬件资源映射，并明确未经过的分支。
3. 在任一关键阶段根据符号、IR operation、pass 或故障症状反向检索源码。
4. 能区分 tracing、lowering、编译、缓存命中、dispatch 和设备执行发生的时机。
5. 能实现并验证 primitive、lowering、StableHLO/HLO pass、Pallas kernel，以及取得源码后的 libtpu compiler/runtime/LLO 修改。
6. 能从固定源码构建匹配的 jaxlib/编译器产物，替换运行环境，证明修改生效并安全回滚。
7. 能定位 retracing、重复编译、内存峰值、sharding、collective、Mosaic race 和 TPU 性能问题。
8. 每个主题形成源码导读、索引、可运行实验或回放、分析文章，并汇总成完整架构图、
   capture 集和贡献记录。
9. coverage inventory 中每个适用项均达到经 workload census 审查后的目标深度，所有
   topic/claim/source/capture 引用可由查询工具解析。

## 3. 当前固定基线

| 组件 | 当前基线 | 说明 |
|---|---|---|
| Python | `3.12.3` | `.python-version` |
| uv | `0.12.9` | `pyproject.toml` |
| JAX source | `5832e866449a41c3eea6333416528039119a0fde` | `0.11.2.dev20260830+5832e86644`，editable source |
| XLA source | `496bd4bd49db9ecbffd85da630b49c860b724604` | 由 JAX `MODULE.bazel` 固定 |
| StableHLO source | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` | submodule gitlink |
| Shardy source | `eb23a98329aa70d991aa2d8a51a209af1f8df8fc` | submodule gitlink |
| LLVM source | `ab547095ead5464dc024d66264d9b8a987f429f3` | submodule gitlink |
| CPU jaxlib wheel | `0.11.1` | 当前仅用于启动分析；与 JAX/XLA source 不构成严格运行证据闭包 |
| libtpu | 未固定 | 需要包版本、build ID、源码 commit、ABI、编译 flags |
| TPU target | 未固定 | 需要代际、slice/pod topology、runtime/firmware、device assignment |
| Tokamax | 未接入 | 需要 URL、commit 和代表性纯 JAX/Pallas 入口 |
| 自研框架 | 未接入 | 需要 sanitized revision/opaque provenance、JAX/Pallas 边界和调用契约 |

在 P1 完成前，任何由当前 `jaxlib==0.11.1` wheel 得到的 C++/XLA 运行结论都必须标记为版本错位。Python-level JAX 源码实验可以继续执行，但不能据此声称当前 `upstream/xla` 中的 C++ 路径已经被运行验证。

## 4. 两条主要 TPU 编译路径

### 4.1 普通 JAX/XLA 路径

```text
Python function
→ transformed Jaxpr
→ StableHLO + sharding metadata
→ PJRT compile request
→ libtpu / TPU compiler
→ HLO import, optimization, SPMD, layout, scheduling, memory assignment
→ TPU-specific LLO
→ bundle / machine code
→ TPU runtime / hardware
```

### 4.2 Pallas/Mosaic 路径

```text
Pallas kernel + BlockSpec
→ Pallas kernel Jaxpr
→ Mosaic TPU MLIR (arith/vector/tpu dialects)
→ serialized payload in an outer HLO custom call
→ PJRT / libtpu Mosaic compiler
→ TPU-specific LLO
→ bundle / machine code
→ TPU runtime / hardware
```

必须同时保留外层 HLO custom call 和内层 Mosaic module 两个观察面。Mosaic TPU dialect 不是 LLO，Pallas/Mosaic 也不是所有普通 JAX operation 的共同 lowering 路径。
这两条是当前基线的主干，而不是对所有扩展入口的穷举；workload 中出现的 FFI、
外部 custom call、export/serialization 或其他 backend extension 必须作为条件分支
单独登记。

## 5. 覆盖深度、证据等级与来源限定

### 5.1 覆盖深度

| 等级 | 达成条件 |
|---|---|
| L0 | 有术语定义和所属层级 |
| L1 | 有职责、输入输出和上下游接口 |
| L2 | 有固定 commit 的源码入口、caller/callee 和数据结构 |
| L3 | 有可运行 probe、真实 IR 和自动断言 |
| L4 | 有源码修改、构建、测试、安装与回滚闭环 |
| L5 | 在真实 TPU 上完成数值、LLO、内存、通信和性能验证 |

Tokamax 关键纵向路径目标为 L5；非关键条件分支先达到 L1–L3。是否继续深入由真实 workload、故障或 pass 需求决定。

### 5.2 证据等级

| 标签 | 含义 |
|---|---|
| `RUN-CPU` | 在当前固定 CPU 环境实际执行 |
| `SIM-TPU` | 使用 TPU/Pallas interpreter 或静态模拟执行 |
| `COMPILE-TPU` | 针对固定 TPU target 完成编译，未在设备运行 |
| `RUN-TPU` | 在记录了代际和 topology 的真实 TPU 上运行 |
| `SOURCE-ONLY` | 仅由固定版本源码证明，尚无执行证据 |
| `REPLAY-OFFLINE` | 使用锁定工具重放或分析已有 capture |

`VERSION-SKEW` 不是证据等级，而是可以叠加到任一运行或回放证据上的来源限定：
它表示实际加载的二进制与所引用的源码 revision 不一致。结论必须紧邻证据等级与
限定；CPU 数值等价不得表述成 TPU layout、时序、通信或性能等价。完整规则以
[`docs/contributing/evidence-conventions.md`](docs/contributing/evidence-conventions.md)
为准。

### 5.3 可度量覆盖清单

[`manifests/coverage.json`](manifests/coverage.json) 是长期范围的机器可读真相，结构由
[`manifests/schema/coverage.schema.json`](manifests/schema/coverage.schema.json) 约束。
它同时枚举软件层和 workload feature，每项只能处于：

| 状态 | 含义 |
|---|---|
| `unobserved` | 在范围内，但还没有足够证据达到任何声明深度 |
| `covered` | 已达到条目中明确记录的 `achieved_depth`；不表示自动达到最终目标 |
| `blocked` | 需要尚未取得的源码、target、设备或真实 workload，必须记录解除动作 |
| `not-applicable` | 已证明固定 workload/分支不经过该项，必须记录理由 |

Tokamax 和自研框架接入后，先对固定训练/推理入口做 workload census，枚举实际出现的
transformation、primitive、effect、control flow、dtype、sharding、custom call、
Pallas kernel、IR op、pass 和 runtime API。每个观察项都必须映射到 coverage entry、
topic/claim/source/capture；关键纵向切片达到 L5，其余可达分支至少达到 L1，或者带有
可复核的 `blocked`/`not-applicable` 理由。只完成架构图不等于完成该层源码分析。
JSON Schema 负责结构约束；`tools/validate-coverage.py` 在其上检查 layer/feature ID
全局唯一、依赖图无环、topic/claim/source/capture 的双向绑定，以及 L0–L3 的证据
门槛。`--require-complete` 额外检查 workload census、目标深度和所有阻塞项是否收口。
当前 L3 门禁验证结构化断言、producer/probe/input 身份、claim 与本地 IR 的绑定，以及
IR 字节、格式与最低结构要求；它不声称已经对 IR 做语义解析。parser-aware IR 验证由
Q016/I004 补齐。
L4 的修改/构建/测试/安装/回滚闭环与 L5 的 target-bound TPU 运行契约尚未编码，因此
验证器对 L4/L5 声明保持 fail-closed；该合同与全局索引生成仍由 Q016/I004 完成。

## 6. 目标目录结构

```text
docs/
  architecture/
    00-whole-stack.md
    01-regular-xla-tpu-path.md
    02-pallas-mosaic-tpu-path.md
    03-runtime-control-plane.md
  index/
    glossary.md
    symbols.yaml
    ir-lineage.yaml
    pass-catalog.yaml
    flags-and-dumps.yaml
    symptoms.yaml
  topics/
    01-jax-program-model/
    02-tracing-jaxpr/
    03-jit-and-cache/
    04-automatic-differentiation/
    05-vmap-and-batching/
    06-transformation-composition/
    07-control-flow-effects-remat/
    08-stablehlo-lowering/
    09-pass-infrastructure/
    10-sharding-and-shardy/
    11-xla-hlo-pipeline/
    12-pjrt-runtime/
    13-pallas-program-model/
    14-mosaic-tpu/
    15-libtpu-compiler/
    16-tpu-runtime-control-plane/
    17-llo-and-tpu-hardware/
    18-debugging-and-performance/
    19-extension-build-contribution/
workloads/
  synthetic/
  tokamax/
labs/
patches/
manifests/
  baseline.json
  status.json
  coverage.json
  schema/
  build-fingerprints/
artifacts/                 本地持久但默认不进 Git 的大产物
  builds/<build-id>/
articles/
tools/
```

目录按需创建。空目录和没有证据的占位文章不进入仓库。

## 7. 每个主题的固定证据包

```text
README.md                         原理、边界、关键问题和已验证结论
topic.json                        claim、证据等级、capture 引用和局限
source-index.json                 revision、symbol、path、line 和职责
callgraph.mmd                     Python/C++/ABI/IR 边界与控制流（按需）
invariants.md                     shape、dtype、effect、layout、sharding 等契约（按需）
probes/                           最小、无隐藏状态的可运行探针
captures/<capture-id>/            manifest 与适合进入 Git 的小型确定性产物
tests/                            数值、结构、缓存、pass 或 ABI 断言（按需）
patches/                          可检查、应用和反向应用的源码改动（L4 必需）
build-and-replay.md               构建、运行、dump、回放和清理命令（L4 必需）
failures.md                       症状 → 观测点 → 源码 → 原因 → 验证方法（按需）
article.md                        从本主题已审查证据生成的分析文章或文章章节（完成时必需）
```

不要求每个主题第一天就具备全部文件。进入 L3 时必须具备 probe、capture 和自动断言；
进入 L4 时必须具备 patch、build/replay 和回滚验证。大型、受限或含敏感信息的产物不
直接提交；manifest 使用不泄露内部路径的 URI、SHA-256 和获取说明定位它们。

每个主题使用同一张交付矩阵，状态同步到 coverage inventory：

| 交付物 | L1 | L3 | L4/L5 或主题完成 |
|---|---|---|---|
| 源码导读与边界 | `README.md` | 用 capture 校正动态路径 | 写明目标相关的完整 caller/callee 与限制 |
| 可检索索引 | `source-index.json` | 关联 claim/capture | 汇入全局 symbol/IR/pass/runtime 索引并通过漂移检查 |
| 实验 | 可执行 source inspection 或 fixture parser | CPU/simulator/device probe 与自动断言 | build/modify/replay/rollback；无设备时可先用离线 replay，最终 L5 仍需真机 |
| 文章 | 建立提纲 | 只引用已验证结论 | 发布 `article.md` 或 `articles/<topic-id>.md` |

长期完成标准要求所有主题具备这四类材料；阶段性 `blocked` 只允许保留明确的输入缺口
和下一动作，不能用空占位文件冒充完成。

底层源码修改能力按组件闭环，而不是只做一次泛化的 C++ 改动：

| 组件 | 最小 L4 闭环 | 主要阶段 |
|---|---|---|
| JAX Python | 可逆 patch、定向测试、加载路径证明、回滚 | P3 |
| jaxlib binding / public PJRT adapter | 源码构建、隔离 wheel、诊断改动、运行时 binary 指纹、回滚 | P1/P6 |
| StableHLO / Shardy | pass 或 verifier patch、原生 target/test、before/after fixture、回滚 | P5/P7 |
| XLA HLO | 原生 `HloModulePass` 或等价 patch、pipeline test、重新链接/加载证明、回滚 | P5/P13 |
| libtpu compiler/runtime | 固定 ABI/build ID、源码构建、诊断改动、替换与回滚 | P10/P11 |
| LLO | target-bound pass patch、compiler/bundle diff、验证器和回滚 | P12 |

## 8. 长期阶段与验收门槛

### P0：项目章程、索引规范与版本清单

目标：让长期任务可以跨会话恢复，且每项结论有明确边界。

- [x] 确定读者：刚进入 JAX 的 kernel 开发者。
- [x] 确定主平台：TPU；CPU 为当前实验平台。
- [x] 确定边界：JAX API 到 TPU-specific LLO/硬件映射。
- [x] 区分普通 XLA/TPU 与 Pallas/Mosaic 两条路径。
- [x] 建立本长期计划。
- [x] 建立机器可读的 `manifests/baseline.json`。
- [x] 建立 evidence、source index、capture manifest schema。
- [x] 固化公开边界、两条主要载荷路径和私有/设备待验证区的初版架构骨架与术语表。
- [x] 建立 `manifests/status.json` 和只读状态校验/摘要入口。
- [x] 建立 `manifests/coverage.json`、coverage schema 和按深度度量的完成规则。
- [x] 建立 evidence/coverage 语义校验器及隔离负例 selftest；L4/L5 在合同完成前
  fail-closed。
- [ ] 早期接入并固定 Tokamax 与自研框架 revision，记录精确 JAX/Pallas 边界和调用契约。
- [ ] 对固定训练/推理入口生成首份 workload feature census，再决定专题优先级。
- [ ] 建立由 topic/coverage 派生的全局索引 schema、生成器、校验器和查询 CLI。

以上三项分别由 Q007、Q008 和 Q016 跟踪，不阻塞已经达到验收条件的 P0 core。
P0 core 验收：新会话只读取本文件、baseline、coverage 和状态文件，就能判断当前
版本、初版架构边界、已完成深度、阻塞项和下一项任务。Tokamax/自研框架 intake
是收到外部源码后立即执行的早期 gate；它不阻止 P1 和 synthetic labs，但必须先于
真实 workload 专题取舍及生产级 topology rewrite。

### P1：源码与运行二进制一致的构建闭环

目标：消除 JAX `0.11.2.dev` source 与 `jaxlib==0.11.1` wheel 的证据错位。

- [ ] 记录当前 wheel 的版本、路径、git hash 和动态依赖。
- [x] 记录 JAX/XLA 使用的 Bazel、Clang、Python 和固定构建配置。
- [x] 将核心源码、uv 和系统工具链恢复固化为环境锁、同步脚本及固定 Docker 环境；
  宿主机 CPU 与容器严格 preflight 分别验证，保留原二进制哈希门禁。
- [ ] 固定并归档 Bazel module graph、resolved repositories、registry/module extension
  输入和下载完整性，使外部依赖闭包可以审查与重放。
- [ ] 从固定 JAX/XLA commit 构建 CPU jaxlib wheel。
- [ ] 将 wheel 保存到版本化 artifact 目录并生成 SHA-256。
- [ ] 在隔离环境安装，验证 JAX、jaxlib、XLA commit provenance。
- [ ] 做一个最小 C++ 诊断改动，重新构建并证明运行时使用了该改动。
- [ ] 记录安装、切换、清缓存和回滚步骤。
- [ ] 构建后续需要的 StableHLO/XLA inspection tools。

验收：修改一处 C++，能够构建、安装、观察变化、执行测试并恢复基线。

### P2：JAX 程序模型

目标：建立后续所有 transformation 的共同语言。

- [ ] PyTree flatten/unflatten 和 treedef。
- [ ] concrete value、abstract value、shape、dtype、weak type。
- [ ] Primitive 定义、bind、impl、abstract evaluation。
- [ ] Trace、Tracer、`trace_ctx.trace`、`parent_trace` 链和
  `unsafe_get_trace_stack`；旧 `MainTrace`/sublevel 只作版本迁移对照。
- [ ] Jaxpr、consts、`ClosedJaxpr` 兼容别名、invar/outvar/equation/effect。
- [ ] 写一个最小 Jaxpr pretty-printer 和 evaluator。
- [ ] 用一个 primitive 从 Python wrapper 跟到 bind 和 eager implementation。

验收：能逐条执行一个包含常量、多个 primitive 和 effect 信息的 Jaxpr，并把每项结构定位到源码。

### P3：变换系统与组合性

目标：源码级回答 `jit/grad/vmap` 为什么能够组合。

- [ ] `jit` callable 创建、首次 tracing、lowering、compile、cache hit 时序。
- [ ] retracing、re-lowering、recompile 和 redispatch 的区分。
- [ ] JVP、linearize、partial evaluation、transpose、VJP、backward pass。
- [ ] primal、tangent、cotangent、residual 和 zero tangent。
- [ ] BatchTracer、batch dimension 和 primitive batching rules。
- [ ] `jit/grad/vmap` 六种排列的 trace stack、Jaxpr、StableHLO 与缓存对照。
- [ ] `cond/while/scan/remat/custom_jvp/custom_vjp` 与组合性。
- [ ] 微型 JAX：primitive/rule/interpreter/JVP/VJP/batching/staging 的最小实现。

验收：新增一个 primitive，并同时实现 abstract evaluation、JVP/transpose、batching 和 lowering，使其在关键组合中通过测试。

### P4：Jaxpr 到 StableHLO

目标：解释语义如何跨过 Python/JAX 与编译器边界。

- [ ] primitive lowering registry 和 lowering context。
- [ ] Jaxpr equation 到 MLIR operation 的映射。
- [ ] StableHLO type、shape、dynamic dimension、tuple 和 effect/token 表达。
- [ ] source location 从 Python 传播到 MLIR/HLO。
- [ ] `lower()`、`compiler_ir()`、export、serialization 和 compatibility。
- [ ] custom call、FFI 与外部 kernel 边界。
- [ ] 建立 primitive → lowering rule → StableHLO op 交叉索引。

验收：任意选定 primitive 都能给出源码入口、lowering rule、生成 IR、约束和可执行验证。

### P5：Pass 基础设施与模型计算图改写

目标：先建立 pass 观察、匹配、pipeline stage 和缓存隔离机制；生产级安全改写在
P6/P7 的 runtime、alias/donation 与 sharding 基础完成后进入 P13 纵向案例。

- [ ] MLIR rewrite、canonicalization、pass manager 和 pass instrumentation。
- [ ] StableHLO pass 注册、单 pass runner、FileCheck 和 reducer。
- [ ] HloModule/HloComputation/HloInstruction 数据模型。
- [ ] HLO pass pipeline、ordering、pre/post scheduler 边界。
- [ ] HLO dump、before/after diff、pass disable 和 bisect。
- [ ] 使用 `jax.extend.xla.register_hlo_module_transformation` 完成故意改变语义的
  `sin → cos` 机制探针；断言输出发生预期变化，不把它表述成正确性优化。
- [ ] 隔离或禁用 persistent cache，验证 callback、注册顺序、清理和 pre/post-scheduler
  stage；结构改写必须选择合法 stage 并说明 schedule 的维护方式。
- [ ] 为两条兼容 `dot` 合并成宽 `dot + slice` 编写 detection-only matcher、拒绝原因和
  pass contract；此阶段不把 synthetic 命中当成生产 rewrite 完成。
- [ ] 记录 transformation 不进入 persistent cache key 的当前版本行为和可复现规避方式。
- [ ] 为 StableHLO、Shardy 和 XLA 原生 pass 各建立至少一个 build/test/rollback 入口。

验收：在 source-built jaxlib 上完成语义改变机制探针和 detection-only matcher，保存
before/after HLO，证明 stage、缓存隔离、清理与回滚行为；语义保持的生产改写由 P13 验收。

### P6：IFRT、PJRT 与运行时

目标：解释编译请求和 executable 如何到达设备。

- [ ] JAX backend discovery 与 plugin 初始化。
- [ ] Python binding、jaxlib、IFRT、PJRT C++ 与 PJRT C API 边界。
- [ ] client/device/topology/compiler/executable/buffer/event 生命周期。
- [ ] host-device transfer、async dispatch、ready event 和同步点。
- [ ] buffer donation、alias、ownership 和删除语义。
- [ ] in-memory/persistent compilation cache 与 executable serialization。
- [ ] CPU PJRT 和 phase-compile sample plugin 实验。

验收：能从一次 `jax.jit` 调用追踪到 CPU PJRT executable 执行，并用 probe 观察关键对象生命周期。

### P7：Sharding、Shardy 与分布式

目标：覆盖项目会用到的全部 JAX 分片入口，重点理解共同编译语义。

- [ ] `Mesh`、`PartitionSpec`、`NamedSharding` 和 global/local shape。
- [ ] `jit`/`pjit` sharding、`pmap`、`shard_map` 的语义和源码入口。
- [ ] sharding constraint、manual/auto axes 和 reshard。
- [ ] Shardy import、propagation、export pipeline。
- [ ] SPMD partitioning 和 collective operation。
- [ ] donation、layout、pass 改写与 sharding 的交互。
- [ ] 使用多个强制 CPU device 做可执行实验。
- [ ] 多 host 初始化、rendezvous、failure 和 hang 路径源码索引。

验收：一个分片 MLP/代表子图能在多 CPU device 运行，并产出带完整 sharding lineage 的 IR；TPU 网络结论保持待验证状态。

### P8：Pallas 程序模型

目标：从 kernel 作者视角理解 Pallas 如何嵌入 JAX transformations。

- [ ] Grid、BlockSpec、Ref、index map、scratch、alias。
- [ ] `pallas_call` primitive、kernel Jaxpr 和 outer Jaxpr。
- [ ] Pallas 的 abstract evaluation、JVP、transpose 和 batching rules。
- [ ] interpret mode 与编译模式的语义差异。
- [ ] CPU 上的 TPU interpret、HBM/VMEM 模拟、DMA、barrier、semaphore。
- [ ] 越界、race 和同步错误实验。

验收：一个代表 kernel 在 interpreter 中完成数值与 race 验证，且能解释其内外两层 Jaxpr。

### P9：Mosaic TPU

目标：解释 Pallas kernel 如何变成 libtpu 可消费的 TPU kernel 表示。

- [ ] Pallas primitive 到 arith/vector/tpu dialect lowering。
- [ ] Mosaic TPU module、layout、tiling、memory space 和 pipeline。
- [ ] DMA、semaphore、barrier、collective 与 hardware generation 参数。
- [ ] Mosaic module serialization 和 outer custom call backend config。
- [ ] TPU dialect verifier、canonicalization 和 serde。
- [ ] 固定 Mosaic MLIR fixture、round-trip 和静态检查。

验收：本地生成并保存代表 kernel 的 Mosaic TPU MLIR，能够从每个 op 反查 Pallas 源码和 lowering rule。

### P10：libtpu 与 TPU compiler

目标：在获得匹配源码/二进制后闭合公开源码之外的 TPU 编译路径。

- [ ] 固定 libtpu package、`libtpu.so` build ID、source commit 和构建工具链。
- [ ] 固定 PJRT C API/extension ABI。
- [ ] 固定 TPU 代际、topology、target description、runtime/firmware。
- [ ] 确认 offline compiler、HLO replay、pass registry 和单 pass runner。
- [ ] 追踪普通 StableHLO/HLO 输入的 TPU pipeline。
- [ ] 追踪 Mosaic custom call 输入的专用 pipeline。
- [ ] 保存 pre/post-SPMD、optimized/scheduled HLO、layout、buffer 与 memory assignment。
- [ ] 构建或接入 LLO printer/parser 和 bundle inspection tools。
- [ ] 从固定源码构建 libtpu/compiler 产物，记录 flags、ABI、build ID、二进制 hash 和
  实际加载路径。
- [ ] 做一个最小 compiler 诊断改动，完成替换、运行或 replay 证明以及安全回滚。

验收：固定输入、target 和 flags 后，可重复生成可归因的 TPU compiler artifacts；无 TPU 时标记为 `COMPILE-TPU`。

### P11：TPU runtime 与 control plane

目标：把 compile、load 和 execute 分开，解释 executable 如何由 libtpu runtime 到达设备。

- [x] 建立 `docs/architecture/03-runtime-control-plane.md` 初版骨架，区分公开 PJRT seam、
  libtpu 私有实现假设和设备证据。
- [ ] 取得 libtpu 与 TPU capture 后，用固定源码和运行证据补齐初版骨架中的待验证区。
- [ ] 追踪 PJRT compile/load/execute API 到 libtpu compiler/runtime 的分流和会合点。
- [ ] executable serialization、load/unload、program launch 与 device assignment 生命周期。
- [ ] host/device buffer、donation/alias、event、async dispatch、同步和错误传播。
- [ ] driver/firmware 边界、队列、infeed/outfeed 以及 profile/counter 接口。
- [ ] 多 host 初始化、rendezvous、collective launch、取消和 hang 路径。
- [ ] 固定 runtime build ID、配置、实际加载 binary 和 target manifest。
- [ ] 修改一个 runtime 诊断点，重新构建/替换并完成 load/launch 证明与回滚。

验收：`COMPILE-TPU` 能证明 compiler output/serialization，并静态映射 load 接口；
`RUN-TPU` capture 才能证明真实 device load，并从一次 PJRT execute 追踪到 program
launch、buffer/event 完成和设备 profile。CPU PJRT 只能验证公共对象语义，不能
替代此验收。

### P12：TPU-specific LLO 与硬件映射

目标：把机器相关 IR 与 TPU 资源建立可检索关系。

- [ ] 固定目标代际的 LLO schema、opcode、operand 和验证规则。
- [ ] LLO pass pipeline、schedule 和 target machine bundle 生成；证据确认后再注明是否为 VLIW。
- [ ] 从目标代际 schema/source 枚举实际地址空间和生命周期；HBM、CMEM、VMEM、
  SMEM、IMEM 仅作为待核验候选，不预设每代全部存在或同义。
- [ ] 从目标资料枚举实际 compute/vector/data-movement、sequencer 和 interconnect
  资源；名称与能力绑定 target manifest。
- [ ] HLO/Mosaic source location → LLO → bundle 的 lineage。
- [ ] 静态 cost report 与真实 XProf/counter 的差异。
- [ ] 修改一个 LLO pass，验证预期指令或调度变化。

验收：任取一个关键 LLO 指令组，可以向上追到 Tokamax/JAX 源码，向下解释使用的硬件单元，并在真实 TPU 上验证行为。

### P13：Tokamax 与自研框架纵向案例

目标：用真实 workload 连接此前独立建立的各层材料。

- [ ] 复用早期 intake 中固定的 Tokamax、自研框架 revision、依赖、JAX/Pallas 边界和
  sanitized invocation contract；上层框架逻辑仍不进入分析范围。
- [ ] 从两者抽取可复现的纯 JAX/Pallas fixture，并证明 static args、RNG、dtype、mesh、
  sharding、donation 和编译配置没有在抽取时丢失。
- [ ] 由 workload census 决定覆盖集合；至少检查 elementwise/fusion、matmul/MLP、
  reduction/softmax、control flow/effect、Pallas、custom call 与 sharded collective 的
  observed/not-applicable 状态，不用预选列表代替实际枚举。
- [ ] 为训练入口保存 `jit(grad(...))` 全链产物。
- [ ] 为推理入口保存 compile、dispatch 和 runtime 全链产物。
- [ ] 先写 detection-only pass，记录模式命中和拒绝原因。
- [ ] 在 P6/P7 基础上完成两条兼容 `dot` → 宽 `dot + slice` 的语义保持计算图改写，
  再完成适用的 TPU kernel 替换。
- [ ] 比较数值、gradient、compile time、HBM、cycles 和硬件利用率。

验收：每个选定 workload 都形成一个可独立复现的 white-box dossier。

### P14：故障、性能和通信诊断

目标：把源码地图转化为现实排障工具。

- [ ] retracing、cache miss、重复 lowering/compile。
- [ ] 编译时间、pass 热点和 compiler crash/reducer。
- [ ] buffer donation、remat、HBM/VMEM/SMEM 峰值。
- [ ] sharding propagation、reshard 和 collective 性能。
- [ ] multi-host 初始化、通信 hang 和错误传播。
- [ ] Pallas OOB、race、DMA/semaphore 错误。
- [ ] LLO schedule、资源冲突、低利用率和数值差异。
- [ ] 建立症状 → flag/dump → IR → pass → source 的决策索引。

验收：预先植入的代表性故障可以只使用仓库材料完成定位、解释和验证。

### P15：文章、贡献与版本迁移

目标：使成果可传播并能跟随上游演进。

- [ ] 每个主题形成源码导读、索引、可执行 lab/replay 和中文分析文章；纵向案例另有
  跨层总文。
- [ ] 维护 CodeTour、Marimo/Jupyter 与命令行 probe。
- [ ] 把通用修复整理为 JAX/XLA/StableHLO/Shardy 上游贡献。
- [ ] 对 source index、symbol、IR golden 和 pass pipeline 做漂移检测。
- [ ] 定义升级 JAX/XLA/libtpu/Tokamax baseline 的迁移流程。
- [ ] 保留旧 baseline captures，避免覆盖历史证据。

验收：升级基线后能自动列出失效源码锚点、IR 变化、测试差异和需要重做的 TPU captures。

## 9. “用 pass 修改模型拓扑”旗舰任务

这里的“拓扑”拆成三个可修改对象：

1. 计算图：operation、edge、fusion、custom call 和 composite。
2. 分片图：sharding、device assignment、reshard 和 collective。
3. kernel schedule：layout、tiling、DMA、pipeline 和 LLO schedule。

物理 TPU pod topology 是 compiler input，不能由普通模型 pass 任意改变。

旗舰任务按以下步骤推进：

1. 用当前源码已有的 `jax.extend.xla.register_hlo_module_transformation` 完成故意改变语义的
   `sin → cos` 机制探针，断言结果按预期改变；隔离或关闭 persistent cache，并验证
   callback、合法 stage、注册和清理行为。
2. 为两条兼容 `dot` → 宽 `dot + slice` 建立完整 pass contract，先只检测候选和输出
   拒绝原因。
3. 完成 P6 runtime 与 P7 sharding/alias/donation 基础后，把 detection-only matcher
   应用到 Tokamax 代表子图，保存命中、拒绝和源码位置。
4. 在选定的 StableHLO/HLO stage 实现语义保持的 topology rewrite，并验证 forward、
   gradient、sharding、alias/donation、schedule、缓存和数值稳定性。
5. 通过 attribute/composite/custom call 将识别结果可靠传递到 TPU compiler。
6. 在 libtpu 或 Mosaic pipeline 中消费标记，保存每个关键 pass 前后 IR。
7. 追踪到 LLO 与硬件资源，比较改写前后的 compile time、内存和性能。

若 pass 位于 AD 之后，它看到的是已经生成的前向或反向程序；它必须保持这些程序的语义，但不需要为新图节点注册 JVP/VJP。若在 Jaxpr 层引入新 primitive，则必须实现 abstract evaluation、JVP/transpose、batching 和 lowering。

所有 topology rewrite 至少检查：

- shape、dtype、dynamic dimension；
- side effect、token、control dependency；
- sharding 和 collective semantics；
- layout、schedule、alias 和 donation；
- source location 与 debug metadata；
- forward、gradient 和数值容差；
- compile cache 与 persistent cache；
- 未命中和安全拒绝路径。

## 10. Workload 与 capture 规范

每个 workload 固定：

- 来源 URL、commit 和许可证；
- 独立的纯 JAX/Pallas entry function；
- 输入 shape、dtype、static args、sharding、donation；
- 参数和小规模数值 golden；
- 训练/推理模式；
- 期望经过的 compiler branch；
- CPU、TPU simulator、offline TPU compiler、真实 TPU 的支持状态。

一个完整纵向 dossier 用适用性矩阵检查以下观察面。每项必须标记 `present`、
`not-applicable`、`unavailable` 或 `blocked`；普通 JAX 不强制产生 Mosaic，未启用
Shardy 的程序不伪造 Shardy IR，CPU 或 source-only 工作也不伪造 LLO/profile：

1. 原始和 transformation 后的 Jaxpr；
2. StableHLO text/bytecode 与 Shardy IR；
3. pre/post-SPMD 和 pre/post-scheduler HLO；
4. layout、schedule、buffer assignment、memory assignment；
5. Pallas 案例的内层 Mosaic TPU MLIR；
6. LLO text/proto 与关键 pass snapshots；
7. bundle/machine-code metadata；
8. compile options、target、topology 和 flags；
9. 数值结果、profile、memory 和 collective trace；
10. 生成以上文件的单一 replay 入口。

单个 capture 只保存该次固定 revision、命令、backend、target 和 evidence level 实际
产生的文件及断言，不要求同时横跨全部十个观察面。一个 dossier 可以引用多个相互
独立的 `RUN-CPU`、`SIM-TPU`、`COMPILE-TPU`、`RUN-TPU` 或 `REPLAY-OFFLINE`
capture；缺失层必须保留原因和解除动作，不能用较弱 capture 提升证据等级。

大型、不可稳定复现或包含敏感信息的原始 capture 不直接进入 Git；仓库保存经过裁剪的 fixture、内容 hash、生成脚本和来源说明。

## 11. 资料索引

长期索引必须同时支持以下查询方向：

- API → transformation → primitive → lowering → IR op → pass → runtime → hardware；
- source symbol → caller/callee → artifact；
- IR operation → producer/consumer/pass；
- pass → pipeline stage → invariants → tests；
- 错误/性能症状 → flags/dumps → responsible layer；
- Tokamax source location → Jaxpr/StableHLO/HLO/Mosaic/LLO；
- hardware counter/LLO opcode → kernel/model source。

行号只作为固定 commit 下的辅助信息；稳定身份由 repository、commit、path 和 symbol 共同确定。

全局索引是独立里程碑，不靠手工维护多份副本：

- [ ] I001：为 symbol、IR lineage、pass catalog、flags/dumps 和 symptoms 定义 schema。
- [ ] I002：从 topic-local `source-index.json`、claim、capture 与 `coverage.json` 确定性生成
  全局索引，并拒绝悬空引用和重复稳定身份。
- [ ] I003：提供一个无界面 query CLI，支持上述七种正向/反向查询，输出精确 revision、
  source symbol、证据等级、coverage depth 和 replay 入口。
- [ ] I004：把 index regeneration、schema validation、coverage completeness 和固定 commit
  的 symbol drift 检查接入里程碑验收。

验收：在 fresh checkout 中只用锁定环境重建索引；对一个 Tokamax source location、
一个 HLO/LLO operation 和一个故障症状分别完成跨层反查，结果没有人工补链。

## 12. 文章系列

文章由已经完成证据闭环的材料生成，不先写脱离实验的长篇教程。候选主题：

1. 一个 JAX 值从 Python 到 TPU 的旅程；
2. Tracer、Primitive 和 Jaxpr 如何组成可扩展解释器；
3. `jit/grad/vmap` 为什么可以组合；
4. JVP、partial evaluation 与 transpose 如何构成反向模式 AD；
5. Jaxpr 如何 lowering 到 StableHLO；
6. Shardy 与 SPMD 如何改变分片计算图；
7. 如何观察并修改 XLA HLO pass pipeline；
8. PJRT 如何管理编译、buffer 和异步执行；
9. Pallas 如何成为 Mosaic TPU kernel；
10. libtpu runtime 如何 load、launch 并完成 buffer/event；
11. 从 Mosaic TPU MLIR 到 LLO 与硬件；
12. 用 pass 改写一个 Tokamax 计算图；
13. 从源码构建、替换和调试 JAX/libtpu 软件栈。

## 13. 当前阻塞项与处理方式

| 缺口 | 当前处理方式 | 解除条件 |
|---|---|---|
| JAX source 与 jaxlib wheel 错位 | 标记 `VERSION-SKEW`，P1 自建 wheel | source-built jaxlib 验证通过 |
| Tokamax 未在工作区 | 先用 synthetic workloads 验证工具；coverage 保持 `blocked` | 固定 URL/commit/entry 并完成 census |
| 自研框架未在工作区 | 只定义 sanitized JAX boundary intake 契约；coverage 保持 `blocked` | 固定 revision/opaque provenance 和 JAX/Pallas entry |
| libtpu 未锁定 | 公开边界标 `SOURCE-ONLY`，先完成接口图 | 提供匹配 source/binary/build ID |
| 本地没有 TPU | 完成 CPU、simulation、compile-only 工作 | 获得真实 TPU 环境 |
| TPU/LLO 随代际变化 | 所有 capture 绑定 target manifest | 固定首个 TPU 代际和 topology |

这些缺口不会阻止 P0 core、P1–P9 的大部分 synthetic/public-source 工作，但禁止提前
宣称真实 workload coverage、TPU runtime、LLO 或性能已经验证。

## 14. 执行协议

后续代理或会话按以下顺序恢复任务：

1. 完整读取本文件。
2. 检查根仓库和相关 submodule 的 `git status`，保护已有用户改动。
3. 读取 `manifests/baseline.json`、`manifests/coverage.json` 和最近的状态记录。
4. 从“当前执行队列”选择第一个未完成且未阻塞的任务；新观察到的 workload feature
   必须先进入 coverage inventory。
5. 先查固定版本源码，再形成最小 probe；不要凭最新版文档替代锁定源码。
6. 保存真实产物和 manifest，再形成源码解释。
7. 只有跨层链路复杂时建立 CodeTour/交互 notebook；自动验证使用命令行 probe/test。
8. 修改 `upstream/*` 时优先维护外层可逆 patch；实验结束检查 submodule 状态。
9. 完成任务后运行范围相称的验证，更新 coverage、全局索引、本文件的 checkbox、
   状态和决策记录。
10. 不因上下文压缩重新开始已完成任务；以仓库状态和测试结果为准。
11. 每完成一个可独立审阅的较大步骤，形成范围清晰的 commit，并推送到当前 GitHub
    远端；推送失败时把认证或远端状态写入状态记录，修复后继续同一里程碑。

每个工作单元遵循：

```text
问题
→ 最小复现
→ 捕获 IR/日志
→ 源码调用链
→ 明确假设与不变量
→ 可逆修改
→ 构建/运行验证
→ 负例与失败路径
→ 文档与索引
```

## 15. 当前执行队列

按依赖顺序处理；可以并行的任务使用独立文件，避免多个执行者改同一位置。

- [x] Q001：创建 `manifests/baseline.json`，自动采集当前 commit、包版本和设备。
- [x] Q002：定义 topic evidence bundle、capture manifest 和 source-index schema。
- [x] Q003：建立全栈、普通 TPU、Pallas/Mosaic 和 runtime/control-plane 初版架构骨架。
- [x] Q004：建立初版 glossary，重点统一 trace/tracing、Jaxpr、HLO、Mosaic 和 LLO。
- [ ] Q005：记录当前 wheel provenance 和 version-skew probe。（wheel/native library 的
  路径、指纹、build git hash 与实际加载路径已记录；待补动态依赖和 loader resolution。）
- [x] Q006：编写 source-built jaxlib 的可审计构建设计、严格前置检查和持久状态/日志
  wrapper。（Bazel 外部依赖闭包、完整 wheel 构建与隔离验证由 P1 继续跟踪。）
- [ ] Q007：早期接入并固定 Tokamax 与自研框架，记录 sanitized provenance、JAX/Pallas
  边界、训练/推理入口和调用契约。（等待外部源码。）
- [ ] Q008：对固定入口生成 workload feature census，并回填 `manifests/coverage.json`。
- [ ] Q009：创建 Lab 模板和无界面验证入口。
- [ ] Q010：扩展 Lab 001 的 cache-key/retrace/recompile 覆盖。
- [ ] Q011：实现 Tracer/Primitive/Jaxpr Lab。
- [ ] Q012：实现 AD Lab。
- [ ] Q013：实现 `vmap`/batching Lab。
- [ ] Q014：实现 `jit/grad/vmap` 组合 Lab。
- [ ] Q015：实现隔离 persistent cache 的 HLO transformation 机制与 detection-only Lab。
- [ ] Q016：实现由 topic/coverage 生成的全局索引、query CLI、漂移检查、parser-aware
  IR 验证，以及 L4/L5 证据合同和完成性门禁。
- [ ] Q017：建立 libtpu/TPU target/runtime baseline schema。

当前最近的里程碑是：

> 拿一个固定 synthetic workload，在 source-built CPU jaxlib 上解释
> `jit(grad(vmap(f)))` 的组合机制并展示 Jaxpr/StableHLO；完成隔离缓存的 `sin → cos`
> 机制探针和 `dot` detection-only matcher，证明注册、stage、清理与回滚。真实 workload
> 的语义保持 rewrite 在 Tokamax intake 及 P6/P7 基础完成后进入 P13。

## 16. 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-09-07 | 面向 JAX 初学 kernel 开发者组织材料 | 材料应自包含，不依赖个人已有知识 |
| 2026-09-07 | TPU 为主平台，CPU 为当前公共路径验证平台 | 本地暂时没有 TPU，但 CPU 可覆盖 transformations、StableHLO/HLO 和部分 PJRT |
| 2026-09-07 | Tokamax 只作为 workload fixture | 保持分析边界在 JAX API 及以下 |
| 2026-09-07 | 普通 XLA/TPU 与 Pallas/Mosaic 分开建图 | 两者进入 libtpu 和 LLO 的方式不同 |
| 2026-09-07 | 使用证据等级和版本 manifest | 防止把静态源码推断或 CPU 模拟写成 TPU 运行事实 |
| 2026-09-07 | 第一个 pass 实验使用现有 HLO transformation extension | 当前锁定源码已经提供可在 CPU 验证的 pre/post-scheduler 接口 |
| 2026-09-08 | 把 `VERSION-SKEW` 定义为来源限定，而非证据等级 | 运行位置与源码/二进制对齐状态是两个独立维度 |
| 2026-09-08 | topic 使用 JSON 元数据和 topic-local capture | 与 schema 及仓库级语义校验器保持单一规范 |
| 2026-09-08 | 每个较大里程碑验证、提交并推送 GitHub | 让长期工作可以按稳定检查点恢复和审阅 |
| 2026-09-08 | 用 coverage inventory 度量白盒范围 | 架构骨架、真实 feature 覆盖和外部阻塞必须可区分 |
| 2026-09-08 | 单个 capture 原子化，纵向 dossier 聚合分支相关 capture | 避免要求 CPU、普通 JAX 或 source-only 证据伪造 TPU/Mosaic/LLO 产物 |
| 2026-09-08 | 单列 TPU runtime/control-plane 阶段 | compile、load、execute 与硬件 profile 需要不同证据 |
| 2026-09-08 | 用真实跨仓库问题补充背景与动机，同时服务现有工程师和新人 | 研究需要支撑定位、选择修改层、上下游协作与验证；历史 issue/PR 不提升固定基线 coverage |
| 2026-09-08 | 以组织成员实际提出的场景作为背景主线 | 分别核对提出者归属与仓库可见性；组织外部案例只作为补充对照 |

## 17. 状态更新记录

### 2026-09-07

- 长期任务已启动。
- 根仓库和已检查核心 submodule 当前干净。
- 上游源码清单包含 102 个登记 submodule；已有验证报告显示 85 个初始化、17 个 lazy。
- 已有材料：源码树说明、CPU/GPU/TPU 粗粒度阅读路径、Lab 001 `jax.jit` CPU 路径。
- 已确认主要缺口：source-built jaxlib、transformations 系列实验、pass 实验、Tokamax fixture、libtpu baseline、TPU/LLO captures。
- 下一项：Q001、Q002、Q003、Q004 可并行开展。

### 2026-09-08

- Q001–Q004 已形成初版：baseline、证据 schema、公开边界架构骨架和术语表；架构
  包含全栈、普通 TPU、Pallas/Mosaic 与 runtime/control-plane 四个观察面，不代表
  libtpu/LLO/硬件内部已经覆盖。
- `manifests/coverage.json` 首次明确记录已达到的深度、未观察项和外部阻塞；Tokamax、
  自研框架、libtpu、TPU target/device 尚未接入。
- CPU runtime 仍为 `jaxlib==0.11.1`，固定 JAX source 为 `0.11.2.dev`；运行证据带
  `VERSION-SKEW` 来源限定。
- 首轮交叉审查已修正证据强度约束、公开/私有 TPU 边界、Pallas 双层载荷和当前
  Jaxpr/Trace 术语。
- `manifests/status.json` 与 `tools/project-status.py` 提供跨会话恢复入口。
- `tools/validate-evidence.py` 与 `tools/validate-coverage.py` 已把 schema 之上的路径、
  revision、artifact、claim/source/capture 绑定、依赖图和 L0–L3 depth gate 变成可执行
  约束；隔离 selftest 覆盖正例与伪造/拼接负例。完整性模式当前按预期失败，因为真实
  workload、libtpu 和 TPU 证据尚未接入。
- Clang 18.1.3 与 Bazel 8.7.0 已就绪；固定 JAX/XLA 的 CPU jaxlib 探索性源码构建
  曾启动，仅用于 cache warming，其 live state/result 不形成持久证据。正式证据将用
  P1 固定的外部 Bazel 依赖闭包与 `tools/build-jaxlib.py` 复用 cache，产生持久日志、wheel
  SHA-256 和隔离环境验证。
- GitHub 远端为 `origin`；每个后续较大里程碑完成后提交并推送。
- 本机已恢复五个核心 submodule；`tools/sync-environment.py` 直接获取锁定 commit，
  不自动暂存，并为中断下载、脏源码、路径/哈希和缓存提供检查。
- `env/environment.lock.json` 与 Dockerfile 固定系统环境。Ubuntu Python
  `3.12.3-1ubuntu0.15` 的官方包已证明匹配原构建 Python 字节，宿主机 `0.16` 保留。
- 环境同步 17 项隔离测试、宿主机/容器 baseline 与历史 evidence、当前 CPU Lab、恢复
  状态和容器严格 preflight 已通过；容器使用独立 venv。历史 capture 不重写为当前 HEAD，
  新采集的 live-source gate 保留。Q005 与 Bazel 外部依赖闭包仍是后续顺序。

### 2026-09-08：背景与动机补充

- 根据发起者的 AI Infra/kernel 优化工作背景，补充跨层定位、选择修改层和上下游需求协作的动机。
- 增加 14 个公开历史案例及来源元数据；受限补充材料仅保存 opaque locator 和 manifest hash。
- 案例包括开放问题、已合并变更、未合并提案、下游绕过、预期行为及不计划支持的请求；没有增加固定基线执行证据或 coverage depth。
- 本轮只完善背景、动机和读者定位；P1 的恢复队列保留。恢复命令仍为 `.venv/bin/python -B tools/project-status.py --check`，源码基线恢复后再继续其打印的第一项 ready action。
- 本轮最初的状态门禁因源码 checkout 缺失/不匹配失败；源码随后可用，复查 project-status、evidence validator 及两者 selftest 均通过。另已检查本地链接、代码围栏、来源快照哈希、JSON 结构与公开材料边界；没有重跑案例中的设备实验。

### 2026-09-08：以组织成员场景重排背景

- 按提出者归属补查公开上游记录，主线改为 16 组组织贡献者提出的工程场景；首轮其他公开案例保留作对照。
- 增补内部配套 PR、remat/residual、编译内存、tracing、初始化和配置传递场景，原始成员与私有 PR 材料继续使用受限归档。
- 核实 HLO 编辑 API 请求、多 slice verifier、alias、mesh 选择、Pallas lowering、FP8 和 XProf 可见性需求；对 PR 状态和实际代码落地分别留证。
- 此次完善不改变 P1 队列与 coverage。下一恢复命令仍为 `.venv/bin/python -B tools/project-status.py --check`，再读取摘要中的 first-ready-action。
- 通过成员匹配、作者限定查询完整性、快照哈希、元数据/链接和公开范围检查；project-status、evidence validator 及对应 selftest 均通过。
