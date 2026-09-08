# JAX → TPU 白盒源码分析执行计划

> 状态：Active
> 启动日期：2026-09-07
> 当前阶段：P0 项目章程与可复现基线
> 当前工作重点：把长期计划、证据规则、版本闭包和最小执行单元固化到仓库中

## 1. 任务定义

本项目面向刚进入 JAX 软件栈的 kernel 开发者，建设一套版本固定、源码可跳转、实验可执行、编译产物可回放、源码修改可验证的 JAX → TPU 白盒资料库。

分析边界从 JAX API 开始，覆盖：

```text
JAX API
→ primitive / tracing / Jaxpr
→ jit / AD / batching / partial evaluation
→ JAX MLIR lowering
→ StableHLO / Shardy
→ XLA HLO passes
→ IFRT / PJRT / jaxlib
→ libtpu / TPU compiler
→ TPU-specific LLO
→ bundle / machine code
→ TPU runtime 与硬件资源
```

Tokamax 作为固定 commit 的真实 workload。Tokamax 和自研框架只负责提供纯 JAX/Pallas 入口、输入契约和数值 golden；其上层框架设计不属于分析范围。

CPU 是当前可执行验证平台，用于公共 JAX 前端、StableHLO/HLO、pass、PJRT 和分片语义实验。TPU 编译器、LLO、真实通信与性能结论分别在取得匹配的 libtpu 源码/二进制和 TPU 环境后补齐。

## 2. 完成标准

项目达到长期目标时，应能做到：

1. 对一个 Tokamax 训练或推理入口解释 `jit`、`grad`、`vmap` 为什么能够组合，并给出对应的源码规则。
2. 从 Python 调用逐层得到 Jaxpr、StableHLO、Shardy IR、HLO、Mosaic TPU MLIR、LLO 和硬件资源映射。
3. 在任一关键阶段根据符号、IR operation、pass 或故障症状反向检索源码。
4. 能区分 tracing、lowering、编译、缓存命中、dispatch 和设备执行发生的时机。
5. 能实现并验证 primitive、lowering、StableHLO/HLO pass、Pallas kernel，以及取得源码后的 libtpu/LLO pass。
6. 能从固定源码构建匹配的 jaxlib/编译器产物，替换运行环境，证明修改生效并安全回滚。
7. 能定位 retracing、重复编译、内存峰值、sharding、collective、Mosaic race 和 TPU 性能问题。
8. 形成完整架构图、源码导读、可运行实验集、可重放 captures、系列文章和贡献记录。

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

在 P1 完成前，任何由当前 `jaxlib==0.11.1` wheel 得到的 C++/XLA 运行结论都必须标记为版本错位。Python-level JAX 源码实验可以继续执行，但不能据此声称当前 `upstream/xla` 中的 C++ 路径已经被运行验证。

## 4. 两条 TPU 编译路径

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

## 5. 覆盖深度与证据标签

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

### 5.2 证据标签

| 标签 | 含义 |
|---|---|
| `RUN-CPU` | 在当前固定 CPU 环境实际执行 |
| `SIM-TPU` | 使用 TPU/Pallas interpreter 或静态模拟执行 |
| `COMPILE-TPU` | 针对固定 TPU target 完成编译，未在设备运行 |
| `RUN-TPU` | 在记录了代际和 topology 的真实 TPU 上运行 |
| `SOURCE-ONLY` | 仅由固定版本源码证明，尚无执行证据 |
| `VERSION-SKEW` | 运行二进制与所引用源码 commit 不一致 |

结论必须紧邻证据标签；CPU 数值等价不得表述成 TPU layout、时序、通信或性能等价。

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
    16-llo-and-tpu-hardware/
    17-debugging-and-performance/
    18-extension-build-contribution/
workloads/
  synthetic/
  tokamax/
labs/
captures/
  <case>/<platform>/<topology>/<build-id>/
patches/
manifests/
  baseline.json
  build-fingerprints/
articles/
tools/
```

目录按需创建。空目录和没有证据的占位文章不进入仓库。

## 7. 每个主题的固定证据包

```text
README.md             原理、边界、关键问题和已验证结论
source-index.yaml     symbol、commit、path、line、caller/callee、职责
callgraph.mmd         Python/C++/ABI/IR 边界与控制流
invariants.md         shape、dtype、effect、layout、sharding 等契约
probes/               最小、无隐藏状态的可运行探针
artifacts/            Jaxpr/MLIR/HLO/LLO、日志和 manifest
tests/                数值、结构、缓存、pass 或 ABI 断言
patches/              可检查、应用和反向应用的源码改动
build-and-replay.md   构建、运行、dump、回放和清理命令
failures.md           症状 → 观测点 → 源码 → 原因 → 验证方法
```

不要求每个主题第一天就具备全部文件。进入 L3 时必须具备 probe、artifact 和 test；进入 L4 时必须具备 patch、build/replay 和回滚验证。

## 8. 长期阶段与验收门槛

### P0：项目章程、索引规范与版本清单

目标：让长期任务可以跨会话恢复，且每项结论有明确边界。

- [x] 确定读者：刚进入 JAX 的 kernel 开发者。
- [x] 确定主平台：TPU；CPU 为当前实验平台。
- [x] 确定边界：JAX API 到 TPU-specific LLO/硬件映射。
- [x] 区分普通 XLA/TPU 与 Pallas/Mosaic 两条路径。
- [x] 建立本长期计划。
- [ ] 建立机器可读的 `manifests/baseline.json`。
- [ ] 建立 evidence、source index、capture manifest schema。
- [ ] 固化完整架构图和术语表。
- [ ] 接入并固定 Tokamax workload commit。

验收：新会话只读取本文件、baseline manifest 和状态文件，就能判断当前版本、已完成内容、证据强度和下一项任务。

### P1：源码与运行二进制一致的构建闭环

目标：消除 JAX `0.11.2.dev` source 与 `jaxlib==0.11.1` wheel 的证据错位。

- [ ] 记录当前 wheel 的版本、路径、git hash 和动态依赖。
- [ ] 记录 JAX/XLA 推荐的 Bazel、Clang、Python 和构建配置。
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
- [ ] Trace、Tracer、main trace、sublevel 和 trace stack。
- [ ] Jaxpr、ClosedJaxpr、constvar/invar/outvar/equation/effect。
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

目标：建立从观察、匹配到安全改写的 pass 工程闭环。

- [ ] MLIR rewrite、canonicalization、pass manager 和 pass instrumentation。
- [ ] StableHLO pass 注册、单 pass runner、FileCheck 和 reducer。
- [ ] HloModule/HloComputation/HloInstruction 数据模型。
- [ ] HLO pass pipeline、ordering、pre/post scheduler 边界。
- [ ] HLO dump、before/after diff、pass disable 和 bisect。
- [ ] 使用 `jax.extend.xla.register_hlo_module_transformation` 完成 `sin → cos` 机制实验。
- [ ] 实现两条兼容 `dot` 合并为宽 `dot + slice` 的结构改写。
- [ ] 验证 shape、dtype、layout、schedule、sharding、alias/donation、effect 和数值契约。
- [ ] 记录 transformation 不进入 persistent cache key 的当前版本行为。

验收：在 source-built jaxlib 上完成结构性计算图改写，保存 before/after HLO，并通过数值、结构、缓存和回滚测试。

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

验收：固定输入、target 和 flags 后，可重复生成可归因的 TPU compiler artifacts；无 TPU 时标记为 `COMPILE-TPU`。

### P11：TPU-specific LLO 与硬件映射

目标：把机器相关 IR 与 TPU 资源建立可检索关系。

- [ ] 固定目标代际的 LLO schema、opcode、operand 和验证规则。
- [ ] LLO pass pipeline、schedule 和 bundle/VLIW 生成。
- [ ] HBM/CMEM/VMEM/SMEM/IMEM 地址空间和生命周期。
- [ ] MXU/VPU/XLU、DMA、sequencer、ICI 等资源映射。
- [ ] HLO/Mosaic source location → LLO → bundle 的 lineage。
- [ ] 静态 cost report 与真实 XProf/counter 的差异。
- [ ] 修改一个 LLO pass，验证预期指令或调度变化。

验收：任取一个关键 LLO 指令组，可以向上追到 Tokamax/JAX 源码，向下解释使用的硬件单元，并在真实 TPU 上验证行为。

### P12：Tokamax 纵向案例

目标：用真实 workload 连接此前独立建立的各层材料。

- [ ] 固定 Tokamax URL、commit、依赖和接口。
- [ ] 抽取不依赖上层框架的纯 JAX/Pallas workload。
- [ ] 覆盖 elementwise/fusion、matmul/MLP、reduction/softmax、Pallas、sharded collective。
- [ ] 为训练入口保存 `jit(grad(...))` 全链产物。
- [ ] 为推理入口保存 compile、dispatch 和 runtime 全链产物。
- [ ] 先写 detection-only pass，记录模式命中和拒绝原因。
- [ ] 完成语义保持的计算图改写和 TPU kernel 替换。
- [ ] 比较数值、gradient、compile time、HBM、cycles 和硬件利用率。

验收：每个选定 workload 都形成一个可独立复现的 white-box dossier。

### P13：故障、性能和通信诊断

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

### P14：文章、贡献与版本迁移

目标：使成果可传播并能跟随上游演进。

- [ ] 每个纵向案例形成中文源码分析文章。
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

1. 用当前源码已有的 `jax.extend.xla.register_hlo_module_transformation` 完成 `sin → cos`，验证 callback、stage、注册、清理和缓存行为。
2. 实现语义保持的两条 `dot` → 宽 `dot + slice`，建立完整 pass contract。
3. 对 Tokamax 代表子图实现 detection-only pass，输出命中、拒绝和源码位置。
4. 在 StableHLO/HLO 层实现 topology rewrite，并验证 forward、gradient、sharding、alias/donation 和数值稳定性。
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

每个 capture 至少保存：

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
10. 从 Mosaic TPU MLIR 到 LLO 与硬件；
11. 用 pass 改写一个 Tokamax 计算图；
12. 从源码构建、替换和调试 JAX/libtpu 软件栈。

## 13. 当前阻塞项与处理方式

| 缺口 | 当前处理方式 | 解除条件 |
|---|---|---|
| JAX source 与 jaxlib wheel 错位 | 标记 `VERSION-SKEW`，P1 自建 wheel | source-built jaxlib 验证通过 |
| Tokamax 未在工作区 | 先用 synthetic workloads，保留 fixture schema | 固定 URL/commit/entry |
| libtpu 未锁定 | 公开边界标 `SOURCE-ONLY`，先完成接口图 | 提供匹配 source/binary/build ID |
| 本地没有 TPU | 完成 CPU、simulation、compile-only 工作 | 获得真实 TPU 环境 |
| TPU/LLO 随代际变化 | 所有 capture 绑定 target manifest | 固定首个 TPU 代际和 topology |

这些缺口不会阻止 P0–P9 的大部分工作，但禁止提前宣称 TPU runtime、LLO 或性能已经验证。

## 14. 执行协议

后续代理或会话按以下顺序恢复任务：

1. 完整读取本文件。
2. 检查根仓库和相关 submodule 的 `git status`，保护已有用户改动。
3. 读取 `manifests/baseline.json` 和最近的状态记录。
4. 从“当前执行队列”选择第一个未完成且未阻塞的任务。
5. 先查固定版本源码，再形成最小 probe；不要凭最新版文档替代锁定源码。
6. 保存真实产物和 manifest，再形成源码解释。
7. 只有跨层链路复杂时建立 CodeTour/交互 notebook；自动验证使用命令行 probe/test。
8. 修改 `upstream/*` 时优先维护外层可逆 patch；实验结束检查 submodule 状态。
9. 完成任务后运行范围相称的验证，更新本文件的 checkbox、状态和决策记录。
10. 不因上下文压缩重新开始已完成任务；以仓库状态和测试结果为准。

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

- [ ] Q001：创建 `manifests/baseline.json`，自动采集当前 commit、包版本和设备。
- [ ] Q002：定义 topic evidence bundle、capture manifest 和 source-index schema。
- [ ] Q003：建立全栈、普通 TPU、Pallas/Mosaic 三张架构图。
- [ ] Q004：建立初版 glossary，重点统一 trace/tracing、Jaxpr、HLO、Mosaic 和 LLO。
- [ ] Q005：记录当前 wheel provenance 和 version-skew probe。
- [ ] Q006：编写 source-built jaxlib 的可复现构建设计与前置检查。
- [ ] Q007：创建 Lab 模板和无界面验证入口。
- [ ] Q008：扩展 Lab 001 的 cache-key/retrace/recompile 覆盖。
- [ ] Q009：实现 Tracer/Primitive/Jaxpr Lab。
- [ ] Q010：实现 AD Lab。
- [ ] Q011：实现 `vmap`/batching Lab。
- [ ] Q012：实现 `jit/grad/vmap` 组合 Lab。
- [ ] Q013：实现 HLO transformation Lab。
- [ ] Q014：接入并固定 Tokamax workload。
- [ ] Q015：建立 libtpu/TPU target baseline schema。

当前最近的里程碑是：

> 拿一个固定 workload，解释 `jit(grad(vmap(f)))` 的组合机制，展示 Jaxpr 和 StableHLO；通过真实 HLO transformation 改写其计算图，并在 source-built CPU jaxlib 上证明修改生效、缓存行为正确且可以回滚。

## 16. 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-09-07 | 面向 JAX 初学 kernel 开发者组织材料 | 材料应自包含，不依赖个人已有知识 |
| 2026-09-07 | TPU 为主平台，CPU 为当前公共路径验证平台 | 本地暂时没有 TPU，但 CPU 可覆盖 transformations、StableHLO/HLO 和部分 PJRT |
| 2026-09-07 | Tokamax 只作为 workload fixture | 保持分析边界在 JAX API 及以下 |
| 2026-09-07 | 普通 XLA/TPU 与 Pallas/Mosaic 分开建图 | 两者进入 libtpu 和 LLO 的方式不同 |
| 2026-09-07 | 使用证据等级和版本 manifest | 防止把静态源码推断或 CPU 模拟写成 TPU 运行事实 |
| 2026-09-07 | 第一个 pass 实验使用现有 HLO transformation extension | 当前锁定源码已经提供可在 CPU 验证的 pre/post-scheduler 接口 |

## 17. 状态更新记录

### 2026-09-07

- 长期任务已启动。
- 根仓库和已检查核心 submodule 当前干净。
- 上游源码清单包含 102 个登记 submodule；已有验证报告显示 85 个初始化、17 个 lazy。
- 已有材料：源码树说明、CPU/GPU/TPU 粗粒度阅读路径、Lab 001 `jax.jit` CPU 路径。
- 已确认主要缺口：source-built jaxlib、transformations 系列实验、pass 实验、Tokamax fixture、libtpu baseline、TPU/LLO captures。
- 下一项：Q001、Q002、Q003、Q004 可并行开展。
