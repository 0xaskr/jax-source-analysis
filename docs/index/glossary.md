# JAX → TPU 术语表

本表规定本仓库使用术语时的唯一含义。读者假定是刚进入 JAX 软件栈的 kernel
开发者，因此同一个词在 Python、编译器和硬件语境中的不同含义会显式拆开。

## 证据记法

- `SOURCE-ONLY`：定义或接口可由固定公开源码确认，尚未代表本机执行过该源码；
- `RUN-CPU`：由与记录版本一致的 CPU probe 实际确认；
- `SIM-TPU`：由 Pallas/TPU interpreter 或静态模拟确认；
- `COMPILE-TPU`：由固定 libtpu 与固定 TPU target 的编译产物确认；
- `RUN-TPU`：由记录代际和 topology 的真实 TPU 执行确认；
- `VERSION-SKEW`：运行二进制与引用源码 commit 不一致。

完整规则见[执行计划](../../PLAN.md#52-证据标签)。本表带公开源码链接的结构性定义
均为 **证据：SOURCE-ONLY**；它们不自动升级为运行证据。LLO/libtpu 私有实现和硬件
行为会单独标记 `COMPILE-TPU` 或 `RUN-TPU`。

## Trace、tracing 与 transformation

### Transformation

接收一个函数并返回具有新语义的 callable，例如 `jit(f)`、`grad(f)` 和 `vmap(f)`。
在 JAX 中，组合性主要来自 primitive 及各解释层的 rules，不应描述成固定的一串
Python AST rewrite。API 入口见 [api.py](../../upstream/jax/jax/_src/api.py)，解释器
rules 见 [jax/_src/interpreters](../../upstream/jax/jax/_src/interpreters/)。
**证据：SOURCE-ONLY。**

### tracing

一个过程：JAX 用 `Tracer` 代表输入或中间值执行 Python 函数，使 primitive
operation 由当前解释规则处理，并收集或变换程序。tracing 不是编译，也不保证产生
设备 executable。**证据：SOURCE-ONLY。**

### `Trace`

解释层的对象/类，决定该层如何处理 primitive、call、map 等操作。固定源码中的
`Trace` 定义见 [core.py](../../upstream/jax/jax/_src/core.py)。文档写代码实体时保留
首字母大写和反引号。**证据：SOURCE-ONLY。**

### `Tracer`

tracing 期间流经 Python 程序的代理值。它携带 abstract value，并关联一个 `Trace`；
对它执行 JAX operation 会进入 primitive/trace 规则。定义见
[core.py](../../upstream/jax/jax/_src/core.py)。`Tracer` 不是 device buffer，也不是
运行时 profile trace。**证据：SOURCE-ONLY。**

### trace

避免单独使用，因为它可能指 tracing 的一次实例，也可能指 profiler event stream。
本仓库分别写“tracing 过程”“trace level/stack”和“runtime profile trace”。

### trace level / trace stack

用于说明嵌套 transformations 中解释层的相对位置和动态上下文。它帮助回答哪个
transformation 正在解释哪个 primitive，但不等同于 Python 调用栈或硬件执行栈。
**证据：SOURCE-ONLY。**

### Primitive

JAX transformation 系统识别的原子 operation。`Primitive.bind` 把调用交给当前
trace 或 eager implementation；primitive 通常还需要 abstract evaluation、AD、
batching 和 lowering 等 rules。类定义见
[core.py](../../upstream/jax/jax/_src/core.py)。primitive 不等于 HLO opcode，也不
保证一一对应某条硬件指令。**证据：SOURCE-ONLY。**

### rule

某个 primitive 在特定解释或 lowering 语境下的处理函数，例如 JVP rule、transpose
rule、batching rule 或 MLIR lowering rule。不同 registry 的 rule 输入输出契约不同，
不能因名称相似而互换。**证据：SOURCE-ONLY。**

### abstract value / aval

tracing 和 type checking 使用的值描述，通常包含 shape、dtype、weak type 等信息，
而不携带完整运行时数组内容。相关类型位于
[core.py](../../upstream/jax/jax/_src/core.py)。aval 不是抽象语法树。
**证据：SOURCE-ONLY。**

### abstract evaluation

只根据输入 aval 和 primitive 参数推导输出 aval/effects，并检查部分静态约束的过程。
它不执行设备 kernel，也不做性能预测。**证据：SOURCE-ONLY。**

### staging

把一段计算表示为可供后续 transformation/lowering 的 IR，而不是立即以普通 Python
值求出全部结果。JAX 的 partial evaluation/staging 实现在
[partial_eval.py](../../upstream/jax/jax/_src/interpreters/partial_eval.py)。staging、
lowering 和 compilation 是三个不同阶段。**证据：SOURCE-ONLY。**

### partial evaluation

根据哪些输入在 tracing 时已知，把已知计算和需要留到后续执行的计算分开的解释
过程。它参与 `jit` staging、residual 处理等机制，但不等于常量折叠这一单个 compiler
pass。源码见 [partial_eval.py](../../upstream/jax/jax/_src/interpreters/partial_eval.py)。
**证据：SOURCE-ONLY。**

## Jaxpr 及自动变换

### Jaxpr

JAX 的 typed functional IR。它包含 variables、literals、equations、in/out vars、
effects 和 debug/source 信息；每个 equation 调用一个 primitive 并携带 params。
数据结构见 [core.py](../../upstream/jax/jax/_src/core.py)。Jaxpr 不是 StableHLO，
也不是 Python AST。**证据：SOURCE-ONLY。**

### `ClosedJaxpr`

把 Jaxpr 与其 closed-over constants 绑定后的容器，便于把“程序”和捕获常量一起传递。
定义见 [core.py](../../upstream/jax/jax/_src/core.py)。**证据：SOURCE-ONLY。**

### Jaxpr equation / eqn

Jaxpr 中一次 primitive application，记录输入 atoms、输出 vars、primitive、params、
effects 和 source info。一个 eqn 不保证对应一个 StableHLO op、HLO instruction、LLO op
或硬件指令。**证据：SOURCE-ONLY。**

### effect

JAX IR 对非纯数据依赖行为的显式分类和排序信息。ordered/unordered effects 可能在
lowering 时需要 token 或附加 metadata。effect 不是 Python exception，也不应只凭
源码顺序推断执行顺序。相关定义见
[effects.py](../../upstream/jax/jax/_src/effects.py)。**证据：SOURCE-ONLY。**

### token

在 IR 中表达某些 effect/control dependency 的值。token 建立顺序约束，不承载普通
tensor payload，也不等同于线程锁。**证据：SOURCE-ONLY。**

### JVP

Jacobian-vector product；把 primal 与 tangent 向前传播的 AD 规则。实现入口见
[ad.py](../../upstream/jax/jax/_src/interpreters/ad.py)。**证据：SOURCE-ONLY。**

### VJP / transpose

VJP 是 vector-Jacobian product；JAX reverse-mode 的关键步骤会使用 linearization、
residuals 和 transpose rules。`grad` 不能简单描述成“把每个算子符号求导一次”。
实现入口见 [ad.py](../../upstream/jax/jax/_src/interpreters/ad.py)。
**证据：SOURCE-ONLY。**

### batching rule / `vmap`

batching rule 描述 primitive 在输入携带 batch dimension 时如何传播或重排该维度；
`vmap` 建立相应的 batching transformation。实现入口见
[batching.py](../../upstream/jax/jax/_src/interpreters/batching.py)。它不保证生成硬件
层面的 SIMD instruction。**证据：SOURCE-ONLY。**

### `jit`

建立 staged compilation/caching 边界的 transformation。`jax.jit(f)` 返回 wrapper；
真正 tracing、lowering 或 compilation 可以由首次调用、显式 `lower()`/`compile()`
或缓存状态决定。入口见 [api.py](../../upstream/jax/jax/_src/api.py)和
[pjit.py](../../upstream/jax/jax/_src/pjit.py)。**证据：SOURCE-ONLY。**

### retracing

再次执行 tracing 以得到新的或重新建立的 Jaxpr/trace result。它可能由 aval、static
arguments、pytree、sharding、配置或缓存生命周期变化触发。不要把 retracing、
re-lowering、recompilation 和 re-dispatch 当同义词；必须分别用 probe 证明。

## IR 层级

### MLIR

支持多 dialect、多层 lowering 和 pass infrastructure 的编译器基础设施。本项目中
StableHLO、Shardy `sdy` 和 Mosaic `tpu` 都以 MLIR dialect 形式出现，但它们语义和
pipeline 不同。锁定实现位于
[llvm-project/mlir](../../upstream/llvm-project/mlir/)。**证据：SOURCE-ONLY。**

### StableHLO

具有版本化/可移植契约的 MLIR dialect，用于在框架 lowering 与编译器之间表达 tensor
程序。operation/type 定义见
[stablehlo/dialect](../../upstream/stablehlo/stablehlo/dialect/)。StableHLO 不是
“优化后的 HLO”，也不是 TPU-specific IR。**证据：SOURCE-ONLY。**

### HLO

本文特指 XLA 的 `HloModule`、`HloComputation` 和 `HloInstruction` 图，以及相应
passes/schedule。数据结构见
[xla/hlo/ir](../../upstream/xla/xla/hlo/ir/)。它与 StableHLO 有转换关系，但不是
同一个 in-memory IR。普通 TPU backend 对 HLO 做哪些 target-specific 变换，需要
匹配 libtpu 产物确认。公开数据结构为 **证据：SOURCE-ONLY**；TPU pipeline 为
**待验证：COMPILE-TPU。**

### Shardy / `sdy`

Shardy 是分片表示与传播系统，`sdy` 是其主要 MLIR dialect namespace。它表达 mesh、
tensor sharding、constraints、data-flow edge 和部分 explicit collectives。定义见
[sdy IR](../../upstream/shardy/shardy/dialect/sdy/ir/)，passes 见
[transforms](../../upstream/shardy/shardy/dialect/sdy/transforms/)。Shardy 不是 TPU
runtime，也不等于物理 ICI topology。**证据：SOURCE-ONLY。**

### Mosaic

在本文中指 JAX/Pallas 使用的 kernel compiler/IR 家族；必须继续注明 target，例如
“Mosaic TPU”或“Mosaic GPU”。不要把它与 MosaicML 混淆，也不要把所有 JAX→TPU
lowering 都称为 Mosaic。Pallas TPU 的公开实现入口在
[jax/_src/pallas/mosaic](../../upstream/jax/jax/_src/pallas/mosaic/)。
**证据：SOURCE-ONLY。**

### Mosaic TPU MLIR / `tpu` dialect

Pallas TPU kernel lowering 产生的 MLIR module，可以包含 `arith`、`scf`、`vector`、
`memref` 和 `tpu` operations。TPU dialect 定义见
[xla/mosaic/dialect/tpu](../../upstream/xla/xla/mosaic/dialect/tpu/)。这是当前公开可读
的 kernel IR，不是 LLO。**证据：SOURCE-ONLY。**

### LLO

本文只用“TPU-specific LLO”指 HLO/Mosaic 以下、bundle/machine code 以上的目标相关
层。当前固定公开源码没有 LLO schema、parser/printer、pass pipeline 或 encoding，
也没有为缩写提供可核验的唯一展开，因此正文不展开 `LLO`。具体 opcode、pass、
数据结构和 Mosaic/HLO 映射必须从固定 libtpu 源码与 target dump 建立。
**待验证：COMPILE-TPU。**

### bundle / VLIW bundle

本文用它指 TPU compiler 为目标代际生成、将可并行设备操作编码在一起的机器级程序
包。当前公开树没有足以证明目标编码的 schema；只有拿到 target-specific packer、
dump 和 executable metadata 后才能描述字段。**待验证：COMPILE-TPU。** bundle 在
真实设备上的发射和 stall 行为仍需 **RUN-TPU**。

### machine code / hardware code

可由目标 TPU runtime 加载执行的最终设备程序。它不等于 StableHLO、HLO、Mosaic
或未经打包的 LLO。格式和兼容性按 target/libtpu 基线取证。
**待验证：COMPILE-TPU、RUN-TPU。**

## Lowering、pass 与 compilation

### lowering

把一个表示转换成更接近某个后端或更低抽象层的表示。本文必须写明两端，例如
“Jaxpr → StableHLO lowering”或“kernel Jaxpr → Mosaic TPU lowering”；单独写
“lowering”容易掩盖两条 TPU 路径。JAX 普通 MLIR lowering 见
[mlir.py](../../upstream/jax/jax/_src/interpreters/mlir.py)，Mosaic lowering 见
[mosaic/lowering.py](../../upstream/jax/jax/_src/pallas/mosaic/lowering.py)。
**证据：SOURCE-ONLY。**

### pass

读取并可能分析、验证或改写某一级 IR 的编译器单元。pass 的合法性只对它声明的 IR
和 pipeline 位置成立；StableHLO MLIR pass、Shardy pass、HLO pass、Mosaic pass 与
LLO pass 不能混称。XLA HLO pass 基础设施见
[hlo_pass_pipeline.h](../../upstream/xla/xla/hlo/pass/hlo_pass_pipeline.h)。
**证据：SOURCE-ONLY。**

### pipeline

按约束顺序组织的一组 passes/stages。知道单个 pass 的代码不等于知道目标 backend
实际启用了它；需要 pass trace、flags 和 target manifest 证明具体 pipeline。

### scheduler / schedule

决定满足 dependency 的 instruction 执行次序或表示该次序的数据结构。HLO schedule、
Mosaic schedule 和硬件 sequencer 是不同层。当前 JAX HLO transformation API 暴露
pre-scheduler/post-scheduler 两个位置，见
[xla_transform.py](../../upstream/jax/jax/_src/xla_transform.py)。
**证据：SOURCE-ONLY。** TPU 实际 scheduler 位置与结果需 **COMPILE-TPU**。

### layout

描述 logical tensor dimensions 如何映射到物理/编译器存储组织。JAX array layout、
HLO layout、Mosaic vector layout 和 TPU physical placement 是不同对象；文档必须写明
层级。真实 TPU layout 结论需 `COMPILE-TPU`，性能影响需 `RUN-TPU`。

### buffer assignment

把经过调度的 logical values/liveness 映射到可复用存储 allocation 的编译阶段或结果。
它不同于 Python garbage collection、JAX array ownership 和 PJRT buffer donation。
TPU 结果 **待验证：COMPILE-TPU。**

### memory-space assignment

在多个 memory spaces 之间选择 value placement/copies 的 compiler 决策。它不等于
Pallas programmer 在 `BlockSpec` 中提出的 memory space，也不等于 runtime 实测峰值。
TPU 编译决策需 `COMPILE-TPU`，实际 traffic/stall 需 `RUN-TPU`。

### compilation

本文指 backend 把 lowered program 与 compile options/target/device assignment 转成
executable 的过程。JAX 调用入口见
[compiler.py](../../upstream/jax/jax/_src/compiler.py)。tracing 和 lowering 可以先于
compilation 发生。**证据：SOURCE-ONLY。**

### executable / loaded executable

编译结果及其可由 client 调用的加载态对象。PJRT 对象和操作见
[pjrt_c_api.h](../../upstream/xla/xla/pjrt/c/pjrt_c_api.h)。它不是运行结果数组；创建
成功也不证明设备执行成功。**证据：SOURCE-ONLY。**

### dispatch

JAX/jaxlib 选择已编译 executable、准备 arguments 并把执行请求交给 runtime 的过程。
dispatch 可以命中缓存并重复发生，而不重新 tracing 或 compilation。

### asynchronous dispatch

host 调用返回时设备工作可能仍未完成；结果 array 的 readiness 由 event/future 或后续
同步操作管理。具体 TPU overlap 和时序必须由真实 trace 证明。
**待验证：RUN-TPU。**

### synchronization point

迫使 host 等待相关设备工作完成的操作，例如显式 readiness/blocking 或把值取回 host。
不要用 Python 函数返回时间替代 device completion 时间。

## Runtime 与 ABI

### jaxlib

JAX 的编译扩展、MLIR bindings、Python/C++ runtime bindings 等二进制组件。当前项目
使用 editable JAX source 加 `jaxlib==0.11.1` wheel，因而在 source-built wheel 前
标记 `VERSION-SKEW`。源码位于 [upstream/jax/jaxlib](../../upstream/jax/jaxlib/)。

### backend

JAX 用于 lowering/compile/execute 的平台 client，例如 CPU、TPU 或 GPU backend。
backend 不是单个 kernel library。发现与注册逻辑见
[xla_bridge.py](../../upstream/jax/jax/_src/xla_bridge.py)。**证据：SOURCE-ONLY。**

### plugin

实现 PJRT 平台接口并由 JAX/JAXlib 注册或动态加载的组件。当前 TPU 路径由
`xla_bridge.make_tpu_client` 加载 `libtpu.so` 并初始化 `tpu` plugin。
**证据：SOURCE-ONLY。** plugin 内部实现尚未固定。

### PJRT

编译器与设备 runtime 的 client/device/topology/program/executable/buffer/event 接口。
本项目的稳定观察边界是
[PJRT C API](../../upstream/xla/xla/pjrt/c/pjrt_c_api.h)，其 C++ adapter 在
[c_api_client](../../upstream/xla/xla/pjrt/c_api_client/)。PJRT 不等于 XLA HLO，也
不是 TPU ISA。**证据：SOURCE-ONLY。**

### IFRT

JAXlib 使用的更高层 runtime abstraction，用于 arrays、sharding、compile/execute 等
对象，并可由 PJRT-compatible implementation 承接。公开接口位于
[xla/python/ifrt](../../upstream/xla/xla/python/ifrt/)。本文写调用链时明确标出
Python binding、IFRT 和 PJRT C API 边界。**证据：SOURCE-ONLY。**

### ABI

独立编译组件之间的 binary interface。本文主要关注 jaxlib 与 PJRT plugin 的 C ABI
版本、struct size 和 extension chain。Python 包版本相同不自动证明 ABI/build ID
匹配。

### custom call

IR 中把执行或编译语义交给命名 target/backend config 的扩展 operation。它是一条
显式边界，不表示其内部只有一个硬件 instruction。Pallas TPU 的封装见
[tpu_custom_call.py](../../upstream/jax/jax/_src/tpu_custom_call.py)。
**证据：SOURCE-ONLY。**

### libtpu

JAX 通过 PJRT plugin 接口动态加载的 TPU compiler/runtime library；公开加载入口见
[xla_bridge.py](../../upstream/jax/jax/_src/xla_bridge.py)。当前项目尚未固定其 package
version、`libtpu.so` build ID、source commit、ABI 和 flags，因此只可确认边界，不能
描述内部实现。边界为 **证据：SOURCE-ONLY**；内部为 **待验证：COMPILE-TPU 或
RUN-TPU。**

## Pallas 与 TPU kernel

### Pallas

用 JAX/Python 表达显式 grid、block、Ref、memory 和同步语义的 kernel programming
系统。它仍使用 JAX primitive/tracing 基础设施，但有自己的 kernel Jaxpr 和后端
lowering。入口见 [jax/_src/pallas](../../upstream/jax/jax/_src/pallas/)。
**证据：SOURCE-ONLY。**

### `pallas_call`

把 Pallas kernel 嵌入外层 JAX 程序的 primitive。outer Jaxpr 的 equation 携带 inner
kernel Jaxpr、GridMapping、aliases、compiler params 等；定义见
[pallas_call.py](../../upstream/jax/jax/_src/pallas/pallas_call.py)。
**证据：SOURCE-ONLY。**

### Grid / program instance

Grid 描述 kernel program instances 的逻辑索引空间；`program_id` 标识当前 instance
在相应 axis 上的位置。它不是 TPU pod topology，也不直接等于芯片数量。
**证据：SOURCE-ONLY。**

### `BlockSpec`

描述每个 program instance 看到的 array window，包括 block shape、index map 和可选
memory space。定义见 [pallas/core.py](../../upstream/jax/jax/_src/pallas/core.py)。
它是 programmer/compiler contract，不证明最终物理 layout。
**证据：SOURCE-ONLY。**

### `Ref`

Pallas kernel 中可读写 memory reference 的抽象，load/store/swap 等 primitive 对它
操作。`Ref` 不是普通 immutable `jax.Array`，也不等同于裸设备地址。
**证据：SOURCE-ONLY。**

### HLO interpret mode

把 Pallas kernel 语义用普通 JAX/HLO operation 解释执行的调试路径，实现在
[hlo_interpreter.py](../../upstream/jax/jax/_src/pallas/hlo_interpreter.py)。它适合
数值/索引原型，不执行 Mosaic-to-LLO。**证据：SOURCE-ONLY。**

### TPU interpret mode

在 CPU 上模拟部分 TPU memory、DMA、semaphore、barrier 和 race 行为的 Pallas 调试
路径，实现在
[mosaic/interpret](../../upstream/jax/jax/_src/pallas/mosaic/interpret/)。实验结果标
`SIM-TPU`，不能外推 cycle timing 或真实硬件性能。**证据：SOURCE-ONLY。**

### HBM / CMEM / VMEM / SMEM / IMEM

本文用这些名字表示 TPU memory hierarchy 中不同职责的空间。公开 Pallas/Mosaic
源码只暴露其中部分 programmer/compiler abstractions；具体容量、banking、地址规则、
lifetime 和 traffic 必须绑定 TPU 代际与 libtpu target。编译映射需 `COMPILE-TPU`，
实际访问和 stall 需 `RUN-TPU`。

### MXU / VPU / XLU / DMA / sequencer

本文用作 TPU matrix、vector、cross-lane、data movement 和 instruction sequencing
资源类别。任何 Mosaic/LLO operation 到资源或周期的一一映射都必须由目标代际的
compiler/hardware 证据建立。**待验证：COMPILE-TPU、RUN-TPU。**

## Sharding 与 topology

### sharding

描述 global array/computation 如何分布到逻辑 device set。JAX 的实现入口在
[sharding_impls.py](../../upstream/jax/jax/_src/sharding_impls.py)。sharding 不等于
physical topology，也不单独决定 collective algorithm。**证据：SOURCE-ONLY。**

### `Mesh`

把 devices 按命名 axes 组织成逻辑多维结构，供 `PartitionSpec`/sharding rules 使用。
它是逻辑视图；相同物理设备可以被组织成不同 Mesh。**证据：SOURCE-ONLY。**

### `PartitionSpec`

描述 array 各 logical dimensions 如何使用 Mesh axes 分片。它不记录实际 device
coordinates、link bandwidth 或 runtime route。**证据：SOURCE-ONLY。**

### SPMD

Single Program Multiple Data：同一分区后程序在多个 logical devices 上处理不同
shards，并通过生成的 collectives/reshards 协作。SPMD 是编译/执行模型，不是某个
单独 JAX API。

### `pjit`

历史上独立暴露、当前仍是 JAX sharded compilation 实现中的重要名称。阅读固定
源码时以 [pjit.py](../../upstream/jax/jax/_src/pjit.py) 的实际实现为准，不根据旧版
教程推断当前 `jit`/`pjit` 边界。**证据：SOURCE-ONLY。**

### `pmap`

以 named axis 和 per-device mapped execution 为核心的 parallel map API。它的用户
语义、collective rules 和 lowering 历史与 `jit` sharding 不完全相同，不能只写成
“旧版 pjit”。固定入口见 [pmap.py](../../upstream/jax/jax/_src/pmap.py)。
**证据：SOURCE-ONLY。**

### `shard_map`

显式描述某个 Mesh 上 per-device function 与 in/out specs 的 transformation。它常用
于需要手工控制 Pallas custom call 分区和 collectives 的场景。入口见
[shard_map.py](../../upstream/jax/jax/_src/shard_map.py)。**证据：SOURCE-ONLY。**

### collective

跨 logical participants 的通信 operation，例如 all-reduce、all-gather、all-to-all
或 collective-permute。IR operation 只证明语义；实际 algorithm、route、channel、
overlap 和带宽需要 target compiler 与 profile。IR 为 `SOURCE-ONLY`，设备行为为
`RUN-TPU`。

### device assignment

compiler/executable 中 logical replica/partition 到具体 devices 的映射。它连接 logical
sharding 与 runtime devices，但不改变物理网络。

### topology

必须加限定词：

- **model graph topology**：IR nodes/edges；可由 Jaxpr、StableHLO 或 HLO pass 改写；
- **logical Mesh topology**：用户命名 axes 与 shape；可由 JAX sharding 配置改变；
- **PJRT device topology**：backend 暴露的 devices/coordinates/attributes；
- **physical TPU topology**：芯片和 ICI 链路的真实连接，compiler pass 不能改变。

因此“使用 pass 修改模型拓扑”默认指 graph rewrite；若目标是通信，必须另写 sharding、
device assignment 或 collective placement。前三层接口有 `SOURCE-ONLY` 证据；物理
行为需要 `RUN-TPU`。

## 最常见的错误等式

以下等式在本仓库一律视为错误：

```text
tracing = compilation
Jaxpr = StableHLO
StableHLO = HLO
HLO = LLO
Mosaic TPU dialect = LLO
pallas_call = one hardware instruction
Mesh = physical TPU topology
compiler success = device execution success
SIM-TPU = RUN-TPU
CPU numerical equivalence = TPU timing/layout/performance equivalence
```

遇到无法判断层级的名称时，记录其 producer、consumer、serialization format、固定
source symbol 和可观察 artifact，再把它加入本表；不要根据缩写或相似 opcode 名称
猜测等价关系。
