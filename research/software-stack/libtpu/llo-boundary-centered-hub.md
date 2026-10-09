# TPU TensorCore 后端（libtpu）：原生 LLO

[查看图](llo-boundary-centered-hub.svg) · [返回总览](../overview/overview-software-stack-components-layered.md)

`libtpu.so` 向 JAX 提供 TPU 的 PJRT 实现，连接编译器、程序加载器和运行时。外层程序在 HLO 层组织计算与通信；TensorCore（TC）后端进一步生成低层程序、指令束和机器代码；SparseCore（SC）有独立的编译分支。执行时，运行时把真实 Buffer 绑定到已经编译好的程序。

本文以 TC 后端的**原生 LLO**为核心抽象，围绕它的定义、产生、变换和消费展开。HLO 作为上游输入说明，SC 的 MLO/LLVM 路线作为独立分支说明。

本文有两类证据：**公开源码**指工作区锁定的 JAX/XLA；**内部答复**指 [libtpu-agent #76][internal-answer] 的源码转述，并以[后续更正][internal-corrections]修订细节。bot 没有提供可绑定到 wheel 的内部 revision，也没有公开私有类名和定义文件。因此，下文的“原生 LLO 模块”“桥接构建器”等是职责名称，不是可以直接检索的 C++ 符号。内部路线用于解释结构，不能冒充本地已核验的实现或编译实测。

## 1. 定义：这一层有哪些组件和核心抽象

### 1.1 组件的分工

| 组件 | 负责什么 | 交付什么 |
| --- | --- | --- |
| PJRT provider / Client | 接收编译、加载、执行和存储请求，适配公开 C API。 | `PJRT_Executable`、`PJRT_LoadedExecutable`、Buffer 与事件等对象。 |
| TPU 编译器 | 优化 HLO，规划布局、调度和存储；分别生成 TC、SC 及需要的 host 子程序。 | 目标程序与编译元数据。 |
| 程序封装与加载器 | 组织代码、常量、存储和调用信息；按目标设备加载程序并处理地址绑定。 | 与设备关联的已加载程序。 |
| 异步运行时 | 管理本次执行的 Buffer、依赖、提交和完成状态，调用硬件接口。 | 输出存储及完成或错误状态。 |

公开入口可定位到 [`make_tpu_client`][plugin]、[`PJRT_Client_Compile`][compile] 和 [`PJRT_LoadedExecutable_Execute`][execute]；libtpu 内部的上述分工来自[内部答复 A.1][internal-answer]。

### 1.2 原生 LLO 及相关对象的生命周期

| 抽象 | 含义与结构 | 谁产生 | 谁变换 | 谁消费、得到什么 |
| --- | --- | --- | --- | --- |
| `PJRT_Program` | 程序字节及格式的提交容器。 | `SerializeProgram` 序列化 MLIR 或 HLO proto，`InitializeArgsAndCompile` 填入容器。 | 这里进行版本化序列化与包装。 | 插件的 `PJRT_Client_Compile` 消费程序与编译选项。 |
| `HloModule` | 由 `HloComputation`、`HloInstruction` 组成的张量计算图，携带 layout、sharding、backend_config 和 execution_thread 等信息。 | MLIR 导入或 HLO proto 构造。 | HLO passes 改写计算、布局、通信与分片。 | 调度、存储规划及 TC/SC/host 后端读取图。 |
| `HloSchedule` | 每个非 fusion computation 内 HLO 指令的有序序列。 | 基础 HLO 调度器。 | LHS 重排；插入、删除或重算操作时维护序列。 | 存活期分析和后端 lowering 消费顺序。 |
| `BufferAssignment` | 编译期存储方案，将逻辑值映射到 allocation 及其 offset/size。 | `BufferAssigner::Run` 依据存活期、别名、对齐与内存空间约束构造。 | 随编译规划更新；不是运行时内存对象。 | 后端生成地址访问；运行时据元数据准备实际存储。 |
| Mosaic TPU MLIR | Pallas kernel 的专用程序，包含 `tpu`、`vector`、`memref`、`scf`、`arith` 等操作。 | Pallas 从 kernel Jaxpr 和 GridMapping lowering。 | Mosaic 做布局、向量切分和目标 lowering。 | TC 路径生成 MLIR `llo`；SC 路径转入 MLO。 |
| MLIR `llo` 方言 | Mosaic TC 后半段的低层 MLIR 表示，仍使用 MLIR 的操作、值和控制流。 | `lower-to-llo`。 | `eliminate-llo-extensions`、`finalize-llo` 等处理。 | 桥接构建器将它转换为原生 LLO。 |
| 原生 LLO | TC 后端的程序对象：区域、循环、指令、值、局部存储及同步依赖。 | 普通 HLO emitter，或 Mosaic 的 MLIR→原生桥接构建器。 | LLO 优化、指令调度、寄存器与局部存储分配。 | 指令束打包与编码器生成 TC 目标程序。 |
| SC 的 MLO / `sparse_core` 表示 | 面向 SC 子程序的 MLIR 方言族。 | SC HLO lowering，或 Mosaic SC 的 `lower-to-mlo`。 | MLO 优化及到 LLVM IR 的 lowering。 | SC 目标后端生成对应的程序镜像。 |
| 可执行对象 | 目标代码及其常量、存储、调用和设备信息。 | 编译器与封装器构造；加载器产生已加载状态。 | 加载、重定位及运行时状态管理。 | Execute 消费程序和 Buffer，产生输出及完成状态。 |

前四项的公开定义与调用分别见 [C API 程序序列化][capi-compile]、[`HloModule`][hlo-module]、[`HloSchedule`][hlo-schedule]、[`BufferAssignment`][buffer-assignment]。公开 [Pallas 设计说明][mosaic-design]确认 Mosaic 会生成 LLO；两种 LLO、SC MLO 路线和内部消费者的细分来自[内部答复 B.4–6][internal-answer]。

### 1.3 “LLO”具体表示什么

在这份内部答复中，LLO 指 Low-Level Optimizer 使用的低层程序，但需要区分两种表示。

**MLIR `llo`** 是 Mosaic TC lowering 的中间方言。逻辑向量和内存访问经过布局处理后，变成更接近目标操作的 MLIR 指令；部分控制流仍由 `scf` 等方言表达。它仍是 `mlir::ModuleOp` 中的程序。

**原生 LLO** 是 TC 后端使用的 C++ 程序模型。按内部答复，它以层级区域组织指令、循环、条件区域和子区域；在寄存器分配前，值之间通过定义和使用关系连接。程序还持有常量、局部 VMEM/SMEM 分配和同步资源；后续变换逐步附加物理寄存器及指令束位置等信息。普通 HLO 与 Mosaic TC 最终汇合到这一表示。

bot 在[后续更正的附录][internal-corrections]中确认，`llo`、`mlo`、`sparse_core` 是方言的实际名称，`lower-to-llo`、`lower-to-mlo` 是 pass 注册名称；被替换为职责描述的是私有 C++ 类、函数和 Proto 消息名。这些内部表示没有跨 libtpu 版本的稳定外部 schema 承诺。

合法性需要同时满足程序依赖和目标硬件约束：定义必须先于相应使用，控制流汇合要保持值的语义；DMA 写入的存储必须在使用前就绪，缓冲区复用不能覆盖仍在使用的数据；寄存器、队列、指令槽位及地址编码必须符合目标要求。具体 verifier、类型定义和稳定的外部 schema 没有公开，不能据这份描述自行编造解析接口。[内部答复 B.4、C.7][internal-answer]

[`XLA_TpuProgram`][opaque] 是另一种公开的不透明程序句柄。句柄声明不定义 LLO，也不表示当前 JAX 的 PJRT 调用必须先经过它。Mosaic TPU MLIR、MLIR `llo`、原生 LLO、机器码和 Executable 序列化字节分别属于不同的对象层次。

### 1.4 存储空间与存储对象

HBM、VMEM、SMEM 指存储空间；`BufferAssignment` 是编译方案；`PJRT_Buffer` 是运行时存储对象。三者描述的层次不同。固定 JAX 的 [`MemorySpace.color`][memory-colors]给出提交到 HLO 布局约束中的颜色编号：

| 空间 | 用途 | 本版本 HLO 颜色 |
| --- | --- | --- |
| HBM | 设备主存，容纳输入、输出及较大的中间数据。 | `0` |
| VMEM | 向量计算使用的片上存储，例如计算块和流水线窗口。 | `1` |
| SEMAPHORE_MEM | 表达搬运及计算之间的同步资源。 | `2` |
| SMEM | 标量数据、索引和控制相关的片上存储。 | `4` |
| HOST | 主机存储空间。 | `5` |

这些是指定接口中的编号，不是所有内部枚举的通用序号。bot 已[撤回首轮答复中的 `SMEM=2`][internal-corrections]。在 SC/TC 场景中还要区分存储所属核心和访问方式，不能仅凭“出现 TC VMEM”就判断 SC lowering 非法。

## 2. 产生：两条 TC 路线在哪里汇合

### 2.1 从 JAX 到插件的公开调用链

<a id="产生公开的-client-与编译请求"></a>

[`make_tpu_client`][plugin] 加载并初始化 TPU PJRT 插件。jaxlib/IFRT/PJRT 提交的外层程序经 [`SerializeProgram` 和 `InitializeArgsAndCompile`][capi-compile] 进入插件：

```text
外层 MLIR Module / XlaComputation + CompileOptions
  → SerializeProgram：按插件版本序列化 MLIR，或序列化 HLO proto
  → InitializeArgsAndCompile：PJRT_Program + 序列化选项
  → 插件 PJRT_Client_Compile
  → PJRT_LoadedExecutable
  → PjRtCApiLoadedExecutable → IFRT / jaxlib
```

这里没有传入数组实际数据。编译器接收程序、静态约束和选项；运行时 Buffer 留到 Execute 再传入。公开 [`MlirToXlaComputation`][mlir-hlo] 与 `HloModule::CreateFromProto` 提供 MLIR/HLO 对象构造能力；内部答复称 TPU Client 使用这套 HLO 类层次，并通过后端专用配置表达 TPU 语义，没有另外定义一套同名的 HLO IR。

### 2.2 普通 HLO 与 Mosaic TC

下面是[内部答复 B.5][internal-answer]描述的对象路线，省略可选优化和重试，箭头不代表固定的完整 pass 顺序：

```text
普通 JAX 张量程序
  → 外层 Module → HLO → 优化 / 调度 / 存储规划
  → HLO、融合与流水线 emitter ───────────────────────┐
                                                   ↓
                                              原生 LLO
                                                   ↑
Pallas kernel → Mosaic TPU Module                  │
  → 序列化到外层 tpu_custom_call                    │
  → 外层 HLO 编译遇到该调用                         │
  → Mosaic 布局与向量 lowering → MLIR llo → 桥接构建器 ┘
```

**普通 HLO 路线不要求先转换成 Mosaic。** 后端根据 HLO、窗口配置和存储方案直接构造原生 LLO 的计算、循环和搬运操作。

**Mosaic TC 路线先编译内嵌程序。** 公开的 [`pallas_call_tpu_lowering_rule`][pallas-lowering] 构造 Mosaic Module；[`_lower_mosaic_module_to_asm`][mosaic-serialize] 运行版本化序列化，输出 MLIR bytecode；[`lower_module_to_custom_call`][mosaic] 将它接入外层程序。虽然函数名带 `asm`，这里的输出不是硬件汇编。

内部答复称，后端将内嵌 Module 降到 MLIR `llo`，再由桥接构建器把指令和控制流写入外层原生 LLO 的当前区域。BlockSpec 路径还可由流水线 emitter 构造窗口循环和 DMA；显式管理搬运的 kernel 使用相应的调用约定。默认情况下，TC kernel 随外层设备程序一起编译、加载；函数去重可以提取设备子函数，最后再链接到程序中。因此，外层有 custom call 并不意味着主机运行时必然单独启动一个 kernel。

### 2.3 SparseCore 是另一条后端路线

按[内部答复 B.6][internal-answer]，SC HLO 子图和 Mosaic SC kernel 都进入 MLO / `sparse_core` 方言族，再降低到 LLVM IR，由 SC 目标后端生成其程序。Mosaic SC 的入口是 `lower-to-mlo`；它不经过 TC 的原生 LLO 打包流水线。TC 与 SC 的程序及调用信息最终共同组成可执行对象，运行时通过异步启动、共享存储和同步资源协调。

公开 JAX 已能确认核类型分流和接口限制：[`_get_device_type`][core-type] 扫描 `tpu.core_type`，单个 Mosaic Module 同时含 TC 与 SC 函数时直接拒绝。这个限制不等于“SC 不能向 TC 的存储传数据”：固定源码的 [SC→TC VMEM 测试][sc-tc-test]使用两个 kernel，通过远程 DMA 和信号量交接数据。[更正答复第 8 项][internal-corrections]确认，SC lowering 支持把标注了 TC 所属核心的 VMEM 当作远程 DMA 地址；这与在 SC 内执行 TC 专属本地计算或访存是不同的情况。

## 3. 变换：HLO 与低层程序分别决定什么

### 3.1 HLO 层改写计算与全局规划

TPU 内部仍使用公开 HLO 对象；目标相关 passes、配置及其组合属于 libtpu。按内部答复，主要职责如下：

| 处理 | 输入及规则 | 结果供谁使用 |
| --- | --- | --- |
| 规范化与 SPMD | 化简计算；依据分片和拓扑把全局程序转为局部计算及通信。 | 布局、融合和后端优化。 |
| 布局与融合 | 确定物理布局，在语义、存储及目标约束下合并计算。 | emitter 的计算块及窗口规划。 |
| 异步改写 | 把适用通信或卸载任务拆为启动与等待，保留数据和控制依赖。 | 调度器寻找重叠机会。 |
| HLO 调度 | 基础调度提供合法顺序，LHS 结合依赖、资源、代价和内存压力重排。 | 存活期分析与 lowering。 |
| 重物化与存储规划 | 在时间、存活期、别名和容量之间取舍；按需插入重算、预取或搬运。 | `BufferAssignment` 及后续地址生成。 |

这些处理可能重复、按选项跳过，存储压力也可能触发后续重试，不能画成跨版本固定的一遍式列表。公开的 [`LatencyHidingScheduler::RunImpl`][lhs]要求已有基础调度，调用 `ScheduleComputation` 后以 `set_sequence` 写回。`memory_limit` 参与启发式选择和有限次重试，不是“任何候选超限就必定被拒绝”的证明。

`HloSchedule` 给出 HLO 层顺序，**不等于主机逐条解释 HLO，也不是逐周期机器指令表**。按内部答复，TC emitter 用它组织设备控制流；HLO 的循环和分支降低成设备程序中的控制流。后续低层调度仍会在合法范围内重新安排具体操作。

### 3.2 原生 LLO 把计算落实到目标资源

| 处理者（职责名称） | 消费什么 | 改变什么、产生什么 |
| --- | --- | --- |
| LLO 优化与目标 lowering | 含虚拟值、控制流及局部存储的原生 LLO。 | 改写循环和地址，提升可放入寄存器的存储，拆解伪操作；仍是 LLO。 |
| 指令调度与 DMA 处理 | 数据、别名、同步和硬件资源依赖。 | 在合法范围内提前搬运、推迟等待，安排计算及队列使用；仍是 LLO。 |
| 寄存器与局部存储分配 | 虚拟值及其存活期、窗口、scratch 和同步资源。 | 绑定物理寄存器与局部存储位置，必要时加入 spill/fill；修改 LLO 及其规划。 |
| 指令束打包器 | 已完成必要物理分配的低层操作。 | 按目标延迟、槽位和端口约束组织可共同发射的指令，产生指令束序列。 |
| 编码与链接器 | 指令束、常量、符号和重定位信息。 | 编码目标程序，并把需要的设备子函数组织进程序镜像。 |

这张表描述职责和对象变化，不承诺某个构建的所有处理严格按行执行一次。内部答复还描述了局部存储不足时收缩融合窗口并重新 lowering 的反馈过程。具体策略依赖目标与构建，不能把某一代芯片的周期数、寄存器数或指令束宽度写成 LLO 的通用定义。[内部答复 C.7][internal-answer]

外层 `BufferAssignment` 与算子内局部存储也有区别：前者描述跨操作的静态存储关系；后者包括流水线窗口、局部 scratch、spill 与同步资源，须在低层结合更细的存活期处理。

### 3.3 SC/TC 没有数据依赖，为什么仍不保证并行

没有直接数据依赖只提供调度机会。控制依赖、effects、存储别名、异步资源配额、代价估计及内存压力都可能限制重叠；即便 HLO 把 TC 放在 SC 的 start/done 之间，设备资源竞争也会影响实际时间线。`cost_estimate` 可以提供调度线索，但不应据此承诺必然 overlap，或断言未提供它就绝不可能重叠。bot 在[更正答复第 2 项][internal-corrections]中确认了启发式偏好与硬性合法性约束的区别。

[公开 LHS 实现][lhs]可检查调度算法；[SC→TC 测试][sc-tc-test]展示数据交接写法。测试代码的存在、源码允许某种顺序、编译后的实际顺序和设备上的性能重叠，是不同层次的证据；本文没有执行该测试或采集设备时间线。

## 4. 消费：怎样成为可执行程序并运行

### 4.1 从低层程序到 Executable

按[内部答复 C.8][internal-answer]，目标程序封装代码、常量、未初始化存储需求以及重定位等信息；可执行对象进一步组织 TC/SC 程序、HLO 与存储元数据、编译选项和需要的 host 通信/回调信息。加载器把这些内容与具体设备关联，形成已加载状态。

公开 API 对应两种能力：`PJRT_Executable` 提供编译产物与序列化接口，`PJRT_LoadedExecutable` 提供设备执行接口。`PJRT_Client_Compile` 可以把编译和加载合并在一次调用中返回后者；内部有多个对象不意味着上层一定要手动逐步加载。

[`PJRT_Executable_Serialize`][serialize]返回平台特定的字节，格式不保证跨时间稳定。它是可执行对象的封装，不能把整体称为 LLO 或裸机器码。可选 LLO 调试信息也应与运行时程序、完整可执行序列化格式分别讨论。

### 4.2 一次 Execute 中主机与设备各做什么

<a id="消费执行与序列化分别消费什么"></a>
<a id="执行接口0407"></a>

公开的 [`PjRtCApiLoadedExecutable::Execute`][capi-execute]把按设备组织的实参转换为 `PJRT_Buffer` 列表，准备输出句柄容器并调用插件。返回后包装输出 Buffer，并在请求执行状态时，用 `ConvertCEventToCppFuture` 转换逐设备完成事件。

插件内部的运行过程按[答复 C.9][internal-answer]可理解为：

1. **主机准备**：检查参数及其就绪依赖，根据存储规划和 donation/alias 约定准备输出与临时空间，创建结果对象和完成状态。
2. **绑定并提交**：依赖满足后，把本次实际存储的基地址接入设备程序的参数约定，再提交已加载程序。答复描述了 SMEM 参数表等实现机制；其具体 ABI 需与构建对应。
3. **设备执行**：设备程序驱动循环、地址计算、DMA、向量/矩阵计算和同步。编译器已经生成控制流，主机通常不逐个解释 HLO；程序若包含 host 回调或卸载，仍需相应主机协作。
4. **传播结果状态**：运行时把完成或错误写入事件，后续计算或主机等待结果时观察这些状态。IFRT 再将输出 Buffer 分片组织为逻辑数组。

TC 的硬件原理可结合[公开 Pallas 说明][tpu-units]理解：标量控制配合宽向量计算，HBM 数据由 DMA 搬到较低层存储，MXU 等单元可相对主指令流异步工作。编译器必须安排好“发起、可用、消费、复用”的关系。

| 观察点 | 能说明什么 |
| --- | --- |
| Execute 返回、拿到输出句柄 | 已取得本次结果对象；设备可能尚未开始，也可能已经完成，不能仅由函数返回判断。 |
| 某次 DMA 的同步条件满足 | 相应搬运或交接完成；不代表整个程序结束。 |
| `device_complete_events[i]` 就绪 | 公开契约中，对应设备的本次执行完成；执行错误也须按接口检查。 |
| 上层 Array/Future 就绪 | 按上层对象覆盖的设备与依赖范围汇总状态，不应与单个 DMA 信号量混同。 |

按[更正答复第 4 项][internal-corrections]，内部单设备执行事件可以聚合该逻辑设备的 TC、SC 及相关传输，但不是跨所有 `PjRtDevice` 或所有 host 的全局屏障。这与公开 C API 的逐设备事件契约一致。

### 4.3 用矩阵乘加一贯穿两条路线

以下是**结构示意，不是编译 dump**。普通函数可以写成 `jit(lambda a, b: a @ b + 1)`；Pallas 则在 kernel 内显式读取输入块、计算并写回输出块。要做数值等价比较，还必须统一 dot 的输出类型、精度设置和舍入位置，例如不能把“先转 bf16 再加一”与“f32 加一后才转 bf16”当成相同语义。

| 层次 | 普通 JAX 路线 | Pallas TC 路线 |
| --- | --- | --- |
| 前端程序 | 外层 Module 表达 dot 与 add。 | 外层 custom call 携带内层 Mosaic Module；内核含读取、矩阵乘、加法与写回。 |
| HLO | 依据目标与选项优化 dot/add；是否融合要看实际编译。 | 外层 HLO 组织调用及其依赖，配置保留内嵌 kernel。 |
| 低层程序 | HLO emitter 直接构造原生 LLO。 | Mosaic 先到 MLIR `llo`，桥接后进入原生 LLO。 |
| 目标程序 | 根据具体规划形成搬运、计算、同步和控制流。 | 也形成设备程序，但不保证与普通路线有相同指令、窗口或性能。 |
| 执行 | 绑定输入输出 Buffer，提交程序，观察完成状态。 | 使用同一层 PJRT 执行契约。 |

### 4.4 LLO 调试信息由谁产生、谁消费

编译器还可以把低层程序转换为诊断数据。公开 XLA profiler schema 定义了 [`llo_proto`][llo-metadata] 元数据键；它确认了这一采集字段，但没有公开内部 Proto 的完整结构。

按[更正答复第 5 项][internal-corrections]，启用相应调试功能时，内部 LLO 序列化器在调度、打包之后生成低层程序的调试 Proto，保存区域、指令及指令束关联等信息。Dumper 可以写出它，profiler 也可以按模块身份缓存，再随采集结果放入 XSpace，交给 XProf 分析。这里的“序列化”消费的是 LLO 调试表示，和 `PJRT_Executable_Serialize` 的输入、目的及格式不同。

因此，原生 C++ LLO 对象结束生命周期，不等于所有 LLO 调试信息都随之消失。具体 flag、导出能力和解析器必须与构建对应，本文没有验证任何 libtpu wheel 的调试开关或产物格式。

## 5. 证据范围与维护

公开源码版本为 JAX `361c43e072cce92b7d3e9bdaf4dd16db26c49043`、XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69`，以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。内部结构来自 [#76 首轮答复][internal-answer]；经[逐项追问][internal-followup]后，bot 在[更正答复][internal-corrections]中修正了内存空间编号、LHS 硬约束、BF16 示例、异步完成范围、调试信息、公开 SPMD 类名、离线编译前提和跨核 DMA 限制。本页以更正后的范围描述这些关系。

此次工作包含源码检查和图的生成检查，未运行 CPU 计算、TPU 模拟、离线 TPU 编译或真实 TPU 执行。bot 也将首轮的“编译/IR 观察”标签更正为静态源码中的可观察接口与验证方法，未提供本次编译或执行产物。普通 `lower(...).compile()` 不能自动视为无设备的离线 TPU 编译：还需确认 TPU backend、目标拓扑及相应构建的支持。若以后采集编译或运行证据，应记录 libtpu 二进制身份、设备/拓扑、输入、选项及产物；二进制与引用源码不匹配时标记 `VERSION-SKEW`。

配图内容在 [`software_stack_tpu_views.py`](../../../tools/software_stack_tpu_views.py)，生成命令：

```bash
python3 -B tools/diagram_environment.py run tools/render_software_stack_component_flows.py --component tpu
```

继续阅读：[PJRT Buffer](../pjrt/buffer-centered-hub.md) · [Pallas 内外层程序](../jax/jaxpr-centered-hub.md#pallas) · [XLA HLO](../xla/hlo-centered-hub.md)。

[program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L739
[opaque]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/tpu/tpu_ops_c_api.h#L39
[plugin]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_bridge.py#L205
[compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L756
[execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L2051
[serialize]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L2258
[mosaic]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
[mosaic-serialize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L491
[pallas-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[mosaic-design]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/docs/pallas/design/design.md#L515
[hlo-module]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L95
[hlo-schedule]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_schedule.h#L153
[buffer-assignment]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/buffer_assignment.h#L474
[capi-compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L608
[capi-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3421
[mlir-hlo]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/mlir_to_hlo.cc#L99
[tpu-units]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/docs/pallas/tpu/details.rst#L41
[core-type]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L525
[sc-tc-test]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/tests/pallas/tpu_pallas_mpmd_test.py#L93
[lhs]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/latency_hiding_scheduler.cc#L4288
[internal-answer]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064434151
[internal-followup]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064481581
[internal-corrections]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064767267
[memory-colors]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L120
[llo-metadata]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/tsl/profiler/utils/xplane_schema.cc#L421
