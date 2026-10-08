# JAX 软件栈：从jax到 tpu/cpu/gpu

本文是[分层架构 SVG](overview-software-stack-components-layered.svg) 的配套说明

## 1. 不同组件的核心抽象概念

每个组件选取一个核心抽象概念，并围绕此进行展开，从该概念描述了哪些东西以及具体的数据结构如何表示？数据结构如何产生？会经过哪些变换？会被哪些组件消费？ 这四个角度来进行探索。

| 组件 | 核心抽象 | 含义 | 生命周期 |
|---|---|---|---|
| JAX | [Jaxpr](overview-software-stack-components-layered.svg#outer) | 用于表达jax/pallas 代码的中间表示, 描述输入如何通过原语及其组合产生输出 | 由 Python 函数和抽象输入经[追踪][src-trace]产生，经过[自动微分（如 JVP）][src-jvp-jaxpr]、[批处理（vmap）][src-batch-jaxpr]、[部分求值][src-partial-eval-jaxpr]和[死代码消除][src-dce]等变换；随后由 JAX 的 [lowering][src-lower] 按原语规则生成外层 MLIR Module，交给原生编译路径。 |
| jaxlib | [MLIR Module（StableHLO/Mosaic）](overview-software-stack-components-layered.svg#binding_module) | 以 [MLIR Module][src-module-op] 为容器，使用 [StableHLO][src-stablehlo-add]、[Mosaic 等专用 IR][src-mosaic-module] 表示计算图的中间表示。 | 由 [JAX][src-lower]/[Pallas lowering][src-pallas-lowering] 从 Jaxpr 产生; jaxlib 把MLIR module 封装成ifrt:HloProgram, ifrt通过调用pjrt交给具体的设备后端的编译接口。|
| IFRT | [Array](overview-software-stack-components-layered.svg#ifrt_array) | 跨设备的逻辑数组。[`ifrt::Array`][src-ifrt-array] 通过 `ArraySpec` 描述元素类型、形状、分片和布局；| 将输入数据对应的设备 Buffer 与类型、形状、分片信息[组合为逻辑数组][src-array-create]，供 [LoadedExecutable 执行][src-ifrt-execute]；执行返回时，再将输出 Buffer [组织为新的逻辑数组][src-ifrt-outputs]。 |
| PJRT | [PjRtBuffer](overview-software-stack-components-layered.svg#pjrt_buffer) | 对[设备上的数据存储的统一抽象][src-pjrt-buffer]，描述所属设备、内存空间、数据形状与布局，并提供所有权和[就绪状态][src-ready]的管理接口。 | 在接收[输入数据][src-buffer-from-host]或准备输出存储时创建，可按需要[复制][src-buffer-copy]或复用；作为 [PjRtLoadedExecutable][src-pjrt-execute] 的执行输入和输出，返回的输出 Buffer 由 IFRT [组织为逻辑数组][src-ifrt-outputs]，不再使用时[释放存储引用][src-buffer-delete]。 |
| XLA | [HLO（HloModule）](overview-software-stack-components-layered.svg#hlo) | 用于优化、规划和代码生成的计算图。[`HloModule`][src-hlo] 包含入口及其他 `HloComputation`，由 `HloInstruction` 表达计算和依赖，并携带形状、布局及分片等约束。 | 由 [StableHlo 导入][src-import]产生，经 [HLO pass pipeline][src-hlo-passes] 规范化并进行目标相关优化，再交给 [CPU][src-cpu-backend]／[GPU 后端][src-gpu-backend]组织存储规划、目标代码与执行计划，形成相应 `Executable`。 |
| XLA CPU 后端 | [LLVM IR](overview-software-stack-components-layered.svg#cpu_ir) | 面向 CPU 代码生成的低层中间表示。描述具体的数据运算，内存访问和控制流，并携带目标平台与数据布局信息。 | 由 CPU后端根据优化后的HLO、存储规划和目标信息生成，经 [LLVM 优化][src-cpu-ir-passes]后交给[目标代码生成][src-cpu-machine-code]得到对象文件；链接后的函数库与 thunk 执行计划共同组成 `CpuExecutable`。 |
| XLA GPU 后端 | [LLVM IR](overview-software-stack-components-layered.svg#gpu_ir) | 面向 GPU kernel 代码生成的[低层中间表示][src-llvm-module]，描述 kernel 内的数据运算、内存访问和控制流，并通过目标相关的指令与约定表达线程协作、地址空间和 kernel 入口。 | 由 [GPU 后端][src-gpu-emit]根据优化后的 HLO、存储规划和目标设备信息生成，按需链接设备 bitcode，经 [LLVM 优化][src-gpu-ir-passes]和目标代码生成得到设备代码；其中 [NVIDIA 路径][src-nvptx-binary]生成 PTX 并编译为 cubin。设备代码与存储规划、thunk 执行计划共同组成 [`GpuExecutable`][src-gpu-backend]，交回 PJRT。 |
| TPU provider / libtpu | [LLO（待版本证据）](overview-software-stack-components-layered.svg#tpu_boundary) | 本概览保留的 TPU 私有程序表示研究项。当前固定公开源码尚不能确认 LLO 的结构、语义和约束；Mosaic TPU MLIR 是另一种已知表示。 | 公开可确认的[编译边界][src-pjrt-program]是：程序与选项进入 TPU provider，返回已加载可执行对象供 PJRT 执行。LLO 在内部从何产生、经过哪些变换、由谁消费，仍待固定版本证据。 |
| 目标汇编表示 | [ASM](overview-software-stack-components-layered.svg#asm) | 以目标指令及其操作数描述计算、数据访问和控制转移的可读程序表示，可面向硬件指令集或 PTX 等虚拟指令集。 | 由编译器的目标代码生成产生，经汇编或[进一步目标编译][src-nvptx-binary]转换为机器代码，供后续链接和加载；也可由已有机器代码反汇编得到，供开发者检查代码生成结果和分析性能。 |
| 设备运行时与驱动 | [执行提交](overview-software-stack-components-layered.svg#submit) | 一次程序运行的请求与[执行上下文][src-cpu-submit]，关联已加载程序、输入输出存储、执行参数和依赖关系，描述执行什么、使用哪些数据以及何时可以执行。 | 由 [PJRT 执行接口][src-pjrt-execute]进入具体后端，结合可执行对象和输入 Buffer [准备所需存储][src-cpu-execute]，按执行计划和依赖关系[调度计算、通信与数据搬运][src-thunk-execute]；执行结果写入输出 Buffer，并向上层传播[完成或错误状态][src-ready]。 |
| 指令集接口与硬件 | [ISA](overview-software-stack-components-layered.svg#isa) | 软件与硬件之间的指令级语义约定，规定指令及其编码、寄存器等可见状态，以及执行指令时的计算、访存和控制转移行为。 | 由处理器体系结构规范定义，并随[架构版本与扩展演进][doc-isa-evolution]；编译器依据目标支持的[指令与特性][src-cpu-target]生成[机器代码][src-cpu-machine-code]，硬件按相应指令语义执行程序，更新寄存器、内存和控制状态。 |

图中 JAX 和 jaxlib 居中；IFRT/PJRT、XLA/编译后端、ASM/设备运行时左右并列；硬件位于底部。这个分布同时表达组件职责和调用关系，纵向位置不代表所有请求都必须依次经过每个框。

填色和粗边框用于核心节点。蓝线表示编译与程序传递，绿线表示执行与输入数据，紫线表示程序变换，橙色虚线表示编译结果返回，粉色实线表示输出对象返回，粉色虚线表示完成状态。灰色虚线说明结构、持有或观察关系；棕色虚线说明 TPU 内部证据边界。线上的拱桥表示交叉处不连接。

## 2. 完整的编译与执行往返

### 2.1 编译请求：函数到后端

| 阶段 | 输入与调用 | 结果或下一步 |
|---|---|---|
| 追踪 | Python 函数、静态参数和抽象输入进入 `trace_to_jaxpr_dynamic` 等追踪机制。 | 构造 Jaxpr；函数变换可以参与追踪或继续改写程序。 |
| lowering | `lower_jaxpr_to_module` 组织模块构造，`jaxpr_subcomp` 遍历方程并调用 lowering 规则。 | 产生外层 MLIR Module；包含 StableHLO 及相应配套方言操作。 |
| JAX → jaxlib，C1 | `PyClient::CompileAndLoad(Module, options)` 接收编译输入。 | 克隆 Module，构造 `ifrt::HloProgram` 和 IFRT 编译选项。 |
| jaxlib → IFRT，C2 | 默认 `Compiler::CompileAndLoad(program, options)` 接收 Program。 | 本路径中的 HloProgram 承载 MLIR Module。 |
| IFRT → PJRT，C3 | 适配实现取出 Module 与 XLA 选项，调用 `PjRtClient::CompileAndLoad`。 | 由具体 provider 决定编译和加载过程。 |
| CPU/GPU，C4、C5a、C5b | 公开编译实现将 Module 导入 HLO，运行优化、规划和目标后端代码生成。 | 形成 CPU 或 GPU 的 XLA Executable。 |
| TPU，C4t | Module 与编译选项进入 TPU provider；外层程序可能携带 Mosaic payload。 | 后续以公开的已加载可执行对象返回；私有中间阶段保持待确认。 |

对应源码：[`trace_to_jaxpr_dynamic`][src-trace]、[`lower_jaxpr_to_module`][src-lower]、[`jaxpr_subcomp`][src-subcomp]、[`PyClient::CompileAndLoad`][src-py-compile]、[IFRT 编译调用][src-ifrt-compile]、[PJRT 编译调用][src-pjrt-compile]、[CPU 的 Module 编译入口][src-cpu-module]。

`HloProgram` 是这里的 IFRT Program 实现名称，其内部仍可承载 MLIR Module。转换成 XLA `HloModule` 是后续公开编译路径中的另一个步骤。

### 2.2 编译结果：逐层加载与包装

编译返回的是可供后续执行使用的对象。各层对象的类型、所有权和附加元信息不同。

| 端口 | 返回或持有关系 | 具体含义 |
|---|---|---|
| R0a / R0b | CPU / GPU 后端 → XLA Executable | 目标代码与执行计划、存储规划等元信息构成编译产物。 |
| R0 | XLA Executable → provider → PJRT | CPU 的具体例子是 `CpuExecutable → PjRtCpuExecutable → LoadInternal → PjRtCpuLoadedExecutable`。 |
| Rt | TPU provider → PJRT | 以 `PjRtLoadedExecutable` 公共接口返回具体实现。 |
| R1 | PJRT → IFRT | IFRT 的适配对象包装 provider 返回的 `PjRtLoadedExecutable`。 |
| R2 | IFRT → jaxlib | 等待编译 future 并取得 `LoadedExecutableRef`，构造 `PyLoadedExecutable`。 |
| R3 | jaxlib → JAX 的持有关系 | `ExecuteReplicated.xla_executable` 持有 Python 可执行包装，供执行入口使用。 |

CPU 的包装和加载是图中明确展开的实例，不能据此认定 GPU、TPU 具有相同的内部类和加载步骤。参见 [CPU 编译产物包装][src-cpu-wrap]、[`LoadInternal`][src-cpu-load]、[IFRT 包装][src-ifrt-wrap]、[`PyLoadedExecutable` 包装][src-py-wrap]和 [`ExecuteReplicated`][src-dispatch]。

### 2.3 执行输入：程序与数据在提交处汇合

程序通路携带已经加载的代码、入口和执行计划；数据通路携带本次调用的数组及其存储。输入数组的关系为：

```text
主机值 / jax.Array → PyArray → IFRT Array → 各设备上的 PjRtBuffer
```

其中 `PyArray → Array` 是持有关系，PJRT-backed `Array → Buffer` 是逻辑数组与分片存储的映射；它们不表示每层都复制一次数据。

1. `ExecuteReplicated.__call__` 经 `in_handler(args)` 取得输入 PyArray，处理需要的 effects/tokens，再调用 `execute_sharded(input_bufs)`（E1）。
2. `PyLoadedExecutable::ExecuteSharded` 从 `PyArray.ifrt_array()` 取得 `ArrayRef[]`，调用 IFRT 的执行接口（E2）。
3. PJRT-backed IFRT 的 `LoadedExecutable::Execute` 读取各数组的底层 Buffer，按计算和设备组织 `PjRtBuffer*[][]`，向 PJRT 提交实参和选项（E3）。
4. provider 的执行实现消费已加载程序、输入 Buffer、donation/alias 约束及数据依赖，准备输出存储并调度工作（E4）。
5. CPU 任务与主机代码、GPU kernel/设备库或 TPU provider 的提交机制分别连接相应硬件（E5）。

对应源码：[`ExecuteReplicated`][src-dispatch]、[`PyArray`][src-py-array]、[`ExecuteSharded`][src-py-execute]、[IFRT 的 PJRT-backed 执行实现][src-ifrt-execute]、[`PjRtLoadedExecutable::Execute`][src-pjrt-execute]、[CPU provider 执行实例][src-cpu-execute]。

### 2.4 输出对象返回与设备完成

输出对象沿与输入对应的层级返回：

```text
provider 的输出存储
  → PjRtBuffer[][]
  → IFRT PjRtArray / ExecuteResult.outputs
  → PyExecuteResults / PyArray
  → Python 可见的 jax.Array
```

O0 表示 provider 经 PJRT 返回输出 Buffer；O1 表示 IFRT 按输出重新聚合分片，用 `PjRtArray::Create` 加入 dtype、shape、sharding 等元信息；O2/O3 表示 jaxlib 接收 IFRT 结果，经 `consume_with_handlers` 组织 Python 可见数组。参见 [IFRT 输出数组构造][src-ifrt-outputs]和 [`ConsumeWithHandlers`][src-py-outputs]。

对象返回和数据就绪分别表达。输出句柄可以先返回，设备完成或错误随后更新相应状态。`PjRtBuffer::GetReadyFuture()` 查询 Buffer 的就绪状态；PJRT Execute 可以按需返回完成 futures；IFRT `ExecuteResult.status` 仅在 `fill_status=true` 时填充。不能仅凭取得 Array/PyArray 对象判断计算已经完成。参见 [`GetReadyFuture`][src-ready]、[PJRT Execute 的返回约定][src-pjrt-execute]和 [`ExecuteResult`][src-ifrt-result]。

## 3. 各模块的定义、产生、变换与消费

### 3.1 JAX：Jaxpr

- **定义**：`Jaxpr` 将输入、附带常量、输出、方程与 effects 组织为程序。当前固定版本用 `_all_invars`、`_consts`、`_outvars` 和 `_eqns` 等字段保存这些信息；`JaxprEqn` 记录 primitive、参数和输入输出，`Var` 的 aval 表达抽象类型。见 [`core.py` 中的 Jaxpr][src-jaxpr]、[`JaxprEqn`][src-eqn]和 [`Var`][src-var]。
- **产生**：`trace_to_jaxpr_dynamic` 等追踪入口根据函数和抽象输入收集方程，得到 Jaxpr。追踪的结果是程序表示，实际数组数据通过执行通路传递。见 [`partial_eval.py`][src-trace]。
- **变换**：自动微分、批处理与部分求值参与程序构造和改写；图中以 `dce_jaxpr` 为具体例子，消费已有 Jaxpr，移除未被使用的纯计算，形成整理后的程序。见 [`dce_jaxpr`][src-dce]。
- **消费**：`lower_jaxpr_to_module` 组织 lowering，`jaxpr_subcomp` 消费方程并分派规则，输出 MLIR 操作和模块。Pallas 方程在这个过程中进入平台专用规则。见 [模块 lowering][src-lower]、[方程 lowering][src-subcomp]和 [Pallas lowering 分派][src-pallas-lowering]。

`kernel Jaxpr` 是专用 kernel 的内层程序，图中归入 Jaxpr 的结构关系；外层 Module 和 Mosaic Module 分别是相关 lowering 产物。

### 3.2 jaxlib：MLIR Module（StableHLO）

- **定义**：核心对象是 MLIR Module，在原生编译入口以 `mlir::ModuleOp` 传递。操作语义由 StableHLO 等方言规定，模块还可包含配套方言和 custom call；`jaxlib` 提供这些对象的绑定。见 [StableHLO 绑定入口][src-stablehlo]和 [Module 编译入口][src-py-compile]。
- **产生**：JAX 的 lowering 使用绑定构造 Module、操作和类型；C1 将生成的外层 Module 交到 jaxlib 的编译入口。见 [`lower_jaxpr_to_module`][src-lower]。
- **变换**：JAX lowering 中的模块 pass 可以更新 Module，例如按配置运行的 `sdy-lift-inlined-meshes`；`PyClient::CompileAndLoad` 还会克隆输入 Module，以交给后续编译过程。见 [Shardy 模块 pass][src-shardy]和 [`PyClient::CompileAndLoad`][src-py-compile]。
- **消费**：`PyClient::CompileAndLoad` 将 Module 包装为 `ifrt::HloProgram`，连同编译选项交给 IFRT Compiler。编译结果再包装为 `PyLoadedExecutable`，数组结果则走 PyArray 的独立通路。见 [IFRT 编译调用][src-ifrt-compile]和 [Python 可执行包装][src-py-wrap]。

### 3.3 IFRT：Array

- **定义**：`ifrt::Array` 表示一个逻辑数组，包含 dtype、shape、sharding 和 layout 等约束；可以由多个设备分片承载。见 [`array.h`][src-ifrt-array]。
- **产生**：输入侧由上层数组构造和放置流程提供 Array，PyArray 持有其引用；输出侧的 `PjRtArray::Create` 将各分片 Buffer 与输出元信息组合成 Array。见 [`PyArray`][src-py-array]和 [输出构造位置][src-ifrt-outputs]。
- **变换**：图中主要展开 Array 与设备 Buffer 的适配：`Execute` 将按数组组织的分片重排为按计算组织的实参；返回时再按输出重新聚合。这个过程不等于对数组内容做算术变换，也不要求无条件复制所有分片。见 [执行适配][src-ifrt-execute]和 [输出重组][src-ifrt-outputs]。
- **消费**：`LoadedExecutable::Execute(ArrayRef[], options, devices)` 消费逻辑数组；PJRT-backed 实现检查输入类型和分片 Buffer 数量等条件，再调用 PJRT。输出放入 `ExecuteResult.outputs`，状态按 `fill_status` 控制。见 [执行实现][src-ifrt-execute]和 [结果结构][src-ifrt-result]。

### 3.4 PJRT：PjRtBuffer

- **定义**：`PjRtBuffer` 是设备存储接口，关联形状、设备、内存空间、所有权、生命周期和就绪状态。它与编译期 `BufferAssignment` 分属运行时存储和编译规划两个阶段。见 [`PjRtBuffer`][src-pjrt-buffer]和 [`BufferAssignment`][src-buffer-assignment]。
- **产生**：输入侧由 provider 的存储与放置实现提供 Buffer；执行侧准备输出存储并以 Buffer 返回。图中以 CPU provider 的执行实现为可检查实例。见 [CPU 执行入口][src-cpu-execute]和 [PJRT Execute 约定][src-pjrt-execute]。
- **变换**：实际存储操作由 provider 实现；执行选项中的 donation 等约束影响输入存储能否被消费或复用。输入、输出句柄的所有权以及是否需要复制，应结合具体实现判断。图中的完成状态更新通过 Buffer 的就绪接口观察。见 [Execute 选项与实参约定][src-pjrt-execute]和 [`GetReadyFuture`][src-ready]。
- **消费**：`PjRtLoadedExecutable::Execute` 接收按计算组织的 Buffer 实参和执行选项，返回各计算的输出 Buffer，并可提供完成 futures；IFRT 将返回的 Buffer 重新包装为逻辑数组。见 [PJRT Execute][src-pjrt-execute]和 [IFRT 输出包装][src-ifrt-outputs]。

`Client`、`LoadedExecutable` 和 provider 是围绕存储与执行协作的接口或实现对象。它们负责接收程序、返回可执行对象和驱动执行，PJRT 本身不充当 HLO 与 LLVM IR 之间的一种 IR。

### 3.5 XLA：HLO（HloModule）

- **定义**：`HloModule` 持有入口和其他 `HloComputation`，计算由 `HloInstruction` 及其依赖关系构成；shape、layout、分片和编译配置共同约束程序。见 [`hlo_module.h`][src-hlo]。
- **产生**：公开 CPU/GPU 编译路径消费 Module；`ConvertMlirHloToHloModule` 将相应 MLIR 程序导入为 HloModule。见 [MLIR → HLO 转换][src-import]和 [CPU Module 编译入口][src-cpu-module]。
- **变换**：HLO pass pipeline 对程序进行规范化、代数重写、融合、分片和目标相关优化；`HloPassPipeline::RunImpl` 组织 pass 运行。具体顺序依目标和配置而定。见 [pass pipeline][src-hlo-passes]和 [CPU 编译组织][src-cpu-jit]。
- **消费**：目标后端消费 HLO 与编译规划。`BufferAssignment` 将 HLO 值映射到分配切片，后端代码生成与执行计划使用这些约束；结果汇集成 XLA Executable 后交还 provider。见 [存储规划][src-buffer-assignment]、[CPU 后端][src-cpu-backend]、[GPU 后端][src-gpu-backend]和 [Executable][src-executable]。

### 3.6 CPU 后端：LLVM IR

- **定义**：Module、Function、BasicBlock 和 Instruction 组织低层程序，表达 CPU 代码生成所需的计算与控制流。
- **产生**：`CpuCompiler::RunBackend` 所组织的后端流程消费目标 HLO 与规划，通过 lowering/emitter 构造低层程序。见 [`cpu_compiler.cc`][src-cpu-backend]。
- **变换**：LLVM 优化与目标代码生成处理该表示，编译和链接形成目标代码。图中把这些后端内部阶段聚合在 LLVM IR 的产生、变换和消费关系中。
- **消费**：目标代码、运行时任务或 thunk、库调用及元信息共同形成 `CpuExecutable`，再由 CPU provider 包装与加载。见 [CPU 后端返回][src-cpu-backend]、[provider 包装][src-cpu-wrap]和 [加载][src-cpu-load]。

更细的对象与调用位置见 [CPU LLVM IR 详细图](../xla/cpu-llvm-ir-centered-hub.svg)。

### 3.7 GPU 后端：LLVM IR

- **定义**：本概览选 LLVM IR 作为 GPU 代码生成的核心表示，Module/Function 组织低层设备程序；Triton 方言属于相关代码生成路径。
- **产生**：`GpuCompiler::RunBackend` 所组织的流程依据计算和目标选择 emitter、Triton 或设备库实现。代码生成路径构造相应 IR；库调用可以进入执行计划。见 [`gpu_compiler.cc`][src-gpu-backend]。
- **变换**：相关 lowering、LLVM 优化和目标代码生成将程序处理为设备代码。NVIDIA 路径可涉及 PTX 和目标二进制；图不将所有表示画成每次编译都必经的固定序列。
- **消费**：`GpuExecutable` 汇集目标代码、元信息与执行计划。设备库调用保留旁路，不要求为每次调用生成自有 kernel，也不能将一个 HLO 操作与一个硬件指令一一对应。见 [GPU 后端实现][src-gpu-backend]。

更细的分支见 [GPU IR 详细图](../xla/gpu-ir-centered-hub.svg)。

### 3.8 TPU 后端：LLO 的证据边界

- **定义**：LLO 在本图中是待版本证据确认的研究对象。公开提交接口不能证明其内部结构、类型系统或程序语义。
- **产生**：可以定位的公开输入是外层 Module 与编译选项；其中可能携带 Mosaic payload。当前图没有证据将这个输入绑定到某个确定的私有 LLO 产生函数。
- **变换**：私有 IR 的 pass、转换顺序和内部产物布局保持待确认。图中的棕色节点仅说明未知范围，不组成一条已确认的私有编译流水线。
- **消费**：公开结果是 provider 返回的 `PjRtLoadedExecutable`，随后可以通过 PJRT 提交执行；这不能推出哪个私有函数消费 LLO。见 [已加载程序接口][src-pjrt-loaded]和 [执行接口][src-pjrt-execute]。

Mosaic TPU MLIR 是 JAX/Pallas 分支产生的模块，不称为 LLO。相关证据范围见 [TPU 边界详细图](../libtpu/llo-boundary-centered-hub.svg)。

### 3.9 目标汇编表示：ASM

- **定义**：ASM 通过助记符、寄存器、地址等形式表达目标程序，是程序表示或观察形式。
- **产生**：图中 N0 从 XLA 编译产物分出按需观察路径；目标代码可以通过 dump 或反汇编得到可读表示。编译过程不保证保留汇编文本。
- **变换**：汇编器和反汇编工具可在相应文本与目标代码之间转换；具体格式与能力取决于平台和工具。
- **消费**：检查工具或开发者读取这些表示以诊断代码生成。该节点未绑定某个固定工具的实现函数，不能被解读为所有设备执行前必经的独立软件层。

图不将 `ASM → ISA` 表示为强制转换。ASM 描述目标程序，ISA 规定目标指令的语义。

### 3.10 设备运行时与驱动：执行提交

- **定义**：提交行为汇合已加载程序、入口与执行计划、输入 Buffer、执行选项和数据依赖。它是各后端共有的职责，具体实现分别展开。
- **产生**：PJRT `Execute` 进入 provider 的执行实现；后者消费程序与数据，准备输出存储并形成待执行工作。见 [PJRT Execute][src-pjrt-execute]和 [CPU 的 `CpuPjRtRawLoadedExecutable::Execute`][src-cpu-execute]。
- **变换**：后端按输入可用性、donation/alias 和任务依赖组织工作。CPU 可以使用任务、thunk 与主机库；GPU 使用相应的 stream、kernel、设备库和驱动机制；TPU 由 provider 管理提交及依赖。
- **消费**：硬件与后端执行机制消费提交的工作；输出句柄沿 O0 返回，完成或错误沿 F1 更新状态。图把创建输出对象和更新其 readiness 画成两个关系。见 [Buffer 就绪约定][src-ready]和 [IFRT 结果状态][src-ifrt-result]。

图中的 CPU/TPU 不因此被解释为经过 CUDA 或 StreamExecutor。GPU 的更细实例见 [执行提交详细图](../stream-executor/submission-centered-hub.svg)。

### 3.11 指令集接口与硬件：ISA

- **定义**：ISA 规定指令操作、编码与可见行为；内存模型和 ABI 补充相应的存储与调用约束。
- **产生**：ISA 由目标体系结构定义，编译器产生符合目标约束的程序。一次 JAX 编译产生的是目标代码等产物，而不是一套新的 ISA。
- **变换**：图中编译器变换的是程序表示与目标代码；没有把 ISA 契约本身描述为一个 IR pass 的输入输出。
- **消费**：物理执行单元执行目标工作，读写寄存器和内存，并与通信、传输及完成状态协作。硬件行为不对应某个统一的软件函数，具体指令和时序需要目标设备证据；本图不推断 TPU 私有指令。

## 4. Pallas 内外层程序如何接合

Pallas 分支嵌在 JAX 模块中。外层 Jaxpr 仍描述完整程序，内层 kernel Jaxpr 描述块内计算；二者通过 `pallas_call` 方程参数连接。

| 阶段 | 具体位置与输入 | 输出及关系 |
|---|---|---|
| kernel 追踪 | `_trace_kernel_to_jaxpr` 消费 kernel 函数及其抽象参数。 | 内层 kernel Jaxpr；Ref 读写表达块内计算。 |
| 调用映射 | `get_grid_mapping` 构造 `GridMapping`。 | 网格、块与调用约定信息。 |
| 外层方程 | `pallas_call_p.bind` 接收数组实参，并通过参数携带 kernel Jaxpr、GridMapping 等。 | 外层 Jaxpr 中的 `pallas_call` 方程；方程输入输出继续连接外层数据流。 |
| lowering 分派 | `jaxpr_subcomp` 消费外层方程，`_pallas_call_lowering` 按平台选择规则。 | 本图展开 Mosaic TPU 且 `interpret=False` 的分支。 |
| 内层模块 | `pallas_call_tpu_lowering_rule` 经 `lower_jaxpr_to_pipelined_module` 消费 kernel Jaxpr 与映射。 | 独立的 Mosaic TPU MLIR Module。 |
| 编译期封装 | `lower_module_to_custom_call` 组织模块处理与序列化，并构造 custom call。 | payload 与配置进入 `backend_config`。 |
| 回接外层 | 构造目标为 `tpu_custom_call` 的外层 custom call。 | 操作的 IR 结果值继续连接外层 Module；完整外层 Module 交给后端编译。 |

对应源码：[`_trace_kernel_to_jaxpr`][src-kernel-trace]、[`get_grid_mapping`][src-grid-build]、[`GridMapping`][src-grid]、[`pallas_call_p.bind`][src-pallas-bind]、[Pallas lowering 分派][src-pallas-lowering]、[TPU lowering 规则][src-tpu-lowering]、[Mosaic 模块构造][src-mosaic-module]、[custom call 封装][src-custom-call]和 [custom call 操作构造][src-custom-emit]。

payload 是编译期嵌入的数据，不是运行时 Array/Buffer，也不是当前 kernel 执行得到的输出。普通 JAX 方程与这个 custom call 可以共存于外层程序；普通程序不要求先经过 Pallas。细节见 [JAX 图的 Pallas 内外层程序区域](../jax/jaxpr-centered-hub.svg#inner_jaxpr)及[配套章节](../jax/jaxpr-centered-hub.md#pallas)。

## 5. MLIR 与 Shardy 的跨层位置

MLIR 提供 Module、Operation、Region、Type 和 pass 等基础设施。JAX lowering 通过 jaxlib 绑定使用它，StableHLO、`sdy`、Mosaic 等方言规定相应操作语义；相关后端路径也可继续使用 MLIR。MLIR 的使用范围跨越多个组件，不是在图中插入一个所有请求串行经过的运行时服务。

Shardy 的图内实例是 JAX lowering 中按配置运行的 `sdy-lift-inlined-meshes`。输入和输出都是外层 Module，处理的是相应分片表示。这个实例不代表全部 Shardy pipeline，也不意味着 HLO 之后必须有一个固定的“Shardy IR 层”。见 [模块 pass 的调用位置][src-shardy]。

## 6. 跨组件接口索引

下表与 SVG 的 30 个命名端口一一对应。节点名称可跳转到图内；接口描述省略部分所有权模板、`StatusOr` 和可选参数。关系方向表示图中相应的调用、包装、持有或状态更新，并非全部是同步函数调用。

| 端口 | 图中关系 | 接口或对象 |
|---|---|---|
| C1 | [外层 MLIR Module](overview-software-stack-components-layered.svg#module) → [MLIR Module（StableHLO）](overview-software-stack-components-layered.svg#binding_module) | CompileAndLoad(Module, options)；外层 Module → jaxlib 编译输入 |
| C2 | [PyClient::CompileAndLoad](overview-software-stack-components-layered.svg#py_compile) → [HloProgram](overview-software-stack-components-layered.svg#ifrt_program) | Compiler::CompileAndLoad(program, options)；HloProgram(Module) + IFRT CompileOptions |
| C3 | [Compiler::CompileAndLoad](overview-software-stack-components-layered.svg#ifrt_compile) → [PjRtClient::CompileAndLoad](overview-software-stack-components-layered.svg#pjrt_compile) | PjRtClient::CompileAndLoad；Module + xla::CompileOptions |
| C4 | [Backend provider](overview-software-stack-components-layered.svg#provider) → [MLIR / StableHLO → HLO](overview-software-stack-components-layered.svg#import) | provider 调用公开 XLA 编译；CPU/GPU：Module + 编译 / 设备配置 |
| C5a | [调度 / BufferAssignment](overview-software-stack-components-layered.svg#plan) → [HLO / emitter / RunBackend](overview-software-stack-components-layered.svg#cpu_lowering) | CPU RunBackend；目标 HLO + 编译规划 |
| C5b | [调度 / BufferAssignment](overview-software-stack-components-layered.svg#plan) → [原生 emitter / Triton / 库](overview-software-stack-components-layered.svg#gpu_lowering) | GPU RunBackend；目标 HLO + 编译规划 |
| C4t | [Backend provider](overview-software-stack-components-layered.svg#provider) → [外层 Module + 可选 Mosaic payload](overview-software-stack-components-layered.svg#tpu) | TPU provider 的编译 / 加载入口；外层 Module + CompileOptions |
| R0a | [CpuExecutable / CPU 目标代码](overview-software-stack-components-layered.svg#cpu) → [XLA Executable + 执行计划](overview-software-stack-components-layered.svg#executable) | CPU 编译产物；CpuExecutable + 执行计划 |
| R0b | [GpuExecutable / 代码与库调用](overview-software-stack-components-layered.svg#gpu) → [XLA Executable + 执行计划](overview-software-stack-components-layered.svg#executable) | GPU 编译产物；GpuExecutable + 执行计划 |
| R0 | [XLA Executable + 执行计划](overview-software-stack-components-layered.svg#executable) → [xla::PjRtLoadedExecutable](overview-software-stack-components-layered.svg#pjrt_loaded) | Executable → provider 包装 / 加载；CPU：PjRtCpuExecutable → LoadInternal |
| Rt | [TPU 的 PjRtLoadedExecutable](overview-software-stack-components-layered.svg#tpu_loaded) → [xla::PjRtLoadedExecutable](overview-software-stack-components-layered.svg#pjrt_loaded) | TPU provider 返回 PJRT 对象；PjRtLoadedExecutable |
| R1 | [xla::PjRtLoadedExecutable](overview-software-stack-components-layered.svg#pjrt_loaded) → [ifrt::LoadedExecutable](overview-software-stack-components-layered.svg#ifrt_loaded) | 包装 provider 返回的可执行对象；PjRtLoadedExecutable → IFRT LoadedExecutable |
| R2 | [ifrt::LoadedExecutable](overview-software-stack-components-layered.svg#ifrt_loaded) → [PyLoadedExecutable](overview-software-stack-components-layered.svg#py_loaded) | 编译 future → Await()；LoadedExecutableRef → PyLoadedExecutable |
| R3 | [PyLoadedExecutable](overview-software-stack-components-layered.svg#py_loaded) → [ExecuteReplicated.__call__](overview-software-stack-components-layered.svg#dispatch) | ExecuteReplicated.xla_executable；持有返回的 PyLoadedExecutable |
| D1 | [主机值 / jax.Array](overview-software-stack-components-layered.svg#inputs) → [PyArray](overview-software-stack-components-layered.svg#py_array) | 输入数组放置 / 包装；主机值或 jax.Array → PyArray |
| D2 | [PyArray](overview-software-stack-components-layered.svg#py_array) → [Array](overview-software-stack-components-layered.svg#ifrt_array) | PyArray 持有 IFRT Array；逻辑数组引用 / Python 元信息 |
| S1 | [Array](overview-software-stack-components-layered.svg#ifrt_array) → [PjRtBuffer](overview-software-stack-components-layered.svg#pjrt_buffer) | PJRT-backed Array → Buffer 映射；一个逻辑数组对应各设备上的分片存储 |
| D0 | [PjRtBuffer](overview-software-stack-components-layered.svg#pjrt_buffer) → [输入 Buffer / 执行依赖](overview-software-stack-components-layered.svg#runtime_buffers) | 输入 Buffer 与数据可用依赖；provider 执行实现接收设备实参 |
| P0 | [xla::PjRtLoadedExecutable](overview-software-stack-components-layered.svg#pjrt_loaded) → [已加载程序 / 执行计划](overview-software-stack-components-layered.svg#runtime_program) | 已加载程序 / 入口 / 执行计划；provider 持有，执行实现消费 |
| E1 | [ExecuteReplicated.__call__](overview-software-stack-components-layered.svg#dispatch) → [PyLoadedExecutable::ExecuteSharded](overview-software-stack-components-layered.svg#py_execute) | execute_sharded(input_bufs)；PyArray[] + 执行选项 |
| E2 | [PyLoadedExecutable::ExecuteSharded](overview-software-stack-components-layered.svg#py_execute) → [LoadedExecutable::Execute](overview-software-stack-components-layered.svg#ifrt_execute) | Execute(args, options, devices)；PyArray.ifrt_array() → ArrayRef[] |
| E3 | [LoadedExecutable::Execute](overview-software-stack-components-layered.svg#ifrt_execute) → [PjRtLoadedExecutable::Execute](overview-software-stack-components-layered.svg#pjrt_execute) | Execute(argument_handles, opts, futures)；Array 分片 → PjRtBuffer*[][] |
| E4 | [PjRtLoadedExecutable::Execute](overview-software-stack-components-layered.svg#pjrt_execute) → [执行提交](overview-software-stack-components-layered.svg#submit) | provider 的 Execute 实现；已加载程序 + 输入 Buffer + 执行选项 |
| E5 | [任务 / kernel / 库 / provider](overview-software-stack-components-layered.svg#runtime) → [CPU / GPU / TPU 执行单元](overview-software-stack-components-layered.svg#hardware) | 后端提交目标工作；CPU 任务 / GPU kernel 与库 / TPU provider |
| O0 | [存储句柄与依赖跟踪](overview-software-stack-components-layered.svg#handles) → [PjRtBuffer[][] + 完成 future](overview-software-stack-components-layered.svg#pjrt_outputs) | Execute 返回输出 Buffer；PjRtBuffer[][]；完成 future 按需请求 |
| O1 | [PjRtBuffer[][] + 完成 future](overview-software-stack-components-layered.svg#pjrt_outputs) → [ExecuteResult / PjRtArray](overview-software-stack-components-layered.svg#ifrt_outputs) | PjRtArray::Create；输出 Buffer 分片 + 元信息 → ArrayRef[] |
| O2 | [ExecuteResult / PjRtArray](overview-software-stack-components-layered.svg#ifrt_outputs) → [PyExecuteResults → PyArray](overview-software-stack-components-layered.svg#py_outputs) | IFRT ExecuteResult；ArrayRef[]；status 按 fill_status 填充 |
| O3 | [PyExecuteResults → PyArray](overview-software-stack-components-layered.svg#py_outputs) → [jax.Array / PyArray](overview-software-stack-components-layered.svg#outputs) | consume_with_handlers(handlers)；返回 Python 可见数组 |
| F1 | [CPU / GPU / TPU 执行单元](overview-software-stack-components-layered.svg#hardware) → [完成、错误与 Future](overview-software-stack-components-layered.svg#runtime_completion) | 设备完成 / 输出可用 / 错误；更新 Future / readiness |
| N0 | [XLA Executable + 执行计划](overview-software-stack-components-layered.svg#executable) → [目标代码 / 对象文件](overview-software-stack-components-layered.svg#asm_code) | 可选 dump / 反汇编观察；目标代码 → 可观察的汇编表示 |

## 7. 源码版本与维护入口

本文沿用 SVG 元数据中已核验的固定源码锚点：JAX 为 `361c43e072cce92b7d3e9bdaf4dd16db26c49043`，XLA 为 `dcf304bc5dca1932b99f740b911dbd73631a1a69`。修订版本以 [`upstream-sources.lock`](../../../upstream-sources.lock)、[`source-archives.lock`](../../../source-archives.lock) 和[环境锁](../../../env/environment.lock.json)为准。源码链接使用固定提交，具体函数和类的位置以这些链接为入口。

SVG 由 [`render_overview_software_stack_flows.py`](../../../tools/render_overview_software_stack_flows.py) 生成；节点内容和唯一核心概念映射位于 [`software_stack_overview_flow_data.py`](../../../tools/software_stack_overview_flow_data.py)。原 [`render_overview_software_stack_components.py`](../../../tools/render_overview_software_stack_components.py) 的命令入口也转交同一生成器。

在仓库根目录运行：

```bash
python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_flows.py
```

命令更新 `overview-software-stack-components-layered.svg`，仅输出 SVG。使用 `--output-dir /tmp/jax-stack-flows` 可以先在临时目录核验。生成器检查固定源码、每个模块恰好一个核心节点、关键路径连通性、文字布局和连线避让；本文是配套说明，修改图的语义时需同步维护。

XLA 完整检出缺失时，生成器使用 [`overview_flow_sources.json`](../../../tools/overview_flow_sources.json) 锁定提交及 SHA-256 的源码缓存，并校验内容。缓存缺失时可以显式加 `--fetch-sources` 获取固定提交的文件；正常重建离线进行。绘图依赖由 [`env/diagrams.lock.json`](../../../env/diagrams.lock.json) 锁定，源码与布局校验不构成编译成功或设备执行证据。

<a id="extension-interfaces"></a>

## 8. 扩展与观测接口版

[扩展版 SVG](overview-software-stack-components-extended.svg) 在同一组件分布和完整往返路径上，增加以下辅助接口。它保留 11 个核心概念；接口卡片属于相应模块的扩展或观察入口，字母按钮将卡片与主图中的作用位置连接起来。

| 接口 | 作用位置 | 输入、行为与输出 |
|---|---|---|
| [A · primitive lowering](overview-software-stack-components-extended.svg#xla-interface-a) | JAX 的 Jaxpr → MLIR | `mlir.register_lowering` 注册通用或平台专用规则；规则接收 lowering context、MLIR operands 和参数，返回 MLIR 结果值。抽象求值、AD 和 batching 规则分别定义。 |
| [B · custom_partitioning](overview-software-stack-components-extended.svg#xla-interface-b) | JAX 记录规则，分片编译消费 | `def_partition` 记录分片行为。Shardy 启用时使用 `sharding_rule`；相应 partition 回调返回 mesh、`lower_fn` 和输入输出 shardings，分片计算再追踪和 lowering 为 MLIR bytecode。 |
| [C · 原生 FFI](overview-software-stack-components-extended.svg#xla-interface-c) | 编译输入及后端执行 | `register_ffi_target` 注册原生 target；`ffi_call` 的 lowering 生成 custom call。匹配的平台、target 与 ABI 支持实际执行，结果进入输出返回通路。Pallas 的 Mosaic payload 是另一种 custom call 用法。 |
| [D · HLO hook](overview-software-stack-components-extended.svg#xla-interface-d) | 后端 HLO pipeline | `register_hlo_module_transformation` 注册宿主 Python 回调：`HloModuleProto bytes → callback → bytes / None`。返回 bytes 更新编译中的程序，返回 None 保留原程序。 |
| [E · IR 与编译结果检查](overview-software-stack-components-extended.svg#xla-interface-e) | lowering 和编译结果 | `Lowered.compiler_ir`、`as_text`、`Compiled.cost_analysis`、`memory_analysis` 和 `runtime_executable` 提供不同层级的表示与元信息。支持范围依后端及版本而定。 |
| [F · 运行观测](overview-software-stack-components-extended.svg#runtime-observation) | 提交、后端工作和完成状态 | `jax.profiler.trace` 等入口采集事件，`TraceAnnotation` 标注宿主范围，采集数据供时间线与性能工具分析。观测范围要结合异步完成与输出就绪判断。 |

A–C 的源码入口分别见 [`register_lowering`][src-ext-lowering]、[`custom_partitioning`][src-ext-partition]、[分片计算 lowering][src-ext-shard-lowering]、[`register_ffi_target`][src-ext-ffi]和 [`ffi_call_lowering`][src-ext-ffi-lowering]。B 接口中的分片契约不接收完整 HloModuleProto，也不等同于主图中单独标出的 `sdy-lift-inlined-meshes` pass。

### 8.1 HLO hook 的注册与返回

注册按 `platforms` 选择后端。CPU 直接调用 `_xla.register_xla_transform`；其他平台经 `get_pjrt_plugin` 和 `_xla.register_xla_transform_c_api` 进入插件扩展。插件必须支持相应扩展。TPU 测试定义可以证明源码中存在该用法，本次工作没有执行这个测试。见 [注册实现][src-ext-hook]和 [TPU 测试定义][src-ext-tpu-test]。

`PRE_SCHEDULER` 和 `POST_SCHEDULER` 是 HLO 调度前后的挂点，具体插入位置由后端决定。CPU 的 POST 挂点发生在 `CreateBufferAssignment` 之前；PRE 不表示所有 HLO 优化之前，POST 不表示机器指令调度之后。见 [CPU PRE][src-ext-cpu-pre]、[CPU POST][src-ext-cpu-post]、[GPU PRE][src-ext-gpu-pre]和 [GPU POST][src-ext-gpu-post]。

扩展图在 XLA 框中画出 `HLO pipeline → 挂点 → Python callback → 继续编译` 的可选循环。回调可以解析序列化 HLO、修改计算图或 schedule，再返回 bytes；同一 stage 的回调按注册顺序运行。单独修改导出的 HLO 副本不会自动更新已有 executable；hook 处理的是外层 HLO，custom call 内的 Mosaic payload 是另一套表示。

### 8.2 观察对象的可变性与执行证据

在当前固定实现中，`MeshComputation.stablehlo()` 返回 `self._hlo`，因此 `lowered.compiler_ir('stablehlo')` 不能被概括为只读副本；`compiler_ir('hlo')` 则走显式导出分支。Module 原地修改可能影响尚未发生的编译，但这不是稳定的编辑契约，也不会重写已经编译的 executable。调试表示与可移植序列化用途分别对待，后者使用 `jax.export`。见 [Module 返回位置][src-ext-stablehlo-object]、[HLO 导出][src-ext-hlo-export]和 [Lowered 接口][src-ext-ir]。

`jax.profiler.trace` 和 `TraceAnnotation` 是采集入口与宿主标记。可见设备事件取决于后端；需要覆盖异步计算完成时，应在采集范围内等待结果就绪。本文新增的是接口和源码关系，没有据此报告实际采集、CPU 执行或 TPU 执行结果。见 [trace 入口][src-ext-trace]和 [宿主标记][src-ext-annotation]。

扩展图的生成命令为 `python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_extensions.py`，仅输出 SVG；可加 `--output-dir /tmp/jax-stack-extensions` 先在临时目录核验。生成器见 [`render_overview_software_stack_extensions.py`](../../../tools/render_overview_software_stack_extensions.py)。

[src-ext-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1003
[src-ext-partition]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/custom_partitioning.py#L272
[src-ext-shard-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/custom_partitioning.py#L158
[src-ext-ffi]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/ffi.py#L54
[src-ext-ffi-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/ffi.py#L656
[src-ext-hook]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_transform.py#L48
[src-ext-tpu-test]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/tests/xla_transform_test.py#L452
[src-ext-cpu-pre]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1178
[src-ext-cpu-post]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1833
[src-ext-gpu-pre]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L1026
[src-ext-gpu-post]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L3286
[src-ext-stablehlo-object]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/pxla.py#L1224
[src-ext-hlo-export]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/stages.py#L237
[src-ext-ir]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/stages.py#L671
[src-ext-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/profiler.py#L315
[src-ext-annotation]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/profiler.py#L354

[src-buffer-assignment]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/buffer_assignment.h#L476
[src-cpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2139
[src-cpu-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1602
[src-cpu-jit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L764
[src-cpu-load]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L709
[src-cpu-module]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L820
[src-cpu-wrap]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1037
[src-custom-call]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
[src-custom-emit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L461
[src-dce]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1204
[src-dispatch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/pxla.py#L338
[src-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L460
[src-executable]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/executable.h#L263
[src-gpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2943
[src-grid]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L946
[src-grid-build]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L1268
[src-hlo]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L95
[src-hlo-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/pass/hlo_pass_pipeline.cc#L298
[src-ifrt-array]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/array.h#L66
[src-ifrt-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L412
[src-ifrt-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L835
[src-ifrt-outputs]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L1106
[src-ifrt-result]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/executable.h#L272
[src-ifrt-wrap]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L798
[src-import]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6232
[src-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L105
[src-kernel-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L788
[src-lower]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327
[src-mosaic-module]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008
[src-pallas-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1358
[src-pallas-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L845
[src-pjrt-buffer]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1108
[src-pjrt-compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L763
[src-pjrt-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1457
[src-pjrt-loaded]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1390
[src-py-array]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_array.h#L141
[src-py-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475
[src-py-execute]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L484
[src-py-outputs]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L260
[src-py-wrap]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L430
[src-ready]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1380
[src-shardy]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1483
[src-stablehlo]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/lib/mlir/dialects/__init__.py#L62
[src-subcomp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L2129
[src-tpu-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[src-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2133
[src-var]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L539

[src-array-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L158
[src-array-disassemble]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L353
[src-buffer-copy]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1306
[src-buffer-from-host]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L966
[src-cpu-emit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1882
[src-cpu-ir-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L354
[src-cpu-machine-code]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L477
[src-cpu-submit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1809
[src-cpu-target]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L255
[src-gpu-emit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/compile_module_to_llvm_ir.cc#L212
[src-gpu-ir-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc#L252
[src-llvm-module]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Module.h#L67
[src-module-op]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/mlir/include/mlir/IR/BuiltinOps.td#L33
[src-nvptx-binary]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/nvptx_compiler.cc#L583
[src-pjrt-program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L739
[src-stablehlo-add]: https://github.com/openxla/stablehlo/blob/7b1b15781ccbd770f50c7eef4b0c3e03834649fd/stablehlo/dialect/StablehloOps.td#L831
[src-thunk-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/runtime/thunk_executor.cc#L248
[doc-isa-evolution]: https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-a-profile-architecture-developments-2025

[src-jvp-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L1044
[src-batch-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/batching.py#L416
[src-partial-eval-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L655

[src-buffer-delete]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1273
