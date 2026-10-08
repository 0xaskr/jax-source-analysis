# XLA CPU 后端：LLVM IR

CPU 后端用 LLVM IR 描述计算的低层实现：具体的数据运算、内存访问和控制流。它根据 HLO、存储规划和目标 CPU 信息生成 LLVM 程序，再通过 LLVM 优化和代码生成得到对象代码，与 thunk 执行计划共同组成 `CpuExecutable`。

[查看 SVG](cpu-llvm-ir-centered-hub.svg) · [软件栈总览](../overview/overview-software-stack-components-layered.md)

## 1. 定义：表示了什么

[`llvm::Module`][module] 是这一层的程序单元，持有函数、全局变量和元数据。它记录 target triple 和 DataLayout，分别标识目标平台并规定数据大小、对齐和指针等低层表示。

| 结构 | 表示的内容 |
|---|---|
| [`Function`][function] | 函数签名、参数和函数体。CPU 后端生成的 host kernel 与嵌套计算都以函数表达。 |
| [`BasicBlock`][block] | 一段顺序执行的指令，末尾的终结指令连接其他块或返回。 |
| [`Instruction`][instruction] | 具体运算、加载、存储、调用和控制转移；操作数连接 SSA 值。 |
| target triple、DataLayout、函数属性 | 目标架构、数据表示和影响代码生成的特性。 |

host kernel 的函数签名需要符合 CPU 运行时的调用约定，内存访问还要与 `BufferAssignment` 的安排一致。[`IrEmitter2`][emitter] 负责生成这类函数，[`ThunkEmitter`][thunk] 则组织何时调用它们，以及何时调用 oneDNN、Eigen 等库实现。

因此，一个完整的 CPU 程序同时需要 LLVM 代码和执行计划：前者描述函数内部怎样计算，后者描述一次调用中有哪些工作及其依赖。

## 2. 产生：从 HLO 生成计算函数

[`CpuCompiler::RunBackend`][backend] 将 HLO 交给 [`CompileCpuExecutable`][compile]。该函数根据 `IrCompiler` 创建 `TargetMachine`，再创建 LLVMContext 和 Module，并设置目标 triple、DataLayout 以及需要的 PIC/PIE 属性。

调度完成后，`CreateBufferAssignment` 为程序准备存储规划。随后在同一个编译入口中构造三类协作对象：

| 构造或调用 | 输入 | 产生的结果 |
|---|---|---|
| `IrEmitter` | HLO、BufferAssignment、目标信息和 LLVM Module。 | 嵌套计算函数以及小常量的全局定义。 |
| [`IrEmitter2`][emitter] | HLO、LLVM Module 和嵌套 emitter。 | fusion、元素级计算等所需的 host kernel 函数与符号信息。 |
| [`ThunkEmitter::EmitEntryComputation`][thunk] | 入口计算、存储规划、目标特性和代码生成能力。 | `ThunkSequence`，并在生成过程中收集额外 kernel 模块。 |
| `ThunkEmitter::ConsumeKernels` | 前一步积累的 kernel。 | 独立的 LLVM 编译单元及名称。 |

产生的 LLVM IR 可能分布在主模块和额外 kernel 模块中。`CompileCpuExecutable` 会对这些模块调用 `VerifyLlvmModule`，然后交给模块整理与编译流程。

## 3. 变换：整理编译单元并优化 LLVM 程序

### 按编译配置拆分

| 函数 | 输入及条件 | 结果 |
|---|---|---|
| [`ExtractKernelsFromModule`][extract] | 主模块，以及具有额外后端选项的一组 kernel。 | 将这些 kernel 抽到单独模块，以便使用对应选项编译。 |
| [`llvm::SplitModule`][split] | 剩余主模块和并行代码生成的分片数量。 | 多个可独立编译的 LLVM Module；局部依赖随相关 kernel 保留。 |
| [`RemoveUnusedSymbols`][remove] | 抽取或拆分得到的模块片段。 | 移除该片段不需要的符号，收集其待解析函数，再整理为线程安全模块。 |

没有拆分的主模块和额外 kernel 模块可以直接提交 `AddModule`。这些分支决定编译任务怎样组织；数组的设备分片由更上层的程序与运行时处理。

### LLVM 优化

[`IrCompiler::operator()`][ir-compiler] 在编译一个 Module 时，先调用 [`RunIrPasses`][passes]，再调用目标代码生成。`RunIrPasses` 根据 `TargetMachine`、优化级别和后端选项建立分析与 pass pipeline，处理内联、循环、向量化等优化机会。输出仍然是 LLVM Module，函数体和指令可能已经改变。

优化前后 hook 可以读取对应阶段的 Module；dump 记录的文本适合比较某个目标配置下的代码变化。

## 4. 消费：生成对象代码并组成 CpuExecutable

模块编译器的 [`AddModule`][add] 接收 LLVM 编译单元；`Compile` 再按已收集的 kernel 和 comparator 符号完成编译与符号解析。当前 JIT 路径将编译得到的函数组织为 `FunctionLibrary`。

[`IrCompiler::EmitMachineCode`][machine-code] 消费优化后的 Module 和 `TargetMachine`，通过 LLVM 目标代码生成 pass 将对象代码写入内存，返回 `MemoryBuffer`。这一入口可以直接产出对象代码，查看汇编时可另外反汇编对象文件。

```text
LLVM Module
  → AddModule / Compile
  → IrCompiler：LLVM 优化 → 对象代码
  → 链接与符号解析 → FunctionLibrary
```

[`CompileCpuExecutable`][compile] 最后调用 `CpuExecutable::Create`，汇合函数库、HLO、BufferAssignment、ThunkSequence、常量和目标信息。返回的 `CpuExecutable` 继续由 CPU PJRT 实现包装和加载。它被调用时，thunk 按执行计划使用函数库，并把实际存储传给生成的计算函数或库调用。

AOT 路径在同一入口选择 `AotLlvmMultipleModuleCompiler`，保存对象代码与符号等信息，供后续加载使用；这里的 JIT 链接过程对应的是当前进程可调用的函数库。

另外，[`VerifyLlvmModule`][verify] 直接读取 Module 并返回验证状态；编译 hook 和 dump 返回或保存程序文本。这些消费方式用于检查 IR，无需调用 CPU kernel。

---

本页以 [`upstream-sources.lock`](../../../upstream-sources.lock) 中的 XLA `dcf304bc`、LLVM `75a45c37` 为准，链接指向完整固定提交。已核对源码中的生成、变换和消费接口，未进行 CPU 编译或执行实验。生成和校验方法见[图文维护](../index.md#图文维护)。

[module]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Module.h#L67
[function]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Function.h#L65
[block]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/BasicBlock.h#L61
[instruction]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Instruction.h#L64
[backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2139
[compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1722
[emitter]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1882
[thunk]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1891
[extract]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1583
[split]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2030
[remove]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1427
[passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L354
[ir-compiler]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L289
[add]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1986
[machine-code]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L477
[verify]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1274
