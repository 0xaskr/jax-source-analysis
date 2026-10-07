# XLA CPU：LLVM IR 与执行计划的协作

[查看 SVG](cpu-llvm-ir-centered-hub.svg) · [定位核心概念](cpu-llvm-ir-centered-hub.svg#core) · [返回总览](../overview/overview-software-stack-components-layered.svg#cpu_ir)

本图的唯一核心概念是 **CPU LLVM IR / `llvm::Module`**。
XLA CPU 后端把 HLO 计算落实为 LLVM host kernel，同时生成组织 kernel 与库调用的 `ThunkSequence`。
两类产物在 `CpuExecutable` 中汇合；读取 IR 时需要同时理解执行计划和存储约定。

图区导航：[定义](cpu-llvm-ir-centered-hub.svg#view_definition) · [产生](cpu-llvm-ir-centered-hub.svg#view_production) · [变换](cpu-llvm-ir-centered-hub.svg#view_transformation) · [消费](cpu-llvm-ir-centered-hub.svg#view_consumption) · [后续结果](cpu-llvm-ir-centered-hub.svg#downstream)。定义块独立说明静态结构，其他三个块保留真实调用链。直接提交旁路只跳过可选提取与拆分；编译调用内部的 LLVM 优化仍经跨区箭头展开。

## 定义：目标相关的 LLVM 程序

[`llvm::Module`][module] 管理函数、全局值、元数据与目标属性。
[`Function`][function] 包含 [`BasicBlock`][block]，基本块再包含 [`Instruction`][instruction]；指令的操作数和使用关系连接计算，终结指令表达控制流。
这套结构采用 LLVM 的类型与语义约束。

| 结构或约定 | 本图关注的含义 | 图中位置 |
| --- | --- | --- |
| Module / Function | 程序容器、函数签名和函数体 | [`d0`](cpu-llvm-ir-centered-hub.svg#d0) |
| BasicBlock / Instruction | 控制流、操作数与计算 | [`d1`](cpu-llvm-ir-centered-hub.svg#d1) |
| TargetTriple / DataLayout | 目标体系结构、数据大小及对齐等低层表示 | [`d2`](cpu-llvm-ir-centered-hub.svg#d2) |
| Host kernel 调用约定 | 参数如何关联存储，函数如何由运行时调用 | [`d3`](cpu-llvm-ir-centered-hub.svg#d3) |

`ThunkSequence` 组织执行工作，LLVM IR 定义待编译代码，二者不是可互换的程序对象。
`BufferAssignment` 与常量分配还承担 IR 之外的存储约定。

## 产生：从 HLO 到 kernel 和 thunks

产生区由 [`CpuCompiler::RunBackend`][backend] 进入 [`CompileCpuExecutable`][compile]。
后者创建 `LLVMContext` 和 `llvm::Module`，从 `TargetMachine` 设置 target triple 与 data layout，再组织代码生成上下文。

[`IrEmitter2` 的构造位置][emitter] 展示了它与 HLO、LLVM 主模块及 nested emitter 的关系。
紧随其后的 [`ThunkEmitter::EmitEntryComputation` 调用][thunk] 使用入口计算和存储、目标信息，返回 `ThunkSequence`；`ConsumeKernels` 取出额外生成的 kernel 模块。
图中 [`p1 → p2`](cpu-llvm-ir-centered-hub.svg#p1) 的灰色关联表示 `ThunkEmitter` 使用 `IrEmitter2` 的代码生成能力，不表示必须先完整生成所有 LLVM 代码再生成执行计划。

| 调用位置 | 主要输入 | 可核对的输出 |
| --- | --- | --- |
| `CompileCpuExecutable` 的模块创建部分 | HLO、IR compiler、目标配置 | 设置目标属性的 LLVM Module |
| `IrEmitter2` 与 nested emitter | HLO、LLVM Module、嵌套计算上下文 | 主模块中的 host kernel、相关函数与符号信息 |
| `ThunkEmitter::EmitEntryComputation` | emitter、入口计算、存储分配与目标特性 | `ThunkSequence` |
| `ThunkEmitter::ConsumeKernels` | 发射过程中累计的 kernel | 独立 LLVM kernel 模块 |

`hlo → p0 → p1 → p2 → produced` 汇总 LLVM 程序的来源与生成能力，具体产物位于 [`produced`](cpu-llvm-ir-centered-hub.svg#produced)，不连入静态定义卡。另一路 `p2 → thunks` 保留同步形成的计划结果。
[`thunks`](cpu-llvm-ir-centered-hub.svg#thunks) 位于灰色后续区：其中既可以有生成代码的调用，也可以有 oneDNN、Eigen 等库调用。
上游程序结构与存储规划的背景见 [HLO 文档](hlo-centered-hub.md)。

## 变换：整理编译单元与优化 IR

图中 `produced → units` 从 LLVM Module 进入可选的编译单元整理，再交给模块编译器。
[`units`](cpu-llvm-ir-centered-hub.svg#units) 汇总条件分支，不能理解为每个模块都固定执行同一组拆分操作。

| 变换 | 触发条件与输入 | 输出 |
| --- | --- | --- |
| [`ExtractKernelsFromModule`][extract] | 部分 kernel 携带不同 backend extra options，需要分别编译 | 调整后的主模块和抽出的 kernel 模块 |
| [`llvm::SplitModule` 的调用][split] | 满足并行代码生成拆分条件的主模块 | 多个 LLVM 编译片段 |
| [`RemoveUnusedSymbols`][remove] | 通过 `add_module_for_compilation` helper 提交的模块片段 | 去除无用符号后的片段 |
| [`IrCompiler::RunIrPasses`][passes] | Module、TargetMachine 与优化配置 | 经 LLVM pass pipeline 修改的 Module |

固定源码在拆分条件中处理大常量和 AOT 等情况；未拆分主模块及 `ConsumeKernels` 取得的模块还存在直接 `AddModule` 的分支。
因此，符号整理的调用范围应回到具体提交分支判断。
这里的 module splitting 是编译工作划分，不是 JAX 数组或设备分片。

提取或拆分后的片段沿 `t2 / t3 → t1 → consume_input` 经 helper 整理；未拆分主模块与额外 kernels 沿 `produced → consume_input` 直接提交。图中 `t2 → t3` 指提取后的剩余主模块按配置继续拆分，不表示每个抽出的 kernel 都再次拆分。

[`IrCompiler::operator()`][ir-compiler] 在构建目标机器后调用优化前 hook、`RunIrPasses`、优化后 hook，再进入目标代码生成。
`RunIrPasses` 通过 LLVM `PassBuilder` 配置分析与优化流程，优化级别为 O0 时采用相应分支。
输入输出仍属于同一个核心概念：变换改变 LLVM 程序，并未开始执行它。
图中 `c1 → t0 → after → c0` 因此是进入编译消费后，返回变换区执行优化，再进入目标发射；不能把直接提交旁路读成跳过 `RunIrPasses`。

## 消费：目标代码生成与 CpuExecutable 聚合

主图 `consume_input → c1 → t0 → after → c0 → symbols → executable` 展开模块编译器内部的代码路径：

1. [`AddModule`][add] 接收 LLVM 模块，模块编译器汇集待编译单元及 kernel、comparator 符号。
2. `Compile(compiled_symbols)` 组织编译和符号解析；其 IR 编译阶段使用 `IrCompiler`。
3. [`EmitMachineCode`][machine-code] 消费优化后的 Module 与 TargetMachine，输出目标对象字节。
4. 编译所得函数库与符号继续参与 `CpuExecutable` 构造。

图中 [`symbols`](cpu-llvm-ir-centered-hub.svg#symbols) 合并表达目标对象与解析后的符号两个相关产物。
LLVM 可以输出可读汇编作诊断，但这条代码生成链不要求先把汇编文本写入文件。

`thunks → executable` 是独立的执行计划汇入路径，位于灰色后续区。
在 [`CompileCpuExecutable`][compile] 的收尾处，函数库、`ThunkSequence`、HLO 模块、`BufferAssignment`、常量和目标配置共同形成后端可执行结果。
只看到机器代码还不足以重建该可执行程序的全部调用顺序和存储行为。
上层 provider 后续负责加载、包装及绑定执行数据，参见 [PJRT Buffer 文档](../pjrt/buffer-centered-hub.md)。

<a id="观测时应区分的证据"></a>

### 与代码生成并列的验证与观察

[`VerifyLlvmModule`][verify] 检查生成的 Module，编译过程对主模块和额外 kernels 均有相应检查。
IR dump 和编译前后 hooks 可以观察特定阶段的程序文本，但它们不是 CPU kernel 的执行记录。
图中 `produced → c2 → r2` 是直接验证的路径；`c1 / after → c3 → r3` 区分编译内优化前后观察。验证或观察已有 LLVM 程序，无须先生成目标对象。

| 观察结果 | 支持的结论 | 仍需另行确认 |
| --- | --- | --- |
| Module 通过 verifier | 满足所用 LLVM verifier 的检查条件 | 数值结果、性能和实际执行 |
| 优化前后 IR | 某目标配置下的程序变换 | 运行时走到哪些 thunk |
| 目标对象或解析符号 | 代码生成及编译产物 | 输入数据、调用次数和完成时间 |
| CpuExecutable 的构造关系 | 代码与执行计划如何汇合 | 当前安装二进制是否来自同一源码版本 |

## 源码版本与证据范围

| 源码树 | 固定提交 |
| --- | --- |
| XLA | `dcf304bc5dca1932b99f740b911dbd73631a1a69` |
| LLVM | `75a45c373407c13a44c7abb28a78d891a97fe665` |

链接复用 SVG 的固定 `source_anchors`；版本依据为 [`upstream-sources.lock`](../../../upstream-sources.lock)，文件哈希见[组件证据清单](../../../tools/component_diagram_sources.json)。
本页是源码检查，未运行 CPU 数值或性能实验，也未据此确认任何安装二进制与源码匹配。
若结合不同构建的运行产物分析，应显式标记 `VERSION-SKEW`。

继续阅读：[HLO](hlo-centered-hub.md) · [GPU LLVM IR](gpu-ir-centered-hub.md) · [组件索引](../index.md)。

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
