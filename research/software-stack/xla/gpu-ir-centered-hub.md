# XLA GPU：LLVM IR、Triton 分支与目标编译

[查看 SVG](gpu-ir-centered-hub.svg) · [定位核心概念](gpu-ir-centered-hub.svg#core) · [返回总览](../overview/overview-software-stack-components-layered.svg#gpu_ir)

本图的唯一核心概念是 **GPU LLVM IR**。
原生代码生成和 Triton 编译都可以产生这一表示；设备库调用则进入执行计划。
后半部分以 NVIDIA 目标代码生成为例，并单独展开 Triton 独立 kernel 的编译路径。

图区导航：[定义](gpu-ir-centered-hub.svg#view_definition) · [产生](gpu-ir-centered-hub.svg#view_production) · [变换](gpu-ir-centered-hub.svg#view_transformation) · [消费](gpu-ir-centered-hub.svg#view_consumption) · [后续结果](gpu-ir-centered-hub.svg#downstream)。定义块只说明静态 LLVM 结构。Triton 的 MLIR passes 放在产生块，因为它们负责产生本图研究的 LLVM IR；LLVM 链接和优化才属于该核心对象的变换。

## 定义：主模块和独立 kernel 都是 LLVM 程序

[`llvm::Module`][module] 管理函数、全局值与元数据；函数中的指令表达待编译 GPU 程序。
target triple、data layout、kernel 标记及设备特性共同参与目标代码生成。
一个 Module 可以承载常规模块代码，也可以是独立 kernel 的编译单元。

| 对象 | 在图中的角色 | 应保持的区别 |
| --- | --- | --- |
| GPU LLVM IR | 唯一核心概念，位于 [`core`](gpu-ir-centered-hub.svg#core) | 程序表示，尚未等同于设备二进制 |
| TritonKernelSource / MLIR Module | Triton 分支的编译输入与中间表示 | 由后续 pipeline 和翻译产生 LLVM IR |
| ThunkSequence | kernel、库调用及相关工作的执行计划 | 运行组织信息与 LLVM 指令分开 |
| PTX / cubin | NVIDIA 路径中的目标代码产物 | PTX 与最终设备二进制处于不同环节 |

Triton IR 没有被列为第二个核心概念；本图关注它如何作为一条生产路径连接到 GPU LLVM IR。

<a id="产生原生triton-与设备库三条分支"></a>

## 产生：原生与 Triton 如何形成 LLVM IR

产生区由 [`GpuCompiler::RunBackend`][backend] 组织后端编译，HLO、设备与存储等信息进入 [`CompileModuleToLlvmIr`][emission]。
`IrEmitterContext` 与 `ThunkEmitter` 共同组织代码生成结果、常量和执行工作。
图中 [`p0`](gpu-ir-centered-hub.svg#p0) 按 HLO 与目标配置选择分支，一个可执行程序可以组合多种分支。

| 分支 | 主要输入 | 产物及去向 |
| --- | --- | --- |
| 原生 emitter，[`native`](gpu-ir-centered-hub.svg#native) | HLO、目标配置与发射上下文 | LLVM 函数、kernel 与相关常量 |
| Triton fusion，[`p1`](gpu-ir-centered-hub.svg#p1) | 选中的 fusion、块级参数和设备信息 | Triton MLIR 经编译得到 LLVM kernel 与包装信息 |
| 设备库旁路，[`library`](gpu-ir-centered-hub.svg#library) | 适合库实现的 HLO 与调用配置 | cuBLAS、cuDNN 等调用直接进入 thunks，不作为 LLVM 产生步骤 |

设备库分支不要求现场为每个操作生成 LLVM kernel。
[`thunks`](gpu-ir-centered-hub.svg#thunks) 因而接收普通代码生成、设备库和独立 kernel 的工作结果，不能只从 LLVM Module 推出完整执行计划。

<a id="变换triton-到-llvm与-llvm-目标优化"></a>

### Triton 分支的实际接口

[`TritonFusion::GenerateTritonKernelAndWrapper`][fusion] 从 fusion 配置取得 `BlockLevelParameters`，调用 `CreateTritonModule` 形成 `TritonKernelSource`。
它再调用 `KernelCompiler::CompileTritonToLlvm`；CUDA 实现的[对应入口][to-llvm] 可以在线程池执行编译任务。

实际编译由 [`CompileTritonToLLVM`][triton-compiler] 接收 kernel 名称、HLO、设备信息、块级参数、目标 triple、data layout 和 MLIR 模块。
主图 `p1 → t2 → t0 → t1 → triton_result → produced` 将其内部过程展开为：

1. [`CreateTritonXlaPipeline`][xla-pipeline] 组织 XLA / 高层操作的合法化与相关整理。
2. [`CreateTritonPipeline`][target-pipeline] 按 CUDA 或 ROCm 目标、warp / CTA / stage 等配置建立目标 pipeline。
3. 运行 MLIR passes 后，通过 [`TranslateLLVMToLLVMIR` 的调用][translate] 得到 LLVM Module。
4. 保留目标属性、launch、shared memory 等元数据，形成 `TritonWrapperResult`。

`TritonWrapperResult` 除代码外还携带后续包装、启动所需的信息。
它的返回不意味着 kernel 已启动，也不要求把所有生成代码合并到同一个全局 LLVM 主模块。
[`produced`](gpu-ir-centered-hub.svg#produced) 汇总原生和 Triton 两种来源的 LLVM 程序实例；[`d1`](gpu-ir-centered-hub.svg#d1) 与 [`d2`](gpu-ir-centered-hub.svg#d2) 在产生区说明 Triton 输入容器及块级配置。它们解释来源，没有扩展 LLVM 核心概念的静态定义。

<a id="llvm-层的链接与优化"></a>

## 变换：LLVM 层的链接与优化

[`LinkAndOptimizeModule`][optimize] 读取 GPU LLVM Module、TargetMachine 与选项，按需链接设备 bitcode，再运行 LLVM 优化。
这是图中 [`t3`](gpu-ir-centered-hub.svg#t3) 的程序变换，结果位于 [`after`](gpu-ir-centered-hub.svg#after)，仍是供目标代码生成消费的 LLVM 程序。
Triton 的 MLIR pass pipeline 与这里的 LLVM 优化属于不同表示层，诊断时应记录各自的输入输出阶段。
常规模块路径中的 `c0 → t3 → after → ptx` 展开目标编译内部的 LLVM 处理。图中 `produced → consume_input` 表示选择消费者，不表示编译路径可以跳过内部链接与优化；直接验证和观察则可以使用已有 LLVM 程序。

## 消费：两条目标编译路径

<a id="常规模块路径1319"></a>

### 常规模块路径

[`GpuCompiler::CompileSingleModule`][single] 接收单个 LLVM Module、HLO 配置与设备描述，按配置验证和打印 IR，再委派目标二进制编译。
NVIDIA 分支进入 [`NVPTXCompiler::CompileTargetBinary`][nvptx]，在通常的 LLVM 编译分支中执行链接与优化、生成 PTX，再由 compilation provider 编译或链接得到 cubin 等结果；源码另有加载调试 LLVM / PTX 的选项。

具体读图路径是 `produced → consume_input → c1 → c0 → t3 → after → ptx`。LLVM Module 在 [`ptx`](gpu-ir-centered-hub.svg#ptx) 被目标代码生成消费；后续 `ptx → device_compile → binary → executable` 处理目标汇编、设备二进制和可执行对象，灰色区域明确其对象已经离开 LLVM IR。

[`binary`](gpu-ir-centered-hub.svg#binary) 表示 `BackendCompileResult` 中的目标代码及编译信息。
`binary → executable` 与 `thunks → executable` 在 [`executable`](gpu-ir-centered-hub.svg#executable) 汇合：二进制与 thunks、常量和存储约定共同构成 `GpuExecutable`。
ROCm 使用自己的目标后端；图中的 PTX / cubin 不能外推为全部 GPU 平台的统一产物。

<a id="triton-独立-kernel-路径2022"></a>

### Triton 独立 kernel 路径

Triton fusion 的 `Emit` 后续逻辑取得独立 LLVM 模块，对包装函数、参数和 launch 属性作处理，再调用 `KernelCompiler::CompileToTargetBinary`。
CUDA 的 [`CubinCustomKernelCompiler::CompileToTargetBinary`][custom-binary] 接收 `LlvmKernelSource`，返回包含二进制字节的 future；它可以直接编译，也可以安排到线程池。

接着，[`CustomKernelThunk` 的构造位置][custom-thunk] 把 kernel 二进制、参数、launch 维度、共享内存和相关元数据组织为可执行工作。
结果回到 `ThunkSequence`，再随整体计划进入可执行对象。
图中 `consume_input → kernel_compile → kernel_binary → thunks → executable` 表达这一条独立路径。`kernel_binary → thunks` 是编译结果的汇合，不是再次执行 Triton 的高层优化。
collective fusion 在源码中还有专门处理分支，本图不展开其内部实现。

<a id="观测与执行边界"></a>

### 并列的验证、观察与执行边界

[`LLVM verifier` 的调用位置][verify] 可报告目标编译前的非法 LLVM IR；dump 则记录某个编译阶段的文本。
这些证据用于判断程序如何生成、优化，不能单独证明真实 GPU 的吞吐、并发重叠或完成时间。
图中 `consume_input → c2 → r2` 保留直接验证入口，`c1 → c2` 则标明常规模块编译内部的验证；`c1 / after → c3 → r3` 说明编译选项控制的 dump 和优化后 hook。这些消费者不会把 Module 变成运行时设备 Buffer。

| 需要回答的问题 | 应查看的对象或位置 |
| --- | --- |
| 某 HLO 为什么进入 Triton？ | fusion 配置、块级参数与分派上下文 |
| Triton 是否已产出 LLVM？ | `CompileTritonToLLVM` 与返回的 LLVM kernel source |
| 二进制属于主模块还是独立 kernel？ | `CompileSingleModule` 与 `CompileToTargetBinary` 各自调用链 |
| 库调用和 kernel 如何组合？ | ThunkSequence 与可执行对象构造 |
| 何时提交、何时完成？ | [StreamExecutor 执行提交文档](../stream-executor/submission-centered-hub.md) |

编译产物返回后，执行时才由相应运行时绑定 Buffer、kernel 参数和 stream，并向设备提交工作。

## 源码版本与证据范围

| 源码树 | 固定提交 |
| --- | --- |
| XLA | `dcf304bc5dca1932b99f740b911dbd73631a1a69` |
| LLVM | `75a45c373407c13a44c7abb28a78d891a97fe665` |

版本依据为 [`upstream-sources.lock`](../../../upstream-sources.lock)，链接复用 SVG 内嵌源码锚点。
文件证据见[组件清单](../../../tools/component_diagram_sources.json)及[overview 清单](../../../tools/overview_flow_sources.json)。
本页核对的是 XLA 中的 Triton 接入和固定 LLVM 定义，未声称枚举 Triton 全部内部 pass，也没有运行 GPU 编译或设备性能实验。

继续阅读：[HLO](hlo-centered-hub.md) · [CPU LLVM IR](cpu-llvm-ir-centered-hub.md) · [执行提交](../stream-executor/submission-centered-hub.md) · [组件索引](../index.md)。

[module]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Module.h#L67
[backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2943
[emission]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/compile_module_to_llvm_ir.cc#L212
[fusion]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/fusion.cc#L148
[to-llvm]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/cubin_custom_kernel_compiler.cc#L137
[triton-compiler]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/xtile_compiler.cc#L496
[xla-pipeline]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/compilation_pipeline.cc#L33
[target-pipeline]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/compilation_pipeline.cc#L106
[translate]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/xtile_compiler.cc#L611
[optimize]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc#L252
[single]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2703
[nvptx]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/nvptx_compiler.cc#L583
[custom-binary]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/cubin_custom_kernel_compiler.cc#L91
[custom-thunk]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/codegen/triton/fusion.cc#L202
[verify]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2727
