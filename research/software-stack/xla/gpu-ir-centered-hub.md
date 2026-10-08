# XLA GPU 后端：LLVM IR

GPU 后端用 LLVM IR 表达 kernel 的低层计算：数据运算、内存访问和控制流，以及目标相关的线程操作、地址空间和 kernel 入口约定。它由 HLO 的代码生成路径产生，经 LLVM 优化和目标代码生成得到设备代码，最后与执行计划共同组成 `GpuExecutable`。

[查看 SVG](gpu-ir-centered-hub.svg) · [软件栈总览](../overview/overview-software-stack-components-layered.md)

## 1. 定义：表示了什么

[`llvm::Module`][module] 持有函数、全局值和元数据；函数体由基本块与指令组成。GPU 目标的 triple、DataLayout、调用约定、属性和 intrinsic 进一步说明这些函数怎样在设备上执行。

| 内容 | 在 LLVM 程序中的表达 |
|---|---|
| kernel 内的计算 | 算术、比较、控制流及函数调用等指令。 |
| 数据访问 | 指针、加载与存储、地址空间及对齐信息。 |
| 并行执行 | 目标相关的线程索引、同步等操作，以及 kernel 入口标记和属性。 |
| 目标要求 | triple、DataLayout、设备能力和影响优化的函数属性。 |

一个 LLVM Module 可以承载一组函数，也可以只作为某个 kernel 的独立编译单元。[`CompileModuleToLlvmIr`][emission] 同时组织代码生成、存储规划和 thunk 执行计划。kernel 的 LLVM 程序描述内部计算，thunk 记录 kernel 怎样启动，以及它与设备库、复制和其他工作的关系。

## 2. 产生：从 HLO 的实现选择到 LLVM Module

### 后端组织代码生成

[`GpuCompiler::RunBackend`][backend] 根据 HLO 和目标设备进入编译流程。[`CompileModuleToLlvmIr`][emission] 先准备 BufferAssignment 和输出信息，再构造 `IrEmitterContext`、`ThunkEmitter`，调用 `EmitHloEntryComputation`。

`IrEmitterContext` 向生成器提供 HLO、存储分配、设备描述、目标 triple、DataLayout 和 kernel 编译器。生成器按运算选择适合的实现；需要现场生成的 kernel 形成 LLVM 编译单元，cuBLAS、cuDNN 等设备库调用则直接形成相应 thunk。源码还会单独检查常量模块，非空时调用 `CompileToTargetBinary` 编译。

### Triton 路径如何产生 LLVM

[`TritonFusion::GenerateTritonKernelAndWrapper`][fusion] 从 fusion 配置中取得 `BlockLevelParameters`，调用 `CreateTritonModule` 构造 `TritonKernelSource`。这个阶段的载体是 MLIR Module，随后由 [`CompileTritonToLlvm`][to-llvm] 调用 [`CompileTritonToLLVM`][triton-compiler]。

| 调用 | 输入 | 处理与结果 |
|---|---|---|
| `CreateTritonModule` | HLO fusion、块级参数和设备描述。 | 生成内核的 MLIR 表示，并整理为 Triton 编译输入。 |
| [`CreateTritonXlaPipeline`][xla-pipeline] | Triton 模块和 XLA 相关编译选项。 | 处理 XLA/Triton 接入需要的合法化与重写。 |
| [`CreateTritonPipeline`][target-pipeline] | 模块、设备能力、warp/CTA/stage 配置。 | 按目标运行低层 MLIR pipeline。 |
| [`TranslateLLVMToLLVMIR` 的调用][translate] | 经上述 pass 处理的 MLIR Module。 | 生成原生 `llvm::Module`。 |

`CompileTritonToLLVM` 还检查共享内存等资源要求，整理目标信息，并返回 `TritonWrapperResult`。其中的 `LlvmKernelSource` 持有 LLVMContext 和 Module，其他字段携带线程维度、共享内存及 TMA 等元信息，供后续 kernel 包装使用。

上述 MLIR passes 是 LLVM IR 的产生过程。进入下一节时，处理对象才是已经生成的 `llvm::Module`。

## 3. 变换：链接设备实现并优化 LLVM 程序

[`LinkAndOptimizeModule`][optimize] 接收 LLVM Module、设备能力、调试选项、设备 bitcode 路径、模块链接器和 `TargetMachine`。它先调用目标链接器，根据需要补入设备函数实现，再通过 `PassBuilder` 建立分析和优化 pipeline。

优化依据目标信息与配置改变函数体、控制流和指令组合。函数返回状态，更新后的程序仍保存在输入 Module 中，接着用于目标代码生成。若启用 LLVM dump，编译器可以记录相应 pass 前后的文本。

Triton 生成的独立 LLVM kernel 在提交目标编译前，还会按 XLA 的调用方式整理包装函数、参数与 launch 属性。线程维度、共享内存和参数列表同时保留为编译元信息，后续构造 thunk 时继续使用。

## 4. 消费：从 LLVM 程序到设备可执行代码

### 从 kernel 编译器进入目标代码生成

生成器把 LLVM 编译单元交给 `KernelCompiler::CompileToTargetBinary`。CUDA 的 [`CubinCustomKernelCompiler`][custom-binary] 接收 `LlvmKernelSource`，直接编译或安排到线程池，并通过 future 返回二进制。

它使用的 LLVM 编译回调由 [`CompileToBackendResult`][kernel-compiler] 构造：回调调用 [`GpuCompiler::CompileSingleModule`][single]，后者验证 LLVM Module，再委派目标二进制编译。Triton 独立 kernel 也经过这条调用链。

NVIDIA 实现是 [`NVPTXCompiler::CompileTargetBinary`][nvptx]。通常的 LLVM 输入分支调用 `nvptx::CompileToPtx`，在其中完成链接、LLVM 优化和 PTX 生成，再将 PTX 交给 compilation provider 编译或链接为 cubin。

```text
LlvmKernelSource
  → CubinCustomKernelCompiler::CompileToTargetBinary
  → 注入的编译回调 → GpuCompiler::CompileSingleModule
  → NVPTXCompiler::CompileTargetBinary
      → 链接与 LLVM 优化 → PTX → 设备二进制
```

这条目标代码生成流程以 NVIDIA 为例；其他 GPU 目标使用各自的实现和二进制格式。调试选项还可以让编译器读取指定 LLVM/PTX 文件。

### 编译结果如何进入 GpuExecutable

`CompileSingleModule` 返回 `BackendCompileResult`，回调从中取出二进制。接下来按编译对象的用途组织结果：

| 编译对象 | 消费二进制的位置 | 放入整体程序的方式 |
|---|---|---|
| Triton 等独立 kernel | [`CustomKernelThunk` 的构造][custom-thunk]。 | 将二进制、kernel 名称、参数、线程与 block 维度、共享内存等组织为一次工作，加入 ThunkSequence。 |
| 常量模块 | [`CompileModuleToLlvmIr`][emission] 中的 `constants_binary`。 | 作为后端结果的 binary，直接传给 `GpuExecutable::Create`。 |

[`GpuExecutable::Create`][executable-create] 汇合常量模块二进制、ThunkExecutor、常量信息、输出信息、存储分配及目标设备描述。独立 kernel 的代码由相应 thunk 持有，设备库调用也已包含在执行计划中。设备实现随后完成 PJRT 包装与加载。

运行时调用可执行对象时，才绑定实际 Buffer、kernel 参数和 stream，并提交工作；这部分见 [设备运行时与驱动](../stream-executor/submission-centered-hub.md)。

[`LLVM verifier` 的调用][verify] 和 IR dump 是另外两类消费者：前者返回 IR 合法性检查结果，后者保存阶段文本，用于检查代码生成和优化。

---

本文按 [`upstream-sources.lock`](../../../upstream-sources.lock) 中的 XLA `dcf304bc`、LLVM `75a45c37` 核对。链接指向完整固定提交；内容来自源码检查，未运行 GPU 编译或设备实验。生成和校验方法见[图文维护](../index.md#图文维护)。

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
[executable-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L3038
[kernel-compiler]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2844
