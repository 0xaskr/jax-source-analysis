# XLA：HLO（HloModule）

HLO 是 XLA 内部表达张量计算程序的中间表示。`HloModule` 把运算、数据依赖和控制流组织在一起，并携带形状、分片、布局及编译配置。XLA 在这一层逐步确定计算如何实现，再交给设备后端生成代码和执行计划。

[查看 SVG](hlo-centered-hub.svg) · [软件栈总览](../overview/overview-software-stack-components-layered.md)

## 1. 定义：表示了什么

[`HloModule`][module] 持有一个入口计算和其他被引用的计算。每个 [`HloComputation`][computation] 包含一组 [`HloInstruction`][instruction]，并指定根指令作为结果。模块的入口计算定义程序输入与输出，其他计算可以用于函数调用、循环、条件分支或 fusion 的内部实现。

| 结构 | 表示的内容 |
|---|---|
| `HloModule`、`HloModuleConfig` | 完整程序、入口计算、编译配置和输入输出布局等约定。 |
| `HloComputation` | 一段由指令组成的计算，以及它的参数和根结果。 |
| `HloInstruction` | 操作码、操作数、结果形状、属性及被调用的计算。`operands` 和 `users` 连接数据依赖。 |
| Shape、Layout、Sharding | 值的元素类型和形状、内存排列、设备间的分布方式。 |
| 控制依赖和副作用 | 计算的额外顺序要求，以及可能影响外部状态的行为。 |

[`HloVerifier`][verifier] 根据验证配置检查模块中的不变量，部分检查对布局敏感。一个变换在某阶段是否合法，既取决于运算语义，也取决于该阶段已确定的布局、分片和调用约定。

## 2. 产生：从前端表示构造 HLO

### 从 MLIR 导入

[`ConvertMlirHloToHloModule`][import] 接收 `mlir::ModuleOp` 和 `MlirToHloConversionOptions`，调用 `ConvertMlirHloToHlo` 导出 `HloProto`，构建并补充 `HloModuleConfig`，最后调用 [`HloModule::CreateFromProto`][from-proto]。输出是新的 `HloModule`，其中的计算和指令已按 HLO 的对象关系组织。

具体设备入口还会补充设备分配和编译配置。例如 CPU 的 [`SetupMlirCompilation`][cpu-mlir] 调用 `MlirToXlaComputation`，先得到 `XlaComputation` 中的 HLO proto；`PjRtCpuClient` 的编译流程再通过 [`CreateFromProto` 调用][cpu-create]构造模块，并由 [`JitCompile`][cpu-jit] 进入 HLO passes 和后端编译。

```text
前端 MLIR Module
  → 导出 HLO proto，并准备编译配置
  → HloModule::CreateFromProto
  → HloModule、HloComputation 与 HloInstruction
```

这里的前端输入来自 jaxlib/IFRT/PJRT 传递的 Module。上游的 `ifrt::HloProgram` 包装的是 MLIR Module，HLO 对象在导入阶段才构造，参见 [jaxlib 文档](../jaxlib/mlir-module-centered-hub.md)。

### 从序列化描述或计算对象构造

`CreateFromProto` 也可以直接接收已有 `HloModuleProto` 和配置，重建计算、指令与引用关系。另一种入口是创建 `HloModule`，再用 [`AddEntryComputation`][entry] 等接口加入已经构建的 `HloComputation`。这两类入口常用于还原程序、测试或编译器内部构造。

## 3. 变换：逐步确定计算的实现

[`HloPassPipeline::RunImpl`][pipeline] 接收模块和 `execution_threads`，读取调试选项，按配置调用各个 pass。返回的 `StatusOr<bool>` 表达错误或“是否发生修改”，变换结果保留在模块中；接收 `unique_ptr<HloModule>&` 的重载还允许替换整个模块。

### 运算重写、分片、融合和布局

| 变换 | 输入与规则 | 改变了什么 |
|---|---|---|
| [`AlgebraicSimplifier`][simplifier] | HLO 指令、代数简化选项及布局条件。 | 按允许的等价规则替换局部计算，减少或重组运算。 |
| [`HloDCE`][dce] | 指令使用关系、根结果和副作用等约束。 | 删除无用指令或计算，按选项处理死参数。 |
| 分片与分区 | 全局计算、分片要求和设备配置。 | 确定各设备负责的计算，并按需要加入通信。 |
| fusion | 可合并的运算及目标后端的代价与实现约束。 | 将多条运算组织为后续生成 kernel 的计算单元。 |
| 布局分配 | 形状、目标操作要求和输入输出约定。 | 确定张量在内存中的排列，必要时引入转换。 |

分片、fusion 和布局的具体 pass 及顺序由后端选定，见 [CPU 的 HLO pipeline][cpu-passes]和 [GPU 的 HLO pipeline][gpu-passes]。`HloModule` 仍表示计算程序，但越靠近代码生成，其中的实现约束越具体。

### 复制、替换和自定义变换

[`Clone`][clone] 复制模块及其中的计算；[`ReplaceComputations`][replace] 根据替换表更新计算的使用关系。它们用于编译器内部组织程序，输入是已有模块与相应选项或映射，输出是新模块或修改后的当前模块。

本工作区 JAX 分支还提供 [`register_hlo_module_transformation`][registration]。回调接收序列化 `HloModuleProto`，返回新的字节时更新程序，返回 `None` 时保留原模块。CPU 在 [调度前的 pipeline][cpu-pre] 和 [调度后的 pipeline][cpu-post] 调用对应变换；调度后的调用位于 BufferAssignment 之前。GPU 也有自己的 [调度后调用位置][gpu-post]。使用这类接口时，需要按后端确认它所在的编译阶段。

## 4. 消费：生成代码与执行计划

### 设备后端

[`CpuCompiler::RunBackend`][cpu-backend] 和 [`GpuCompiler::RunBackend`][gpu-backend] 消费已经完成相应 HLO 处理的模块，同时读取目标设备信息和编译选项。它们继续安排调度、存储和代码生成，返回后端 Executable。

调度确定计算的执行顺序；[`BufferAssignment`][assignment] 记录各个值使用哪些分配和切片，供代码生成与运行时按照同一约定访问存储。随后，后端为需要的计算生成 LLVM IR 或选择设备库实现，并组织 thunk 执行计划。

| 后端 | HLO 的主要去向 | 编译结果 |
|---|---|---|
| CPU | host kernel、嵌套计算函数及库调用。 | 函数库、存储规划、常量和 thunk 等组成 `CpuExecutable`。 |
| GPU | 目标 kernel、Triton 生成路径及设备库调用。 | 设备代码、存储规划、常量和 thunk 等组成 `GpuExecutable`。 |

设备实现随后将这些产物包装并加载为 PJRT 可执行对象。运行时调用它们时，才把本次输入 Buffer 与已确定的程序和存储安排组合起来。代码生成细节分别见 [CPU LLVM IR](cpu-llvm-ir-centered-hub.md) 和 [GPU LLVM IR](gpu-ir-centered-hub.md)。

### 验证、打印和序列化

[`HloVerifier`][verifier] 读取模块并返回验证状态或诊断；[`ToString`][text] 输出 HLO 文本；[`ToProto`][proto] 生成可保存、传输和重新构造的程序描述。这些接口也能直接处理已有 HLO，用于检查 pass 前后的变化。

---

版本依据为 [`upstream-sources.lock`](../../../upstream-sources.lock)：XLA `dcf304bc`，JAX 工作区分支 `361c43e0`。源码链接使用完整固定提交。本文只核对源码中的对象和调用，未运行 CPU/GPU 编译或设备计算。生成和校验方法见[图文维护](../index.md#图文维护)。

[module]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L95
[computation]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_computation.h#L87
[instruction]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_instruction.h#L232
[import]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6232
[from-proto]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L570
[entry]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L111
[pipeline]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/pass/hlo_pass_pipeline.cc#L298
[simplifier]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/transforms/simplifiers/algebraic_simplifier.h#L457
[dce]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/transforms/simplifiers/hlo_dce.h#L40
[clone]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L165
[replace]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L152
[registration]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_transform.py#L48
[cpu-pre]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1178
[cpu-post]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1833
[gpu-post]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L3286
[cpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2139
[gpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2943
[assignment]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/buffer_assignment.h#L476
[verifier]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/hlo_verifier.h#L485
[text]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L505
[proto]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L554
[cpu-mlir]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L276
[cpu-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L989
[cpu-jit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L764
[cpu-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1183
[gpu-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2304
