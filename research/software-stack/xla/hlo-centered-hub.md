# XLA：围绕 HLO 的程序结构与编译流转

[查看 SVG](hlo-centered-hub.svg) · [定位核心概念](hlo-centered-hub.svg#core) · [返回总览](../overview/overview-software-stack-components-layered.svg#component_xla)

本文对应图中的唯一核心概念 **HLO / HloModule**：XLA 用它表示待优化和编译的程序。
图中的 pass、后端入口、存储规划与可执行对象围绕这一程序表示展开。
CPU LLVM IR、GPU LLVM IR 在各自组件文档中继续说明。

图区导航：[定义](hlo-centered-hub.svg#view_definition) · [产生](hlo-centered-hub.svg#view_production) · [变换](hlo-centered-hub.svg#view_transformation) · [消费](hlo-centered-hub.svg#view_consumption) · [后续结果](hlo-centered-hub.svg#downstream)。左侧定义块保持静态；右侧对象链从已有 HLO 出发，按需变换后交给并列消费者，也保留直接观察的旁路。后端消费内部的调度后变换通过跨区箭头展开。

## 定义：HloModule 表达什么

[`HloModule`][module] 管理入口与嵌套 computation，并持有模块配置。
[`HloComputation`][computation] 组织指令与根结果；[`HloInstruction`][instruction] 通过操作数和使用者关系连接计算。
入口 computation 给出整个模块的调用边界和输出；调用、控制流及 fusion 可以引用嵌套 computation。

| 结构 | 语义与约束 | 图中位置 |
| --- | --- | --- |
| Module / Computation | 模块配置、入口、嵌套计算与根结果 | [`d0`](hlo-centered-hub.svg#d0) |
| Instruction / Opcode | 操作码、操作数、结果 shape、专属属性、控制依赖 | [`d1`](hlo-centered-hub.svg#d1) |
| Shape / Layout / Sharding | 类型、形状、存储布局和分片等约束 | [`d2`](hlo-centered-hub.svg#d2) |
| 合法性 | 所处阶段及 verifier 配置要求的不变量 | [`d3`](hlo-centered-hub.svg#d3) |

`HloModule` 是程序对象；`HloModuleProto` 是它的序列化描述。
两者都不承载运行时设备 Buffer 的实际内容。
数据依赖之外还存在控制依赖、效果及后端约束，因此不能仅凭纯数据 DAG 判断重写是否合法。

## 产生：从编译输入构造模块

产生区 [`p0`](hlo-centered-hub.svg#p0) 对应 [`ConvertMlirHloToHloModule`][import]：

1. 接收 `mlir::ModuleOp` 和 `MlirToHloConversionOptions`。
2. 调用 `ConvertMlirHloToHlo` 生成 `HloProto`。
3. 从 proto 构建默认 `HloModuleConfig`，再导入 MLIR 模块属性中的配置。
4. 调用 `HloModule::CreateFromProto`，返回 `StatusOr<unique_ptr<HloModule>>`。

这条链解释 MLIR 与 HLO 的表示转换，不能把输入 MLIR 模块与输出 `HloModule` 当成同一对象。
与上游输入的衔接见 [jaxlib 的 MLIR Module 文档](../jaxlib/mlir-module-centered-hub.md)。

[`p1`](hlo-centered-hub.svg#p1) 单独表示 [`CreateFromProto`][from-proto] 的构造入口：输入已有 `HloModuleProto` 与 config，重建计算、指令及引用关系。
[`p2`](hlo-centered-hub.svg#p2) 则保留直接组织模块的方式：构造模块，再通过 [`AddEntryComputation`][entry] 等接口添加已经构建的计算。
这些是不同构造入口，无须依次经过；它们汇入 [`produced`](hlo-centered-hub.svg#produced) 表示具体程序实例，其结构遵守定义区的 [`core`](hlo-centered-hub.svg#core)。

## 变换：由配置决定的 pass 与挂点

[`HloPassPipeline::RunImpl`][pipeline] 接收模块和 `execution_threads`，读取 debug options，再调用内部 pass 调度。
固定源码中同时有接受 `HloModule*` 与 `unique_ptr<HloModule>&` 的入口，适配原地更新及允许替换模块的情形。
调用结果为 `StatusOr<bool>`：错误与“是否改变”分别表达，变换后的程序保留在模块对象中。

| 变换入口 | 输入与规则 | 结果 |
| --- | --- | --- |
| [`AlgebraicSimplifier`][simplifier] | HLO 与简化选项；按操作语义和布局相关条件重写 | 更新 HLO，并报告是否改变 |
| [`HloDCE`][dce] | HLO 与死参数等选项；移除满足删除条件的无用指令、计算 | 更新 HLO，并报告是否改变 |
| [`Clone`][clone] | 模块与复制选项 | 新的模块及计算对象 |
| [`ReplaceComputations`][replace] | 当前模块与计算替换映射 | 重接 computation 的使用关系 |

主图 `produced → transform_input → t0 → after` 表示 pipeline 的调用与结果，`t0 → t1 / t2 → after` 展开所配置 pass。
`AlgebraicSimplifier` 和 `HloDCE` 是可展开的实例，虚线表示按配置进入；图没有规定所有平台都执行同一份 pass 列表。
[`after`](hlo-centered-hub.svg#after) 仍是 HLO，表示变换后的状态，不增加第二个核心概念。
复制及计算替换由 `transform_input → t3 → after` 单独表达；随后 `after → consume_input` 将变换结果交给消费者。

### PRE / POST_SCHEDULER 的实际位置

[`register_hlo_module_transformation`][registration] 注册 `(bytes) -> bytes | None` 回调。
输入是序列化 `HloModuleProto`；返回 bytes 表示提交更新后的模块，返回 `None` 表示保留当前程序。
CPU 使用直接注册入口，插件后端经 PJRT C API 扩展注册，能否使用取决于相应支持。

[`hook`](hlo-centered-hub.svg#hook)、[`callback`](hlo-centered-hub.svg#callback) 与 [`hook_result`](hlo-centered-hub.svg#hook_result) 表示调用阶段关联与回调往返，不是所有后端在 `RunBackend` 之前统一完成的一段顺序流程。
固定 CPU 源码把 [`PRE_SCHEDULER`][cpu-pre] 加入 HLO pass pipeline；[`POST_SCHEDULER`][cpu-post] 在调度后运行，随后才创建 `BufferAssignment`。
GPU 在自己的 [`post-scheduler-xla-transforms` pipeline][gpu-post] 调用该挂点，后面仍有其他处理。
图中 `t0 → hook` 专指 CPU pipeline 内的 PRE 调用，回调后接续原 pipeline；`c0 / c1 → hook` 则对应后端调度后的 POST，回调后接续该后端的后续处理与规划。由此应按目标后端核对挂点位置，不能仅按名称推出后续阶段的完整顺序。

## 消费：编译、验证和序列化

[`consume_input`](hlo-centered-hub.svg#consume_input) 接受变换后的 HLO，也接受可直接观察或已经满足目标后端前置条件的已有程序。`produced → consume_input` 不表示任何未优化、未合法化的 HLO 都能直接进入目标代码生成。

目标编译有两个并列消费者：

- [`c0`](hlo-centered-hub.svg#c0)：HLO 交给 [`CpuCompiler::RunBackend`][cpu-backend]，进入 CPU 后端并返回可执行结果。
- [`c1`](hlo-centered-hub.svg#c1)：HLO 与 GPU 目标信息交给 [`GpuCompiler::RunBackend`][gpu-backend]，进入 GPU 代码生成及执行计划组织。

[`plan`](hlo-centered-hub.svg#plan) 表示目标后端内部的调度和存储规划。
[`BufferAssignment`][assignment] 描述逻辑值如何安排存储分配，是编译计划；执行时的设备 Buffer 属于另一个生命周期。
后端产物继续由 provider 包装到上层可执行对象，运行时输入数据不会沿图中的程序编译箭头流转。

[`c2`](hlo-centered-hub.svg#c2) 和 [`c3`](hlo-centered-hub.svg#c3) 保留与编译并列的观察消费者；它们直接消费已有 HLO，不要求先跑目标后端：

| 消费者 | 输入 | 输出与能说明的内容 |
| --- | --- | --- |
| [`HloVerifier`][verifier] | HLO 与验证配置 | 状态或诊断，检查选定不变量 |
| [`ToString`][text] | HLO 与打印选项 | 可阅读的 HLO 文本 |
| [`ToProto`][proto] | 当前模块 | 供传输、保存或再构造的程序描述 |

验证通过不等于数值正确性或设备执行已验证；文本与 proto 也不等于运行轨迹。
追踪变换时，应同时记录输入阶段、目标配置以及变换前后模块，避免把不同后端的 dump 直接当成相邻阶段。

## 源码版本与证据范围

本页使用 SVG 内嵌 `source_anchors` 对应的固定提交链接：

| 源码树 | 固定提交 |
| --- | --- |
| XLA | `dcf304bc5dca1932b99f740b911dbd73631a1a69` |
| JAX 工作区分支 | `361c43e072cce92b7d3e9bdaf4dd16db26c49043` |

版本依据为 [`upstream-sources.lock`](../../../upstream-sources.lock)。
本地引用文件由[组件证据清单](../../../tools/component_diagram_sources.json)及[overview 证据清单](../../../tools/overview_flow_sources.json)约束。
这里确认的是固定源码的接口、调用与对象关系，没有据此声称 CPU 或 GPU 执行结果，也没有将该 JAX 分支的扩展接口视为任意版本都具备的能力。

继续阅读：[CPU LLVM IR](cpu-llvm-ir-centered-hub.md) · [GPU LLVM IR](gpu-ir-centered-hub.md) · [PJRT Buffer](../pjrt/buffer-centered-hub.md) · [组件索引](../index.md)。

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
