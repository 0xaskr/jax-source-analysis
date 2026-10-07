# jaxlib：MLIR Module 的绑定、编译提交与导出

本文对应 [jaxlib 内部架构 SVG](mlir-module-centered-hub.svg)，承接 [software-stack overview](../overview/overview-software-stack-components-layered.md)。图中唯一核心概念是 [MLIR Module（StableHLO）](mlir-module-centered-hub.svg#core)；Module 是程序容器，StableHLO 定义其中相应操作的语义。

图左侧定义块说明 Module 的静态结构；右侧产生、变换和消费三个块通过实际对象与调用关系相连。跨区箭头也展开消费调用内部的 clone 和导出 pass，所以四个视角不是必须依次完成的流水线。灰色区域展示消费结果的包装与后续表示。

图区导航：[定义](mlir-module-centered-hub.svg#view_definition) · [产生](mlir-module-centered-hub.svg#view_production) · [变换](mlir-module-centered-hub.svg#view_transformation) · [消费](mlir-module-centered-hub.svg#view_consumption) · [后续结果](mlir-module-centered-hub.svg#downstream)。节点归属区分 JAX、jaxlib、IFRT 与 XLA 实现。

## 1. 定义：容器、方言与绑定各自负责什么

| 对象或约束 | 具体定义位置 | 在本图中的含义 |
|---|---|---|
| `mlir::ModuleOp` / Python `ir.Module` | [`BuiltinOps.td` 中的 ModuleOp][src-module] | 顶层容器具有单个 graph region 和单个 block，可持有操作；绑定让 Python 构造和访问对应原生对象。 |
| `SymbolTable / IsolatedFromAbove` | [`ModuleOp` traits][src-module] | 管理符号关系，并约束内部操作不得隐式捕获 Module 外定义的 SSA 值。 |
| Operation、类型、属性与 SSA 结果 | [StableHLO `AddOp` 定义实例][src-stablehlo] | 具体操作规定输入、结果和语义约束；Module 本身不规定所有内部计算语义。 |
| JAX `ModuleContext` | [`ModuleContext`][src-context] | 保存 Module 及 JAX lowering 所需的平台、符号、回调等上下文；它与 MLIR 基础 Context 的职责不同。 |
| 模块验证 | [`operation.verify()` 调用位置][src-verify] | 检查所构造 IR 的结构与语义约束；输入是模块，成功只说明验证通过。 |

因此，“MLIR Module（StableHLO）”说明本页选择的程序载体及常见方言，不要求 Module 只包含 StableHLO。`func`、`sdy` 及 custom call 等可以按对应路径共存，jaxlib 的绑定职责也不等于独占这些方言的实现。

<a id="2-产生三个入口返回形式各不相同"></a>

## 2. 产生：构造、还原与辅助解析

| 来源与图中节点 | 具体函数 | 输入 | 输出与注意点 |
|---|---|---|---|
| JAX lowering，[`p0`](mlir-module-centered-hub.svg#p0) | [`lower_jaxpr_to_module`][src-lower] 与 [`ModuleContext`][src-context] | Jaxpr、输入输出类型、平台、分片和效果等上下文。 | `LoweringResult.module`；按配置完成模块 pass 后，成为编译输入。 |
| 可移植产物反序列化，[`p1`](mlir-module-centered-hub.svg#p1) | [`PyDeserializePortableArtifact`][src-deserialize] | StableHLO portable artifact 字节与 MLIR Context。 | 反序列化后包装为 Python Module；失败返回错误。 |
| HLO 兼容入口，[`p2`](mlir-module-centered-hub.svg#p2) | [`PyXlaComputationToMlirModule`][src-from-hlo] | `XlaComputation`。 | 内部构造 StableHLO Module，函数最终返回打印文本；不是直接返回 Python Module 对象。 |
| 辅助转换内部解析，[`mhlo_input`](mlir-module-centered-hub.svg#mhlo_input) | [`PyMhloToStablehlo`][src-mhlo] | 模块文本或字节。 | 先得到含 MHLO 操作的 Module，再交给该接口内部的合法化 pass。 |

JAX lowering 与反序列化的对象汇入 [`produced`](mlir-module-centered-hub.svg#produced)，再由 [`consume_input`](mlir-module-centered-hub.svg#consume_input) 选择接口；这条直接消费路径不要求先做独立变换。HLO 兼容入口则沿 `p2 → compat_module → compat_text` 返回文本。若后续代码需要 Module 对象，不能把这一文本结果直接当作原生 `mlir::ModuleOp` 使用。

## 3. 变换：标明发生在哪个调用内部

| 位置 | 输入及调用条件 | 所做处理 | 输出或改变 |
|---|---|---|---|
| JAX lowering 的 [Shardy 模块 pass][src-shardy]，[`t1`](mlir-module-centered-hub.svg#t1) | 已生成的 Module；启用对应 Shardy 配置时。 | 运行 `builtin.module(sdy-lift-inlined-meshes)`。 | 仍为 Module；分片表示处理发生在进入本图编译绑定之前。 |
| [`PyClient::CompileAndLoad`][src-compile] 内部，[`t0`](mlir-module-centered-hub.svg#t0) | 调用者提交的 `mlir::ModuleOp`。 | `module.clone()`，将独立副本交给后续处理，并允许该副本被原地修改。 | 编译流程持有的克隆 Module；调用者的原模块保留。 |
| [`PyMhloToStablehlo`][src-mhlo] 内部，[`t2`](mlir-module-centered-hub.svg#t2) | 前面已经解析的 Module。 | 运行 MHLO → StableHLO legalization。 | 变换结果仍是 Module；接口随后写成普通 MLIR bytecode，并非所有编译的必经步骤。 |
| XLA 的 [`PrepareForExport`][src-prepare]，[`t3`](mlir-module-centered-hub.svg#t3) | HLO 导出流程中的 Module。 | 组织导出前 pass；按需要处理 shape 操作等。 | 被准备过的 Module，后续导出继续构造 HLO。 |

启用相应 Shardy 路径时，图中的 `p0 → t1 → produced` 展开 lowering 内部处理；否则由 `p0 → produced` 直接返回 Module 实例。静态 [`core`](mlir-module-centered-hub.svg#core) 不充当运行阶段。这里展示一个具体模块 pass 调用位置，不代表全部 Shardy pipeline。

两处嵌套顺序尤其需要保留：**先进入 `PyClient::CompileAndLoad`，再在函数内部 clone；先进入 `ConvertMlirHloToHloModule` 的导出流程，再在其内部调用 `PrepareForExport`。** 图中把内部步骤展开，是为了说明处理位置，不将它们改成调用者必须先自行完成的前置任务。

## 4. 消费：编译提交、HLO 导出与对象返回

### 4.1 从 Module 到 IFRT 编译接口

图中 `produced → consume_input → c0 → t0 → after → ifrt_program` 对应以下对象路径：

```text
Module + CompileOptions
  → PyClient::CompileAndLoad
      → clone() 得到独立 Module
      → IFRT HloProgram(Module)
      → IFRT Compiler::CompileAndLoad
```

[`PyClient::CompileAndLoad`][src-compile] 接收 Module、设备及编译选项，处理后将副本放进 `ifrt::HloProgram`，交给 `CompileAndLoadIfrtProgram`。后者在 [IFRT 调用位置][src-ifrt] 调用默认 Compiler。可以对照图中的 [编译绑定](mlir-module-centered-hub.svg#c0)、[克隆副本](mlir-module-centered-hub.svg#after) 和 [IFRT Program](mlir-module-centered-hub.svg#ifrt_program)。

`HloProgram` 是此处的 IFRT Program 实现名称。这个构造位置明确传入 `mlir::ModuleOp` 的 owning reference，所以名字并不证明对象已经变成 XLA `HloModule`。

### 4.2 下游公开 HLO 转换

图中 `ifrt_program → c1 → t3 → r1` 聚合下游编译路径里的 Module → HLO 转换，并展开导出内部步骤；它不表示 IFRT 公共接口要求每一种 provider 使用同一套内部转换。

[`ConvertMlirHloToHloModule`][src-to-hlo] 的输入是 Module 和转换选项，内部先调用 `ConvertMlirHloToHlo`。后者在导出前调用 [`PrepareForExport`][src-prepare]，再构造 HLO proto；外层函数根据 proto 和配置调用 `HloModule::CreateFromProto`，返回 `HloModule`。

所以 [导出准备](mlir-module-centered-hub.svg#t3) 的结果仍是 Module，而 [HloModule](mlir-module-centered-hub.svg#r1) 是继续导出与构造后的结果。后续 HLO pass 和目标后端见 [XLA / HLO 文档](../xla/hlo-centered-hub.md)。

### 4.3 编译结果沿接口返回

图中 `ifrt_program → r0` 从 IFRT 编译接口返回 [Python 可执行包装](mlir-module-centered-hub.svg#r0)。[`CompileAndLoadIfrtProgram`][src-ifrt] 获取编译结果并构造 `PyLoadedExecutable`。这条返回关系聚合被调用方完成编译与加载后的结果，不是绕过下游编译的另一种实现。

返回的可执行对象用于后续提交数组参数。Module、HloModule 和可执行对象分别处于不同阶段；数组输入输出另走 [IFRT Array](../ifrt/array-centered-hub.md) 与 [PJRT Buffer](../pjrt/buffer-centered-hub.md) 通路。

<a id="5-辅助消费接口与表示边界"></a>

### 4.4 并列的序列化、验证与观察接口

[`consume_input`](mlir-module-centered-hub.svg#consume_input) 也分别连接版本化序列化 [`c2`](mlir-module-centered-hub.svg#c2) 和验证、打印、普通 bytecode [`c3`](mlir-module-centered-hub.svg#c3)。这些消费者与编译提交并列；选择一个不要求先调用其他消费者。

| 接口 | 输入 | 输出 | 使用时需要保留的区别 |
|---|---|---|---|
| [`module_to_bytecode`][src-bytecode] | MLIR Module。 | MLIR bytecode 字节。 | 普通模块序列化不自动等同于指定版本的 StableHLO portable artifact。 |
| [`PySerializePortableArtifact`][src-serialize] | Module、目标版本、Shardy 版本及序列化选项。 | 可移植产物字节。 | 这里引用的是接收 Python Module 的重载；版本化序列化与设备代码生成不同。 |
| [`PyDeserializePortableArtifact`][src-deserialize] | 产物字节和 Context。 | Python Module 对象。 | 还原程序表示，不执行程序。 |
| [`operation.verify()`][src-verify] | 构造后的 Module。 | 验证通过或错误。 | 通过验证不意味着已完成后端编译或已得到设备结果。 |

Pallas 的 Mosaic Module 也使用 MLIR 基础设施，但图中提交编译的是包含 custom call 的完整外层 Module；内层 payload、外层 operands 和运行时数组分属不同角色，见 [JAX 文档中的 Pallas 两层程序](../jax/jaxpr-centered-hub.md#pallas)。

<a id="6-固定版本与证据范围"></a>

## 固定版本与证据范围

本页源码链接直接复用 SVG 中的固定提交与行号，版本以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。

| 源码树 | 固定提交 | 本页使用范围 |
|---|---|---|
| JAX / jaxlib | `361c43e072cce92b7d3e9bdaf4dd16db26c49043` | Python lowering、原生绑定、编译提交及序列化接口。 |
| LLVM / MLIR | `75a45c373407c13a44c7abb28a78d891a97fe665` | `ModuleOp` 定义。 |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` | 方言操作定义实例。 |
| XLA | `dcf304bc5dca1932b99f740b911dbd73631a1a69` | MLIR → HLO 导出及其内部准备过程。 |

证据来自本地固定源码和校验过的源码缓存。本文没有执行 CPU 计算、TPU 模拟、离线 TPU 编译或真实 TPU 执行；只说明这些版本中可定位的结构、输入输出和调用关系。

继续阅读：[JAX / Jaxpr](../jax/jaxpr-centered-hub.md) · [IFRT Array](../ifrt/array-centered-hub.md) · [XLA / HLO](../xla/hlo-centered-hub.md) · [software-stack 索引](../index.md)。

[src-module]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/mlir/include/mlir/IR/BuiltinOps.td#L33
[src-stablehlo]: https://github.com/openxla/stablehlo/blob/7b1b15781ccbd770f50c7eef4b0c3e03834649fd/stablehlo/dialect/StablehloOps.td#L831
[src-context]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L804
[src-verify]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1464
[src-lower]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327
[src-deserialize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/mlir.cc#L192
[src-from-hlo]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/mlir.cc#L109
[src-shardy]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1483
[src-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475
[src-mhlo]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/mlir.cc#L130
[src-prepare]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6066
[src-ifrt]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L412
[src-to-hlo]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6232
[src-bytecode]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L603
[src-serialize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/mlir.cc#L176
