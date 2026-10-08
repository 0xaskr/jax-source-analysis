# jaxlib：MLIR Module

MLIR Module 是这一层传递计算程序的容器。JAX 用 StableHLO 等方言描述外层计算，Pallas 可以另外构造 Mosaic 内核模块；jaxlib 提供 Python 与原生对象的绑定，并把外层 Module 交给编译接口。

[查看 SVG](mlir-module-centered-hub.svg) · [软件栈总览](../overview/overview-software-stack-components-layered.md)

## 1. 定义：表示了什么

[`mlir::ModuleOp`][src-module] 把一组操作组织为完整的程序单元，Python 中的对应对象是 `ir.Module`。它具有一个 Region 和一个 Block，并提供符号表；`IsolatedFromAbove` 约束内部操作不能隐式捕获模块外部的 SSA 值。操作之间通过输入与结果建立数据依赖，函数调用等关系通过符号引用表达。

Module 中的计算含义由所用方言决定。比如 [`stablehlo.add`][src-stablehlo] 定义张量加法及其输入输出约束，`func` 描述函数，`sdy` 描述分片。Pallas 的 Mosaic Module 采用同一种 MLIR 容器，但其中的操作描述内核计算、存储访问和设备相关行为。

| 结构 | 表示的内容 | 约束由谁提供 |
|---|---|---|
| Module、Region、Block | 程序单元及操作的组织方式。 | MLIR 的容器和区域规则。 |
| Operation、Value、Type、Attribute | 运算、数据依赖、值的类型和操作参数。 | 对应方言的操作定义与 verifier。 |
| 函数和符号引用 | 程序入口、被调用函数及其联系。 | 符号表和函数操作约束。 |
| [`ModuleContext`][src-context] | JAX lowering 使用的 Module、平台、符号和回调等上下文。 | JAX 的构造与 lowering 实现。 |

`ModuleContext` 是帮助构造程序的上下文；最终提交的是其中的 Module，以及单独提供的编译选项。运行时数组通过 IFRT Array 和 PJRT Buffer 传递。

## 2. 产生：从 Jaxpr 构造，或从已有表示还原

### JAX 外层程序

[`lower_jaxpr_to_module`][src-lower] 接收 Jaxpr、抽象类型、平台、分片和 effects 等信息。它创建 `ModuleContext`，生成入口函数，再由 `jaxpr_subcomp` 遍历方程，调用每个原语的 lowering 规则。规则接收 MLIR 输入值和原语参数，产生操作与结果值；这些结果继续连接后续方程。

函数返回 `LoweringResult`，其中的 `module` 是外层 MLIR Module。生成后调用 [`operation.verify()`][src-verify] 检查结构、类型和操作约束；启用 Shardy 时，还会进行下一节的 mesh 表示整理。

### Pallas 内核程序

TPU 路径中的 [`pallas_call_tpu_lowering_rule`][src-pallas] 根据 kernel Jaxpr 和 `GridMapping` 调用 [`lower_jaxpr_to_pipelined_module`][src-mosaic]。后者创建一个独立的 `ir.Module`，把内核计算与流水安排写入其中。

外层 Module 表示整个 JAX 函数，Mosaic Module 表示其中一次 Pallas 调用的内核。内核随后被序列化，连同调用配置放入外层 custom call；这两份 Module 的关系在消费一节展开。

### 从已有程序还原

| 函数 | 输入 | 产生的对象及接口返回值 |
|---|---|---|
| [`PyDeserializePortableArtifact`][src-deserialize] | StableHLO 可移植产物字节、MLIR Context。 | 还原 Module，返回 Python Module 对象。 |
| [`PyXlaComputationToMlirModule`][src-from-hlo] | `XlaComputation` 中的 HLO proto。 | 内部调用 `ConvertHloToStablehlo` 构造 Module，最终返回模块文本。 |
| [`PyMhloToStablehlo`][src-mhlo] | MLIR 文本或字节。 | 先解析出 Module，再做方言转换，最终返回 MLIR bytecode。 |

## 3. 变换：在 Module 上改了什么

这些处理发生在不同调用中，使用哪一项取决于模块来源和用途。

| 位置与函数 | 输入及条件 | 处理与结果 |
|---|---|---|
| JAX lowering 中的 [`sdy-lift-inlined-meshes`][src-shardy] | 带分片属性的 Module，且启用 Shardy。 | 将内联 mesh 提升为命名定义，合并相同 mesh，并更新引用；计算仍保存在 Module 中。 |
| [`PyClient::CompileAndLoad`][src-compile] | 调用者提交的 Module。 | 先 `clone()`，后续编译可以修改副本。遇到 Shardy 与旧 GSPMD 属性混用时，按条件回退并导出相应分片表示。 |
| [`PyMhloToStablehlo`][src-mhlo] | 已解析、含 MHLO 操作的 Module。 | 运行 `createHloLegalizeToStablehloPass`，将相应操作转换为 StableHLO。 |
| [`PrepareForExport`][src-prepare] | 将要导出 HLO 的 Module。 | 运行 MHLO 合法化和 StableHLO 导出准备；发现 shape 操作时，再整理和合法化形状计算。 |
| [`_lower_mosaic_module_to_asm`][src-mosaic-serde] | Mosaic TPU Module 和可选的 IR 目标版本。 | 克隆模块，运行 `mosaic-serde` pass，得到适合序列化的模块副本。 |

`verify()` 读取 Module 并检查约束；pass 和克隆则分别改写程序或产生副本。Shardy 的 mesh 整理只是分片处理中的一步，后续分片传播与计算分区由相应编译路径继续完成。

## 4. 消费：交给编译器，或导出程序表示

### 外层 Module 进入 IFRT 和 PJRT

[`PyClient::CompileAndLoad`][src-compile] 把克隆的 Module 放入 [`ifrt::HloProgram`][src-hlo-program]，并构造 IFRT 编译选项。`HloProgram` 在这里持有 `mlir::ModuleOp` 及其所有权，接着由 [`CompileAndLoadIfrtProgram`][src-ifrt] 调用 IFRT 默认 Compiler。

使用 PJRT 的实现中，[`PjRtCompiler::CompileAndLoad`][src-ifrt-compiler] 检查 Program 类型，取出 Module 和 XLA 编译选项，交给 `PjRtLoadedExecutable::Create`；后者调用设备实现的 [`PjRtClient::CompileAndLoad`][src-pjrt-compile]。

```text
外层 MLIR Module + 编译选项
  → jaxlib 克隆 Module，构造 ifrt::HloProgram
  → IFRT PjRtCompiler 取出 Module 与编译配置
  → PJRT 调用具体设备后端的编译与加载接口
```

CPU/GPU 后续路径会导出 HLO 并构造 `HloModule`。公开的 [`ConvertMlirHloToHloModule`][src-to-hlo] 展示了这类转换：先准备 Module、导出 HLO proto，再根据配置调用 `HloModule::CreateFromProto`。具体设备入口怎样调用导出接口，见 [HLO 文档](../xla/hlo-centered-hub.md)。

编译与加载完成后，IFRT 返回 LoadedExecutable，jaxlib 构造 `PyLoadedExecutable` 供 Python 持有。它提供后续执行能力；本次函数调用的数据由执行接口另行传入。

### Mosaic Module 接入外层 custom call

[`lower_module_to_custom_call`][src-custom-call] 组织内核配置，并调用序列化流程。`_lower_mosaic_module_to_asm` 将处理后的 Module 写成 MLIR bytecode；这里函数名中的 `asm` 指向的实际返回值是字节序列。

外层 custom call 的 `backend_config` 携带这份内核程序及配置，操作数和结果连接外层数据流。包含该调用的完整外层 Module 随后进入上述编译接口，由 TPU 专用编译路径处理内核载荷。

### 保存与检查 Module

[`PySerializePortableArtifact`][src-serialize] 接收 Module、目标版本和序列化选项，返回版本化的 StableHLO 可移植产物。普通的 [`module_to_bytecode`][src-bytecode] 则写出 MLIR bytecode。验证接口返回状态或诊断，打印接口返回文本，供开发者检查某个阶段的程序。

---

本页按 [`upstream-sources.lock`](../../../upstream-sources.lock) 中的 JAX `361c43e0`、XLA `dcf304bc`、LLVM `75a45c37` 和 StableHLO `7b1b1578` 核对。链接均指向完整固定提交；结论来自源码检查，未运行编译或设备计算。生成和校验方法见[图文维护](../index.md#图文维护)。

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
[src-pallas]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[src-mosaic]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008
[src-mosaic-serde]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L491
[src-custom-call]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
[src-hlo-program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/hlo/hlo_program.h#L39
[src-ifrt-compiler]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_compiler.cc#L91
[src-pjrt-compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L763
