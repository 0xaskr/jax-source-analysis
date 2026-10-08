# TPU provider / libtpu：LLO 的定义、产生、变换与消费

[查看图](llo-boundary-centered-hub.svg) · [返回总览](../overview/overview-software-stack-components-layered.md)

TPU provider 接收程序和编译选项，返回已加载的可执行对象，随后处理带有输入 Buffer 的执行请求。本页沿用总览中的 **LLO（待版本证据）**，把它作为 TPU 私有程序表示的研究项。当前固定公开源码能确认上述接口，尚不能确认 LLO 的结构和内部流转。下面按四个角度说明能定位到哪里，以及缺少哪一段实现。

## 1. 定义：LLO 表示什么

要准确描述 LLO，需要知道它如何表达计算、数据依赖和状态，由哪些操作、类型或字段组成，以及哪些约束决定程序是否合法。当前证据尚不足以为这些内容给出具体的类、结构或格式定义，因此本页不为 LLO 补写推测性的含义。

公开源码中有三种容易与它混淆的对象：

| 对象 | 已知含义 |
| --- | --- |
| [`PJRT_Program`][program] | 编译接口提交程序的容器，以 `code`、`code_size` 和格式信息描述输入。 |
| [`XLA_TpuProgram`][opaque] | 公开头文件中的不透明类型；声明没有给出内部程序结构。 |
| [Mosaic TPU MLIR][mosaic] | Pallas TPU lowering 产生的专用 kernel 表示，通过外层 custom call 接入编译。 |

这些对象各有用途。程序容器和句柄没有定义后端内部的 IR；Mosaic TPU MLIR 则有自己的操作和语义，不能称为 LLO。图中的[定义区](llo-boundary-centered-hub.svg#view_definition)保留了这一区分。

## 2. 产生：程序从哪里进入，LLO 在哪里构造

<a id="产生公开的-client-与编译请求"></a>

[`make_tpu_client`][plugin] 按需加载并初始化 TPU PJRT 插件，取得供上层使用的 Client。这一步建立的是设备接口。程序本身来自 JAX/Pallas 的 lowering，经 jaxlib、IFRT 和 PJRT 进入设备后端，完整调用关系见[总览的编译流程](../overview/overview-software-stack-components-layered.md#23-jaxlibifrt-和-pjrt-如何传递编译请求)。

在公开 C API 中，[`PJRT_Client_Compile`][compile] 接收 Client、`PJRT_Program` 和序列化编译选项，成功时返回 `PJRT_LoadedExecutable`。`PJRT_Program` 的格式注释列出了 HLO proto 与 MLIR 文本或 bytecode；某个 libtpu 构建实际支持哪些格式，仍由该实现决定。

Pallas 的原生 TPU 路径还会提交内核程序。[`pallas_call_tpu_lowering_rule`][pallas-lowering] 根据 kernel Jaxpr 和 GridMapping 构造 Mosaic TPU Module；[`lower_module_to_custom_call`][mosaic] 把它接入外层调用。序列化函数克隆 Module，运行 `mosaic-serde`，将 MLIR bytecode 放入调用配置。[`_lower_mosaic_module_to_asm`][mosaic-serialize] 名称中的 `asm` 在这里指向这份 bytecode。

这些源码能确定交给 TPU 后端的内容。由哪个函数、在什么调用位置、根据哪种内部输入构造 LLO，尚需对应 libtpu 构建的实现或阶段产物。图中的[产生区](llo-boundary-centered-hub.svg#view_production)因此没有把公开 Compile 入口标为 LLO 构造函数。

## 3. 变换：LLO 经历哪些处理

当前没有可核验的 LLO pass 调用链，无法列出固定的变换顺序。落实这一角度需要以下信息：

| 需要说明的内容 | 对应的实现证据 |
| --- | --- |
| 谁在变换 LLO、何时调用 | 变换类或函数，以及调用它的编译步骤。 |
| 变换改变了什么 | 同一构建中可对应的输入输出表示及格式定义。 |
| 按什么规则变换 | 重写条件、合法性检查及需要保持的语义。 |
| 与后续步骤如何衔接 | 调度、存储规划、代码生成等步骤实际读取和返回的对象。 |

这些内容分别放在图中的[变换区](llo-boundary-centered-hub.svg#view_transformation)，没有串成 pass pipeline。公开 HLO 的优化规则和 Mosaic 的序列化处理可以各自定位，但还不能据此推出 LLO 的变换规则。仅有文件扩展名或一份二进制数据，也不足以确定它是哪种内部表示。

## 4. 消费：谁使用 LLO，产生什么结果

对 LLO 本身，还需要定位读取它的函数、使用哪些程序信息，以及返回哪种表示或执行产物。当前证据停留在编译的外部结果：TPU provider 通过 `PJRT_LoadedExecutable` 提供已加载程序的执行能力。内部 LLO 如何参与形成这一结果尚未确定，图中的[消费区](llo-boundary-centered-hub.svg#view_consumption)保留这一段空缺。

<a id="消费执行与序列化分别消费什么"></a>
<a id="执行接口0407"></a>

公开结果的两种用途可以直接核对：

| 使用位置 | 消费的对象 | 产生的结果 |
| --- | --- | --- |
| [`PJRT_LoadedExecutable_Execute`][execute] | 已加载程序、按设备组织的输入 Buffer 和执行选项。 | 输出 Buffer，以及按需请求的设备完成事件。 |
| [`PJRT_Executable_Serialize`][serialize] | `PJRT_Executable`。 | 平台特定的序列化字节；格式不保证跨时间稳定。 |

输出 Buffer 用于持有结果数据，上层可以继续把它们组织为逻辑数组；完成事件用于观察对应设备何时执行完毕。可执行对象的序列化字节则用于保存编译产物，其格式尚未被确认为 LLO。图中消费区的[公开对象流](llo-boundary-centered-hub.svg#view_consumption)只连接这些已知的输入和输出。

本页依据 [`upstream-sources.lock`](../../../upstream-sources.lock) 固定的 JAX `361c43e072cce92b7d3e9bdaf4dd16db26c49043` 与 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69` 源码。此次工作为源码检查，没有运行 TPU 模拟、离线 TPU 编译或真实 TPU 执行；这些源码版本也不代表已核验某个 libtpu 二进制。后续补充 LLO 证据时，需要同时记录 libtpu 构建、输入程序、编译选项和产物格式。

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
