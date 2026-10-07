# TPU / libtpu：公开接口与 LLO 证据边界

[查看 SVG](llo-boundary-centered-hub.svg) · [定位核心研究对象](llo-boundary-centered-hub.svg#core) · [返回总览](../overview/overview-software-stack-components-layered.svg#tpu_boundary)

本图保留的唯一核心研究对象是 **LLO**。
当前固定开源源码没有提供足以核验它的内部结构、构造者、完整变换链和消费者的证据。
图中左侧为独立的[定义区](llo-boundary-centered-hub.svg#view_definition)，右侧按[产生](llo-boundary-centered-hub.svg#view_production)、[变换](llo-boundary-centered-hub.svg#view_transformation)、[消费](llo-boundary-centered-hub.svg#view_consumption)分成三块，分别记录尚缺少的证据。四区之间没有可核验的 LLO 调用链，因此没有补画内部箭头。
下方[公开接口调用链](llo-boundary-centered-hub.svg#downstream)保留插件加载、编译请求、执行、输出与完成事件，并单独展示 executable 序列化；每条连线只表达公开对象的参数和返回契约。

## 定义：目前能定义到哪一层

对 LLO 本身，尚不能从这些文件落实到具体类、字段、操作集合、类型系统或效果约束。
图中 [`d0`](llo-boundary-centered-hub.svg#d0) 明确保留这一缺口。
下面三种可见对象具有各自的公开定义，但都不足以替代 LLO 的定义。

| 公开对象 | 已知结构或契约 | 不能据此推断的内容 |
| --- | --- | --- |
| [`PJRT_Program`][program] | `code`、`code_size`、`format` 和 `format_size` | libtpu 内部所有 IR 及其顺序 |
| `PJRT_LoadedExecutable` | 编译结果及执行接口使用的不透明句柄 | 句柄内是否及如何保存 LLO |
| [`XLA_TpuProgram`][opaque] | 公开头文件中的不透明类型前向声明 | 具体程序结构，或当前 PJRT 插件的完整实现 |

`PJRT_Program` 的公开格式注释列出 `hlo`、`hlo_with_config`、`mlir`：分别承载相应 HLO proto 或 MLIR 字节码 / 文本。
这是 API 的格式说明，不能直接证明某个实际 libtpu 构建支持全部格式，更不能从格式名推出内部 LLO schema。

## 产生：LLO 的构造者与输入尚未定位

图中 [`llo_producer`](llo-boundary-centered-hub.svg#llo_producer) 保留三个问题：由哪个类或函数构造 LLO、调用发生在哪里、输入输出分别是哪种内部表示。当前源码不足以回答这些问题。

[`make_tpu_client`][plugin] 产生的是插件 Client；[`PJRT_Client_Compile`][compile] 接收公开程序描述并返回不透明 executable。这两个入口不能直接命名为 LLO producer。需要对应 libtpu 构建的实现、格式定义或可核验的阶段产物，才能补上这一块及其调用链。

## 变换：LLO 的四类缺口

图中 [`t0`–`t3`](llo-boundary-centered-hub.svg#t0) 保存研究问题，不把问题画成已确认的 pass pipeline。

| 需要落实的内容 | 当前状态 | 能补足该项的证据 |
| --- | --- | --- |
| 由谁执行变换 | 没有可核验的 LLO pass 类或函数 | 对应构建的实现或版本化接口 |
| 输入输出是什么 | 格式、schema、阶段版本未确认 | 可与构建对应的真实中间产物及解析定义 |
| 按什么规则变换 | verifier、效果和合法重写约束未确认 | 语义规范、验证器或相关实现 |
| 阶段怎样衔接 | 与布局、调度、存储规划、指令选择的关系未确认 | 对应版本的阶段调用链与边界产物 |

任意 `.pb` 文件、MLIR 文本或可执行序列化字节都不能在缺少格式证据时直接标为 LLO。
同样，公开 HLO 的 pass 和 verifier 不能自动用来定义 LLO 的重写规则。

## 消费：LLO 的使用者与结果尚未定位

图中 [`c0`](llo-boundary-centered-hub.svg#c0) 与 [`r0`](llo-boundary-centered-hub.svg#r0) 分别保留使用者和输出格式的缺口。当前没有可核验的具体消费函数、调用位置及 LLO 到下一种表示的转换契约，因此这两张问题卡也不画成已确认的函数调用。

公开 Execute 接收 LoadedExecutable 和 Buffer，Serialize 接收 Executable。它们消费的公开对象可以核对，内部是否读取 LLO、由谁读取及何时读取仍不能从这些 ABI 声明推出。

## 公开接口调用链：从插件到执行结果

本节对应灰色辅助区，处理 Client、程序描述、executable 和 Buffer。它保留可确认的调用与对象流，与上方 LLO 的四个研究区分别阅读。

<a id="产生公开的-client-与编译请求"></a>

### Client 与编译请求

图中 [`p0`](llo-boundary-centered-hub.svg#p0) 对应 [`make_tpu_client`][plugin]：

1. 检查 TPU PJRT 插件是否已加载；需要时从指定路径或 `libtpu.so` 动态加载。
2. 初始化尚未初始化的插件。
3. 通过 `get_c_api_client` 获得供上层提交工作使用的 Client。

这个入口产生 Client，不公开 LLO 的构造函数。
图中 [`d1`](llo-boundary-centered-hub.svg#d1) 与 [`p1`](llo-boundary-centered-hub.svg#p1) 接到 [`PJRT_Client_Compile` 的参数结构与声明][compile]：

| 位置 | 输入 | 输出 |
| --- | --- | --- |
| `PJRT_Client_Compile_Args.client` | 插件 Client | 确定请求的接收端 |
| `program` | `PJRT_Program` 指定的程序与格式 | 编译输入 |
| `compile_options` / `compile_options_size` | 序列化的 `CompileOptionsProto` | 编译配置 |
| `executable` | 由接口填写 | `PJRT_LoadedExecutable*` |

公开头文件还约定 `program` 只需在 Compile 调用期间存活。
接口返回值是 `PJRT_Error*`，与通过参数返回的 executable 分开表达。
编译入口证明请求与结果的契约，不能定位“程序在何处变为 LLO”、该构造函数叫什么或接收哪种内部表示。

<a id="消费执行与序列化分别消费什么"></a>
<a id="执行接口0407"></a>

### 执行接口

[`PJRT_LoadedExecutable_Execute_Args`][execute] 接收已加载程序句柄、执行选项与按设备组织的输入 Buffer。
`argument_lists` 的形状是 `[num_devices, num_args]`；输出通过每设备的 `output_lists` 返回。
这些字段说明执行 API 的数据边界，不是 LLO 的读取接口。

主图将返回结果分成两个节点：

- [`outputs`](llo-boundary-centered-hub.svg#outputs)：输出 `PJRT_Buffer*`，表示结果存储的句柄。
- [`events`](llo-boundary-centered-hub.svg#events)：可选的 `device_complete_events`，在对应设备执行完成时就绪。

调用者可以将 `device_complete_events` 置空；请求该输出时，事件数组长度需与设备输出列表一致。
若 Execute 返回错误，头文件规定不会填充这些事件。
输出列表的存储、返回 Buffer 与返回事件还分别有调用者应履行的释放约定。
因此，得到输出句柄与观察执行完成是两个不同环节，图中用两条独立连线表达。

### 可执行对象序列化

[`PJRT_Executable_Serialize`][serialize] 消费的是 `PJRT_Executable`，返回平台特定的序列化字节及管理这些字节生命周期的对象和 deleter。
公开注释明确说明该序列化不保证跨时间稳定。
这能证明可执行对象存在序列化契约，不能证明字节格式就是 LLO，也不能把它当成稳定、通用的 LLO 交换格式。

图中的 [`executable_input → c2 → serialized_executable`](llo-boundary-centered-hub.svg#executable_input) 从 Serialize 参数开始，和 LoadedExecutable 的执行支路并列。字节只在 `serialized_executable` 存活期间有效；`serialized_executable_deleter` 对拥有这些字节的对象调用一次。

## Pallas / Mosaic 分支如何理解

图中 [`d3`](llo-boundary-centered-hub.svg#d3) 对应另一个可见入口：Pallas 的 TPU lowering 可以产生 Mosaic TPU MLIR。
[`lower_module_to_custom_call`][mosaic] 把该模块接入外层 custom call，之后由后端处理。
**Mosaic TPU MLIR 不是 LLO**，也不是 TPU 机器码；当前公开调用不能补足它在 libtpu 内部如何转化的完整链条。
内外层程序的对应关系见 [JAX 文档中的 Pallas 章节](../jax/jaxpr-centered-hub.md#pallas)。

## 源码版本与证据范围

| 源码树 | 固定提交 |
| --- | --- |
| JAX 工作区分支 | `361c43e072cce92b7d3e9bdaf4dd16db26c49043` |
| XLA 公开头文件 | `dcf304bc5dca1932b99f740b911dbd73631a1a69` |

版本依据为 [`upstream-sources.lock`](../../../upstream-sources.lock)；链接复用 SVG 的固定源码锚点。
公开源文件证据见[组件清单](../../../tools/component_diagram_sources.json)及[overview 清单](../../../tools/overview_flow_sources.json)。
这些提交固定的是可见源码，不等于已经锁定或核验某个 libtpu 二进制的私有实现。

本页只进行源码检查：没有进行 TPU 模拟、离线 TPU 编译或真实 TPU 执行。
后续若结合真实产物，需要记录 libtpu 构建、输入程序、编译选项与产物格式；源码和运行二进制不匹配时标记 `VERSION-SKEW`。
在这些证据补足之前，LLO 的定义、产生、变换与消费仍按上述缺口保留。

继续阅读：[PJRT Buffer](../pjrt/buffer-centered-hub.md) · [Pallas 内外层程序](../jax/jaxpr-centered-hub.md#pallas) · [HLO](../xla/hlo-centered-hub.md) · [组件索引](../index.md)。

[program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L739
[opaque]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/tpu/tpu_ops_c_api.h#L39
[plugin]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_bridge.py#L205
[compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L756
[execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L2051
[serialize]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L2258
[mosaic]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
