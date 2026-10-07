# IFRT：Array 与分片执行

本文对应 [IFRT Array 架构图](array-centered-hub.svg)，解释逻辑数组如何映射到设备分片、参与执行并返回结果。
全栈位置见 [overview 图](../overview/overview-software-stack-components-layered.svg#ifrt_array)及其[配套说明](../overview/overview-software-stack-components-layered.md)。

图将[定义](array-centered-hub.svg#view_definition)独立成块，[产生](array-centered-hub.svg#view_production)、[变换](array-centered-hub.svg#view_transformation)、[消费](array-centered-hub.svg#view_consumption)由实际 Array 实例连接。四个视角不是四个必经步骤：已有数组既可直接消费，也可先按需改变放置或表示；核心定义卡不参与执行调用链。

<a id="核心概念与范围"></a>

唯一核心概念是 [IFRT Array](array-centered-hub.svg#core)：一个具有 dtype、shape、sharding 和 layout 的逻辑数组。
本文展开固定版本的 PJRT-backed 实现；`PjRtArray`、`PjRtBuffer`、已加载程序和 Future 都围绕 Array 的使用展开。

`ArrayRef` 是引用计数句柄。一个 Array 可以覆盖多个设备分片，本进程实际持有的 Buffer 对应可寻址分片。
`PjRtArray → PjRtBuffer` 表示实现与存储映射；同一逻辑数组参与包装或参数重排，并不意味着每层都复制一次数据。
IFRT 的其他实现、字符串数组专用路径及完整跨主机传输协议不在主图展开范围内。

<a id="definition"></a>

## 定义：逻辑值、分片与存储

| 图中节点 | 定义位置与结构 | 输入输出或约束 |
|---|---|---|
| [Array / ArraySpec](array-centered-hub.svg#d0) | [`Array`][src-array] 的 `array_spec()`、`dtype()`、`shape()`、`sharding()`、`layout()`；[`ArraySpec`][src-spec] 组合相关描述。 | 表达逻辑值及放置、布局约束；Array 接口要求实现线程安全。 |
| [Sharding / DeviceList](array-centered-hub.svg#d1) | [`Sharding`][src-sharding] 描述设备集合、memory kind 和分片方式。 | 全局设备列表与当前进程可寻址设备列表需要分别理解。 |
| [PjRtArray / PjRtBuffer](array-centered-hub.svg#d2) | [`PjRtArray`][src-pjrt-array] 保存 `PjRtBuffers`，元素为 `shared_ptr<PjRtBuffer>`。 | 元信息加分片句柄组成 PJRT-backed Array；Buffer 提供实际存储契约。 |
| [复制、捐赠与就绪](array-centered-hub.svg#d3) | [`Value`][src-value] 定义 `GetReadyFuture()`、`Delete()`、`IsDeleted()`；数组操作另外接收 `ArrayCopySemantics`。 | 句柄有效性、可复用性和异步就绪是不同维度。 |

`ArraySpec` 是数组描述，已加载程序则是消费 Array 的执行对象。
因此，图中 `LoadedExecutableRef` 与 Array 在执行入口汇合，不能将其中任何一个理解成另一个的内部成员。

<a id="production"></a>

## 产生：主机数据、分片组装与结果包装

### 主机数据入口

[`PjRtClient::MakeArrayFromHostBuffer`][src-host] 接收主机地址、dtype、shape、byte strides、sharding、layout 和主机缓冲区生命周期约定。
数值数组路径为可寻址设备调用底层 PJRT 的 `BufferFromHostBuffer`，再构造 Array，见图中 [p0](array-centered-hub.svg#p0) → [已有 ArrayRef](array-centered-hub.svg#produced)。

此实现直接支持单设备或完全复制的 sharding，并要求存在可寻址设备。
因此，“主机值生成 Array”不能据此推广成任意全局分片都由这一函数自动完成切分与分发。
主机缓冲区何时可以释放，由输入语义和完成回调决定；取得 ArrayRef 本身不是释放依据。

### 单设备分片组装

[`AssembleArrayFromSingleDeviceArrays`][src-assemble] 接收单设备 Array 列表、目标 dtype/shape/sharding 与复制语义，返回一个 ArrayRef，见 [p1](array-centered-hub.svg#p1) → [已有 ArrayRef](array-centered-hub.svg#produced)。
实现检查输入 dtype、单设备 sharding、可寻址分片数量等条件，然后将底层 Buffer 组装给 `PjRtArray::Create`。

复用与捐赠会改变 Buffer 引用的持有方式。
该固定版本的组装实现对 `kAlwaysCopy` 仍有 TODO，并复用 Buffer；不能从这个枚举名称推断此路径已经深拷贝分片。

### 执行结果包装

[`PjRtLoadedExecutable::Execute`][src-execute] 在 PJRT 返回后，把按设备组织的 Buffer 转为按逻辑输出组织的 Array。
它使用已记录的输出 dtype、shape、sharding，并结合可用的布局信息调用 [`PjRtArray::Create`][src-create]，形成 `ExecuteResult.outputs`。
layout 优先取编译元信息，缺失时可从结果 Buffer 取得；没有输出 Buffer 时还会查询可执行对象的输出布局，该查询未实现时使用默认布局。
对应消费区中的 [输出包装](array-centered-hub.svg#p2) → [输出 ArrayRef](array-centered-hub.svg#r0)。这是执行消费内部再次构造 Array 的位置；不表示输出数组必须先于本次执行产生。

<a id="transformation"></a>

## 变换：改变放置、分片组织或所有权

[已有 ArrayRef](array-centered-hub.svg#produced)按需进入[操作输入](array-centered-hub.svg#transform_input)，分别选择以下操作，再经[变换结果](array-centered-hub.svg#after)交给[消费输入](array-centered-hub.svg#consume_input)。操作输入与结果卡是接口交接的汇总，不代表新增的 IFRT 类型；四条分支彼此独立。

| 操作及图中节点 | 输入与调用位置 | 结果与当前实现边界 |
|---|---|---|
| [`CopyArrays`][src-copy]，[t0](array-centered-hub.svg#t0) | Array 列表、可选目标设备列表、memory kind、复制语义；入口在 `pjrt_client.cc`。 | 返回 Array 列表；本地主机内路径委派 `PjRtArray::Copy`，跨主机路径依 provider 和配置选择。目标设备数量受检查。 |
| [`RemapArrays`][src-remap]，[t1](array-centered-hub.svg#t1) | `RemapPlan`、Array 列表和复制语义。 | 入口委派 `PjRtCompatibleClientRemapArrays`；图表达按 plan 重组分片的接口，不承诺任意 remap 都零拷贝。 |
| [`DisassembleIntoSingleDeviceArrays`][src-disassemble]，[t2](array-centered-hub.svg#t2) | 已有 Array、复制语义及请求全部或可寻址分片的选项。 | 按 sharding 拆出单设备 Array；请求全部分片但存在不可寻址设备时返回错误。 |
| [`BitcastArrays`][src-bitcast]，[t3](array-centered-hub.svg#t3) | Array 列表、目标 `ArraySpec` 列表和复制语义。 | 当前 PJRT-backed 实现只接受 `kDonateInput`；逐 Buffer 调用 `Bitcast`，构造新 Array，随后删除输入 Array。 |

这些操作影响数组描述、放置或生命周期；它们不等同于对数值做加法、乘法等程序计算。
新 Array 与旧 Array 是否共享、复制或转移存储，应看具体操作和复制语义，不能只看返回了一个新句柄。

<a id="consumption"></a>

## 消费：Array 进入已加载程序

[已有 ArrayRef](array-centered-hub.svg#produced)可以沿直接旁路进入[消费输入](array-centered-hub.svg#consume_input)，也可以先经过选定变换。消费输入按用途分派到执行、主机读取、就绪观察或删除，彼此不构成先后顺序。

| 图中路径 | 对应调用或数据关系 | 应关注的输入输出 |
|---|---|---|
| [已加载程序](array-centered-hub.svg#loaded)与消费输入 → [Execute](array-centered-hub.svg#c0) | `LoadedExecutableRef` 与 ArrayRef 参数在执行入口汇合。 | 检查 `PjRtCompatibleArray` 类型与分片数量；类型和存储映射的定义独立留在左栏。 |
| Execute → [参数提取](array-centered-hub.svg#extract) → [PJRT Execute](array-centered-hub.svg#call) | 把“每数组一组分片”重排为“每设备一组参数”。 | 生成 `PjRtBuffer*[][]` 并转发执行选项，容器重排本身不保证发生数据搬移。 |
| PJRT → [输出 Buffer](array-centered-hub.svg#buffers) → [包装](array-centered-hub.svg#p2) → [outputs](array-centered-hub.svg#r0) | 按逻辑输出聚合设备分片。 | ArrayRef 可以在设备完成前返回。 |
| PJRT → [执行 status](array-centered-hub.svg#status) | 常规执行路径合并各设备 Future。 | 仅在 `fill_status=true` 时填入对外结果的 `status`，与 Array 的数据就绪接口分开。 |

详细调用位于 [`pjrt_executable.cc` 的 `PjRtLoadedExecutable::Execute`][src-execute]。
该函数还保留单设备 portable execution 分支，调用 `ExecutePortable`；主图的 PJRT Execute 节点概括 provider 执行边界。
本图中的参数重排也不保证搬移数据：需要区分容器组织与设备间传输。

### 主机读取、就绪与删除

消费区并列展开执行之外的三个消费者：

- [`CopyToHostBuffer`][src-to-host]（[c1](array-centered-hub.svg#c1)）接收主机目标地址和可选 strides，返回传输 Future。当前实现直接支持单分片；完全复制数组可以选取一个分片，其他多分片数组不会在此自动全量聚合。
- [`GetReadyFuture`][src-ready]（[c2](array-centered-hub.svg#c2)）在单 Buffer 时直接返回其 Future，多 Buffer 时合并就绪状态。它观察数组数据的就绪或错误，不返回主机数据副本。
- [`Delete`][src-delete]（[c3](array-centered-hub.svg#c3)）逐 Buffer 删除并标记 Array；该版本返回立即就绪的 OK Future，并留有完成追踪 TODO。

因此需要区分三类观察：`ExecuteResult.outputs` 表示结果句柄已返回，`GetReadyFuture` 表示数据就绪或错误，执行 `status` 表示本次执行的状态观察。
其中任何一个都不能被替换为“对象存在，所以设备已经完成”。

[`Value::Delete`][src-value] 的接口允许不追踪实际删除完成的实现立即返回 OK。
在这里，这个返回值尤其不能证明底层显存已经物理回收；共享引用及尚未结束的异步使用仍影响存储生命周期。

## 固定版本与证据范围

源码版本为 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69`，以仓库的 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。
本文复核了 SVG metadata 对应的本地固定源码；源码链接均指向同一 commit。
缓存文件由 [`component_diagram_sources.json`](../../../tools/component_diagram_sources.json) 的大小与 SHA-256 约束，图中节点、命名关系和引用来自对应 SVG。

证据属于源码检查与静态图文核对。本篇没有运行 CPU 计算、TPU 模拟、离线 TPU 编译或真实 TPU 执行，也不以此推断某个已安装二进制的具体行为。

继续阅读：[PJRT Buffer](../pjrt/buffer-centered-hub.md)、[jaxlib Module](../jaxlib/mlir-module-centered-hub.md)、[执行提交](../stream-executor/submission-centered-hub.md)、[软件栈索引](../index.md)。

[src-array]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/array.h#L66
[src-spec]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/array_spec.h#L44
[src-sharding]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/sharding.h#L86
[src-pjrt-array]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.h#L67
[src-value]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/value.h#L35
[src-host]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1038
[src-assemble]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1214
[src-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L835
[src-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L158
[src-copy]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1300
[src-remap]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1750
[src-disassemble]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L353
[src-bitcast]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1756
[src-to-host]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L396
[src-ready]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L626
[src-delete]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L639
