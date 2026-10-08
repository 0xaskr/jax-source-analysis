# PJRT：PjRtBuffer 如何管理设备存储

`PjRtBuffer` 是对设备数据存储的统一抽象。它描述所属设备、内存空间、数据形状与布局，并提供所有权和就绪状态的管理接口。设备实现负责具体存储，调用方通过同一套接口把 Buffer 交给程序执行、复制数据或读取结果。

[配套图](buffer-centered-hub.svg)按定义、产生、变换、消费展开。本文以 PJRT C++ 接口为依据；CPU、GPU、TPU 的具体实现可以不同。全栈位置见[概览](../overview/overview-software-stack-components-layered.md)。

<a id="核心概念与范围"></a>
<a id="definition"></a>

## 1. 定义：数据存在哪里，以及如何使用

[`PjRtBuffer`][src-buffer] 定义在 `xla/pjrt/pjrt_client.h`。它把数据描述、存储归属和生命周期放在一个运行时接口中，支持数组和 tuple；一个句柄不必对应一段连续分配。

| 内容 | 具体接口 | 表示什么 |
|---|---|---|
| 数据描述 | `element_type()`、`dimensions()`、`on_device_shape()`、`layout()`。 | 元素类型、维度及设备上的排列方式。 |
| 动态形状 | `logical_dimensions()`、`logical_on_device_shape()`。 | 实际有效的动态维度；查询可能需要与设备通信。 |
| 存储归属 | `client()`、`device()`、`memory_space()`。 | 管理该存储的运行时、设备和内存空间。 |
| 数据就绪 | [`GetReadyFuture()`][src-ready]。 | 计算或传输何时完成，或是否发生错误。 |
| 所有权 | [`Delete()`][src-delete]、[`ExternalReference`][src-external]、[`ReleaseDeviceMemoryOwnership()`][src-release]。 | 本句柄、外部使用者与底层存储之间的持有关系。 |

Buffer 可以代表尚未完成的输出。形状和存储句柄已经可用时，实际数据仍可能在传输或计算中。调用方应遵守就绪、捐赠和外部访问的约定。

XLA 的 `BufferAssignment` 记录编译期存储规划；PjRtBuffer 承载运行时的实际存储。执行时，设备实现把规划与本次 Buffer 结合起来。

<a id="production"></a>

## 2. 产生：接收输入数据，或准备执行结果

### 2.1 接收主机数据

[`PjRtClient::BufferFromHostBuffer`][src-host] 接收主机地址、元素类型、维度、strides、目标内存空间与布局，以及 `HostBufferSemantics` 和完成回调，返回 `unique_ptr<PjRtBuffer>`。

设备实现根据主机存储语义和自身能力选择复制或零拷贝等方式。接口返回时，数据传输可以尚未完成。`on_done_with_host_buffer` 通知何时不再使用主机输入；输出 Buffer 的数据就绪则通过 `GetReadyFuture()` 观察。

### 2.2 构造程序输出

[`PjRtLoadedExecutable::Execute`][src-execute] 接收按设备组织的输入 Buffer 和执行选项，返回按设备、按结果组织的 Buffer 列表。设备实现为输出准备存储；编译期别名关系及执行时 donation 约定允许时，也可以复用输入存储。

这些输出 Buffer 随后可以直接参与下一次执行，或由 [IFRT](../ifrt/array-centered-hub.md) 按类型、全局形状和分片信息组织成逻辑数组。返回句柄与完成计算可以发生在不同时间。

<a id="transformation"></a>

## 3. 变换：复制、重解释和附加依赖

变换以已有 Buffer 为输入。下面三个接口分别改变数据位置、数据解释或就绪条件，并返回新 Buffer。

| 接口 | 输入与规则 | 输出及所有权变化 |
|---|---|---|
| [`CopyToMemorySpace`][src-copy] | 原 Buffer 和目标 `PjRtMemorySpace`。 | 返回目标空间中的新 Buffer；目标已经是当前空间时返回错误。 |
| [`Bitcast`][src-bitcast] | 新元素类型、维度和设备布局，要求设备上的总字节数相同。 | 只修改解释数据所需的元信息；底层存储交给新 Buffer，原 Buffer 被捐赠。 |
| [`DonateWithControlDependency`][src-dependency] | 原 Buffer 和额外的依赖 Future。 | 捐赠原 Buffer，返回内容相同的新 Buffer；原数据和额外依赖都就绪时，新 Buffer 才就绪，任一错误都会传播。默认实现返回 `Unimplemented`。 |

例如，Bitcast 保留数据字节，改变类型与形状；数值类型转换则需要执行相应的计算。调用后能否继续使用原 Buffer，由该接口的所有权规则决定。

<a id="consumption"></a>

## 4. 消费：执行、读取与释放存储

### 4.1 程序使用 Buffer 作为实参和结果

[`PjRtLoadedExecutable::Execute`][src-execute] 把已加载程序与每个设备的 Buffer 参数组合起来，交给设备实现。程序规定计算和输入输出约束，Buffer 提供本次运行的数据。

```text
已加载程序 + 各设备的输入 Buffer + 执行选项
  → 设备实现准备存储、处理依赖并执行
  → 输出 Buffer 列表
  → IFRT 组织逻辑输出数组
```

[`PjRtLoadedExecutable` 的接口契约][src-loaded]说明：输入输出别名关系可能使输入在执行时被捐赠。执行选项还会限制哪些输入允许捐赠。原句柄被捐赠后，调用方必须结束对它的普通使用。

调用方可以请求各设备的执行 Future，也可以查询结果 Buffer 的 `GetReadyFuture()`。前者用于观察执行状态，后者用于观察相应数据是否就绪。

### 4.2 读取和外部访问

| 消费者 | 使用方式 | 结果及约束 |
|---|---|---|
| [`ToLiteral`][src-literal] | 读取 Buffer，写入调用方提供的主机 Literal。 | 返回完成 Future；主机目标需保持有效，直到传输完成。 |
| [`GetReadyFuture`][src-ready] | 观察计算、传输或错误状态。 | 返回可等待状态。删除或捐赠前取得的 Future 仍有效；之后再查询会立即得到错误。 |
| [`AcquireExternalReference`][src-external] | 取得底层存储的外部引用和不透明设备指针。 | 保持存储存活；外部读写仍须遵守同步约定。 |

### 4.3 结束使用

[`Delete`][src-delete] 放弃 Buffer 对设备存储的引用，使句柄失效。实际存储会等尚未完成的异步操作和外部引用结束后，再按设备分配规则回收。

[`ReleaseDeviceMemoryOwnership`][src-release] 同样使原句柄失效，但把存储所有权交给返回的 `ExternalReference`。启用等待选项时，它等待相关异步操作完成；不等待时，调用方须先按接口约定完成必要同步，再移交和访问存储。

本文依据 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69` 的源码，版本以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准；未运行设备实验。继续阅读：[IFRT Array](../ifrt/array-centered-hub.md)、[运行时执行提交](../stream-executor/submission-centered-hub.md)。

[src-buffer]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1108
[src-host]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L966
[src-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1457
[src-copy]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1306
[src-bitcast]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1353
[src-dependency]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1366
[src-delete]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1273
[src-release]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1292
[src-loaded]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1390
[src-literal]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1210
[src-external]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1203
[src-ready]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1380
