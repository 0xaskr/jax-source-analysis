# IFRT：Array 如何组织设备上的数据

IFRT 的 Array 表示一个完整的逻辑数组：它把元素类型、形状、分片和布局，与设备上的数据存储联系起来。数组可以放在单个设备上，也可以分片或复制到多个设备上。“逻辑”指的是从完整数组的角度描述数据，不要求一定跨设备或跨主机。

[配套图](array-centered-hub.svg)按定义、产生、变换、消费展开。本文采用 IFRT 的 PJRT 实现：`PjRtArray` 组织本进程可访问的 `PjRtBuffer`，`LoadedExecutable` 使用这些数组执行程序。全栈位置见[概览](../overview/overview-software-stack-components-layered.md)。

<a id="核心概念与范围"></a>
<a id="definition"></a>

## 1. 定义：一个数组及其设备分片

[`ifrt::Array`][src-array] 继承 `Value`，通过 [`ArraySpec`][src-spec] 描述数组属性。调用方以 `ArrayRef` 持有它；`ArrayRef` 是引用计数句柄。Array 接口要求实现线程安全。

| 内容 | 具体结构或接口 | 表示什么 |
|---|---|---|
| 元素类型与形状 | `ArraySpec.dtype`、`shape`；`Array::dtype()`、`shape()`。 | 完整数组的元素类型与维度。 |
| 分片与放置 | [`Sharding`][src-sharding] 的设备列表、memory kind 和分片规则。 | 各设备保存哪些数据；完全复制时，每个分片都可覆盖完整数组。 |
| 布局 | `ArraySpec.layout`、`Array::layout()`。 | 分片内部的数据排列约束。 |
| 实际存储 | [`PjRtArray`][src-pjrt-array] 持有的 `PjRtBuffers`。 | 当前进程可访问的分片存储，元素为 `shared_ptr<PjRtBuffer>`。 |
| 状态与生命周期 | [`Value::GetReadyFuture()`、`Delete()`、`IsDeleted()`][src-value]。 | 数据何时就绪，以及数组是否仍可使用。 |

例如，一个形状为 `[8, 4]` 的逻辑数组可以整体放在一个设备上，也可以沿第一维分成两个 `[4, 4]` 的分片。Array 保留 `[8, 4]` 的全局视图，底层 Buffer 分别承载实际分片。只有当前进程可以访问的设备，才在这里有可直接提交执行的 Buffer。

<a id="production"></a>

## 2. 产生：把存储和数组描述组合起来

PJRT 实现的共同构造位置是 [`PjRtArray::Create`][src-create]。它接收 client、dtype、shape、sharding、Buffer 列表和布局，构造 `PjRtArray`。带 `xla::PjRtLayout` 的重载会先检查设备与 Buffer 数量等输入约束。

| 来源 | 调用位置与输入 | 如何得到 Array |
|---|---|---|
| 主机数据 | [`PjRtClient::MakeArrayFromHostBuffer`][src-host] 接收主机地址、类型、形状、strides、分片、布局和主机存储生命周期约定。 | 为可寻址设备调用 PJRT 的 `BufferFromHostBuffer`，再将 Buffer 与数组描述传给 `PjRtArray::Create`。数值数组的这条入口支持单设备或完全复制的分片。 |
| 已有单设备数组 | [`AssembleArrayFromSingleDeviceArrays`][src-assemble] 接收 Array 列表及目标类型、形状、分片和复制语义。 | 检查类型、单设备分片和数量后，收集或转移 Buffer 引用，组装为一个逻辑数组。 |
| 程序执行结果 | [`PjRtLoadedExecutable::Execute`][src-execute] 取得 PJRT 返回的输出 Buffer。 | 按每个逻辑输出收集设备分片，补上输出类型、形状、分片和布局，构造 `ExecuteResult.outputs`。 |

主机数据何时可以释放，由 `HostBufferSemantics` 和完成回调约定。ArrayRef 返回时，底层传输或计算仍可能在进行。

当前版本的组装实现对 `kAlwaysCopy` 留有 TODO，实际仍复用 Buffer 引用；需要独立存储时，应核对具体复制路径。

<a id="transformation"></a>

## 3. 变换：改变放置、分片组织或数据解释

这些操作接收已有 Array，并返回新的 Array 或数组列表。它们按调用需要选择，不是执行前必须依次经过的步骤。

| 操作 | 输入与处理 | 输出及约束 |
|---|---|---|
| [`PjRtClient::CopyArrays`][src-copy] | 数组列表、目标设备或内存种类、复制语义；本地主机内路径委派 `PjRtArray::Copy`。 | 返回目标放置下的数组；按操作和复制语义决定复用、复制或转移存储。 |
| [`PjRtClient::RemapArrays`][src-remap] | 数组列表、`RemapPlan` 和复制语义；委派 `PjRtCompatibleClientRemapArrays`。 | 按计划重组输入分片，形成输出数组。 |
| [`DisassembleIntoSingleDeviceArrays`][src-disassemble] | 一个 Array、复制语义，以及请求全部或可寻址分片的选项。 | 返回单设备 Array 列表；请求全部分片而存在不可寻址设备时返回错误。 |
| [`PjRtClient::BitcastArrays`][src-bitcast] | 数组列表、新 `ArraySpec` 和复制语义；逐 Buffer 调用 `Bitcast`。 | 按新类型与形状解释已有字节，构造新 Array。当前实现要求 `kDonateInput`，随后删除原数组。 |

返回新 Array 只说明得到新的数组对象；是否分配新存储，要看对应操作。分片组装、拆分和参数重排可以只改变描述及引用关系。

<a id="consumption"></a>

## 4. 消费：执行程序、读取数据和结束使用

### 4.1 已加载程序如何使用 Array

[`PjRtLoadedExecutable::Execute`][src-execute] 接收 Array 列表和执行选项。它先检查输入实现及分片数量，再提取每个 Array 的可寻址 Buffer：输入原本按“每个数组有哪些分片”组织，PJRT 需要的是“每个设备有哪些参数”。

```text
Array 参数列表
  → 提取并按设备排列 PjRtBuffer
  → 调用 PJRT 的已加载程序
  → 按逻辑输出收集结果 Buffer
  → PjRtArray::Create
  → ExecuteResult.outputs
```

输出 Array 的构造复用第二章的机制。IFRT 的 Compiler 和 LoadedExecutable 负责程序的编译与执行接口；Array 负责组织参与运行的数据。程序和数据在 Execute 处汇合。

执行可返回状态 Future；`fill_status=true` 时，IFRT 才填充 `ExecuteResult.status`。这个状态和输出对象分开返回，Array 对象存在时不一定已经完成计算。

### 4.2 读取、等待与删除

| 消费者 | 输入与具体处理 | 结果 |
|---|---|---|
| [`PjRtArray::CopyToHostBuffer`][src-to-host] | Array、主机目标地址和可选 strides；从底层 Buffer 读取数据。 | 写入主机存储并返回传输 Future。当前实现直接支持单分片，完全复制数组可选一个分片；其他多分片数组不会在此自动聚合。 |
| [`PjRtArray::GetReadyFuture`][src-ready] | 读取底层 Buffer 的就绪状态；多个 Buffer 时合并 Future。 | 报告数组数据就绪或错误，不额外传输数据。 |
| [`PjRtArray::Delete`][src-delete] | 逐个调用底层 Buffer 的 `Delete`，标记数组已删除。 | 结束数组使用。当前版本返回立即就绪的 OK Future，实际删除完成追踪仍有 TODO。 |

输出对象返回、数据就绪和存储回收是不同的时刻。Buffer 的异步使用与外部引用仍会影响实际存储的回收。

本文依据 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69` 的源码，版本以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准；未运行设备实验。继续阅读：[PJRT 的设备存储](../pjrt/buffer-centered-hub.md)、[运行时执行提交](../stream-executor/submission-centered-hub.md)。

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
