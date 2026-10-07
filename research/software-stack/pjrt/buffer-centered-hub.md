# PJRT：PjRtBuffer 与设备存储生命周期

本文对应 [PJRT Buffer 架构图](buffer-centered-hub.svg)，说明设备存储句柄如何产生、参与执行并经历复制、捐赠与删除。
全栈位置见 [overview 图](../overview/overview-software-stack-components-layered.svg#pjrt_buffer)及其[配套说明](../overview/overview-software-stack-components-layered.md)。

图将[定义](buffer-centered-hub.svg#view_definition)独立放在左侧，用真实 Buffer 实例连接[产生](buffer-centered-hub.svg#view_production)、[可选变换](buffer-centered-hub.svg#view_transformation)和[消费](buffer-centered-hub.svg#view_consumption)。已有 Buffer 可以绕过变换直接消费；就绪、主机读取和释放所有权都是独立接口。

<a id="核心概念与范围"></a>

唯一核心概念是 [PjRtBuffer](buffer-centered-hub.svg#core)：由 PJRT provider 实现的运行时存储对象。
它提供类型、形状、布局、存储归属、就绪状态与所有权操作，供已加载程序消费。
`PjRtLoadedExecutable` 是与 Buffer 配合的程序对象，`Future` 是状态观察对象，二者都不是本模块新增的核心概念。

本文以固定版本的 PJRT C++ 接口为依据；CPU、GPU、TPU provider 共享接口边界，但各自实现分配、执行和回收。
图中的 provider 节点表示分派边界，不能把 CUDA 的内部提交流程推广为所有设备的实现。

`PjRtBuffer` 与 XLA 编译阶段的 `BufferAssignment` 也需要区分：前者是运行时存储句柄，后者是编译期的分配与别名规划。
它们服务于不同阶段，不能因名称都有 Buffer 就视为同一个对象。编译规划的位置见 [HLO 配套说明](../xla/hlo-centered-hub.md)。

<a id="definition"></a>

## 定义：存储描述与归属

[`PjRtBuffer` 的类定义][src-buffer] 位于 `xla/pjrt/pjrt_client.h`。
图中 [d0–d3](buffer-centered-hub.svg#d0) 从四个方面补充核心对象：

| 方面 | 具体接口 | 输入输出及约束 |
|---|---|---|
| 形状与布局 | `element_type()`、`dimensions()`、`on_device_shape()`、`layout()`、`IsTuple()`。 | 接口可表达数组或 tuple；布局说明设备表示，不等同于主机数组布局。 |
| 动态维度 | `has_dynamic_dimensions()`、`logical_dimensions()`、`logical_on_device_shape()`。 | 动态逻辑形状可能需要与设备通信，不能一概视为无开销的静态字段读取。 |
| 归属 | `client()`、`device()`、`memory_space()`。 | 指定运行时与存储空间；接口从 device 归属向 memory space 归属过渡。 |
| 生命周期与状态 | `Delete()`、`IsDeleted()`、`GetReadyFuture()`、`ExternalReference`。 | 有效句柄、数据就绪、外部持有及实际回收分别受约束。 |

`PjRtBuffer` 不保证是“已经完成计算的一段可直接访问内存”。
它可以代表仍在传输或计算中的结果，读取和修改必须遵循对应的就绪与所有权协议。

<a id="production"></a>

## 产生：主机数据与执行结果

### 主机数据入口

[`PjRtClient::BufferFromHostBuffer`][src-host] 接收主机地址、元素类型、维度、可选 byte strides、主机缓冲区语义、完成回调，以及目标 memory space/layout。
成功时返回 `unique_ptr<PjRtBuffer>`，见图中 [p0](buffer-centered-hub.svg#p0) → [已有 Buffer](buffer-centered-hub.svg#produced)。

接口返回句柄时，传输可能仍在进行。
`on_done_with_host_buffer` 与 `HostBufferSemantics` 规定主机输入可以被复用或释放的时机；它们与输出 Buffer 的就绪状态承担不同职责。
具体重载和 provider 支持范围需要核对实现，不能将头文件中的接口声明理解为所有后端均已支持。

### 执行产生结果 Buffer

[`PjRtLoadedExecutable::Execute`][src-execute] 接收按设备组织的参数句柄和执行选项，成功时返回按设备、按结果组织的 `unique_ptr<PjRtBuffer>` 列表。
这些 Buffer 是本次执行结果的存储句柄；上层 [IFRT Array](../ifrt/array-centered-hub.md) 再为它们补充逻辑数组与分片视图。
程序的编译、加载先于这里的 Execute，返回结果 Buffer 本身也不意味着重新编译了程序。

<a id="transformation"></a>

## 变换：复制、重解释、依赖与所有权

[已有 Buffer](buffer-centered-hub.svg#produced)可按需进入[操作输入](buffer-centered-hub.svg#transform_input)。下面三个分支分别产生新的 Buffer，再交给[消费输入](buffer-centered-hub.svg#consume_input)；它们不组成固定流水线。

| 操作及图中节点 | 输入与定义位置 | 输出及所有权变化 |
|---|---|---|
| [`CopyToMemorySpace`][src-copy]，[t0](buffer-centered-hub.svg#t0) | 原 Buffer 与目标 `PjRtMemorySpace`；声明在 `pjrt_client.h`。 | 返回目标空间的新 Buffer；接口注明目标已经是原空间时返回错误。原句柄不会因“复制”而自动变为捐赠状态。 |
| [`Bitcast`][src-bitcast]，[t1](buffer-centered-hub.svg#t1) | 新元素类型、维度及设备布局。 | 设备字节数必须相同；仅改变元信息，不复制数据；原 Buffer 被捐赠，底层内存交给新 Buffer。 |
| [`DonateWithControlDependency`][src-dependency]，[t2](buffer-centered-hub.svg#t2) | 原 Buffer 与额外 `Future<> dependency`。 | 返回内容相同的新 Buffer，只有原数据和依赖都就绪才就绪；任一错误会传播。默认实现返回 `Unimplemented`。 |

图中 [after](buffer-centered-hub.svg#after)汇总返回的新 Buffer，[invalid](buffer-centered-hub.svg#invalid)汇总捐赠、删除或所有权转移后的原句柄状态。变换返回的新句柄可以继续消费，失效的原句柄不能因此再次作为普通输入使用。两个结果节点说明接口效果，并非 provider 的新增实现类。

`Delete / ReleaseDeviceMemoryOwnership` 已移入消费区：它们结束原句柄的使用资格或移交存储所有权，不产生沿变换主线继续使用的新 Buffer。

执行本身也可能捐赠输入。[`PjRtLoadedExecutable` 的契约][src-loaded]规定：程序声明了输入输出 alias 时，相应输入可能在执行提交中被捐赠。
因此，执行后继续使用输入是否合法，要同时看 executable 的 alias/donation 约定和执行选项，不能只检查是否显式调用过 `Bitcast`。

<a id="consumption"></a>

## 消费：已加载程序与输入 Buffer 汇合

[已有 Buffer](buffer-centered-hub.svg#produced)与[变换结果](buffer-centered-hub.svg#after)分别进入[消费输入](buffer-centered-hub.svg#consume_input)。图中的直接旁路明确表示执行、读取或状态观察不要求先复制或 Bitcast。

| 图中路径 | 源码对应的输入输出 |
|---|---|
| [已加载程序](buffer-centered-hub.svg#loaded)与消费输入 → [Execute](buffer-centered-hub.svg#c0) | `argument_handles` 为 `[num_devices, num_args]`，配合 executable 的设备分配及执行约束。 |
| Execute → [provider](buffer-centered-hub.svg#provider) → [结果 Buffer](buffer-centered-hub.svg#r0) | 具体实现消费程序、实参、执行选项与依赖，返回存储句柄，通常继续由 IFRT 包装。 |
| provider → [执行 Future](buffer-centered-hub.svg#status) | 按需请求各设备的完成 Future；Buffer 自身仍提供独立数据就绪接口。 |
| 消费输入 → 主机读取、就绪或外部引用接口 | 三个接口并列使用有效 Buffer，各自输出数据、Future 或保活引用。 |
| 消费输入 → [Delete / Release](buffer-centered-hub.svg#t3) → [生命周期结果](buffer-centered-hub.svg#invalid) | 删除或移交所有权使原句柄失效；物理回收的时机另受异步使用约束。 |

这些契约集中在 [`PjRtLoadedExecutable::Execute`][src-execute] 的声明和注释中。
接口还提供 `ExecuteSharded` 等入口；主图选取常规多设备 Execute，未将所有入口混成单一调用链。

### 主机读取与外部框架共享

消费区中的读取与共享接口各自输出不同对象：

- [`ToLiteral`][src-literal]（[c1](buffer-centered-hub.svg#c1)）将值异步复制到主机 Literal，返回传输完成 Future；同步便利接口通过等待该 Future 实现。
- 同一类中的 `CopyRawToHost` 复制设备表示的指定字节区间，返回 Future；这与根据 Literal 布局读取值不同。
- [`AcquireExternalReference`][src-external]（[c3](buffer-centered-hub.svg#c3)）提供外部引用和不透明设备地址。引用保活与数据就绪是两个条件，外部框架修改内存前仍需自行保证同步。

因此，拿到 `ExternalReference` 或设备地址不等于得到主机可直接读取的值。
接口的 `WaitUntilBufferReadyOnStream` 面向特定平台互操作，默认实现并不保证所有硬件都支持。

### 输出对象、执行完成与回收

[`Execute`][src-execute] 同时涉及两种返回：函数先返回结果 Buffer 或提交错误；如果请求 `returned_futures`，每个设备还提供完成时就绪的 Future。
返回句柄与设备完成可以发生在不同时间，见图中粉色输出线与完成状态线。

[`GetReadyFuture`][src-ready]（[c2](buffer-centered-hub.svg#c2)）观察 Buffer 数据计算完成或错误。
若在 Buffer 已删除或捐赠后调用，接口约定返回立即带错误的 Future；若在删除或捐赠前取得 Future，则该 Future 不会仅因后续删除而变为错误。

[`Delete`][src-delete] 不等待 stream 工作结束；它让本句柄失效，底层内存的回收需等待异步使用以及相关外部引用满足平台要求。
[图中 t3](buffer-centered-hub.svg#t3) 还包含 [`ReleaseDeviceMemoryOwnership`][src-release]，它将所有权交出：`wait_for_operations_to_complete=true` 会等待；选择 false 时调用者需按接口要求先确认数据就绪。

这也是 [生命周期结果节点](buffer-centered-hub.svg#invalid) 与 [完成状态节点](buffer-centered-hub.svg#status) 分开绘制的原因。
前者回答“谁还可以持有和使用这段存储”，后者回答“本次工作或这份数据是否已完成”。

## 固定版本与证据范围

源码版本为 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69`，以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。
本文复核了本地固定版本 `pjrt_client.h`；引用与对应 SVG metadata 一致，并指向固定 commit。
缓存文件的大小与 SHA-256 记录在 [`component_diagram_sources.json`](../../../tools/component_diagram_sources.json)。

这里验证的是公开 C++ 接口与静态图文关系，没有运行 CPU 计算、GPU/TPU 设备执行、TPU 模拟或离线 TPU 编译。
接口中的 provider 扩展点及默认 `Unimplemented` 路径不能据此当作某个运行时二进制的已验证能力。

继续阅读：[IFRT Array](../ifrt/array-centered-hub.md)、[GPU LLVM IR](../xla/gpu-ir-centered-hub.md)、[执行提交](../stream-executor/submission-centered-hub.md)、[软件栈索引](../index.md)。

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
