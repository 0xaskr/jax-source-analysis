# StreamExecutor / CUDA：执行提交与异步完成

本文对应 [执行提交架构图](submission-centered-hub.svg)，解释 XLA GPU 运行时如何把计划中的工作转成设备提交，以及提交返回和完成观察如何分开。
全栈位置见 [overview 图](../overview/overview-software-stack-components-layered.svg#submit)及其[配套说明](../overview/overview-software-stack-components-layered.md)。

图将[定义](submission-centered-hub.svg#view_definition)独立成块；具体请求在[产生](submission-centered-hub.svg#view_production)、[按需记录或更新](submission-centered-hub.svg#view_transformation)、[消费](submission-centered-hub.svg#view_consumption)三区之间传递。普通 kernel 和 memcpy 可以直接进入平台提交，命令缓冲区则按状态选择更新或复用。后续 Event 与主机等待放在[完成观察辅助区](submission-centered-hub.svg#downstream)。

<a id="核心概念与范围"></a>

唯一核心概念是 [执行提交](submission-centered-hub.svg#core)：本次工作的函数或操作、设备地址、启动参数与 stream 在此汇合。
它是运行时行为的研究对象，源码中分别由 `Stream`、`CudaStream`、Thunk 和 CommandBuffer 相关接口实现，并没有一个统一名为“执行提交”的类。

主图选择公开的 XLA GPU runtime 与 StreamExecutor/CUDA 路径。
kernel 启动、异步 memcpy 和 CommandBuffer 是三条独立分支。核心定义说明共同的提交职责；调用链使用具体参数或命令记录，不经过核心定义卡，也不假定存在统一请求类或统一分派函数。
CPU 任务提交、TPU provider 内部提交、ROCm 实现及驱动内部调度不在这里展开。

<a id="definition"></a>

## 定义：一次提交需要什么

| 图中节点 | 定义或调用位置 | 约束与作用 |
|---|---|---|
| [Stream / CUstream](submission-centered-hub.svg#d0) | [`Stream`][src-stream] 与 CUDA 平台实现。 | Stream 表达工作队列及依赖；平台句柄连接对应驱动资源。 |
| [LaunchKernel 参数](submission-centered-hub.svg#d1) | [`Stream::LaunchKernel`][src-launch-interface]。 | 接收线程与 block 维度、可选 cluster 维度、函数句柄、参数、shared memory 等。 |
| [DeviceAddress / BufferAllocations](submission-centered-hub.svg#d2) | [`KernelThunk::GetKernelAndArgs`][src-args]。 | 由运行时分配表把编译计划中的 slice 解析为本次调用的设备地址。 |
| [Event / Status / 完成](submission-centered-hub.svg#d3) | [`RecordEvent`][src-record]、[`WaitFor`][src-wait]、[`BlockHostUntilDone`][src-sync-interface]。 | 提交结果、设备侧依赖与主机等待承担不同职责。 |

已加载 kernel 决定执行什么，BufferAllocations 决定本次操作的数据位于哪里，launch 参数决定如何启动，stream 决定工作进入哪条队列。
这四者在执行时汇合；其中的设备地址不是编译阶段 `BufferAssignment` 本身。

<a id="production"></a>

## 产生：从执行计划得到具体请求

### 普通 kernel

[`KernelThunk::ExecuteOnStream`][src-kernel]（[p0](submission-centered-hub.svg#p0)）从执行参数取得 stream、executor 和 BufferAllocations。
它按需清零输出区域，调用 [`GetKernelAndArgs`][src-args]，再经 `ExecuteKernelOnStream` 发起启动。
对应 p0 → [args](submission-centered-hub.svg#args) → [本次 kernel 与启动参数](submission-centered-hub.svg#launch_request)，再沿直接旁路交给 CUDA 启动接口。最后一张卡汇总已有接口的参数，不是新定义的运行时类。

`GetKernelAndArgs` 在 executor 对应的 kernel cache 中查找已经加载的 kernel；未完成初始化时返回错误。
随后逐参数解析 `BufferAllocation::Slice` 的设备地址，并按需要构造 TensorMap 参数，输出 `KernelWithArgs`。
这个阶段准备实际启动参数，不能将它解释为重新生成 GPU LLVM IR 或重新编译 kernel。

### 数据传输

`Stream::Memcpy` 的重载接收主机/设备源与目标，以及字节数，见 [p2](submission-centered-hub.svg#p2) → [本次传输参数](submission-centered-hub.svg#copy_request)。
这些请求直接进入传输分支；它们并不需要伪装成普通 kernel 才能提交。
具体 host-memory、地址及对齐要求由接口和平台实现约束。

### 命令缓冲区

[`CommandBufferThunk::ExecuteOnStream`][src-command]（[p1](submission-centered-hub.svg#p1)）从记录的命令和本次 BufferAllocations 组织提交。
它可以创建或复用 executor 对应的 CommandBuffer，根据地址变化更新记录，再调用 `CommandBuffer::Submit(params.stream)`。
这是与普通 kernel 启动并列的路径，一个 CommandBuffer 可包含多个工作项。

<a id="transformation"></a>

## 变换：记录、捕获与更新

这里的“变换”发生在工作记录方式和参数绑定上，不是 Jaxpr/HLO 的程序优化，也不是对一个统一提交对象运行必经 pass。

命令缓冲区的实际路径是 [ExecuteOnStream](submission-centered-hub.svg#p1)按需进入[记录或更新](submission-centered-hub.svg#t2)，得到[可提交记录](submission-centered-hub.svg#command_ready)；不需重录时复用已有记录，随后进入 `CommandBuffer::Submit`。普通 kernel 与传输的直接提交旁路不经过该记录步骤。

| 图中节点 | 输入与具体调用 | 输出或变化 |
|---|---|---|
| [CUDA graph capture](submission-centered-hub.svg#t1) | [`CudaStream::CaptureHandle::BeginCapture`][src-capture] 接收 stream、graph、依赖及 capture mode。 | 建立捕获上下文，CUDA 实现调用 `cuStreamBeginCaptureToGraph`；结束 capture 后相应工作被记录到 graph。 |
| [更新已记录命令](submission-centered-hub.svg#t2) | [`CommandBufferThunk::ExecuteOnStream`][src-command] 比较本次分配与已记录状态，按需调用 `commands_.Record`。 | 首次创建或更新命令参数，使记录引用本次有效地址；随后才能提交该记录。 |

CUDA capture 另有独立路径：[调用者的图与捕获区间](submission-centered-hub.svg#capture_input) → [capture 协议](submission-centered-hub.svg#t1) → [已记录的 CUgraph](submission-centered-hub.svg#captured_graph)。`BeginCapture` 先建立记录区间，其中的 stream 操作记录到调用者提供的 graph，`EndCapture` 检查记录仍使用该 graph。结束捕获只说明记录完成，图的实例化和启动由调用方继续安排。

本图不把这个 CUgraph 直接连成所有 CommandBuffer 的构造来源；CommandBuffer 也可以直接记录命令。
同一 [`kernel_thunk.cc`][src-kernel] 中的 `KernelThunk::Record` 就分别使用 `CreateLaunch` 和 `UpdateLaunch` 记录或修改 kernel 命令。

`CommandBufferThunk::ExecuteOnStream` 还包含条件分支：空命令直接返回；某些 profiling 条件、缺少 persistent allocation indices，或首次 warmup 时可能使用逐 thunk 的回退执行。
因此，图中的 CommandBuffer 路径是具体启用条件满足后的提交方式，不能按一次 API 调用就断言已经走了 CUDA graph。

<a id="consumption"></a>

## 消费：三条独立提交路径

| 图中节点 | 接收的输入 | 源码调用与结果 |
|---|---|---|
| [CudaStream::LaunchKernel](submission-centered-hub.svg#c0) | [本次 kernel 参数](submission-centered-hub.svg#launch_request)：函数、启动维度、参数数组、shared memory、stream。 | [`CudaStream::LaunchKernel`][src-cuda-launch] 调用 `LaunchCudaKernel`；后者依 cluster/PDL 条件选择 `cuLaunchKernelEx` 或 `cuLaunchKernel`。 |
| [CUDA 异步 memcpy](submission-centered-hub.svg#c1) | [本次传输参数](submission-centered-hub.svg#copy_request)：源与目标地址、字节数、stream。 | [`AsynchronousMemcpyH2D`][src-memcpy] 等 helper 调用异步拷贝接口；传输方向决定具体实现。 |
| [CommandBuffer::Submit](submission-centered-hub.svg#command_submit) | [可提交命令记录](submission-centered-hub.svg#command_ready)与目标 stream。 | [`CommandBufferThunk` 的提交位置][src-command] 调用 `cmd_buffer->command_buffer->Submit(params.stream)`，平台实现执行提交。 |

三条分支分别连到[提交 Status](submission-centered-hub.svg#submit_status)，表达调用返回的成功或错误；普通执行模式中成功提交的工作进入[设备队列](submission-centered-hub.svg#work)。两组线对应不同观察时刻，API 返回 `OK` 不能单独证明 kernel 或传输已结束。capture 模式的操作属于前述记录协议，不能据此推断已经执行设备工作。

驱动接收函数和设备地址后如何进行内部调度，不能从本图引用的公开源码完整还原。
这里的“队列中的工作”是对 stream/异步接口语义的说明，不声称已经采集设备执行时间线。

### 完成观察：Event、WaitFor 与主机等待

#### 设备侧依赖

辅助区展开 work → [RecordEvent](submission-centered-hub.svg#event) → [WaitFor](submission-centered-hub.svg#wait)，这是按调用方需要增加的设备侧依赖。
[`Stream::RecordEvent`][src-record] 把事件插入 stream 尾部，此前的工作处理完成后事件才标记完成。
`Stream` 不取得 Event 所有权，调用者必须保证 Event 活到相应完成点之后。

[`Stream::WaitFor(Event*)`][src-wait] 让后续工作依赖已记录的事件；另一重载可建立对另一条 stream 的依赖。
Event 必须先被记录，否则接口注释指出等待可能直接被视为已完成。
这样的设备侧依赖通常不要求主机每次提交后都阻塞；跨 stream 的等待还需要避免形成环。

#### 主机观察完成

主机观察路径为 work → [BlockHostUntilDone](submission-centered-hub.svg#c3) → [主机可观察结果](submission-centered-hub.svg#r3)。
[`Stream::BlockHostUntilDone`][src-sync-interface] 的契约是阻塞主机，等待调用时已经排入该 stream 的工作完成，并返回成功或错误。
[`CudaStream::BlockHostUntilDone`][src-cuda-sync] 调用 `SynchronizeStream`，同时处理 host callback 的失败传播；capture stream 上不允许这一操作。

这里观察的是指定 stream 的工作，不应把一次等待的成功扩大为所有 stream 或所有设备都已经空闲。
`Status` 出现在多个接口中时，含义取决于返回位置：启动调用的 Status 对应提交结果，同步调用的 Status 则对应等待结果及期间检测到的错误。

### 与上层 Array / Buffer 的关系

本图的设备地址由运行时存储提供；上层 [PjRtBuffer](../pjrt/buffer-centered-hub.md) 和 [IFRT Array](../ifrt/array-centered-hub.md) 负责存储句柄及逻辑数组包装。
它们的输出对象可以先返回，上层再用各自的 Future 观察就绪。
本图中的 Event、Stream 等待与上层 Future 处于不同抽象层，不能不经 provider 的具体实现就断言一一对应。

同样，kernel 在此已加载并可被调用；其编译过程见 [GPU LLVM IR](../xla/gpu-ir-centered-hub.md)。
编译结果、执行参数、提交返回和设备完成各有独立节点，便于沿同一请求追踪每个阶段的输入输出。

## 固定版本与证据范围

源码版本为 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69`，以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准。
本文核对了本地固定版本的 `kernel_thunk.cc`、`command_buffer_thunk.cc`、`stream.h` 与 `cuda_stream.cc`；引用均指向同一 commit。
源码缓存由 [`component_diagram_sources.json`](../../../tools/component_diagram_sources.json) 中的大小及 SHA-256 锁定，图中节点与命名关系来自对应 SVG metadata。

本文只提供源码检查与静态图文关系；没有实际启动 CUDA kernel、测量传输或观察驱动时间线，也没有进行 CPU 执行、TPU 模拟、离线 TPU 编译或真实 TPU 执行。

继续阅读：[PJRT Buffer](../pjrt/buffer-centered-hub.md)、[GPU LLVM IR](../xla/gpu-ir-centered-hub.md)、[overview 说明](../overview/overview-software-stack-components-layered.md)、[软件栈索引](../index.md)。

[src-stream]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/stream.h#L67
[src-launch-interface]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/stream.h#L308
[src-args]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/runtime/kernel_thunk.cc#L203
[src-record]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/stream.h#L144
[src-wait]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/stream.h#L137
[src-sync-interface]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/stream.h#L220
[src-kernel]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/runtime/kernel_thunk.cc#L238
[src-command]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/gpu/runtime/command_buffer_thunk.cc#L279
[src-capture]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/cuda/cuda_stream.cc#L200
[src-cuda-launch]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/cuda/cuda_stream.cc#L615
[src-memcpy]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/cuda/cuda_stream.cc#L124
[src-cuda-sync]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/stream_executor/cuda/cuda_stream.cc#L354
