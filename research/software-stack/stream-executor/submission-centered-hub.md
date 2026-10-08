# 设备运行时与驱动：一次执行请求如何完成

执行提交表示一次程序运行的请求与执行上下文：它关联已加载程序、输入输出存储、执行参数和依赖，说明执行什么、使用哪些数据、何时可以执行。

[配套图](submission-centered-hub.svg)按定义、产生、变换、消费展开，以公开的 CPU 和 CUDA 路径说明这个抽象。CPU 使用 thunk 执行上下文，CUDA 使用 kernel 启动参数、stream 和命令记录；源码没有把它们统一成一个请求类。全栈位置见[概览](../overview/overview-software-stack-components-layered.md)。

<a id="核心概念与范围"></a>
<a id="definition"></a>

## 1. 定义：程序、存储和依赖在本次调用中结合

编译产物说明要完成哪些计算，运行时请求把这些工作绑定到本次调用的数据和资源。相同的可执行程序可以运行多次，每次的输入、地址和依赖都可以不同。

| 一次执行需要的内容 | CPU 路径 | CUDA 路径 |
|---|---|---|
| 执行什么 | `CpuExecutable` 中的函数库和 thunk 计划。 | 已加载 kernel、设备库调用或 CommandBuffer 中的命令。 |
| 使用哪些存储 | Buffer 表与 `cpu::BufferAllocations`，进入 [`Thunk::ExecuteParams`][src-cpu-context]。 | [`GetKernelAndArgs`][src-args] 从 `BufferAllocations` 解析参数 slice 的设备地址。 |
| 使用哪些资源 | 线程池、通信与自定义调用上下文等执行参数。 | [`LaunchKernel`][src-launch-interface] 的启动维度、参数、shared memory 和 stream。 |
| 何时可以执行 | 输入就绪事件、控制依赖和执行选项。 | [`Stream`][src-stream] 内的顺序，以及 event / stream 之间的等待。 |
| 如何观察结果 | 输出 Buffer、执行事件和对外的 Future。 | 提交 Status、设备工作完成事件，以及主机同步接口。 |

存储必须在相关工作使用期间保持有效。运行时还要处理存储复用、输入捐赠和错误传播；具体规则由 PJRT Buffer 与设备实现共同约束。

<a id="production"></a>

## 2. 产生：从已加载程序和输入准备请求

### 2.1 CPU：准备一次程序执行

[`PjRtCpuLoadedExecutable::Execute`][src-cpu-execute] 接收按设备组织的 Buffer 参数、执行选项和可选的 Future 返回位置。它选择对应设备的执行路径，结合程序的存储规划取得输入存储、准备输出和临时存储，并收集输入就绪信息。

在 `cpu_client.cc` 的执行准备中，实际地址被整理成 `cpu::BufferAllocations`。随后构造 [`cpu::Thunk::ExecuteParams`][src-cpu-context]，传入函数库、存储表、线程池、通信和自定义调用上下文等。本次 CPU 请求就由这些运行时数据具体表达。

### 2.2 GPU：准备 kernel、传输或命令缓冲区

| 请求来源 | 输入与构造位置 | 准备好的内容 |
|---|---|---|
| 普通 kernel | [`KernelThunk::ExecuteOnStream`][src-kernel] 从执行参数取得 stream、executor 和 BufferAllocations，调用 [`GetKernelAndArgs`][src-args]。 | 从缓存中取得已加载 kernel，解析参数地址，按需构造 TensorMap，得到 `KernelWithArgs`；再配合启动维度提交。 |
| 数据传输 | `Stream::Memcpy` 的重载接收主机或设备源、目标和字节数。 | 确定传输方向、地址和目标 stream，交给对应平台实现。 |
| 命令缓冲区 | [`CommandBufferThunk::ExecuteOnStream`][src-command] 接收命令计划、本次 BufferAllocations 和 stream。 | 取得或创建 executor 对应的 CommandBuffer，决定记录、更新或直接复用。 |

这些请求使用已经编译并加载的程序。地址解析和参数准备发生在运行时；kernel 如何生成，见 [GPU LLVM IR](../xla/gpu-ir-centered-hub.md)。

<a id="transformation"></a>

## 3. 变换：绑定参数，整理依赖与命令记录

本次请求在提交前，需要把程序中的存储引用换成本次地址，并决定工作何时具备执行条件。可重复使用的命令记录还要处理地址变化。

### 3.1 CPU：输入依赖与执行方式

CPU 实现收集输入 Buffer 的就绪事件，并按需要加入 collective 或前一次提交的依赖。执行选项和运行时配置决定采用当前线程执行还是异步调度。

当输入已经就绪且选择就地执行时，代码直接分配所需存储并调用 thunk；否则，[`ExecuteWhenReady`][src-cpu-dependencies] 等待依赖，随后通过异步工作执行器运行相应回调。回调检查输入错误、准备存储，最后调用 thunk 执行入口。变化的是本次请求的参数绑定和调度条件。

### 3.2 GPU：更新可复用的命令

[`CommandBufferThunk::ExecuteOnStream`][src-command] 比较本次 BufferAllocations 与已有记录。第一次调用、地址发生变化或命令本身要求更新时，它调用 `commands_.Record`；已有记录仍适用时，则直接复用。得到的 CommandBuffer 必须引用本次有效的存储。

该函数也有逐 thunk 回退路径，例如特定 profiling 条件、缺少所需分配信息或首次 warmup。是否使用命令缓冲区，由这些条件共同决定。

### 3.3 CUDA：把一个提交区间记录为图

[`CudaStream::CaptureHandle::BeginCapture`][src-capture] 接收调用方提供的 graph、stream、依赖和 capture mode，调用 `cuStreamBeginCaptureToGraph` 开始记录。区间内的 stream 操作写入这个图，`EndCapture` 结束记录并检查所用 graph。

这一变换的输入是待捕获的操作区间，输出是记录了工作的 `CUgraph`。图的实例化和启动由调用方继续安排。CommandBuffer 也可以直接记录命令；例如 `KernelThunk::Record` 使用 `CreateLaunch` 或 `UpdateLaunch` 记录和修改 kernel 启动。因此，图中把 capture 与普通命令记录分别列出。

<a id="consumption"></a>

## 4. 消费：运行工作，写入输出并传播状态

### 4.1 CPU：由 ThunkExecutor 使用执行上下文

[`ThunkExecutor::Execute`][src-cpu-thunks] 接收 `Thunk::ExecuteParams`，按已有计划执行计算函数、库调用和其他工作。空计划直接完成，单 thunk 直接调用；多个 thunk 可以顺序执行，也可以按依赖关系调度。

执行返回 `AsyncValueRef<ExecuteEvent>`。CPU 设备实现处理事件的完成或错误，将状态传播到本次执行的输出存储和对外 Future。`PjRtCpuLoadedExecutable::Execute` 返回输出 Buffer，供 IFRT 组织逻辑数组或继续执行后续程序。

### 4.2 CUDA：平台接口使用具体请求

| 消费者 | 接收什么 | 做什么、返回什么 |
|---|---|---|
| [`CudaStream::LaunchKernel`][src-cuda-launch] | 已加载函数、参数地址、启动维度、shared memory 与 stream。 | 经 `LaunchCudaKernel`，按 cluster/PDL 条件选择 `cuLaunchKernelEx` 或 `cuLaunchKernel`，返回提交 Status。 |
| [`AsynchronousMemcpyH2D`][src-memcpy] 等传输 helper | 源、目标、字节数和 stream。 | 调用对应方向的异步传输接口，返回提交 Status。 |
| [`CommandBuffer::Submit` 的调用位置][src-command] | 本次有效的命令记录和目标 stream。 | 由平台实现提交记录中的工作，并返回调用状态。 |

普通执行模式下，成功提交的工作按 stream 和 event 依赖推进，计算或传输结果写入指定存储。提交 Status 为 OK 说明调用已成功提交；工作可能仍在排队或执行。capture 模式则按第三章的记录协议处理。

### 4.3 完成、依赖与输出返回

[`Stream::RecordEvent`][src-record] 在队列中记录此前工作的完成点。后续 stream 可以通过 [`WaitFor`][src-wait] 等待该事件；事件须先记录，并在相关工作结束前保持有效。

主机需要等待时，调用 [`BlockHostUntilDone`][src-sync-interface]。CUDA 的[具体实现][src-cuda-sync]同步指定 stream，并传播期间检测到的错误。等待覆盖调用时已排入该 stream 的工作。

上层的 [PjRtBuffer](../pjrt/buffer-centered-hub.md) 和 [IFRT Array](../ifrt/array-centered-hub.md) 可以在计算结束前返回，随后通过各自的就绪接口观察结果。CPU 执行事件、CUDA event 和 PJRT Future 分属不同接口层，状态如何衔接由具体设备实现完成。

本文依据 XLA `dcf304bc5dca1932b99f740b911dbd73631a1a69` 的源码，版本以 [`upstream-sources.lock`](../../../upstream-sources.lock) 为准；未运行 CPU/CUDA 实验。TPU provider 和驱动内部调度不在这些公开实现的证据范围内。

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
[src-cpu-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L2002
[src-cpu-context]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1809
[src-cpu-dependencies]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1875
[src-cpu-thunks]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/runtime/thunk_executor.cc#L248
