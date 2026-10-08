"""Four research views for logical arrays, device storage and execution requests."""
from __future__ import annotations


def _prepare(spec, title, summary, foot):
    spec.update(layout='component_views', title=title, summary=summary, foot=foot,
                sections=[], edges=[], paths=[], details=[],
                avoid_endpoint_captions=True)
    spec['nodes'] = {key: value for key, value in spec['nodes'].items()
                     if key not in spec['omit']}
    spec['omit'] = set()
    return spec['nodes']


def _section(spec, ident, title, owner, rows, question, *, auxiliary=False):
    from software_stack_component_flow_data import section
    section(spec, title, owner, *rows, external=auxiliary)
    spec['sections'][-1].update(id=ident, question=question, auxiliary=auxiliary)


def _wording(nodes, content):
    for key, (title, rows) in content.items():
        nodes[key].update(title=title, rows=rows)


def _definition(spec, owner):
    from software_stack_component_flow_data import edge
    _section(spec, 'view_definition', '定义 · 表示什么', owner,
             [['core'], ['d0'], ['d1'], ['d2'], ['d3']],
             '从数据描述、组成结构和使用约束理解这个抽象。')
    for key in ('d0', 'd1', 'd2', 'd3'):
        edge(spec, 'core', key, '组成与约束', 'neutral')


def _array_views(spec):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(spec, 'IFRT · Array',
        '把类型、形状、分片和布局与设备存储组合起来，形成可交给程序执行的逻辑数组。',
        '依据固定版本的 IFRT PJRT 数值数组实现。Array 可放在单设备上，也可分片或复制到多设备；输出返回、数据就绪和存储回收分别观察。')
    n['produced'] = node('ArrayRef', [], 'MakeArrayFromHostBuffer', 'Assemble arrays', tag='产生结果 · Array', color='data')
    n['transform_input'] = node('Array 与操作参数', [], 'CopyArrays', 'RemapArrays', 'Bitcast arrays', tag='变换输入', color='data')
    n['consume_input'] = node('Array', [], 'ifrt Array', 'Value', tag='消费输入', color='data')
    n['p2']['tag'] = '消费内部 · 构造输出 Array'
    _wording(n, {
        'core': ('IFRT Array', ['一个完整的逻辑数组，关联元信息与设备存储。', 'ArraySpec 描述类型、全局形状、分片和布局。', '单设备、跨设备分片和完全复制都可表达。', 'ArrayRef 以引用计数方式持有数组。']),
        'd0': ('Array / ArraySpec', ['dtype、shape 描述完整数组的类型与形状。', 'sharding、layout 描述数据放置与排列。', 'Array 接口要求实现线程安全。']),
        'd1': ('Sharding：设备与分片', ['设备列表说明数组分片的放置位置。', 'GetShardShape / IndexDomains 描述局部片段。', '当前进程可访问的设备称为可寻址设备。']),
        'd2': ('PjRtArray：元信息与 Buffer', ['PjRtArray 保存数组描述和 PjRtBuffers。', '每个可寻址分片由 PjRtBuffer 承载。', 'Buffer 通过 shared_ptr 被持有。']),
        'd3': ('存储使用与就绪约定', ['ArrayCopySemantics 约定复制、复用或捐赠。', 'GetReadyFuture 观察数据就绪和错误。', 'Delete / IsDeleted 管理数组的使用资格。']),
        'p0': ('从主机数据构造', ['MakeArrayFromHostBuffer 接收地址、类型、形状和放置。', '为可寻址设备创建 Buffer，再调用 PjRtArray::Create。', '数值数组入口支持单设备或完全复制的分片。']),
        'p1': ('从单设备数组组装', ['AssembleArrayFromSingleDeviceArrays 接收分片列表。', '检查类型、单设备分片和数量，收集 Buffer 引用。', '构造目标形状与分片下的 Array。']),
        'produced': ('ArrayRef：构造结果', ['存储引用与数组描述已经组合完成。', '底层传输可以继续异步进行。', '数组可直接执行，也可先改变放置或表示。']),
        'transform_input': ('已有 Array 与操作参数', ['输入一个数组或数组列表。', '同时提供目标设备、分片映射或新 ArraySpec。', '按需要选择一种操作。']),
        't0': ('CopyArrays：改变放置', ['输入数组、目标设备或内存种类、复制语义。', '本地主机内路径委派 PjRtArray::Copy。', '返回目标放置下的数组列表。']),
        't1': ('RemapArrays：重组分片', ['输入数组列表、RemapPlan 和复制语义。', '按 plan 将分片放入输出数组对应位置。', '返回重组后的数组列表。']),
        't2': ('Disassemble：拆成单设备数组', ['输入逻辑数组、复制语义和分片范围。', '按设备为分片构造 Array。', '请求全部分片时要求设备全部可寻址。']),
        't3': ('BitcastArrays：重新解释字节', ['输入数组列表和新的 ArraySpec。', '当前实现要求 DonateInput，逐 Buffer 调用 Bitcast。', '创建新数组，随后删除原数组。']),
        'after': ('变换后的 Array', ['返回新的数组或单设备数组列表。', '存储是否复制，取决于操作与复制语义。', '类型、形状、放置或所有权按规则改变。']),
        'consume_input': ('供调用方使用的 Array', ['已有 Array 或变换返回的 Array。', '执行、读取、等待和删除按用途调用。']),
        'loaded': ('已加载的程序', ['LoadedExecutable 由编译与加载产生。', '执行入口接收 Array 列表和执行选项。']),
        'c0': ('LoadedExecutable::Execute', ['输入逻辑数组列表和执行选项。', '检查数组实现、分片数量与设备对应关系。']),
        'extract': ('按设备排列 Buffer 参数', ['提取每个 Array 的可寻址分片。', '把每数组的分片列表重排为每设备的参数列表。']),
        'call': ('调用 PJRT 执行', ['向设备实现传入 PjRtBuffer 参数和执行选项。', '取得输出 Buffer，并按需取得执行 Future。']),
        'buffers': ('各设备的输出 Buffer', ['结果存储已由设备实现关联到句柄。', '计算可以尚未完成。']),
        'p2': ('将结果组织成 Array', ['按逻辑输出收集 Buffer 分片。', '加入输出类型、全局形状、分片和布局。', '调用 PjRtArray::Create。']),
        'r0': ('ExecuteResult.outputs', ['返回 ArrayRef 列表，供上层继续使用。', '这些数组也可作为后续程序的输入。']),
        'status': ('ExecuteResult.status', ['仅在 fill_status=true 时填充执行状态。', '与 Array 的数据就绪接口分别观察。']),
        'c1': ('CopyToHostBuffer', ['接收主机目标地址和可选 strides。', '单分片或全复制数组可直接读取一个分片。']),
        'r1': ('主机数据与传输 Future', ['Future 成功后，目标存储中的数据有效。', '主机目标地址须保持有效直到传输完成。']),
        'c2': ('GetReadyFuture', ['查询各底层 Buffer 的就绪状态。', '多 Buffer 时合并它们的 Future。']),
        'r2': ('数组数据就绪或错误', ['供调用方等待、衔接依赖或处理失败。', '等待本身不复制数据到主机。']),
        'c3': ('Delete', ['逐 Buffer 调用 Delete，并标记数组已删除。', '结束对原数组的使用。']),
        'r3': ('数组失效', ['当前实现返回立即就绪的 OK Future。', '实际回收还受异步使用及外部引用影响。']),
    })
    _definition(spec, 'IFRT / PJRT-backed 实现')
    _section(spec, 'view_production', '产生 · 构造逻辑数组', 'IFRT / PJRT-backed 实现',
             [['p0', 'p1'], ['produced']],
             '主机数据和已有单设备数组进入 PjRtArray::Create；执行结果在消费区使用同一构造机制。')
    _section(spec, 'view_transformation', '变换 · 放置、分片与表示', 'IFRT / PJRT-backed 实现',
             [['transform_input'], ['t0', 't1'], ['t2', 't3'], ['after']],
             '选择相应操作改变放置、分片组织或数据解释；存储是否复制由接口约定。')
    _section(spec, 'view_consumption', '消费 · 执行、读取与生命周期', 'IFRT / PJRT-backed 实现',
             [['consume_input'], ['loaded', 'c0', 'extract'], ['call', 'buffers', 'p2'],
              ['r0', 'status'], ['c1', 'c2', 'c3'], ['r1', 'r2', 'r3']],
             '程序在 Execute 处使用数组；结果 Buffer 重新组成逻辑数组，读取和生命周期操作按需调用。')
    edge(spec, 'p0', 'produced', '主机入口的 ArrayRef', 'data')
    edge(spec, 'p1', 'produced', '组装后的 ArrayRef', 'data')
    edge(spec, 'produced', 'consume_input', '直接消费 Array', 'data', optional=True)
    spec['edges'][-1]['channel'] = 'direct'
    edge(spec, 'produced', 'transform_input', 'Array + 操作参数', 'data', optional=True)
    for i in range(4):
        edge(spec, 'transform_input', f't{i}', '按需选择操作', 'data', optional=True)
        edge(spec, f't{i}', 'after', '新 Array / Array 列表', 'output')
    edge(spec, 'after', 'consume_input', '按调用约定选择 Array', 'data')
    edge(spec, 'loaded', 'c0', '已加载程序', 'neutral')
    for i in range(4):
        edge(spec, 'consume_input', f'c{i}',
             ('ArrayRef 参数', 'Array + 主机目标', '观察就绪状态', '终止使用资格')[i],
             'data' if i < 2 else 'neutral', optional=True)
    chain(spec, ['c0', 'extract', 'call', 'buffers', 'p2', 'r0'],
          ['提取可寻址分片', '按设备组织 Buffer 参数', '输出 Buffer 句柄',
           '按逻辑输出聚合分片', 'ArrayRef 输出'], 'data')
    for e in spec['edges'][-3:]:
        e['kind'] = 'output'
    edge(spec, 'call', 'status', 'fill_status 时合并执行 Future', 'completion', optional=True)
    for i, label in ((1, '主机数据与传输 Future'), (2, '数据就绪 / 错误'),
                     (3, '失效状态 / 返回 Future')):
        edge(spec, f'c{i}', f'r{i}', label, 'output' if i == 1 else 'completion' if i == 2 else 'neutral')
    spec['paths'].extend([
        ['p0', 'produced', 'consume_input', 'c0', 'extract', 'call', 'buffers', 'p2', 'r0'],
        ['p1', 'produced', 'transform_input', 't3', 'after', 'consume_input', 'c2', 'r2'],
        ['produced', 'consume_input', 'c3', 'r3']])
    return spec


def _buffer_views(spec):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(spec, 'PJRT · PjRtBuffer',
        '统一描述设备上的数据存储，管理它的归属、形状、布局、就绪状态和所有权。',
        '接口由具体设备实现。输入输出别名和执行选项影响存储复用；句柄失效后，存储仍可能等待异步操作或外部引用结束。')
    n['produced'] = node('PjRtBuffer', [], 'BufferFromHostBuffer', 'PjRtBuffer', tag='产生结果 · Buffer', color='data')
    n['transform_input'] = node('Buffer 与操作参数', [], 'CopyToMemorySpace', 'Buffer bitcast', 'Buffer dependency', tag='变换输入', color='data')
    n['consume_input'] = node('有效 Buffer', [], 'PjRtBuffer', 'PJRT Execute', tag='消费输入', color='data')
    n['t3']['tag'] = '消费 · 结束使用或移交所有权'
    _wording(n, {
        'core': ('PjRtBuffer', ['设备数据存储的统一接口。', '描述类型、形状、布局、设备和内存空间。', '管理数据就绪与存储所有权。', '已加载程序使用 Buffer 作为输入和输出。']),
        'd0': ('形状、布局与动态维度', ['element_type / dimensions 描述数组类型和维度。', 'on_device_shape / layout 描述设备表示。', '动态逻辑维度查询可能需要与设备通信。']),
        'd1': ('Client、Device 与 MemorySpace', ['client 关联管理这块存储的运行时。', 'device / memory_space 描述存储归属。', '接口可表达数组或 tuple。']),
        'd2': ('就绪与有效性', ['Buffer 返回时，计算或传输可以尚未完成。', 'GetReadyFuture 报告数据就绪或错误。', 'Delete 或捐赠后，原句柄结束普通使用。']),
        'd3': ('存储所有权', ['ExternalReference 可保持底层存储存活。', 'ReleaseDeviceMemoryOwnership 移交所有权。', '外部读写须遵守同步约定。']),
        'p0': ('BufferFromHostBuffer', ['输入主机地址、类型、形状、strides 和目标位置。', '设备实现依据 HostBufferSemantics 接收数据。', '返回 Buffer；回调通知何时不再使用主机输入。']),
        'produced': ('构造完成的 Buffer 句柄', ['关联本次输入的设备存储。', '数据传输可以尚未完成。', '后续使用遵守就绪与所有权约定。']),
        'transform_input': ('已有 Buffer 与操作参数', ['提供目标内存空间、新数据描述或额外依赖。', '选择所需的复制、Bitcast 或依赖操作。']),
        't0': ('CopyToMemorySpace', ['接收原 Buffer 和目标内存空间。', '返回目标空间中的新 Buffer。', '目标已是当前空间时返回错误。']),
        't1': ('Bitcast', ['接收新类型、维度和布局；设备总字节数须相同。', '修改元信息，保留原有数据字节。', '存储交给新 Buffer，原 Buffer 被捐赠。']),
        't2': ('DonateWithControlDependency', ['接收 Buffer 与额外依赖 Future。', '新 Buffer 在原数据和依赖均就绪后才就绪。', '任一错误都会传播；默认实现不支持此操作。']),
        'after': ('变换返回的新 Buffer', ['复制改变位置，Bitcast 改变数据解释。', '附加依赖改变新 Buffer 的就绪条件。', '按接口规则决定原句柄是否仍可使用。']),
        'consume_input': ('供调用方使用的 Buffer', ['输入为有效 Buffer 及所需调用参数。', '可供执行、读取、状态查询或外部访问。']),
        'loaded': ('PjRtLoadedExecutable', ['已编译并加载的程序。', 'Execute 将程序与本次 Buffer 参数组合起来。']),
        'c0': ('Execute：使用输入 Buffer', ['传入每设备的 Buffer 参数和执行选项。', '输入输出别名及 donation 约定影响存储复用。']),
        'provider': ('设备实现执行请求', ['准备结果存储，处理输入依赖。', '按 CPU、GPU 或 TPU 的运行时机制执行。']),
        'r0': ('各设备的输出 Buffer', ['返回结果存储句柄，计算可以尚未完成。', 'IFRT 再按输出类型、形状和分片组织数组。']),
        'status': ('按需返回的执行 Future', ['用于观察本次设备执行状态。', '输出数据也可通过 Buffer 就绪接口观察。']),
        'c1': ('ToLiteral：读取到主机', ['接收调用方提供的主机 Literal。', '读取 Buffer 数据，并返回传输 Future。']),
        'r1': ('主机数据与传输 Future', ['传输成功后主机数据有效。', '目标存储须保持有效直到传输完成。']),
        'c2': ('GetReadyFuture', ['观察数据的计算、传输或错误状态。', '删除或捐赠前取得的 Future 仍保持有效。']),
        'r2': ('数据就绪或错误', ['供调用方等待或处理失败。', '删除或捐赠后再查询会立即得到错误。']),
        'c3': ('AcquireExternalReference', ['取得底层存储的外部引用。', '外部使用者可访问不透明设备指针。']),
        'r3': ('ExternalReference', ['引用保持存储存活。', '使用者负责满足读写的同步要求。']),
        't3': ('Delete / 移交存储所有权', ['Delete 放弃本句柄的存储引用。', 'Release 将所有权交给 ExternalReference。', '原句柄随之失效。']),
        'invalid': ('原句柄失效或被捐赠', ['原 Buffer 结束普通使用。', '实际回收仍受异步操作和外部引用约束。']),
    })
    _definition(spec, 'PJRT C++ 接口')
    _section(spec, 'view_production', '产生 · 获取设备存储句柄', 'PJRT / provider',
             [['p0'], ['produced']],
             '主机数据通过客户端接口进入设备存储；执行也会创建输出 Buffer，见消费区。')
    _section(spec, 'view_transformation', '变换 · 存储、解释与依赖', 'PJRT / provider',
             [['transform_input'], ['t0', 't1', 't2'], ['after']],
             '按需要改变数据位置、解释方式或就绪条件，并遵守每个接口的所有权规则。')
    _section(spec, 'view_consumption', '消费 · 程序、读取与所有权', 'PJRT / provider',
             [['consume_input'], ['loaded', 'c0', 'provider'], ['r0', 'status'],
              ['c1', 'c2', 'c3'], ['r1', 'r2', 'r3'], ['t3', 'invalid']],
             '已加载程序使用 Buffer 计算；主机读取、就绪查询、外部访问和释放存储各有对应结果。')
    edge(spec, 'p0', 'produced', 'PjRtBuffer · 可异步传输', 'data')
    edge(spec, 'produced', 'consume_input', '直接消费 Buffer', 'data', optional=True)
    spec['edges'][-1]['channel'] = 'direct'
    edge(spec, 'produced', 'transform_input', 'Buffer + 操作参数', 'data', optional=True)
    for i in range(3):
        edge(spec, 'transform_input', f't{i}', '按需选择操作', 'data', optional=True)
        edge(spec, f't{i}', 'after', '新 Buffer / 新依赖', 'output')
    edge(spec, 'after', 'consume_input', '使用返回的新句柄', 'data')
    edge(spec, 'loaded', 'c0', '已加载程序', 'neutral')
    for i, label in enumerate(('每设备 Buffer 参数', 'Buffer + 主机目标', '观察数据就绪', '请求外部引用')):
        edge(spec, 'consume_input', f'c{i}', label, 'data' if i < 2 else 'neutral', optional=True)
    edge(spec, 'consume_input', 't3', '结束使用 / 释放所有权', 'neutral', optional=True)
    chain(spec, ['c0', 'provider', 'r0'], ['provider 执行分派', '输出 Buffer 句柄'], 'data')
    spec['edges'][-1]['kind'] = 'output'
    edge(spec, 'provider', 'status', '按需返回设备执行 Future', 'completion', optional=True)
    for i, label in ((1, '主机数据 / 传输 Future'), (2, '数据就绪 / 错误'), (3, '保活的外部引用')):
        edge(spec, f'c{i}', f'r{i}', label, 'output' if i == 1 else 'completion' if i == 2 else 'neutral')
    for key in ('t1', 't2', 't3'):
        edge(spec, key, 'invalid', '原句柄失效 / 所有权变化', 'neutral')
    edge(spec, 'c0', 'invalid', '按 executable 契约捐赠时', 'neutral', optional=True)
    spec['paths'].extend([
        ['p0', 'produced', 'consume_input', 'c0', 'provider', 'r0'],
        ['produced', 'transform_input', 't2', 'after', 'consume_input', 'c2', 'r2'],
        ['produced', 'consume_input', 't3', 'invalid']])
    return spec


def _submission_views(spec):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(spec, '设备运行时与驱动 · 执行提交',
        '把已加载程序、本次数据存储、执行参数和依赖组合起来，交给具体运行时执行。',
        'CPU 使用函数库、存储表和 thunk 执行上下文；CUDA 使用 kernel 参数、stream 和命令记录。提交成功、输出句柄返回和工作完成分别观察。')
    _wording(n, {
        'core': ('执行提交', ['一次程序运行的请求与执行上下文。', '关联程序、输入输出存储、执行参数和依赖。', '说明执行什么、使用哪些数据、何时可以执行。', 'CPU 与 CUDA 使用不同结构表达这些信息。']),
        'd0': ('执行上下文与工作队列', ['CPU Thunk::ExecuteParams 携带函数库、存储和线程池。', 'GPU Stream 组织异步工作，CUDA 实现关联 CUstream。', '具体结构由设备运行时定义。']),
        'd1': ('CUDA kernel 启动参数', ['LaunchKernel 接收函数句柄、参数和启动维度。', 'shared memory、cluster 等选项约束本次启动。', 'stream 指定工作提交到哪条队列。']),
        'd2': ('程序使用的实际地址', ['BufferAllocations 将存储规划关联到本次地址。', 'GPU GetKernelAndArgs 解析 kernel 的参数切片。', '存储须覆盖工作使用它的生命周期。']),
        'd3': ('依赖、完成与错误', ['CPU 输入事件约束计算何时可运行。', 'CUDA event / stream 等待约束设备侧执行。', '提交返回与工作完成由各自接口报告。']),
        'p0': ('GPU：KernelThunk::ExecuteOnStream', ['取得 stream、已加载 kernel 和 BufferAllocations。', '调用 GetKernelAndArgs 准备本次启动。']),
        'args': ('GPU：GetKernelAndArgs', ['从 executor 的缓存取得已加载 kernel。', '将参数 slice 解析为设备地址，按需准备 TensorMap。', '返回 KernelWithArgs。']),
        'p1': ('GPU：CommandBufferThunk', ['接收命令计划、BufferAllocations 和 stream。', '创建或取得现有 CommandBuffer。', '按本次地址决定记录、更新或复用。']),
        'p2': ('GPU：Stream::Memcpy', ['接收主机或设备源、目标与字节数。', '按传输方向选择具体平台实现。']),
        't1': ('CUDA：记录捕获区间', ['BeginCapture 指定 stream、graph 和依赖。', '区间中的 stream 操作记录到给定图中。', 'EndCapture 结束记录。']),
        't2': ('GPU：记录或更新命令', ['根据当前 BufferAllocations 检查地址变化。', '首次记录、地址变化或命令要求更新时重新记录。', '得到引用本次有效存储的命令记录。']),
        'c0': ('CUDA：LaunchKernel', ['接收函数、设备地址、启动维度与 stream。', '按配置调用 cuLaunchKernel 或 cuLaunchKernelEx。']),
        'c1': ('CUDA：异步数据传输', ['接收源、目标、字节数和 stream。', '例如 H2D helper 调用 cuMemcpyHtoDAsync。']),
        'command_submit': ('GPU：CommandBuffer::Submit', ['接收有效的命令记录和目标 stream。', '由平台实现提交记录中的工作。']),
        'submit_status': ('GPU：提交调用返回', ['启动或传输 API 返回 Status。', '报告提交成功或调用阶段的错误。']),
        'work': ('GPU：队列中的工作', ['普通执行模式下，成功提交的工作等待或执行。', '设备按 stream 和 event 依赖推进。']),
        'event': ('CUDA：RecordEvent', ['在 stream 中记录此前工作的完成点。', '调用方保持事件有效，直到相关工作结束。']),
        'wait': ('CUDA：WaitFor', ['让后续 stream 工作依赖已记录的事件。', '也可以建立对另一条 stream 的等待。']),
        'c3': ('CUDA：BlockHostUntilDone', ['主机等待当前 stream 已提交的工作。', 'CudaStream 调用同步接口并传播错误。']),
        'r3': ('CUDA：等待结果', ['返回等待成功或错误。', '观察范围是指定 stream 的已有工作。']),
    })
    n['core']['refs'] = ('CPU execution context', 'Stream launch')
    n['d0']['refs'] = ('CPU execution context', 'Stream')
    n['d2']['refs'] = ('CPU execution context', 'Kernel arguments')
    n['d3']['refs'] = ('CPU input dependencies', 'Stream record', 'Stream synchronize')
    n['cpu_request'] = node('CPU：PJRT 执行入口', [
        'PjRtCpuLoadedExecutable::Execute 接收各设备 Buffer。',
        '结合已加载程序、执行选项和输入就绪信息。',
        '进入单设备执行准备，安排输出与临时存储。'],
        'CPU execute', 'CPU execution context', tag='产生 · CPU 执行请求', color='data')
    n['cpu_context'] = node('CPU：绑定存储与执行依赖', [
        '把本次存储表关联到函数库和 thunk 计划。',
        '输入事件、执行选项决定就地执行或异步调度。',
        '执行前构造 Thunk::ExecuteParams。'],
        'CPU execution context', 'CPU input dependencies', tag='变换 · CPU 参数与依赖', color='transform')
    n['cpu_execute'] = node('CPU：ThunkExecutor::Execute', [
        '接收 ExecuteParams，取得函数库、地址和线程池。',
        '按计划选择顺序执行或依赖驱动的调度。',
        '调用计算函数、库和其他 thunk 工作。'],
        'CPU thunk execution', tag='消费 · CPU 执行', color='data')
    n['cpu_done'] = node('CPU：执行事件与输出 Buffer', [
        'ThunkExecutor 返回 ExecuteEvent。',
        'CPU 设备实现传播完成或错误状态。',
        '输出 Buffer 关联结果存储，可供后续计算使用。'],
        'CPU execute', 'CPU execution context', 'CPU thunk execution',
        tag='消费结果 · CPU 完成与输出', color='output')
    n['launch_request'] = node('GPU：本次 kernel 调用', [
        '已加载函数与实际设备参数地址。',
        '启动维度、shared memory 和目标 stream。'],
        'Kernel arguments', 'Stream launch', tag='产生 · kernel 参数', color='data')
    n['copy_request'] = node('GPU：本次传输调用', [
        '源与目标地址、字节数和目标 stream。',
        '传输方向决定使用哪个 Memcpy 入口。'],
        'CUDA memcpy', 'Stream', tag='产生 · 传输参数', color='data')
    n['capture_input'] = node('CUDA：调用方提供的图', [
        'CUgraph、stream、依赖和 capture mode。',
        '调用方选择需要记录的 stream 操作区间。'],
        'CUDA graph capture', tag='产生 · 捕获输入', color='neutral')
    n['captured_graph'] = node('CUDA：已记录的图', [
        'EndCapture 确认工作写入给定 graph。',
        '后续实例化和启动由调用方安排。',
        '记录结束时，这些工作尚不因此执行。'],
        'CUDA graph capture', tag='变换结果 · 图记录', color='neutral')
    n['command_ready'] = node('GPU：本次可提交的命令', [
        '记录引用本次有效的存储地址。',
        '可以来自首次记录、更新或复用。'],
        'Command buffer submission', tag='变换结果 · 命令记录', color='data')
    n['downstream'] = node('完成状态与结果存储', [
        '计算结果保存在本次关联的输出存储中。',
        '上层通过设备实现取得 Buffer 及其就绪状态。',
        'CPU 事件与 CUDA 等待采用各自的完成机制。'],
        'CPU execute', 'GetReadyFuture', 'CUDA synchronize',
        tag='消费结果 · 返回上层', color='completion')
    for key, value in n.items():
        value['owner'] = ('XLA CPU runtime' if key.startswith('cpu_') else
                          '设备运行时与驱动' if key in {'core', 'd0', 'd2', 'd3', 'downstream'} else
                          'XLA GPU runtime / CUDA')
    _definition(spec, 'CPU runtime / StreamExecutor / CUDA')
    _section(spec, 'view_production', '产生 · 从程序与数据准备请求', 'PJRT / XLA runtime',
             [['cpu_request'], ['p0', 'args', 'launch_request'],
              ['p2', 'copy_request', 'p1'], ['capture_input']],
             'CPU 准备本次执行；GPU 分别准备 kernel、传输和命令记录所需的参数。')
    _section(spec, 'view_transformation', '变换 · 绑定参数、依赖与记录', 'CPU runtime / XLA GPU runtime',
             [['cpu_context'], ['t2', 'command_ready'], ['t1', 'captured_graph']],
             '处理本次存储和执行依赖；命令按需记录或更新，普通 kernel 与传输可直接提交。')
    _section(spec, 'view_consumption', '消费 · 执行工作并传播结果', 'CPU runtime / StreamExecutor / CUDA',
             [['cpu_execute', 'cpu_done'], ['c0', 'c1', 'command_submit'],
              ['submit_status', 'work'], ['event', 'wait'], ['c3', 'r3'], ['downstream']],
             'CPU 调度 thunk；CUDA 提交异步工作。输出存储、提交状态与完成状态分别返回。')
    chain(spec, ['cpu_request', 'cpu_context', 'cpu_execute', 'cpu_done'],
          ['存储与输入依赖', 'ExecuteParams / 依赖满足后', '执行事件与结果存储'], 'data')
    spec['edges'][-1]['kind'] = 'output'
    edge(spec, 'cpu_done', 'downstream', '设备实现传播完成或错误', 'completion')
    chain(spec, ['p0', 'args', 'launch_request', 'c0'],
          ['GetKernelAndArgs', '函数、地址和启动参数', '直接提交 kernel'], 'data')
    spec['edges'][-1]['channel'] = 'direct'
    chain(spec, ['p2', 'copy_request', 'c1'], ['方向与地址', '直接提交传输'], 'data')
    spec['edges'][-1]['channel'] = 'direct'
    edge(spec, 'p1', 't2', '首次记录或需要更新时', 'data', optional=True)
    edge(spec, 't2', 'command_ready', '本次有效的命令记录', 'data')
    edge(spec, 'p1', 'command_ready', '无需重录时复用', 'data', optional=True)
    edge(spec, 'command_ready', 'command_submit', '命令记录与 stream', 'data')
    chain(spec, ['capture_input', 't1', 'captured_graph'],
          ['进入捕获区间', '结束记录'], 'neutral')
    for key in ('c0', 'c1', 'command_submit'):
        edge(spec, key, 'submit_status', '提交调用返回 Status', 'output')
        edge(spec, key, 'work', '普通执行模式下提交成功', 'data')
    edge(spec, 'work', 'event', '按需记录完成点', 'completion', optional=True)
    edge(spec, 'event', 'wait', '后续工作等待事件', 'completion')
    edge(spec, 'work', 'c3', '主机按需等待 stream', 'completion', optional=True)
    edge(spec, 'c3', 'r3', '等待成功或错误', 'completion')
    edge(spec, 'r3', 'downstream', '指定 stream 的完成观察', 'completion')
    spec['paths'].extend([
        ['cpu_request', 'cpu_context', 'cpu_execute', 'cpu_done', 'downstream'],
        ['p0', 'args', 'launch_request', 'c0', 'work', 'c3', 'r3'],
        ['p1', 't2', 'command_ready', 'command_submit', 'submit_status'],
        ['p1', 'command_ready', 'command_submit', 'work', 'event', 'wait'],
        ['p2', 'copy_request', 'c1', 'work']])
    return spec


def apply_runtime_views(spec):
    """Adapt a raw component builder result; leave other component keys intact."""
    adapter = {'ifrt': _array_views, 'pjrt': _buffer_views,
               'runtime': _submission_views}.get(spec['key'])
    return adapter(spec) if adapter else spec
