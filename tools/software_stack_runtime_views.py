"""Object-centred four-view layouts for IFRT, PJRT and execution submission.

The core card defines an abstraction. Concrete handles and requests carry the
operation flow; status, ownership and device completion remain separate results.
"""
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


def _definition(spec, owner):
    from software_stack_component_flow_data import edge
    _section(spec, 'view_definition', '定义 · 对象、结构与约束', owner,
             [['core'], ['d0'], ['d1'], ['d2'], ['d3']],
             '定义独立于具体调用；以下关系说明类型、存储与状态契约。')
    for key in ('d0', 'd1', 'd2', 'd3'):
        edge(spec, 'core', key, '结构与契约', 'neutral')


def _array_views(spec):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(spec, 'IFRT · Array 的定义与操作调用链',
        '定义独立说明逻辑 Array；产生、按需变换和消费以 ArrayRef 实例连接。',
        '展开 PJRT-backed 数值数组路径。数组操作独立可选，执行、主机读取、就绪观察和删除并列；输出句柄、执行状态与物理回收不是同一事件。')
    n['produced'] = node('已有 ArrayRef / 输入数组', [
        '主机数据入口或单设备数组组装得到逻辑数组。',
        '句柄关联 dtype、shape、sharding 和底层分片。',
        '可以直接消费，也可按接口约束变换后消费。'],
        'MakeArrayFromHostBuffer', 'Assemble arrays', tag='产生结果 · Array 实例', color='data')
    n['transform_input'] = node('Array 与操作参数', [
        'Array / 数组列表配合目标设备、映射或新 ArraySpec。',
        '复制语义决定复用、捐赠和复制行为。',
        '四个分支按需选择，不是数值程序计算。'],
        'CopyArrays', 'RemapArrays', 'Bitcast arrays', tag='变换入口 · 放置与表示', color='data')
    n['consume_input'] = node('待消费的 ArrayRef', [
        '接收产生或变换返回的一个或多个 Array。',
        '调用方按目标接口选择数组及相应上下文。',
        '执行、读取、观察与删除的结果各自不同。'],
        'ifrt Array', 'Value', tag='消费入口 · Array 实例', color='data')
    n['p2']['tag'] = '消费内部 · 构造输出 Array'
    _definition(spec, 'IFRT / PJRT-backed 实现')
    _section(spec, 'view_production', '产生 · 构造逻辑数组', 'IFRT / PJRT-backed 实现',
             [['p0', 'p1'], ['produced']],
             '主机存储与已有单设备数组是不同输入来源；执行结果也由消费路径内的包装器构造 Array。')
    _section(spec, 'view_transformation', '变换 · 放置、分片与表示', 'IFRT / PJRT-backed 实现',
             [['transform_input'], ['t0', 't1'], ['t2', 't3'], ['after']],
             '按操作协议处理数据与所有权，返回新的 Array 或分片列表。')
    _section(spec, 'view_consumption', '消费 · 执行、读取与生命周期', 'IFRT / PJRT-backed 实现',
             [['consume_input'], ['loaded', 'c0', 'extract'], ['call', 'buffers', 'p2'],
              ['r0', 'status'], ['c1', 'c2', 'c3'], ['r1', 'r2', 'r3']],
             '程序与数组在 Execute 汇合；其他消费者独立选择，数据就绪与执行 status 分开观察。')
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
    n = _prepare(spec, 'PJRT · Buffer 的定义与操作调用链',
        '定义独立说明设备存储句柄；产生、可选变换与消费以 PjRtBuffer 实例连接。',
        '执行、主机读取、就绪观察、外部引用及删除分别消费 Buffer。Delete / Release 不产生新的可执行输入；捐赠和异步回收遵守 provider 契约。')
    n['produced'] = node('已有 PjRtBuffer / 输入句柄', [
        '主机数据入口返回设备存储句柄。',
        '传输可以尚未完成；使用时遵守就绪与所有权约定。',
        '执行返回的结果 Buffer 也可由调用方继续使用。'],
        'BufferFromHostBuffer', 'PjRtBuffer', tag='产生结果 · Buffer 实例', color='data')
    n['transform_input'] = node('Buffer 与操作参数', [
        '目标 MemorySpace、新类型 / 布局或额外依赖。',
        '按需复制、重解释，或捐赠并附加控制依赖。',
        '这些操作不等同于数值计算或后端编译。'],
        'CopyToMemorySpace', 'Buffer bitcast', 'Buffer dependency', tag='变换入口 · 存储与依赖', color='data')
    n['consume_input'] = node('待消费的有效 Buffer', [
        '接收产生或变换返回的 Buffer 句柄。',
        '调用方按设备、类型和生命周期协议组织使用。',
        '不同消费者独立选择，不组成必经序列。'],
        'PjRtBuffer', 'PJRT Execute', tag='消费入口 · Buffer 实例', color='data')
    n['after']['title'] = '变换结果：新的 Buffer'
    n['after']['rows'] = ['复制、Bitcast 或附加依赖返回新句柄。',
        '新句柄可以仍在等待数据或依赖；就绪另行观察。',
        '捐赠会改变原句柄资格，不能沿原输入继续使用。']
    n['t3']['tag'] = '消费 · 结束使用或移交所有权'
    _definition(spec, 'PJRT C++ 接口')
    _section(spec, 'view_production', '产生 · 获取设备存储句柄', 'PJRT / provider',
             [['p0'], ['produced']],
             'BufferFromHostBuffer 构造输入句柄；Execute 的输出句柄在消费结果中展开。')
    _section(spec, 'view_transformation', '变换 · 存储、解释与依赖', 'PJRT / provider',
             [['transform_input'], ['t0', 't1', 't2'], ['after']],
             '三个操作按需返回新的 Buffer；捐赠影响原句柄，Delete / Release 单列为生命周期消费。')
    _section(spec, 'view_consumption', '消费 · 程序、读取与所有权', 'PJRT / provider',
             [['consume_input'], ['loaded', 'c0', 'provider'], ['r0', 'status'],
              ['c1', 'c2', 'c3'], ['r1', 'r2', 'r3'], ['t3', 'invalid']],
             'Execute 汇合已加载程序和实参；其他消费接口分别产生数据、Future、外部引用或失效状态。')
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
    n = _prepare(spec, 'StreamExecutor / CUDA · 提交的定义与调用链',
        '定义独立说明一次执行提交；具体 kernel、传输与命令计划沿各自调用链进入平台接口。',
        '执行提交是行为抽象，不是统一请求类。命令记录 / 更新只在相应路径使用；CUDA capture 记录到调用者提供的图，不推定所有 CommandBuffer 经 capture。提交成功、排队工作与设备完成分开。')
    n['core']['rows'] = ['执行提交是把具体工作及依赖交给设备运行时的行为。',
        '工作、地址、启动参数与 stream 共同约定这次调用。',
        'Stream / CudaStream 等接口实现它，没有统一请求类。',
        '提交 Status 与异步完成分别观察。']
    n['launch_request'] = node('本次 kernel 与启动参数', [
        '已加载函数、实际设备地址和 launch 维度。',
        'shared memory 与 stream 配合指定本次启动。',
        '这是调用参数的汇总，不是新定义的请求类。'],
        'Kernel arguments', 'Stream launch', tag='产生结果 · 一次 kernel 调用', color='data')
    n['copy_request'] = node('本次传输参数', [
        '主机 / 设备源与目标、字节数和目标 stream。',
        '按方向、地址和平台约束选择 Memcpy 实现。'],
        'CUDA memcpy', 'Stream', tag='产生结果 · 一次传输调用', color='data')
    n['capture_input'] = node('调用者的图与待捕获区间', [
        '输入：CUgraph、stream、依赖和 capture mode。',
        '捕获协议让区间内的提交记录到给定 graph。',
        '本图不假定它就是普通 CommandBuffer 的构造入口。'],
        'CUDA graph capture', tag='产生输入 · 可选捕获协议', color='neutral')
    n['captured_graph'] = node('已记录的 CUgraph', [
        'EndCapture 确认工作写入调用者传入的 graph。',
        '图的后续实例化和启动由其调用方安排。',
        '记录结束不表示所记录的设备工作已经执行。'],
        'CUDA graph capture', tag='变换结果 · 记录到图', color='neutral')
    n['command_ready'] = node('本次可提交的 CommandBuffer', [
        '既有记录可直接复用，或在本次调用中完成记录 / 更新。',
        '记录中的地址必须满足本次存储生命周期约定。',
        '后续 Submit 消费命令记录和目标 stream。'],
        'Command buffer submission', tag='变换结果 / 复用结果 · 命令记录', color='data')
    n['p1']['rows'] = ['取得或创建 executor 对应的 CommandBuffer。',
        '检查首次记录与本次地址变化，决定记录 / 更新或直接复用。',
        '满足命令缓冲区路径条件后，继续调用 Submit。',
        '空命令、profiling、warmup 等分支可能返回或回退。']
    n['p2']['rows'] = ['输入：主机或设备地址、字节数与 stream。',
        '选择传输方向与对应平台入口。',
        '本图将请求参数与实际 CUDA 提交分开表达。']
    n['t1']['tag'] = '变换 · 捕获区间的记录方式'
    n['t1']['rows'] = ['BeginCapture 建立向给定 CUgraph 记录的区间。',
        '区间内的 stream 操作按 capture 协议被记录。',
        'EndCapture 结束区间；本身不是设备完成等待。']
    n['t2']['tag'] = '变换 · 命令记录与参数更新'
    _definition(spec, 'StreamExecutor / CUDA')
    _section(spec, 'view_production', '产生 · 准备本次工作与参数', 'XLA GPU runtime / StreamExecutor',
             [['p0', 'args', 'launch_request'], ['p2', 'copy_request', 'p1'], ['capture_input']],
             'kernel、传输和命令缓冲区分别准备调用；具体工作不经过核心定义卡。')
    _section(spec, 'view_transformation', '变换 · 按需记录或更新工作', 'XLA runtime / CUDA',
             [['t2', 'command_ready'], ['t1', 'captured_graph']],
             '命令记录可更新或复用；capture 是单独的条件记录协议，不是所有提交的必经步骤。')
    _section(spec, 'view_consumption', '消费 · 向平台提交具体工作', 'StreamExecutor / CUDA',
             [['c0', 'c1', 'command_submit'], ['submit_status', 'work']],
             '平台接口读取不同调用参数；同步提交结果与设备侧异步工作分别返回和推进。')
    _section(spec, 'downstream', '提交之后 · 依赖与完成观察', 'StreamExecutor / CUDA',
             [['event', 'wait', 'c3', 'r3']],
             '记录事件、后续 stream 等待、主机同步各自可选；完成观察不自动回传数组。', auxiliary=True)
    chain(spec, ['p0', 'args', 'launch_request', 'c0'],
          ['GetKernelAndArgs', '函数 / 地址 / launch 参数', '普通 kernel 直接提交'], 'data')
    spec['edges'][-1]['channel'] = 'direct'
    chain(spec, ['p2', 'copy_request', 'c1'], ['传输方向与地址', 'Memcpy 直接提交'], 'data')
    spec['edges'][-1]['channel'] = 'direct'
    edge(spec, 'p1', 't2', '首次记录 / 需要更新时', 'data', optional=True)
    edge(spec, 't2', 'command_ready', '本次有效的命令记录', 'data')
    edge(spec, 'p1', 'command_ready', '无需重录 · 复用现有记录', 'data', optional=True)
    edge(spec, 'command_ready', 'command_submit', '命令记录 + stream', 'data')
    chain(spec, ['capture_input', 't1', 'captured_graph'],
          ['按调用方的 capture 协议', '工作记录到给定 graph'], 'neutral')
    for key in ('c0', 'c1', 'command_submit'):
        edge(spec, key, 'submit_status', 'API 返回 Status', 'output')
        edge(spec, key, 'work', '普通执行模式 · 提交成功后', 'data')
    edge(spec, 'work', 'event', '按需记录此前工作的完成点', 'completion', optional=True)
    edge(spec, 'event', 'wait', '事件 / stream 依赖', 'completion')
    edge(spec, 'work', 'c3', '主机按需等待 stream', 'completion', optional=True)
    edge(spec, 'c3', 'r3', '完成 / 错误', 'completion')
    spec['paths'].extend([
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
