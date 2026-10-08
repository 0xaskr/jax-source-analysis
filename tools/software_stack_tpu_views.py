"""Keep LLO's four questions separate from the public TPU program boundary."""
from __future__ import annotations


def apply_tpu_views(spec):
    if spec['key'] != 'tpu':
        return spec
    from software_stack_component_flow_data import node, edge, chain

    s = spec
    n = s['nodes']
    s.update(
        layout='component_views', sections=[], edges=[], paths=[], details=[],
        title='TPU provider / libtpu · LLO 的定义、产生、变换与消费',
        summary='LLO 保留为 TPU 私有程序表示研究项。公开源码能定位编译输入与返回结果，内部表示需对应 libtpu 版本的证据。',
        foot='四区围绕 LLO；产生与消费区同时保留公开接口作为参照。连线只连接已知的程序、句柄与结果，LLO 内部过程不补画箭头。本页为源码检查。',
    )

    def card(key, title, rows, *refs, tag, color='unknown', uncertain=False):
        n[key] = node(title, rows, *refs, tag=tag, color=color)
        if uncertain:
            n[key]['uncertain'] = True

    card('core', 'LLO（待版本证据）', [
        '总览保留的 TPU 私有程序表示研究项。',
        '结构、语义和阶段位置尚无可核验的版本定义。',
    ], tag='核心抽象', uncertain=True)
    card('d0', '要定义的程序内容', [
        '操作与类型如何表达计算；依赖和状态如何约束执行。',
        '这些内容还未落实到 LLO 的类、字段或格式。',
    ], tag='定义 · 语义与结构', uncertain=True)
    card('d2', '公开句柄提供什么', [
        'XLA_TpuProgram 只有不透明类型声明。',
        '它没有公开内部程序的结构与语义。',
    ], 'TPU opaque program', tag='定义参照 · 程序句柄', color='neutral')
    card('d3', 'Mosaic TPU MLIR', [
        'Pallas 可生成的专用 kernel 表示，通过 custom call 接入。',
        '它有自己的定义，不能称为 LLO。',
    ], 'lower_module_to_custom_call', tag='定义参照 · 另一种已知表示', color='neutral')

    card('llo_producer', '构造 LLO 的位置', [
        '待定位：调用位置、构造函数及输入表示。',
        '公开编译入口只确定提交程序，不暴露此步骤。',
    ], tag='产生 · 内部构造', uncertain=True)

    card('t0', '变换实现', [
        '待定位执行变换的类或函数及其调用者。',
    ], tag='变换 · 由谁处理', uncertain=True)
    card('t1', '输入与产物', [
        '待确认变换前后表示的格式与版本。',
    ], tag='变换 · 改变什么', uncertain=True)
    card('t2', '变换规则', [
        '待确认合法性检查、重写条件与语义约束。',
    ], tag='变换 · 按什么规则', uncertain=True)
    card('t3', '阶段关系', [
        '待确认与调度、存储规划、代码生成的关系。',
    ], tag='变换 · 在哪里发生', uncertain=True)

    card('c0', '使用 LLO 的编译步骤', [
        '待定位读取 LLO 的函数及其所需信息。',
        '当前无法把它连接到具体代码生成入口。',
    ], tag='消费 · 内部使用者', uncertain=True)
    card('r0', '消费 LLO 后得到什么', [
        '待确认下一种程序表示或执行产物的格式。',
        '公开结果是已加载程序句柄，二者之间的路径未知。',
    ], tag='消费 · 内部产物', uncertain=True)
    s['omit'].difference_update({'c0', 'r0'})

    # Observable contracts are not producers or consumers of the unknown IR.
    card('p0', 'make_tpu_client', [
        '按需加载、初始化 TPU 插件，取得 Client。',
    ], 'TPU plugin', tag='公开上下文 · 插件入口', color='program')
    card('d1', 'PJRT_Program', [
        'code 与 format 描述提交的程序字节及格式。',
        'API 列出 MLIR / HLO 格式，由具体实现接收。',
    ], 'PJRT Program', tag='公开上下文 · 编译输入', color='program')
    card('p1', 'PJRT_Client_Compile', [
        '接收 Client、程序与序列化 CompileOptions。',
        '成功时返回已加载可执行对象。',
    ], 'PJRT compile ABI', tag='公开上下文 · 编译接口', color='program')
    card('loaded', 'PJRT_LoadedExecutable', [
        '封装已加载程序的执行能力。',
    ], 'PJRT compile ABI', tag='公开上下文 · 编译结果', color='result')
    card('input', '输入 Buffer 与执行选项', [
        '输入 Buffer 按设备组织；执行选项描述本次调用。',
    ], 'PJRT execute ABI', tag='公开上下文 · 运行时输入', color='data')
    card('execute', 'PJRT_LoadedExecutable_Execute', [
        '用已加载程序处理输入 Buffer。',
    ], 'PJRT execute ABI', tag='公开上下文 · 执行接口', color='data')
    card('outputs', '输出 Buffer', [
        '返回结果存储的句柄，供上层包装数组。',
    ], 'PJRT execute ABI', tag='公开上下文 · 结果存储', color='output')
    card('events', '设备完成事件', [
        '按需请求 device_complete_events，观察执行完成。',
        'Execute 返回错误时不填充事件输出。',
    ], 'PJRT execute ABI', tag='公开上下文 · 完成状态', color='completion')
    card('executable_input', 'PJRT_Executable', [
        '可执行对象序列化接口的输入。',
    ], 'PJRT serialize ABI', tag='公开上下文 · 可序列化对象', color='neutral')
    card('c2', 'PJRT_Executable_Serialize', [
        '从可执行对象取得平台特定的序列化字节。',
    ], 'PJRT serialize ABI', tag='公开上下文 · 产物观察', color='neutral')
    card('serialized_executable', '可执行对象的序列化字节', [
        '格式不保证跨时间稳定，也未被确认为 LLO。',
    ], 'PJRT serialize ABI', tag='公开上下文 · 序列化产物', color='neutral')

    def part(ident, title, question, owner, rows):
        s['sections'].append(dict(
            id=ident, title=title, question=question, owner=owner,
            rows=rows,
        ))
        for row in rows:
            for key in row:
                if key:
                    n[key]['owner'] = owner

    part('view_definition', '定义 · 表示什么',
         'LLO 的语义、结构与约束尚需版本定义。', 'TPU provider / libtpu',
         [['core'], ['d0'], ['d2'], ['d3']])
    part('view_production', '产生 · 从哪里来',
         '上方是待定位的构造步骤；下方是已知的提交入口。', 'TPU provider / libtpu；公开入口为 JAX / PJRT',
         [['llo_producer'], ['p0', 'd1'], ['p1']])
    part('view_transformation', '变换 · 怎样处理',
         '以下是缺少的实现信息，不是已确认的 pass 顺序。', 'TPU provider / libtpu',
         [['t0', 't1'], ['t2', 't3']])
    part('view_consumption', '消费 · 交给谁',
         '上方是待定位的消费者；下方是编译结果的公开用途。', 'TPU provider / libtpu；公开结果为 PJRT 对象',
         [['c0', 'r0'], ['loaded', 'input', 'execute'],
          ['outputs', 'events'],
          ['executable_input', 'c2', 'serialized_executable']])

    edge(s, 'p0', 'p1', '选择 TPU Client')
    edge(s, 'd1', 'p1', '程序 + 编译选项')
    edge(s, 'p1', 'loaded', '已加载程序', 'result')
    edge(s, 'loaded', 'execute', '复用编译结果', 'data')
    edge(s, 'input', 'execute', '输入存储', 'data')
    edge(s, 'execute', 'outputs', '返回存储句柄', 'output')
    edge(s, 'execute', 'events', '按需返回事件', 'completion', optional=True)
    chain(s, ['executable_input', 'c2', 'serialized_executable'],
          ['待序列化对象', '平台特定字节'], 'neutral')
    s['paths'].extend([
        ['p0', 'p1', 'loaded', 'execute', 'outputs'],
        ['d1', 'p1', 'loaded', 'execute', 'events'],
        ['input', 'execute', 'outputs'],
    ])
    return s
