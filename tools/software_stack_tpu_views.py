"""Separate LLO research questions from the verified public TPU ABI flow."""
from __future__ import annotations


def apply_tpu_views(spec):
    if spec['key'] != 'tpu':
        return spec
    from software_stack_component_flow_data import node, edge, chain

    s = spec
    n = s['nodes']
    s.update(layout='component_views', sections=[], edges=[], paths=[], details=[],
             title='TPU / libtpu · LLO 四视角与公开接口调用链',
             summary='定义、产生、变换、消费分别保留 LLO 的证据缺口；已核验的插件编译与执行调用链单独展开。',
             foot='四个研究区不补造 LLO 内部调用。灰色附区只表达公开 ABI 的参数、返回与生命周期；不透明句柄、Mosaic TPU MLIR 和序列化字节均不能替代 LLO 定义。')
    n['llo_producer'] = node('LLO 的构造者与输入 · 待取证', [
        '具体类、函数、调用位置与输入 IR 尚未确认。',
        '公开 Compile 接口只给出程序输入与 executable 输出。',
        '需要对应 libtpu 构建的实现或可核验的阶段产物。',
    ], tag='产生 · 内部证据缺口', color='unknown')
    n['llo_producer']['uncertain'] = True
    n['c0']['tag'] = '消费 · 内部证据缺口'
    n['r0']['tag'] = '消费输出 · 内部证据缺口'
    for key in ('c0', 'r0'):
        n[key].update(color='unknown', uncertain=True)
    s['omit'].difference_update({'c0', 'r0'})

    # Serialize takes a PJRT_Executable. Do not invent a direct call from the
    # PJRT_LoadedExecutable returned by Compile to this distinct ABI argument.
    n['c2'].update(title='PJRT_Executable_Serialize', tag='公开 · 可执行对象序列化', color='neutral',
                   rows=['输入：PJRT_Executable，不是已确认的 LLO 对象。',
                         '返回平台特定字节、拥有这些字节的对象与 deleter。',
                         '格式不保证跨时间稳定，不能据此定义 LLO。'])
    n['executable_input'] = node('待序列化的 PJRT_Executable', [
        '此处从 Serialize 的公开参数开始。',
        '编译返回的 LoadedExecutable 是另一类句柄。',
    ], 'PJRT serialize ABI', tag='公开 · 序列化输入', color='neutral')
    n['serialized_executable'] = node('平台特定序列化字节', [
        'serialized_bytes / serialized_bytes_size 描述返回数据。',
        'serialized_executable 拥有数据；调用 deleter 释放。',
        '当前证据不足以将字节格式标为 LLO。',
    ], 'PJRT serialize ABI', tag='公开 · 序列化结果与所有权', color='neutral')
    n['p0']['tag'] = '公开 · 插件 Client 入口'
    n['d1']['tag'] = '公开 · 编译输入契约'
    n['p1']['tag'] = '公开 · 消费提交的程序'
    n['execute']['tag'] = '公开 · 消费 executable 与 Buffer'
    n['events']['rows'] = [
        '调用者可请求每设备完成事件，或将事件输出设为空。',
        '已请求的事件在对应设备完成时就绪；错误时不填充。',
        '只核验接口声明，不构成真实 TPU 执行证据。',
    ]

    def part(ident, title, owner, rows, external=False):
        s['sections'].append(dict(id=ident, title=title, owner=owner,
                                  rows=rows, external=external, auxiliary=external))
        for row in rows:
            for key in row:
                if key:
                    n[key]['owner'] = owner

    part('view_definition', '定义 · LLO 的语义与证据边界', 'libtpu · 待版本证据',
         [['core'], ['d0'], ['d2'], ['d3']])
    part('view_production', '产生 · 构造者与输入待确认', 'libtpu · 待版本证据',
         [['llo_producer']])
    part('view_transformation', '变换 · 规则与阶段待确认', 'libtpu · 待版本证据',
         [['t0', 't1'], ['t2', 't3']])
    part('view_consumption', '消费 · 使用者与结果待确认', 'libtpu · 待版本证据',
         [['c0'], ['r0']])
    part('downstream', '公开接口调用链 · 不等同于 LLO 内部流程', '公开 · JAX / PJRT ABI',
         [['p0', 'd1', 'p1', 'loaded'],
          ['input', 'execute', 'outputs', 'events'],
          ['executable_input', 'c2', 'serialized_executable']], external=True)

    edge(s, 'p0', 'p1', 'Client + 插件编译入口')
    edge(s, 'd1', 'p1', 'PJRT_Program + CompileOptions')
    edge(s, 'p1', 'loaded', '不透明 LoadedExecutable', 'result')
    edge(s, 'loaded', 'execute', '已加载程序句柄', 'data')
    edge(s, 'input', 'execute', 'Buffer 参数', 'data')
    edge(s, 'execute', 'outputs', '输出存储句柄', 'output')
    edge(s, 'execute', 'events', '按需请求完成事件', 'completion', optional=True)
    chain(s, ['executable_input', 'c2', 'serialized_executable'],
          ['待序列化的可执行对象', '平台特定字节与释放约定'], 'neutral')
    s['paths'].extend([['p0', 'p1', 'loaded', 'execute', 'outputs'],
                       ['d1', 'p1', 'loaded', 'execute', 'events'],
                       ['input', 'execute', 'outputs']])
    return s
