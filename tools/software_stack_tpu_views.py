"""TPU abstractions: pinned public interfaces and attributed internal accounts."""
from __future__ import annotations

TPU_INTERNAL_EVIDENCE = {
    'url': 'https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064434151',
    'followup_url': 'https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064481581',
    'corrections_url': 'https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064767267',
    'kind': 'Bot account of private implementation; independently checked public interfaces are separate',
    'internal_revision': None,
    'libtpu_build': None,
    'symbol_scope': 'Private component names denote responsibilities, not disclosed C++ symbols',
    'execution_evidence': 'No compilation, simulation or device execution performed; the bot corrected observation claims to static-source verification methods',
}


def apply_tpu_views(spec):
    if spec['key'] != 'tpu':
        return spec
    from software_stack_component_flow_data import node, edge, chain

    s = spec
    s.update(
        layout='component_views', nodes={}, sections=[], edges=[], paths=[],
        details=[], omit=set(),
        title='TPU TensorCore 后端（libtpu） · 原生 LLO',
        summary='普通 HLO 与 Mosaic TC 汇合到原生 LLO；SC 走独立的 MLO / LLVM 路线。公开接口与内部答复分别标注。',
        foot='私有类名未披露，emitter / 桥接器等是职责名称。内部路线据 issue #76 转述，无内部 revision 或 wheel 映射；阶段可重复或重试。',
        evidence_note='证据：固定公开源码 + issue #76 内部实现转述；完成 SVG 静态检查，未做 CPU / TPU 计算、离线 TPU 编译或 TPU 模拟。',
        external_evidence=[TPU_INTERNAL_EVIDENCE],
        view_reading=[
            '研究对象：TC 原生 LLO。左侧区分 HLO、Mosaic、MLIR llo 与原生 LLO；右侧连接产生、变换、消费。',
            '“内部答复”节点链接至 bot 原文，职责名称不冒充私有源码符号。SC 分支单列，最终与 TC 程序共同封装。',
        ],
    )
    n = s['nodes']

    def card(key, title, rows, *refs, tag, color='program', internal=False):
        n[key] = node(title, rows, *refs, tag=tag, color=color)
        if internal:
            n[key].update(link=TPU_INTERNAL_EVIDENCE['url'],
                          link_label='内部答复 · issue #76 ↗')

    card('core', '原生 LLO', [
        'TC 后端的低层程序对象。',
        '层级区域组织指令、循环、条件区域和子区域。',
        '值、局部存储及同步依赖约束后续调度和编码。',
    ], tag='定义 · 内部答复', color='unknown', internal=True)
    card('d0', '区域 / 指令 / 值 / 局部存储', [
        '分配前以定义和使用关系连接虚拟值。',
        '后续绑定物理寄存器、存储位置与指令束。',
        '依赖、DMA 可用性及目标资源约束必须保持。',
        '未取得私有类定义或稳定的外部 schema。',
    ], tag='定义 · 内部答复', color='neutral', internal=True)
    card('hlo_definition', 'HLO 与编译期规划', [
        'HloModule 表达张量计算和依赖。',
        'HloSchedule 是 HLO 顺序，不是逐周期指令表。',
        'BufferAssignment 是存储方案，不是设备 Buffer。',
    ], 'HloModule', 'HloSchedule', 'BufferAssignment',
         tag='定义参照 · 公开对象', color='neutral')
    card('d3', 'Mosaic TPU MLIR ≠ LLO', [
        'Pallas kernel 使用 tpu / vector / memref / scf 等操作。',
        '公开设计说明 Mosaic 会继续生成 LLO。',
        'MLIR bytecode 的序列化也不等于机器码。',
    ], 'lower_module_to_custom_call', 'Mosaic to LLO design',
         tag='定义参照 · 公开设计', color='neutral')
    card('mlir_llo_definition', 'MLIR llo 与原生 LLO', [
        'MLIR llo 是 Mosaic TC 的过渡方言。',
        '桥接构建器将其转为后端原生程序对象。',
        '普通 HLO 直接进入原生 LLO，不要求经过 Mosaic。',
    ], tag='定义参照 · 内部答复', color='neutral', internal=True)
    card('d2', '不透明句柄与低层表示', [
        'XLA_TpuProgram 的公开声明不定义 LLO。',
        'PJRT Executable 序列化是平台封装，不能直接称为 LLO。',
        '可选 LLO 调试信息可交给 Profiler / XProf 消费。',
    ], 'TPU opaque program', 'PJRT serialize ABI', 'LLO profiling metadata',
         tag='定义参照 · 公开接口', color='neutral')

    card('p0', 'make_tpu_client', [
        '按需加载并初始化 libtpu.so 的 PJRT 插件。',
    ], 'TPU plugin', tag='产生 · 公开 Client 入口')
    card('d1', 'PJRT_Program + CompileOptions', [
        'SerializeProgram 序列化外层 MLIR 或 HLO proto。',
        '传递程序与静态选项，实际 Buffer 在执行时传入。',
    ], 'C API program serialization', tag='产生 · 公开编译输入')
    card('p1', 'PJRT_Client_Compile', [
        'InitializeArgsAndCompile 调用插件函数指针。',
        '内部编译并加载，公开结果为 LoadedExecutable。',
    ], 'C API compile call', tag='产生 · 公开提交接口')
    card('hlo_input', 'TPU HLO、调度与存储规划', [
        '沿用 HloModule / Computation / Instruction。',
        '优化、分片、布局、融合、异步改写与调度后分流。',
        'HloSchedule 与存储规划供后端 lowering 消费。',
    ], tag='产生 · 内部答复', internal=True)
    card('plain_emit', '普通 HLO emitter', [
        '读取 HLO、窗口配置和存储方案。',
        '直接构造原生 LLO 的循环、搬运与计算。',
    ], tag='产生 · TC 路线一 / 内部答复', internal=True)
    card('mosaic_input', 'Mosaic TC 内嵌 Module', [
        'Pallas lowering 后序列化到 tpu_custom_call。',
        '外层编译遇到调用，再编译内嵌 kernel。',
    ], 'lower_module_to_custom_call', tag='产生 · TC 路线二')
    card('mosaic_llo', 'Mosaic lowering → MLIR llo', [
        '布局与向量 lowering 后，lower-to-llo 生成过渡方言。',
        'eliminate-llo-extensions / finalize-llo 继续处理。',
        '桥接器把内核写入外层原生 LLO 的当前区域。',
    ], tag='产生 · 内部答复', internal=True)
    card('llo_producer', '两条路线汇合：原生 LLO', [
        '普通 HLO 与 Mosaic TC 接入同一设备程序。',
        '默认内联；函数去重可提取子函数后再链接。',
    ], tag='产生结果 · 内部答复', internal=True)
    card('sc_input', 'SC HLO / Mosaic SC', [
        'SC 执行线程与 TC 分流。',
        '单 Mosaic Module 混合 TC / SC 在公开前端被拒绝。',
        '两个 kernel 仍可通过跨核 DMA 交接数据。',
    ], 'TPU core classification', tag='产生 · SC 分支', internal=True)
    card('sc_mlo', 'MLO / sparse_core', [
        'SC HLO lowering 或 Mosaic lower-to-mlo 产生。',
        '不接入 TC 的原生 LLO 流水线。',
    ], tag='产生结果 · 内部答复', internal=True)

    card('t0', '原生 LLO 输入', [
        '区域、虚拟值、局部存储和同步依赖。',
    ], tag='变换输入 · 内部答复', internal=True)
    card('t1', 'LLO 优化与目标 lowering', [
        '改写循环和地址，提升存储，拆解伪操作。',
        '按数据、别名和资源依赖安排指令与 DMA。',
        '对象仍为原生 LLO；不是新一层张量 IR。',
    ], tag='变换 · 内部答复', color='transform', internal=True)
    card('t2', '寄存器与局部存储分配', [
        '虚拟值绑定寄存器；窗口、scratch 与同步资源确定位置。',
        '必要时插入 spill / fill，修改低层程序。',
        '容量压力可引起窗口调整与重新 lowering。',
    ], tag='变换 · 内部答复', color='transform', internal=True)
    card('t3', '指令束打包与编码', [
        '满足目标延迟、槽位、端口和控制流约束。',
        'LLO → 指令束序列 → 编码、重定位和链接。',
        '生成 TC 程序；固定硬件常数不能跨代际套用。',
    ], tag='消费 LLO · 内部答复', color='transform', internal=True)
    card('sc_codegen', 'SC 优化与 LLVM 后端', [
        'MLO 优化后降低为 LLVM IR。',
        'SC 目标后端生成相应子程序与存储信息。',
    ], tag='变换 · SC 独立路线 / 内部答复', color='transform', internal=True)

    card('c0', 'TC 程序镜像', [
        '组织目标代码、常量、存储需求及重定位信息。',
    ], tag='消费结果 · 内部答复', color='result', internal=True)
    card('sc_program', 'SC 程序镜像', [
        '与 TC 程序配套，通过调用信息、共享存储和同步协作。',
    ], tag='消费结果 · 内部答复', color='result', internal=True)
    card('executable_input', 'PJRT_Executable', [
        '封装编译产物；内部答复描述 TC / SC 程序及编译元数据。',
        '公开提供序列化等能力。',
    ], 'PJRT serialize ABI', tag='消费 · 产物封装', color='result', internal=True)
    card('loaded', 'PJRT_LoadedExecutable', [
        '与设备关联的已加载程序执行能力。',
        '实际加载、存储绑定与提交由 provider 负责。',
    ], 'PJRT compile ABI', tag='消费 · 加载结果', color='result')
    card('c2', 'PJRT_Executable_Serialize', [
        '输出平台特定的封装字节；格式不保证跨时间稳定。',
    ], 'PJRT serialize ABI', tag='消费 · 产物观察', color='neutral')
    card('serialized_executable', '可执行对象的序列化字节', [
        '不能将整体直接称为 LLO 或裸机器码。',
    ], 'PJRT serialize ABI', tag='消费结果 · 公开契约', color='neutral')
    card('input', '输入 Buffer 与执行选项', [
        '按设备提供实参，运行时遵守就绪与所有权约定。',
    ], 'PJRT execute ABI', tag='运行时输入', color='data')
    card('execute', 'PJRT_LoadedExecutable_Execute', [
        'C API 包装层按设备整理参数并调用插件。',
        '输出列表是句柄容器；provider 管理实际设备存储。',
        '运行时绑定 Buffer 并提交已编译程序。',
    ], 'C API execute call', tag='消费 · 执行入口', color='data')
    card('outputs', '输出 Buffer → 逻辑数组', [
        '输出对象可以在计算完成前返回。',
        'IFRT 收集输出分片并补充类型、形状与分片信息。',
    ], 'PJRT execute ABI', tag='运行结果 · 存储对象', color='output')
    card('events', '逐设备完成事件 → Future', [
        '按需返回 device_complete_events。',
        'ConvertCEventToCppFuture 转换完成或错误状态。',
        '一次 DMA 完成不等于整个设备执行完成。',
    ], 'PJRT execute ABI', 'C API execute call', tag='运行结果 · 完成状态', color='completion')

    def part(ident, title, question, owner, rows):
        s['sections'].append(dict(id=ident, title=title, question=question, owner=owner, rows=rows))
        for row in rows:
            for key in row:
                if key:
                    n[key]['owner'] = owner

    part('view_definition', '定义 · TC 原生 LLO 表示什么',
         '核心结构据内部答复；公开 HLO 和接口对象提供参照。', 'libtpu；公开参照为 JAX / XLA',
         [['core'], ['d0'], ['hlo_definition'], ['d3'], ['mlir_llo_definition'], ['d2']])
    part('view_production', '产生 · 两条 TC 路线与 SC 分支',
         'HLO 与 Mosaic TC 汇合到原生 LLO；SC 单独降低。', 'libtpu；公开提交为 JAX / PJRT',
         [['p0', 'd1', 'p1'], ['hlo_input'], ['plain_emit', 'mosaic_input', 'sc_input'],
          [None, 'mosaic_llo', None], ['llo_producer', None, 'sc_mlo']])
    part('view_transformation', '变换 · 从虚拟值到目标程序',
         '按依赖和目标资源改写程序；处理可重复，容量不足可重试。', 'libtpu 编译器 / TC 与 SC 后端',
         [['t0', 'sc_codegen'], ['t1', None], ['t2', None], ['t3', None]])
    part('view_consumption', '消费 · 封装、加载与 Buffer 执行',
         '机器程序、可执行对象、结果存储和完成状态分别表达。', 'libtpu provider / 加载器 / 运行时',
         [['c0', 'sc_program'], ['executable_input'], ['c2', 'loaded'],
          ['serialized_executable', 'input'], ['execute'], ['outputs', 'events']])

    edge(s, 'p0', 'p1', '选择 TPU Client')
    edge(s, 'd1', 'p1', '程序与编译选项')
    edge(s, 'p1', 'hlo_input', '导入与 TPU HLO 编译')
    edge(s, 'hlo_input', 'plain_emit', '普通 TC 计算')
    edge(s, 'hlo_input', 'mosaic_input', '遇到 TC custom call')
    edge(s, 'hlo_input', 'sc_input', 'SC 执行线程', optional=True)
    edge(s, 'plain_emit', 'llo_producer', '直接构造低层程序')
    chain(s, ['mosaic_input', 'mosaic_llo', 'llo_producer'], ['Mosaic lowering', '桥接到外层原生 LLO'])
    edge(s, 'sc_input', 'sc_mlo', 'SC lowering')
    chain(s, ['llo_producer', 't0', 't1', 't2', 't3', 'c0'],
          ['交给 TC 后端', '虚拟值与局部存储', '目标资源规划', '完成必要物理分配', 'TC 目标程序'], 'transform')
    chain(s, ['sc_mlo', 'sc_codegen', 'sc_program'], ['MLO → LLVM IR', 'SC 目标程序'], 'transform')
    edge(s, 'c0', 'executable_input', 'TC 代码与元数据', 'result')
    edge(s, 'sc_program', 'executable_input', '可选 SC 程序与调用信息', 'result', optional=True)
    edge(s, 'executable_input', 'loaded', '加载并关联设备', 'result')
    chain(s, ['executable_input', 'c2', 'serialized_executable'], ['序列化对象', '平台封装字节'], 'neutral')
    edge(s, 'loaded', 'execute', '复用已加载程序', 'data')
    edge(s, 'input', 'execute', '实参与依赖', 'data')
    edge(s, 'execute', 'outputs', '返回结果存储', 'output')
    edge(s, 'execute', 'events', '按需返回完成状态', 'completion', optional=True)
    s['paths'].extend([
        ['p1', 'hlo_input', 'plain_emit', 'llo_producer', 't0', 't1', 't2', 't3', 'c0', 'executable_input', 'loaded', 'execute', 'outputs'],
        ['input', 'execute', 'events'],
    ])
    return s
