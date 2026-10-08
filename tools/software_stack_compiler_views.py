"""Four views of the selected compiler objects, with real call/object edges.

Adapters operate on fresh legacy recipes. Imports of their helpers stay local
so the shared recipe module can install these adapters without a module cycle.
"""
from __future__ import annotations


def _prepare(s, core_name, owner):
    n = s['nodes'] = {key: value for key, value in s['nodes'].items()
                      if key not in s.get('omit', set())}
    s.update(sections=[], edges=[], paths=[], details=[], omit=set(),
             layout='component_views', avoid_endpoint_captions=True,
             title=f'{owner} · {core_name} 的定义、产生、变换与消费',
             summary=f'围绕 {core_name}，说明它表示什么、从哪里产生、如何变换，以及交给谁使用。')
    return n


def _part(s, ident, title, owner, *rows, auxiliary=False):
    from software_stack_component_flow_data import section
    section(s, title, owner, *rows, external=auxiliary)
    s['sections'][-1].update(id=ident, auxiliary=auxiliary, rows=[list(row) for row in rows])


def _definitions(s, keys=('d0', 'd1', 'd2', 'd3')):
    from software_stack_component_flow_data import edge
    _part(s, 'view_definition', '定义 · 语义、结构与约束', s['nodes']['core'].get('owner') or '核心对象',
          ['core'], *([key] for key in keys))
    for key in keys:
        edge(s, 'core', key, '结构 / 语义与边界', 'neutral')


def _finish(s, old_ids, old_refs):
    _rewrite_content(s)
    consumption = next(p for p in s['sections'] if p['id'] == 'view_consumption')
    for part in list(s['sections']):
        if part['id'] == 'downstream':
            for row in part['rows']:
                consumption['rows'].extend(row[i:i + 3] for i in range(0, len(row), 3))
            s['sections'].remove(part)
    assigned = [key for part in s['sections'] for row in part['rows'] for key in row if key]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(s['nodes']):
        raise ValueError(f'{s["key"]}: each node must occur in exactly one view: '
                         f'{set(s["nodes"]) - set(assigned)}')
    if not old_ids <= set(s['nodes']):
        raise ValueError(f'{s["key"]}: lost stable node IDs')
    refs = {ref for n in s['nodes'].values() for ref in n['refs']}
    if not old_refs <= refs:
        raise ValueError(f'{s["key"]}: lost existing source references')
    pairs = {(e['source'], e['target']) for e in s['edges']}
    for path in s['paths']:
        if not all(pair in pairs for pair in zip(path, path[1:])):
            raise ValueError(f'{s["key"]}: incomplete call/object path {path}')
    return s


def _jaxlib(s):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(s, 'MLIR Module', 'jaxlib')
    n['produced'] = node('可供后续处理的 MLIR Module', [
        'JAX lowering 或 portable artifact 反序列化得到对象。',
        'Module 携带程序；编译选项与运行时数组另行提供。'],
        'lower_jaxpr_to_module', 'deserialize artifact', tag='产生结果 · 普通程序实例')
    n['consume_input'] = node('Module + 所选消费接口', [
        '编译提交、版本化序列化和验证 / 打印按用途选择。',
        '这些接口相互并列；并非先序列化才可以编译。'],
        'PyClient CompileAndLoad', 'serialize artifact', tag='消费输入 · 程序与调用选项')
    n['compat_module'] = node('HLO 兼容入口内部的 Module', [
        'ConvertHloToStablehlo 构造临时 MLIR 程序。',
        '本接口继续打印该模块，不直接返回 Python Module。'],
        'HLO to Module', tag='产生结果 · 兼容入口内部对象')
    n['compat_text'] = node('兼容入口返回的 Module 文本', [
        'PyXlaComputationToMlirModule 返回打印文本。',
        '若调用方需要 Module 对象，还需要相应解析步骤。'],
        'HLO to Module', tag='来源接口的返回值', color='neutral')
    n['mhlo_input'] = node('辅助转换入口解析 Module', [
        'PyMhloToStablehlo 接收模块文本或字节。',
        '解析得到含 MHLO 的 Module，再应用合法化 pass。'],
        'MHLO to StableHLO', tag='产生 · 辅助转换内部解析')
    n['mhlo_module'] = node('合法化后的 StableHLO Module', [
        'Module 内的 MHLO 操作经 pass 合法化为 StableHLO。',
        '辅助接口随后把这一对象写成普通 MLIR bytecode。'],
        'MHLO to StableHLO', tag='变换结果 · Module 实例', color='transform')
    n['mhlo_bytes'] = node('辅助转换返回的 MLIR bytecode', [
        '这是 PyMhloToStablehlo 的公开返回值。',
        '与指定目标版本的 portable artifact 分开。'],
        'MHLO to StableHLO', tag='后续封装 · 普通模块字节', color='neutral')
    n['t0']['tag'] = '变换 · 编译调用内克隆'
    n['t1']['tag'] = '变换 · lowering 条件 pass'
    n['t3']['tag'] = '变换 · HLO 导出内准备'
    n['t2']['tag'] = '变换 · 已解析 Module 的合法化'
    n['t2']['rows'] = ['PyMhloToStablehlo 内部消费已解析的 Module。',
                       '运行 MHLO → StableHLO legalization。',
                       '变换结果仍是 Module；bytecode 是随后封装结果。']
    _definitions(s)
    _part(s, 'view_production', '产生 · 构造、还原与辅助解析', 'JAX / jaxlib',
          ['p0', 'p1', 'produced'], ['p2', 'compat_module', 'compat_text'], ['mhlo_input'])
    _part(s, 'view_transformation', '变换 · Module pass 与克隆', 'JAX / jaxlib / XLA',
          ['t1', 't0', 'after'], ['t2', 'mhlo_module', 't3'])
    _part(s, 'view_consumption', '消费 · 编译、导出与观察', 'jaxlib / XLA',
          ['consume_input'], ['c0', 'c1'], ['c2', 'c3'])
    _part(s, 'downstream', '消费结果 · 框架包装、HLO 与字节', '下游接口与返回对象',
          ['ifrt_program', 'r0', 'r1'], ['r2', 'r3', 'mhlo_bytes'], auxiliary=True)
    edge(s, 'p0', 't1', '启用 Shardy · lowering 内部', 'transform', optional=True)
    edge(s, 't1', 'produced', '模块 pass 后返回 Module')
    edge(s, 'p0', 'produced', '未启用该 pass · Module', optional=True)
    edge(s, 'p1', 'produced', '反序列化后的 Module')
    chain(s, ['p2', 'compat_module', 'compat_text'], ['内部构造 StableHLO Module', '接口打印并返回文本'])
    chain(s, ['mhlo_input', 't2', 'mhlo_module', 'mhlo_bytes'],
          ['解析后的 Module', '合法化更新 Module', 'SerializeUsingBytecode'])
    s['edges'][-3]['kind'] = s['edges'][-2]['kind'] = 'transform'
    edge(s, 'produced', 'consume_input', '直接选择消费 · Module')
    s['edges'][-1]['channel'] = 'direct'
    for key, label in [('c0', 'Module + CompileOptions'), ('c2', 'Module + 目标版本'), ('c3', 'Module + 观察选项')]:
        edge(s, 'consume_input', key, label, optional=True)
    chain(s, ['c0', 't0', 'after', 'ifrt_program'],
          ['进入调用后 clone', '独立 Module 副本', 'HloProgram 包装副本'])
    s['edges'][-3]['kind'] = s['edges'][-2]['kind'] = 'transform'
    chain(s, ['ifrt_program', 'c1', 't3', 'r1'],
          ['公开下游路径 · Module', '导出内部 PrepareForExport', '准备后继续导出 HloModule'])
    s['edges'][-2]['kind'] = 'transform'
    edge(s, 'ifrt_program', 'r0', '编译加载完成后包装返回', 'result')
    edge(s, 'c2', 'r2', '版本化 portable artifact', 'neutral')
    edge(s, 'c3', 'r3', '判定 / 文本 / 普通 bytecode', 'neutral')
    s['paths'].extend([['p0', 'produced', 'consume_input', 'c0', 't0', 'after', 'ifrt_program'],
                       ['p0', 't1', 'produced', 'consume_input'],
                       ['p1', 'produced', 'consume_input', 'c2', 'r2'],
                       ['produced', 'consume_input', 'c3', 'r3']])
    s['foot'] = ('定义区的 Module 只说明结构。Shardy 是 lowering 内的条件 pass；clone 在 CompileAndLoad 内，'
                 'PrepareForExport 在 HLO 导出内。MHLO 辅助转换返回 bytecode，HLO 兼容入口返回文本。')


def _hlo(s):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(s, 'HloModule', 'XLA / HLO')
    n['produced'] = node('构造出的 HloModule', ['导入 MLIR、还原 proto 或直接组织 computation。',
        '具体程序实例遵守 HLO 的结构与阶段约束。'], 'HloModule', tag='产生结果 · HLO 程序实例')
    n['transform_input'] = node('HloModule + 变换配置', ['根据编译阶段选择 pipeline 或结构重写。',
        '更新方式、目标约束与 changed / status 按接口区分。'], 'HloPassPipeline RunImpl', tag='变换输入 · 已有程序', color='transform')
    n['consume_input'] = node('待检查、序列化或编译的 HLO', ['消费者并列；验证和文本导出可直接使用已有程序。',
        'RunBackend 要求调用方满足对应目标编译的前置条件。'], 'HloVerifier', 'CpuCompiler RunBackend', tag='消费输入 · HLO 与上下文')
    n['hook_result'] = node('调用阶段内的 HLO 状态', ['注册回调返回 bytes 时更新模块，None 时保留。',
        'PRE 接续原 pipeline；POST 接续调度后的处理。'], 'CPU POST_SCHEDULER', 'GPU POST_SCHEDULER', tag='变换结果 · 原调用阶段内', color='transform')
    _definitions(s)
    _part(s, 'view_production', '产生 · 导入、还原与直接构造', 'XLA HLO',
          ['p0', 'p1', 'p2'], ['produced'])
    _part(s, 'view_transformation', '变换 · Pipeline、结构改写与挂点', 'XLA 编译控制',
          ['transform_input', 't0', 't3'], ['t1', 't2', 'after'], ['hook', 'callback', 'hook_result'])
    _part(s, 'view_consumption', '消费 · 目标后端与观察接口', 'XLA',
          ['consume_input'], ['c0', 'c1'], ['c2', 'c3'])
    _part(s, 'downstream', '消费结果 · 规划、可执行对象与观察结果', '后端内部与返回对象',
          ['plan', 'return', 'r2', 'r3'], auxiliary=True)
    for key in ('p0', 'p1', 'p2'):
        edge(s, key, 'produced', 'HloModule 实例')
    edge(s, 'produced', 'transform_input', '按阶段选择变换', 'transform', optional=True)
    edge(s, 'transform_input', 't0', '已配置的 pass pipeline', 'transform', optional=True)
    edge(s, 'transform_input', 't3', '复制 / computation 替换', 'transform', optional=True)
    for key in ('t1', 't2'):
        edge(s, 't0', key, '配置该 pass 时调用', 'transform', optional=True)
        edge(s, key, 'after', '更新模块 / changed', 'transform')
    edge(s, 't0', 'after', 'pipeline 返回的程序状态', 'transform')
    edge(s, 't3', 'after', '新模块 / 更新引用关系', 'transform')
    edge(s, 'after', 'consume_input', '变换后的 HloModule')
    edge(s, 'produced', 'consume_input', '直接观察 / 满足后端前置条件', optional=True)
    s['edges'][-1]['channel'] = 'direct'
    for key in ('c0', 'c1', 'c2', 'c3'):
        edge(s, 'consume_input', key, 'HLO + 接口上下文', optional=True)
    edge(s, 't0', 'hook', 'CPU pipeline 中的 PRE', 'transform', optional=True)
    for key in ('c0', 'c1'):
        edge(s, key, 'hook', '后端调度后的 POST', 'neutral', optional=True)
        edge(s, key, 'plan', '内部调度 / BufferAssignment', 'neutral')
        edge(s, key, 'return', '目标 Executable', 'result')
    edge(s, 'hook', 'callback', '调用注册回调 · proto bytes', 'transform', optional=True)
    edge(s, 'callback', 'hook_result', '返回 bytes 更新 / None 保留', 'transform')
    edge(s, 'hook_result', 'after', 'PRE 返回原 pipeline', 'transform', optional=True)
    edge(s, 'hook_result', 'plan', 'POST 接续后端规划', 'neutral', optional=True)
    edge(s, 'c2', 'r2', '验证状态 / 诊断', 'neutral')
    edge(s, 'c3', 'r3', '文本 / HloModuleProto', 'neutral')
    s['paths'].extend([['p0', 'produced', 'transform_input', 't0', 'after', 'consume_input', 'c0', 'return'],
                       ['produced', 'transform_input', 't3', 'after', 'consume_input'],
                       ['produced', 'consume_input', 'c2', 'r2'],
                       ['produced', 'consume_input', 'c3', 'r3'],
                       ['c1', 'hook', 'callback', 'hook_result', 'plan']])
    s['foot'] = ('定义区是静态 HLO 结构。Pass 与 Clone / Replace 按需选择；验证、打印和目标后端是并列消费者。'
                 '挂点位于各后端自身阶段内；BufferAssignment 和 Executable 属于后续规划与编译产物。')


def _cpu(s):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(s, 'CPU LLVM IR', 'XLA CPU')
    n['produced'] = node('生成的 LLVM 主模块 / host kernels', ['HLO 代码生成得到 LLVM 函数、全局值及独立模块。',
        'ThunkSequence 是并行产生的执行计划，另行汇合。'], 'CPU IrEmitter2', 'CPU ThunkEmitter', tag='产生结果 · LLVM 程序实例')
    n['consume_input'] = node('可提交的 LLVM 编译单元', ['未拆分主模块与额外 kernels 可直接 AddModule。',
        '抽出 / 拆分片段经 helper 整理后也进入该接口。'], 'CPU AddModule', tag='消费输入 · 已有 LLVM Module')
    n['after'] = node('RunIrPasses 后的 LLVM Module', ['LLVM 优化在 IrCompiler 的调用内部进行。',
        '下一步由 EmitMachineCode 消费目标相关程序。'], 'CPU IR passes', tag='变换结果 · LLVM IR', color='transform')
    n['t0']['tag'] = '变换 · 编译内部 LLVM pass'
    n['units']['tag'] = '变换准备 · 条件提取与拆分'
    n['units']['rows'] = ['按 kernel 选项与并行编译配置选择模块处理分支。',
                          'ExtractKernels / SplitModule 产出的片段由 helper 整理。',
                          '未拆分主模块与额外 kernels 可以直接提交。']
    _definitions(s)
    _part(s, 'view_production', '产生 · HLO 生成目标 LLVM 程序', 'XLA CPU 代码生成',
          ['hlo', 'p0'], ['p1', 'p2'], ['produced'])
    _part(s, 'view_transformation', '变换 · 条件拆分与编译内优化', 'XLA CPU / LLVM',
          ['units'], ['t2', 't3', 't1'], ['t0', 'after'])
    _part(s, 'view_consumption', '消费 · 模块编译、目标发射与观察', 'XLA CPU / LLVM',
          ['consume_input', 'c1', 'c0'], ['c2', 'c3'])
    _part(s, 'downstream', '消费结果 · 代码、计划与可执行对象', 'XLA CPU 后端与 provider',
          ['thunks', 'symbols', 'executable'], ['r2', 'r3'], auxiliary=True)
    chain(s, ['hlo', 'p0', 'p1', 'p2', 'produced'],
          ['HLO 与目标配置', 'Module + emitter 上下文', '提供 kernel 生成能力', 'LLVM 主模块 / 独立 kernels'])
    s['edges'][-2]['kind'] = 'neutral'
    edge(s, 'p2', 'thunks', 'EmitEntryComputation → 执行计划', 'result')
    edge(s, 'produced', 'units', '需单独整理的模块', 'transform', optional=True)
    for key in ('t2', 't3'):
        edge(s, 'units', key, '满足相应条件时', 'transform', optional=True)
        edge(s, key, 't1', 'helper 消费模块片段', 'transform')
    edge(s, 't2', 't3', '剩余主模块 · 按配置拆分', 'transform', optional=True)
    edge(s, 't1', 'consume_input', '清理符号并组织 thread-safe 模块')
    edge(s, 'produced', 'consume_input', '未拆分主模块 / 额外 kernels', optional=True)
    s['edges'][-1]['channel'] = 'direct'
    chain(s, ['consume_input', 'c1', 't0', 'after', 'c0', 'symbols', 'executable'],
          ['AddModule / Compile', 'IrCompiler 内部优化', '优化后 LLVM Module',
           '目标发射', '目标对象与符号', '解析代码汇入可执行对象'])
    s['edges'][-5]['kind'] = s['edges'][-4]['kind'] = 'transform'
    s['edges'][-2]['kind'] = s['edges'][-1]['kind'] = 'result'
    edge(s, 'produced', 'c2', '生成后的 Module · verifier', 'neutral')
    edge(s, 'c2', 'r2', 'IR 合法性 / 错误', 'neutral')
    edge(s, 'c1', 'c3', 'IrCompiler 优化前 hook', 'neutral', optional=True)
    edge(s, 'after', 'c3', '优化后 hook / dump', 'neutral', optional=True)
    edge(s, 'c3', 'r3', '观察得到的 IR 文本', 'neutral')
    edge(s, 'thunks', 'executable', '执行计划 / 常量 / 存储约定', 'result')
    s['paths'].extend([['produced', 'consume_input', 'c1', 't0', 'after', 'c0', 'symbols', 'executable'],
                       ['produced', 'units', 't2', 't1', 'consume_input'],
                       ['produced', 'units', 't3', 't1', 'consume_input'],
                       ['produced', 'c2', 'r2'], ['after', 'c3', 'r3']])
    s['foot'] = ('跳过的是可选提取 / 拆分，不是 IrCompiler 内必经的 RunIrPasses。HLO → LLVM 属于产生；'
                 'LLVM pass 改写核心程序；目标对象、thunks 和 CpuExecutable 在灰色后续区汇合。')


def _gpu(s):
    from software_stack_component_flow_data import node, edge, chain
    n = _prepare(s, 'GPU LLVM IR', 'XLA GPU')
    n['produced'] = node('生成的 GPU LLVM Module', ['原生 emitter 或 Triton 编译形成 LLVM 程序。',
        '主模块与独立 kernel 是不同实例，按路径交给消费者。'],
        'GPU module emission', 'Triton LLVM translation', tag='产生结果 · LLVM 程序实例')
    n['consume_input'] = node('LLVM Module + 目标编译上下文', ['常规模块与独立 kernel 选择对应目标编译入口。',
        '查看 / 验证 IR 不要求先执行目标优化或设备代码。'],
        'GPU single module', 'GPU custom binary', tag='消费输入 · 目标程序')
    n['after'] = node('链接 / 优化后的 LLVM Module', ['可按目标选项链接设备 bitcode，完成 LLVM 优化。',
        '其后进入 NVIDIA 目标代码生成，产生 PTX。'],
        'GPU link and optimize', tag='变换结果 · LLVM 程序', color='transform')
    n['device_compile'] = node('PTX compilation provider', ['消费已生成的 PTX，继续编译或链接设备二进制。',
        '这是 LLVM 目标发射之后的处理，依 NVIDIA provider 而定。'],
        'NVPTX target binary', tag='后续处理 · 目标汇编到二进制')
    n['t3']['tag'] = '变换 · 目标编译内部链接优化'
    for key in ('t0', 't1', 't2'):
        n[key]['tag'] = '产生内部 · Triton MLIR 到 LLVM'
        n[key]['color'] = 'program'
    n['d1']['tag'] = '来源约束 · Triton 的输入表示'
    n['d2']['tag'] = '来源约束 · kernel 生成配置'
    n['ptx']['title'] = 'LLVM 目标发射 → PTX'
    n['ptx']['rows'] = ['NVIDIA CompileToPtx 路径消费优化后的 LLVM Module。',
                        '目标代码生成输出 PTX，随后交给 compilation provider。']
    n['ptx']['tag'] = '消费 · NVIDIA 目标代码生成'
    _definitions(s, ('d0', 'd3'))
    _part(s, 'view_production', '产生 · 原生与 Triton 的 LLVM 来源', 'XLA GPU 代码生成',
          ['hlo', 'p0'], ['native', 'p1'], ['d1', 'd2', 't2'],
          ['t0', 't1', 'triton_result'], ['produced'])
    _part(s, 'view_transformation', '变换 · LLVM 链接与优化', 'XLA GPU / LLVM',
          ['t3'], ['after'])
    _part(s, 'view_consumption', '消费 · 目标编译与 IR 观察', 'XLA GPU / LLVM',
          ['consume_input'], ['c1', 'c0', 'kernel_compile'], ['c2', 'c3', 'ptx'])
    _part(s, 'downstream', '消费结果与旁路 · 二进制、库调用和执行计划', 'GPU 后端 / compilation provider',
          ['device_compile', 'binary', 'executable'], ['library', 'kernel_binary', 'thunks'],
          ['r2', 'r3'], auxiliary=True)
    edge(s, 'hlo', 'p0', 'HLO + target / buffer 信息')
    for target, label in [('native', '原生 emitter'), ('p1', '选中 Triton fusion'), ('library', '设备库调用')]:
        edge(s, 'p0', target, label, optional=True)
    edge(s, 'native', 'produced', 'LLVM 主模块 / kernels')
    chain(s, ['p1', 't2', 't0', 't1', 'triton_result', 'produced'],
          ['TritonKernelSource', '编译内部 XLA / Triton pass', '目标 MLIR pipeline',
           '后续翻译为 LLVM', 'LLVM kernel 与包装代码'])
    edge(s, 'd1', 't2', '输入的 MLIR 容器', 'neutral')
    edge(s, 'd2', 't2', '块级配置与设备约束', 'neutral')
    edge(s, 'produced', 'consume_input', '已有 LLVM 程序 → 所选接口')
    s['edges'][-1]['channel'] = 'direct'
    edge(s, 'consume_input', 'c1', '常规模块编译路径', optional=True)
    edge(s, 'consume_input', 'kernel_compile', '独立 LLVM kernel 路径', optional=True)
    chain(s, ['c1', 'c0', 't3', 'after', 'ptx', 'device_compile', 'binary', 'executable'],
          ['CompileTargetBinary', 'CompileToPtx 内的链接 / 优化', 'LLVM 程序更新',
           '目标代码生成', 'PTX', '编译 / 链接 cubin', '目标二进制'])
    s['edges'][-6]['kind'] = s['edges'][-5]['kind'] = 'transform'
    for item in s['edges'][-3:]:
        item['kind'] = 'result'
    edge(s, 'c1', 'c2', '编译前 verifyModule', 'neutral')
    edge(s, 'consume_input', 'c2', '直接检查 LLVM 程序', 'neutral', optional=True)
    edge(s, 'c2', 'r2', '验证状态 / 错误', 'neutral')
    edge(s, 'c1', 'c3', '编译内按选项 dump / hook', 'neutral', optional=True)
    edge(s, 'after', 'c3', '优化后 Module', 'neutral', optional=True)
    edge(s, 'c3', 'r3', 'IR 文本 / 编译观察', 'neutral')
    edge(s, 'kernel_compile', 'kernel_binary', '二进制 + launch / 参数', 'result')
    edge(s, 'kernel_binary', 'thunks', 'CustomKernelThunk', 'result')
    edge(s, 'p0', 'thunks', '同步组织执行计划', 'neutral')
    edge(s, 'library', 'thunks', '设备库 thunk', 'neutral')
    edge(s, 'thunks', 'executable', '计划 + 常量 / 存储约定', 'result')
    s['paths'].extend([['native', 'produced', 'consume_input', 'c1', 'c0', 't3', 'after', 'ptx', 'device_compile', 'binary', 'executable'],
                       ['triton_result', 'produced', 'consume_input', 'kernel_compile', 'kernel_binary', 'thunks', 'executable'],
                       ['produced', 'consume_input', 'c2', 'r2'], ['after', 'c3', 'r3']])
    s['foot'] = ('Triton MLIR passes 是产生 LLVM IR 的内部过程；LLVM 变换区只改写 LLVM Module。'
                 '常规模块进入目标编译后执行链接 / 优化，不能据直接消费箭头跳过它；设备库独立进入执行计划。')



def _rewrite_content(s):
    """Use the agreed overview vocabulary while preserving linked node IDs."""
    from software_stack_component_flow_data import node, edge, chain
    texts = {
        'jaxlib': {
            'core': ('MLIR Module（StableHLO / Mosaic）', '以 MLIR Module 为容器，组织计算程序。|StableHLO 表达外层张量计算；Mosaic 表达专用内核。|函数、操作及其输入输出共同说明程序的行为。'),
            'd0': ('ModuleOp 的结构', '一个 Region 和一个 Block 容纳模块中的操作。|符号表组织函数等命名对象，内部 SSA 值受隔离规则约束。'),
            'd1': ('操作与数据依赖', 'Operation 保存操作数、结果、属性和嵌套区域。|Value 连接数据依赖，Type 描述值的类型。|各方言定义操作语义及其约束。'),
            'd2': ('构造程序的上下文', 'MLIR Context 管理方言、类型和属性。|JAX ModuleContext 保存 Module、平台、符号及回调等 lowering 信息。'),
            'd3': ('模块的合法性', 'operation.verify() 检查结构、类型和操作约束。|验证规则由 MLIR 基础设施及所用方言提供。'),
            'p0': ('JAX lowering 构造外层 Module', 'lower_jaxpr_to_module 接收 Jaxpr、抽象类型和平台等信息。|逐方程调用原语 lowering，连接 MLIR 输入与结果值。|返回 LoweringResult.module。'),
            'p1': ('还原可移植程序', 'PyDeserializePortableArtifact 接收字节和 Context。|反序列化 StableHLO 产物，返回 Python Module。'),
            'p2': ('从 HLO 构造 Module', 'PyXlaComputationToMlirModule 接收 XlaComputation。|ConvertHloToStablehlo 在内部构造对应模块。'),
            'produced': ('外层 MLIR Module', '来自 JAX lowering，或从可移植产物还原。|程序操作保存在模块中，编译选项由调用者另行提供。'),
            'compat_module': ('HLO 转换得到的 Module', 'ConvertHloToStablehlo 将 HLO proto 转成 StableHLO 操作。|兼容入口随后打印这一模块。'),
            'compat_text': ('Module 文本', 'PyXlaComputationToMlirModule 的返回值。|需要原生 Module 的调用者可再次解析文本。'),
            'mhlo_input': ('解析待转换的 Module', 'PyMhloToStablehlo 接收 MLIR 文本或字节。|解析后取得含 MHLO 操作的模块。'),
            't0': ('编译调用中克隆 Module', 'PyClient::CompileAndLoad 调用 module.clone()。|后续编译持有副本，可以在其中修改程序。'),
            'after': ('编译持有的 Module 副本', '副本与调用者的原模块分开持有。|接着包装为 HloProgram，交给 IFRT Compiler。'),
            't1': ('整理分片中的 mesh 表示', '启用 Shardy 时运行 sdy-lift-inlined-meshes。|把内联 mesh 提升为命名定义，合并重复项并更新引用。'),
            't2': ('将 MHLO 操作转成 StableHLO', 'PyMhloToStablehlo 运行对应合法化 pass。|操作表示改变，结果仍保存在 Module 中。'),
            'mhlo_module': ('完成方言转换的 Module', '原来的 MHLO 操作已转成 StableHLO。|辅助接口接着写出普通 MLIR bytecode。'),
            't3': ('为 HLO 导出整理 Module', 'PrepareForExport 运行合法化与导出准备 pass。|存在 shape 操作时，继续整理并合法化形状计算。'),
            'consume_input': ('Module 与调用参数', '编译需要目标设备和编译选项。|保存程序需要格式与目标版本；检查接口读取模块本身。'),
            'c0': ('PyClient::CompileAndLoad', '接收外层 Module、设备及编译选项。|克隆模块后，构造 HloProgram 和 IFRT 编译选项。'),
            'ifrt_program': ('ifrt::HloProgram', '持有 MLIR Module 及其所有权。|jaxlib 将 Program 和选项交给 IFRT 默认 Compiler。'),
            'c1': ('将 Module 导出为 HLO', 'ConvertMlirHloToHloModule 整理模块并导出 HLO proto。|结合配置调用 CreateFromProto，返回 HloModule。'),
            'r1': ('XLA HloModule', '计算和指令按照 HLO 对象结构组织。|后续由 XLA passes 和设备后端继续处理。'),
            'r0': ('返回 PyLoadedExecutable', '设备实现完成编译和加载后，IFRT 返回程序对象。|jaxlib 将其包装为 Python 可持有的执行接口。'),
            'c2': ('保存版本化 StableHLO 产物', 'PySerializePortableArtifact 接收 Module 和目标版本。|返回可传输、保存及再次还原的程序字节。'),
            'r2': ('StableHLO 可移植产物', '保存程序及其版本化表示。|反序列化接口可以据此还原 Module。'),
            'c3': ('验证、打印或写出 bytecode', 'verify() 检查模块约束。|打印和 module_to_bytecode 分别导出文本与普通字节。'),
            'r3': ('诊断、文本或 MLIR bytecode', '诊断用于定位非法 IR。|文本和字节用于检查、保存与传递程序。'),
            'mhlo_bytes': ('辅助转换返回 MLIR bytecode', 'PyMhloToStablehlo 将转换后的 Module 序列化。|调用者得到普通 MLIR 模块字节。'),
        },
        'xla': {
            'core': ('HLO（HloModule）', 'XLA 内部表达张量计算程序的中间表示。|描述运算、数据依赖和控制流。|携带形状、分片、布局及编译配置。'),
            'd0': ('模块与计算', 'HloModule 持有入口及其他 HloComputation。|入口定义程序输入输出；其他计算用于调用、控制流或 fusion。'),
            'd1': ('指令与依赖', 'HloInstruction 保存 opcode、operands、shape 和属性。|数据使用关系连接计算，根指令给出结果。|控制依赖与副作用约束额外的执行顺序。'),
            'd2': ('形状、布局和分片', 'Shape 描述值的类型与形状。|Layout 描述内存排列，Sharding 描述设备间的数据分布。'),
            'd3': ('阶段内的合法性', 'HloVerifier 按配置检查计算和指令的不变量。|重写需满足运算语义，以及当前阶段的布局等约束。'),
            'p0': ('从前端 MLIR 导入', 'ConvertMlirHloToHloModule 接收 Module 和转换选项。|导出 HLO proto，准备 config，再构造 HloModule。'),
            'p1': ('从 HLO proto 还原', 'CreateFromProto 接收 HloModuleProto 和配置。|重建计算、指令及引用关系，返回 HloModule。'),
            'p2': ('直接组织计算对象', '创建 HloModule，再调用 AddEntryComputation 等接口。|将已构建的计算加入模块，确定入口。'),
            'produced': ('构造完成的 HloModule', '包含入口计算、指令和编译配置。|随后用于检查、变换或目标后端编译。'),
            'transform_input': ('模块与变换配置', '编译器根据目标和所处阶段选择 pass。|结构复制与替换接口还接收选项或计算映射。'),
            't0': ('HloPassPipeline 调用各个 pass', 'RunImpl 读取模块、execution_threads 和调试选项。|顺序运行已配置的 pass，返回状态和是否修改。'),
            't1': ('代数简化', 'AlgebraicSimplifier 根据语义、选项和布局条件匹配指令。|用允许的等价计算替换局部程序。'),
            't2': ('删除无用计算', 'HloDCE 根据使用关系、根结果和副作用判断可删除内容。|更新指令、计算及按选项处理的死参数。'),
            't3': ('复制或替换计算', 'Clone 创建独立模块。|ReplaceComputations 按替换表更新计算的使用关系。'),
            'after': ('变换后的 HloModule', '程序经过计算简化、融合、分片或布局等处理。|具体 pass 及顺序由后端和配置决定。'),
            'hook': ('调度前后的自定义变换', '后端在配置的阶段调用 ApplyXlaTransforms。|CPU 的调度后变换位于 BufferAssignment 之前。'),
            'callback': ('Python HLO 变换回调', '输入是序列化 HloModuleProto。|返回新字节时更新模块，返回 None 时保留。|CPU 直接注册；插件需要对应 PJRT 扩展支持。'),
            'hook_result': ('回调后的模块状态', '更新结果留在当前编译阶段。|调用者继续原 pipeline 或调度后的后续处理。'),
            'consume_input': ('HLO 与消费接口的上下文', '后端读取已经满足相应阶段要求的程序。|验证与导出接口读取已有模块和各自选项。'),
            'c0': ('CpuCompiler::RunBackend', '消费 HLO 和 CPU 编译选项。|安排存储、生成计算函数和 thunk 执行计划。'),
            'c1': ('GpuCompiler::RunBackend', '消费 HLO、GPU 目标和编译选项。|组织 kernel 生成、Triton 及设备库调用。'),
            'plan': ('调度与存储规划', '调度给出执行顺序。|BufferAssignment 记录值使用哪些分配与切片，供代码生成和运行时使用。'),
            'return': ('CpuExecutable / GpuExecutable', '代码、常量、存储规划和 thunk 组成后端产物。|设备实现继续包装并加载为 PJRT 可执行对象。'),
            'c2': ('检查 HLO 的不变量', 'HloVerifier 遍历计算与指令。|依据配置返回验证状态或错误诊断。'),
            'r2': ('验证状态与诊断', '给出约束检查结果及出错位置。|也可用于 pass pipeline 内部的检查。'),
            'c3': ('打印或序列化 HLO', 'ToString 生成可读文本。|ToProto 写出可保存、传输和还原的模块描述。'),
            'r3': ('HLO 文本 / HloModuleProto', '记录某个阶段的程序结构。|用于比较变换前后差异，或重新构造模块。'),
        },
        'cpu': {
            'core': ('CPU LLVM IR（llvm::Module）', '面向 CPU 代码生成的低层程序表示。|描述具体运算、内存访问和控制流。|携带目标平台、DataLayout 和函数属性。'),
            'd0': ('Module 与 Function', 'Module 持有函数、全局变量和元数据。|Function 描述参数、返回类型与函数体。'),
            'd1': ('基本块与指令', 'BasicBlock 组织顺序执行的指令，以终结指令连接控制流。|Instruction 的操作数引用 SSA 值。'),
            'd2': ('目标与数据表示', 'TargetMachine 决定目标 triple 和 DataLayout。|CPU 特性、对齐和数据大小参与代码生成。'),
            'd3': ('kernel 的调用约定', 'host kernel 的签名符合 CPU 运行时的调用方式。|内存访问与 BufferAssignment 一致，ThunkEmitter 组织函数和库调用。'),
            'hlo': ('HLO 与 CPU 目标信息', 'RunBackend 进入 CompileCpuExecutable。|编译过程准备调度、存储规划和目标配置。'),
            'p0': ('创建目标 LLVM Module', 'CompileCpuExecutable 创建 LLVMContext 与 Module。|从 TargetMachine 设置 triple、DataLayout 及相关属性。'),
            'p1': ('生成嵌套函数与 host kernel', 'IrEmitter 生成嵌套计算与小常量。|IrEmitter2 为 fusion、元素级计算等生成 LLVM 函数。'),
            'p2': ('生成入口计算的工作计划', 'ThunkEmitter 调用 emitter 并收集额外 kernel 模块。|EmitEntryComputation 返回 ThunkSequence。'),
            'produced': ('主模块与独立 kernel 模块', '程序函数已表示为 LLVM 指令。|函数符号、参数和存储约定供后续编译使用。'),
            'units': ('选择编译单元的组织方式', '额外后端选项可能要求抽出特定 kernel。|并行代码生成配置决定是否拆分剩余主模块。'),
            't2': ('提取需单独编译的 kernel', 'ExtractKernelsFromModule 接收模块及 kernel 名称集合。|返回独立模块，并调整原模块。'),
            't3': ('拆分 LLVM Module', 'SplitModule 根据并行编译数量生成多个模块。|相关局部函数随 kernel 保留。'),
            't1': ('整理模块片段的符号', 'RemoveUnusedSymbols 删除片段中无用的符号。|调用方收集函数名称，并组织为线程安全模块。'),
            't0': ('运行 LLVM 优化', 'IrCompiler::RunIrPasses 根据目标、优化级别和选项建立 pipeline。|进行内联、循环和向量化等优化。'),
            'after': ('优化后的 LLVM Module', '函数与指令已按目标配置改写。|接着交给 EmitMachineCode 生成对象代码。'),
            'consume_input': ('LLVM 编译单元', '主模块、整理后的片段与额外 kernel 均可提交。|每个单元保留自己的目标和后端选项。'),
            'c1': ('AddModule / Compile', '模块编译器接收 LLVM Module，并收集待解析符号。|JIT 路径完成编译与链接，返回 FunctionLibrary。'),
            'c0': ('生成对象代码', 'IrCompiler::EmitMachineCode 接收 Module 和 TargetMachine。|LLVM 目标 pass 将对象代码写入 MemoryBuffer。'),
            'symbols': ('对象代码与 FunctionLibrary', 'JIT 链接对象代码，解析 kernel 和 comparator 符号。|函数库供 CPU 运行时调用。'),
            'thunks': ('ThunkSequence', '组织生成的 kernel、库调用、复制和控制流工作。|与常量及存储规划一起确定执行行为。'),
            'executable': ('CpuExecutable', 'Create 汇合函数库、存储规划、HLO、thunk 和常量。|设备实现继续包装、加载并交回 PJRT。'),
            'c2': ('验证 LLVM Module', 'VerifyLlvmModule 检查主模块及额外 kernel。|错误作为编译诊断返回。'),
            'r2': ('LLVM 验证结果', '报告 IR 是否满足验证约束。|出错时提供相应诊断。'),
            'c3': ('读取优化前后的 LLVM IR', 'IrCompiler 在相应阶段调用 hook。|dump 将当前 Module 保存为文本。'),
            'r3': ('LLVM IR 文本', '记录某个编译阶段的程序。|用于比较目标配置和优化造成的变化。'),
        },
        'gpu': {
            'core': ('GPU LLVM IR（llvm::Module）', '面向 GPU kernel 代码生成的低层程序表示。|描述运算、内存访问和控制流。|目标相关指令与约定表达线程操作、地址空间和 kernel 入口。'),
            'd0': ('LLVM 程序与 GPU 目标', 'Module 持有函数、指令、全局值和元数据。|triple、DataLayout 及函数属性约束目标代码生成。'),
            'd3': ('kernel 与执行计划', 'LLVM 程序描述 kernel 内部计算。|thunk 记录 kernel 启动和设备库调用，关联参数与存储。'),
            'hlo': ('HLO 与 GPU 编译配置', '后端根据运算、fusion 和设备能力选择实现。|kernel 生成与设备库调用共同组成程序。'),
            'p0': ('组织代码生成与存储规划', 'CompileModuleToLlvmIr 创建 BufferAssignment 和输出信息。|构造 IrEmitterContext、ThunkEmitter，并处理入口计算。'),
            'native': ('生成目标 kernel', '相应 emitter 生成 LLVM 函数和常量。|kernel 编译器接收独立编译单元及目标信息。'),
            'p1': ('构造 TritonKernelSource', 'TritonFusion 读取 HLO fusion 和块级参数。|CreateTritonModule 产生内核的 MLIR 表示。'),
            'd1': ('Triton 的 MLIR 输入', 'TritonKernelSource 持有 MLIR Module。|其中的操作先经 MLIR passes，再翻译为原生 LLVM IR。'),
            'd2': ('块级配置与设备资源', 'BlockLevelParameters 提供 warp、CTA 和 stage 等参数。|设备描述给出共享内存等资源限制。'),
            't2': ('调用 Triton 到 LLVM 编译', 'CompileTritonToLlvm 调用 CompileTritonToLLVM。|传入内核源程序、HLO 配置与目标信息。'),
            't0': ('XLA / Triton 接入处理', 'CreateTritonXlaPipeline 组织合法化与重写。|处理对象仍为内核 MLIR Module。'),
            't1': ('运行目标 MLIR pipeline', 'CreateTritonPipeline 根据设备和线程配置选择 passes。|整理到可翻译为 LLVM IR 的低层表示。'),
            'triton_result': ('返回 LLVM kernel 与元信息', 'TranslateLLVMToLLVMIR 产生 llvm::Module。|TritonWrapperResult 保存 LlvmKernelSource、线程维度和共享内存等信息。'),
            'produced': ('生成的 GPU LLVM Module', '包含目标 kernel 函数、全局值和相关属性。|可作为模块或独立 kernel 编译单元交给目标编译。'),
            't3': ('链接设备函数并优化', 'LinkAndOptimizeModule 按需要链接设备 bitcode。|根据目标信息和选项运行 LLVM 优化 pipeline。'),
            'after': ('优化后的 LLVM Module', '函数体和指令已经过目标相关优化。|继续生成目标代码。'),
            'consume_input': ('LLVM Module 与目标上下文', '包含 kernel 程序，以及编译所需的设备与选项。|提交模块编译或独立 kernel 编译入口。'),
            'c1': ('CompileSingleModule', '接收 LLVM Module、HLO 配置和设备描述。|按配置检查 IR，再调用目标二进制编译。'),
            'c0': ('NVIDIA 目标编译', 'NVPTXCompiler::CompileTargetBinary 接收 Module。|通常通过 CompileToPtx 完成 LLVM 处理和 PTX 生成。'),
            'ptx': ('LLVM Module → PTX', 'NVIDIA 目标代码生成读取优化后的程序。|输出 PTX 文本供设备编译器继续处理。'),
            'device_compile': ('PTX → 设备二进制', 'compilation provider 根据设备能力编译或链接 PTX。|输出 cubin 等目标结果。'),
            'binary': ('BackendCompileResult', '持有目标二进制和编译相关信息。|随后与执行计划汇合。'),
            'kernel_compile': ('编译独立 LLVM kernel', 'KernelCompiler::CompileToTargetBinary 接收 LlvmKernelSource。|CUDA 实现直接编译或通过线程池返回二进制 future。'),
            'kernel_binary': ('CustomKernelThunk', '把二进制、参数、launch 维度和共享内存组织为 kernel 工作。|加入整体 ThunkSequence。'),
            'library': ('设备库调用', 'cuBLAS、cuDNN 等调用形成相应 thunk。|随生成的 kernel 一起进入执行计划。'),
            'thunks': ('ThunkSequence', '组织 kernel 启动、库调用及其他工作。|持有独立 kernel 的代码和调用信息。'),
            'executable': ('GpuExecutable', 'Create 汇合设备代码、ThunkExecutor、常量和存储分配。|交回设备实现，继续 PJRT 包装与加载。'),
            'c2': ('验证 LLVM Module', 'LLVM verifier 按配置检查目标编译输入。|非法 IR 形成编译诊断。'),
            'r2': ('LLVM 验证结果', '返回检查状态及错误信息。|帮助定位代码生成中的非法程序。'),
            'c3': ('检查阶段内的 LLVM IR', '编译 hook 读取 Module，dump 保存文本。|选项决定记录哪些阶段。'),
            'r3': ('LLVM IR 文本与编译信息', '记录 kernel 在编译中的表示。|用于比较生成结果和优化变化。'),
        },
    }
    n = s['nodes']
    if set(texts[s['key']]) != set(n):
        raise ValueError(f'{s["key"]}: content must cover every node: {set(n) ^ set(texts[s["key"]])}')
    for key, (title, rows) in texts[s['key']].items():
        n[key].update(title=title, rows=rows.split('|'))
    s['view_reading'] = [
        '定义说明程序的含义与结构；产生、变换和消费说明具体函数怎样处理它。',
        '箭头标明调用或传递的对象，可选分支使用虚线；点击源码链接可核对实现。',
    ]
    for part in s['sections']:
        if part['id'] == 'view_definition':
            part['title'] = '定义 · 表示了什么'
        elif part['id'] == 'view_production':
            part['title'] = '产生 · 由谁根据什么构造'
        elif part['id'] == 'view_transformation':
            part['title'] = '变换 · 怎样改变程序'
        elif part['id'] == 'view_consumption':
            part['title'] = '消费 · 交给谁，产生什么结果'
    if s['key'] == 'jaxlib':
        n['core']['refs'] += ('lower_jaxpr_to_pipelined_module',)
        n['ifrt_program']['refs'] += ('IFRT HloProgram',)
        n['mosaic'] = node('Pallas 构造 Mosaic TPU Module', [
            'TPU lowering 根据 kernel Jaxpr 和 GridMapping 构造独立模块。',
            'lower_jaxpr_to_pipelined_module 写入内核计算与流水安排。'],
            'pallas_call_tpu_lowering_rule', 'lower_jaxpr_to_pipelined_module',
            tag='产生 · Pallas 内核', owner='JAX / Pallas')
        n['mosaic_transform'] = node('准备 Mosaic 内核序列化', [
            '_lower_mosaic_module_to_asm 克隆 Module。',
            '运行 mosaic-serde pass，按目标版本整理内核表示。'],
            '_lower_mosaic_module_to_asm', tag='变换 · 内核 Module', color='transform', owner='JAX / Mosaic')
        n['mosaic_payload'] = node('内核 bytecode → 外层 custom call', [
            '把 Mosaic Module 写成 MLIR bytecode。',
            '内核程序和配置进入外层 custom call 的 backend_config。'],
            'lower_module_to_custom_call', '_lower_mosaic_module_to_asm',
            tag='消费 · 内核接入外层程序', owner='JAX / Pallas')
        n['ifrt_compiler'] = node('IFRT PjRtCompiler → PJRT', [
            '从 HloProgram 取出 Module 和 XLA 编译选项。',
            '通过 PjRtLoadedExecutable::Create 调用设备的 CompileAndLoad。'],
            'IFRT PjRtCompiler', 'PJRT CompileAndLoad',
            tag='消费 · 委派设备编译', owner='IFRT / PJRT')
        for ident, row in [('view_production',['mosaic']), ('view_transformation',['mosaic_transform']),
                           ('view_consumption',['mosaic_payload','ifrt_compiler'])]:
            next(p for p in s['sections'] if p['id']==ident)['rows'].append(row)
        chain(s, ['mosaic','mosaic_transform','mosaic_payload','produced'],
              ['独立 Mosaic Module','处理后写出 bytecode','外层 Module 含内核 custom call'])
        s['edges'][-3]['kind'] = 'transform'
        for e in s['edges']:
            if e['source']=='ifrt_program' and e['target']=='c1':
                e['source']='ifrt_compiler'
                e['label']='下游 XLA 导出接口'
        for path in s['paths']:
            if 'ifrt_program' in path and 'c1' in path:
                path.insert(path.index('c1'), 'ifrt_compiler')
        edge(s,'ifrt_program','ifrt_compiler','Module + 编译选项')
        s['foot'] = '外层 Module 经 HloProgram、IFRT 和 PJRT 提交设备编译；独立 Mosaic 内核序列化后，通过 custom call 随外层程序传递。'
    elif s['key'] == 'xla':
        n['t0']['refs'] += ('CPU HLO passes', 'GPU HLO passes')
        n['callback']['tag'] = '变换 · Python 回调'
        s['foot'] = 'HLO passes 根据后端和配置重写程序；目标后端继续安排存储、生成代码和 thunk，返回可执行对象。'
    elif s['key'] == 'cpu':
        s['foot'] = 'LLVM Module 描述计算函数；CPU 后端将对象代码链接为函数库，再与存储规划、常量和 thunk 组成 CpuExecutable。'
    else:
        n['consume_input'].update(title='LLVM 编译单元与目标配置', rows=[
            '生成器将 kernel 或常量模块交给 kernel 编译器。',
            '设备信息与后端选项用于后续目标代码生成。'])
        n['kernel_compile'].update(title='提交 LLVM 编译单元', rows=[
            'CubinCustomKernelCompiler 接收 LlvmKernelSource。',
            '其编译回调进入 CompileSingleModule，直接执行或交给线程池。'])
        n['kernel_compile']['refs'] += ('GPU kernel compiler',)
        n['binary'].update(title='目标二进制与编译信息', rows=[
            'CompileSingleModule 返回 BackendCompileResult。',
            '回调取出 binary，分别用于 kernel 或常量存储。'])
        n['executable']['rows'] = [
            '直接持有常量模块二进制、ThunkExecutor 和存储分配。',
            '各 kernel 的代码与调用信息由相应 thunk 持有。']
        s['edges'] = [e for e in s['edges'] if (e['source'], e['target']) not in {
            ('consume_input','c1'), ('kernel_compile','kernel_binary')}]
        for e in s['edges']:
            if (e['source'],e['target']) == ('consume_input','kernel_compile'):
                e.update(label='LlvmKernelSource', optional=False)
            elif (e['source'],e['target']) == ('binary','executable'):
                e.update(label='常量模块的 binary', optional=True)
        edge(s,'kernel_compile','c1','注入的 LLVM 编译回调')
        edge(s,'binary','kernel_binary','kernel binary + 参数与 launch 信息','result')
        s['paths'] = [
            ['p1','t2','t0','t1','triton_result','produced'],
            ['produced','consume_input','kernel_compile','c1','c0','t3','after',
             'ptx','device_compile','binary','kernel_binary','thunks','executable'],
            ['produced','consume_input','c2','r2'], ['after','c3','r3'],
        ]
        s['foot'] = 'kernel 编译器经同一目标编译回调生成代码；kernel 二进制进入 CustomKernelThunk，常量模块二进制直接进入 GpuExecutable。'


def apply_compiler_views(spec):
    adapters = {'jaxlib': _jaxlib, 'xla': _hlo, 'cpu': _cpu, 'gpu': _gpu}
    adapter = adapters.get(spec['key'])
    if adapter is None:
        return spec
    visible = {key: n for key, n in spec['nodes'].items() if key not in spec.get('omit', set())}
    old_ids = set(visible)
    old_refs = {ref for n in visible.values() for ref in n['refs']}
    adapter(spec)
    return _finish(spec, old_ids, old_refs)
