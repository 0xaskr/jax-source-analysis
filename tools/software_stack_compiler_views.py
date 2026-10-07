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
             summary=f'唯一核心概念 {core_name}；定义保持静态，实际程序与调用关系连接产生、变换和消费。')
    return n


def _part(s, ident, title, owner, *rows, auxiliary=False):
    from software_stack_component_flow_data import section
    section(s, title, owner, *rows, external=auxiliary)
    s['sections'][-1].update(id=ident, auxiliary=auxiliary)


def _definitions(s, keys=('d0', 'd1', 'd2', 'd3')):
    from software_stack_component_flow_data import edge
    _part(s, 'view_definition', '定义 · 语义、结构与约束', s['nodes']['core'].get('owner') or '核心对象',
          ['core'], *([key] for key in keys))
    for key in keys:
        edge(s, 'core', key, '结构 / 语义与边界', 'neutral')


def _finish(s, old_ids, old_refs):
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
