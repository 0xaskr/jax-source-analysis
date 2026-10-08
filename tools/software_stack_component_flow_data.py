"""Component-local workflows, ownership boundaries and four-view reference cards."""
from __future__ import annotations

from copy import deepcopy
from software_stack_component_hub_data import DIAGRAMS
from render_jaxpr_centered_hub import DEFINITIONS, PRODUCERS, TRANSFORMS, CONSUMERS


PATHS = {s['key']: s['path'] for s in DIAGRAMS}
PATHS['jax'] = 'jax/jaxpr-centered-hub.svg'
OVERVIEW_IDS = dict(jax='component_jax', jaxlib='component_jaxlib', ifrt='component_ifrt',
                    pjrt='component_pjrt', xla='component_xla', cpu='cpu_ir', gpu='gpu_ir',
                    tpu='tpu_boundary', runtime='component_runtime')
NEIGHBORS = dict(jax=('jaxlib', 'xla'), jaxlib=('jax', 'ifrt'), ifrt=('jaxlib', 'pjrt'),
                 pjrt=('ifrt', 'runtime'), xla=('jaxlib', 'cpu', 'gpu'), cpu=('xla', 'pjrt'),
                 gpu=('xla', 'runtime'), tpu=('pjrt', 'jax'), runtime=('gpu', 'pjrt'))
LABELS = dict(jax='JAX', jaxlib='jaxlib', ifrt='IFRT', pjrt='PJRT', xla='XLA / HLO',
              cpu='CPU 后端', gpu='GPU 后端', tpu='TPU / libtpu', runtime='设备运行时与驱动')


def node(title, rows, *refs, tag='内部接口', color='program', owner=None, link=None):
    return dict(title=title, rows=rows, refs=refs, tag=tag, color=color, owner=owner, link=link)


def base(key):
    s = deepcopy(next(s for s in DIAGRAMS if s['key'] == key))
    runtime = key in {'ifrt', 'pjrt', 'runtime'}
    nodes = {'core': node(s['core'], s['center'], s['core_ref'], tag='定义 · 研究对象',
                          color='unknown' if key == 'tpu' else 'data' if runtime else 'program'),
             'after': node(s['after'], s['after_rows'], s['core_ref'], tag='变换结果',
                           color='data' if runtime else 'transform')}
    for group, prefix, tag in [('definitions', 'd', '定义 · 结构与约束'),
                              ('producers', 'p', '产生 · 输入与构造'),
                              ('transforms', 't', '变换 · 按条件应用'),
                              ('consumers', 'c', '消费 · 接口与结果')]:
        for i, record in enumerate(s[group]):
            color = 'neutral' if prefix == 'd' else 'data' if runtime else 'transform' if prefix == 't' else 'program'
            n = node(record['title'], record['rows'], *record['refs'], tag=tag, color=color)
            if record.get('uncertain'): n['color'], n['uncertain'] = 'unknown', True
            nodes[f'{prefix}{i}'] = n
            if prefix == 'c':
                nodes[f'r{i}'] = node(record['result'], record['result_rows'], *record['refs'],
                                      tag='消费结果', color='output' if runtime else 'result')
    s.update(nodes=nodes, sections=[], edges=[], paths=[], omit=set(), core_id='core',
             title=LABELS[key] + ' · 内部架构与对象流转')
    return s


def section(s, title, owner, *rows, external=False):
    s['sections'].append(dict(title=title, owner=owner, rows=rows, external=external))
    for row in rows:
        for key in row:
            if key: s['nodes'][key]['owner'] = s['nodes'][key].get('owner') or owner


def edge(s, a, b, label, kind='program', *, optional=False):
    s['edges'].append(dict(source=a, target=b, label=label, kind=kind, optional=optional))


def chain(s, ids, labels, kind='program'):
    for a, b, label in zip(ids, ids[1:], labels): edge(s, a, b, label, kind)
    s['paths'].append(ids)


def jax():
    s = dict(key='jax', path=PATHS['jax'], title='JAX · Jaxpr 的定义、产生、变换与消费',
             summary='Jaxpr 用原语及其组合表达 JAX / Pallas 计算；同一程序可用于微分、批量化、解释和 lowering。',
             foot='变换按调用需要选择。JVP 和 batching 构造新的计算；DCE 保留所需计算。lowering 的结果交给 jaxlib 继续编译。',
             nodes={}, sections=[], edges=[], paths=[], details=[], omit=set(), core_id='core',
             avoid_endpoint_captions=True, layout='component_views')
    n = s['nodes']
    n['core'] = node('Jaxpr', ['表达 JAX / Pallas 代码的中间表示。',
        '描述输入如何通过原语及其组合产生输出，', '以及计算中发生的可观察效果。'], 'Jaxpr', tag='定义 · 程序表示')
    n['d0'] = node('输入、输出与常量', ['all_invars 保存全部输入变量。',
        'constvars / consts 对应已附常量；invars 为其余输入。',
        'outvars 引用变量或字面量，确定输出顺序。',
        '本版本 ClosedJaxpr 与 Jaxpr 是同一个类。'], 'Jaxpr', 'ClosedJaxpr alias', tag='定义 · 程序边界', color='neutral')
    n['eqn'] = node('JaxprEqn · 原语应用', ['invars / outvars 连接值依赖。',
        'primitive 决定操作，params 保存静态参数或子 Jaxpr。',
        'eqns 按解释顺序组成程序主体。'], 'JaxprEqn', 'Primitive', tag='定义 · 方程', color='neutral')
    n['var'] = node('Var / Literal / aval', ['Var 表示值的身份，aval 描述抽象类型。',
        'Literal 直接携带字面值。', '类型可包含 shape、dtype，也可表达 Ref 等。'], 'Var', 'Literal', 'AbstractValue', tag='定义 · 值与类型', color='neutral')
    n['d3'] = node('effects 与良构性', ['effects 记录读写等可观察效果。',
        '变量先定义后使用，同一变量只绑定一次。',
        '方程输入输出须满足原语类型与效果规则。'], 'check_jaxpr', 'has_effects', tag='定义 · 语义约束', color='neutral')

    n['p0'] = node('Python 函数与抽象输入', ['展平函数输入输出，取得动态输入的 avals。',
        '_trace_for_jit 把函数与抽象输入交给追踪器。'], '_trace_for_jit', 'flatten_fun', tag='产生 · 输入')
    n['p1'] = node('DynamicJaxprTrace', ['Tracer 代表追踪中的值。',
        '处理 primitive 调用，推导类型与 effects，记录方程。'], 'DynamicJaxprTrace', 'make_eqn', tag='产生 · 记录计算')
    n['p2'] = node('JaxprStackFrame.to_jaxpr', ['汇总输入、输出、常量与方程。',
        '构造 Jaxpr，并按追踪接口返回常量与输出类型。'], 'Frame.to_jaxpr', tag='产生 · 构造对象')
    n['produced'] = node('追踪得到的 Jaxpr', ['追踪接口返回携带常量的程序与输出抽象类型。',
        '可继续变换，也可直接解释、检查或 lower。'], 'trace_to_jaxpr_nocache', tag='产生 · 结果')
    n['caller'] = node('pl.pallas_call 的 kernel 与配置', ['kernel、grid、BlockSpec 定义块内计算及调用方式。',
        'get_grid_mapping 整理映射与 Ref 输入类型。'], 'pallas_call', 'get_grid_mapping', tag='产生 · Pallas 输入')
    n['inner_jaxpr'] = node('内层 kernel Jaxpr', ['_trace_kernel_to_jaxpr 追踪 kernel 并运行 DCE。',
        '从输入 Ref 读取，向输出 Ref 写入；函数返回 None。',
        '结果仍是 core.Jaxpr，另返回允许捕获的常量。'], '_trace_kernel_to_jaxpr', tag='产生 · kernel 程序')
    n['mapping'] = node('GridMapping', ['记录 grid、block mappings 与输入输出约定。',
        '与 kernel Jaxpr 一起传递，指导后续变换和 lowering。'], 'GridMapping', 'get_grid_mapping', tag='产生 · 配套调用信息', color='neutral')
    n['pallas_bind'] = node('pallas_call_p.bind', ['把 kernel Jaxpr、GridMapping 和数组实参交给当前 Trace。',
        '在外层 Jaxpr 追踪中，记录为一条 pallas_call 方程。'], 'pallas_call_p.bind', 'Primitive.bind', tag='产生 · 外层调用')
    n['outer_jaxpr'] = node('含 pallas_call 的外层 Jaxpr', ['方程输入输出连接调用者的数组值。',
        'params 分别携带 kernel Jaxpr 与 GridMapping。'], 'JaxprEqn', 'pallas_call_p.bind', tag='产生 · 调用者程序')

    for i, (_, title, rows, refs) in enumerate(TRANSFORMS[:5]):
        n[f't{i}'] = node(title, rows, *refs, tag='变换 · 按需应用', color='transform')
    n['jvp'] = node('jvp_jaxpr · 前向微分', ['输入程序与切向非零标记，应用原语 JVP 规则。',
        '追踪原值与切向计算，返回新 Jaxpr 和输出非零标记。'], 'jvp_jaxpr', tag='变换 · 构造导数计算', color='transform')
    n['transform_input'] = node('程序与变换参数', ['根据微分、批量化或编译需要选择规则。',
        '各规则分别处理类型、effects 和调用约定。'], 'jvp_jaxpr', 'batch_jaxpr2', tag='变换 · 输入', color='transform')
    n['after'] = node('新程序与附加信息', ['返回改写或拆分后的 Jaxpr。',
        'used_inputs、out_axes、residual 等供调用方衔接。'], 'dce_jaxpr', 'linearize_jaxpr', tag='变换 · 结果', color='transform')
    n['transforms'] = node('Pallas 的 JVP / batching 规则', ['接收 kernel Jaxpr、映射和切向或批维信息。',
        '按分支改写或复用 kernel，调整 Ref 顺序及 GridMapping。',
        '重新绑定 pallas_call，返回调用方所需的数组结果。'],
        '_pallas_call_jvp_rule', '_pallas_call_batching_rule', tag='变换 · kernel 与调用约定', color='transform')

    n['consume_input'] = node('待使用的 Jaxpr', ['消费者接收追踪结果或变换后的程序。',
        '实际值、打印选项或 lowering 上下文由各接口提供。'], 'eval_jaxpr', 'lower_jaxpr_to_module', tag='消费 · 输入')
    n['c0'] = node('eval_jaxpr → 输出值', ['以常量与实参建立值环境。',
        '逐方程调用 primitive.bind，最后读取 outvars。',
        '当前 Trace 决定这些操作如何解释。'], 'eval_jaxpr', tag='消费 · 解释')
    n['c1'] = node('check_jaxpr → 校验结果', ['检查变量绑定、类型和原语约束。',
        '成功返回 None；不满足约束则报告类型错误。'], 'check_jaxpr', tag='消费 · 校验', color='neutral')
    n['c2'] = node('pretty_print → 可读文本', ['读取变量、方程、类型与效果。',
        '按打印选项生成供阅读和诊断使用的文本。'], 'pretty_print', tag='消费 · 观察', color='neutral')
    n['c3'] = node('lower_jaxpr_to_module', ['接收程序、平台、分片及 lowering 上下文。',
        '创建模块和函数，调用 jaxpr_subcomp 处理方程。'], 'lower_jaxpr_to_module', tag='消费 · 构造 MLIR')
    n['lower_rules'] = node('jaxpr_subcomp → 原语规则', ['从变量环境读取输入 IR 值。',
        '调用原语的 lowering 规则，将结果绑定到输出变量。'], 'jaxpr_subcomp', 'register_lowering', tag='消费 · 逐方程转换')
    n['r3'] = node('外层 MLIR Module', ['承载 StableHLO 等操作及程序入口。',
        '作为 LoweringResult 的一部分返回，交给后续编译路径。'],
        'lower_jaxpr_to_module', tag='消费 · 输出', color='result', link='../jaxlib/mlir-module-centered-hub.svg#core')
    n['outer_lowering'] = node('pallas_call 的平台分派', ['读取内层 Jaxpr、GridMapping 和调用配置。',
        '按 interpret、平台与后端选择 kernel lowering。'], '_pallas_call_lowering', tag='消费 · Pallas 方程')
    n['mosaic_module'] = node('Mosaic TPU MLIR Module', ['TPU 规则调用 lower_jaxpr_to_pipelined_module。',
        'kernel 方程与网格映射共同决定模块内容。'], 'pallas_call_tpu_lowering_rule', 'lower_jaxpr_to_pipelined_module', tag='消费 · kernel lowering 结果')
    n['payload'] = node('序列化 kernel 模块', ['运行 mosaic-serde，写出版本化 MLIR 字节码。',
        '编码到 custom_call 的 backend_config。'], '_lower_mosaic_module_to_asm', 'CustomCallBackendConfig.to_json', tag='消费 · 结果封装', color='transform')
    n['outer_module'] = node('stablehlo.custom_call', ['tpu_custom_call 携带 kernel payload。',
        '操作数与返回 IR 值接入完整外层 Module。'], 'emit tpu_custom_call', tag='消费 · 接回调用者')

    sections = [
        ('view_definition', '定义 · Jaxpr 表示什么？', 'JAX core',
         '输入输出、原语方程、抽象类型与 effects 共同定义程序。',
         [['core'], ['d0'], ['eqn'], ['var'], ['d3']]),
        ('view_production', '产生 · 函数如何成为 Jaxpr？', 'JAX tracing / Pallas tracing',
         '普通函数和 kernel 都通过追踪构造程序；Pallas 另带调用映射。',
         [['p0','p1'], ['p2','produced'], ['caller','inner_jaxpr'], ['mapping','pallas_bind'], ['outer_jaxpr']]),
        ('view_transformation', '变换 · 程序如何被改写？', 'JAX interpreters / Pallas 规则',
         '构造微分与批量计算，拆分已知部分，消除无用计算或展开高层表示。',
         [['transform_input'], ['jvp','t3'], ['t1','t2'], ['t0','t4'], ['after','transforms']]),
        ('view_consumption', '消费 · 谁使用 Jaxpr？', 'JAX 解释器 / lowering',
         '解释产生值，校验产生检查结果，打印产生文本，lowering 产生 MLIR。',
         [['consume_input'], ['c0','c1'], ['c2','c3'], ['lower_rules','r3'],
          ['outer_lowering','mosaic_module'], ['payload','outer_module']])]
    for ident, title, owner, question, rows in sections:
        section(s, title, owner, *rows)
        s['sections'][-1].update(id=ident, question=question)

    edge(s, 'core','eqn','eqns 持有方程','neutral')
    edge(s, 'eqn','var','方程引用输入与输出','neutral')
    chain(s, ['p0','p1','p2','produced'], ['函数与 avals','记录方程与常量','构造 Jaxpr'])
    chain(s, ['caller','inner_jaxpr','pallas_bind','outer_jaxpr'], ['追踪 kernel','kernel 与常量','外层追踪记录调用'])
    edge(s, 'caller','mapping','整理网格与块映射')
    edge(s, 'mapping','pallas_bind','独立的映射参数')
    edge(s, 'produced','transform_input','程序与变换参数','transform',optional=True)
    for key in ('jvp','t0','t1','t2','t3','t4'):
        edge(s, 'transform_input',key,'选择相应规则','transform',optional=True)
        edge(s, key,'after','程序与附加信息','transform')
    edge(s, 'inner_jaxpr','transforms','kernel 与专用变换参数','transform',optional=True)
    edge(s, 'transforms','pallas_bind','更新程序与映射后重新绑定','transform')
    edge(s, 'after','consume_input','选择要使用的程序')
    edge(s, 'produced','consume_input','直接使用追踪结果',optional=True)
    s['edges'][-1]['channel']='direct'
    edge(s, 'outer_jaxpr','consume_input','包含 kernel 调用的外层程序',optional=True)
    for key, label in (('c0','提供常量与实参'),('c1','检查结构与类型'),('c2','读取程序结构'),('c3','提供 lowering 上下文')):
        edge(s, 'consume_input',key,label,'neutral' if key in ('c1','c2') else 'program',optional=True)
    chain(s, ['c3','lower_rules','r3'], ['逐方程转换','IR 值与操作汇入模块'])
    edge(s, 'lower_rules','outer_lowering','遇到 pallas_call',optional=True)
    chain(s, ['outer_lowering','mosaic_module','payload','outer_module','r3'],
          ['TPU 原生分支','模块字节码','payload 与调用配置','接入外层模块'])
    s['edges'][-4]['optional']=True
    return s


def jaxlib():
    s=base('jaxlib'); n=s['nodes']
    n['ifrt_program']=node('IFRT HloProgram / Compiler', ['包装克隆后的 mlir::ModuleOp 与 CompileOptions。',
        'HloProgram 名称不表示已经转换为 HloModule。', 'CompileAndLoad 的返回由绑定层包装回 Python。'], 'IFRT CompileAndLoad', 'PyClient CompileAndLoad', tag='外部接口 · 编译程序')
    n['after']['title']='克隆后的 MLIR Module'
    n['after']['rows']=['编译调用持有独立的模块副本。', '后续后端处理可以修改该副本；Python 原输入保留。']
    section(s,'01 · 构造来源与分片准备','外部 · JAX lowering',['p0','t1'],external=True)
    section(s,'02 · Python / 原生对象与编译绑定','jaxlib',['p1','core','p2'],['c0','t0','after'])
    section(s,'03 · 交给框架编译接口','外部 · IFRT',['ifrt_program'],external=True)
    section(s,'04 · 导出 HLO 的下游接口','外部 · XLA',['c1','t3','r1'],external=True)
    section(s,'05 · 编译完成，绑定层返回对象','jaxlib',['r0'])
    chain(s,['p0','t1','core'],['Module · 启用 Shardy 时','模块 pass 后的 Module'])
    edge(s,'p0','core','未启用 Shardy · Module',optional=True)
    edge(s,'p1','core','反序列化得到 Module')
    edge(s,'p2','core','内部构造 Module · 接口导出文本','neutral')
    chain(s,['core','c0','t0','after','ifrt_program'],['Module + CompileOptions','调用内部 clone()','独立 Module 对象','HloProgram + CompileOptions'])
    chain(s,['ifrt_program','c1','t3','r1'],['下游编译路径 · Module','导出内部的 PrepareForExport','继续导出并构造 HloModule'])
    edge(s,'ifrt_program','r0','LoadedExecutable → Python 包装','result')
    n['p2']['tag']='兼容入口 · 内部 Module 导出为文本'
    n['p2']['link']='../xla/hlo-centered-hub.svg'
    n['r1']['link']='../xla/hlo-centered-hub.svg'
    n['ifrt_program']['link']='../ifrt/array-centered-hub.svg'
    n['d0']['owner']=n['d1']['owner']='MLIR / StableHLO 基础设施'
    n['d2']['owner']='JAX lowering 上下文'
    return s


def ifrt():
    s=base('ifrt'); n=s['nodes']
    n['loaded']=node('LoadedExecutableRef', ['由编译与加载接口产生的程序对象。',
        '执行接口接收 ArrayRef 列表，数据与程序在此汇合。'], 'IFRT PjRtLoadedExecutable Execute', tag='执行前提 · 已加载程序')
    n['extract']=node('PjRtArray → 每设备参数列表', ['PJRT-backed 实现读取各可寻址分片的 PjRtBuffer。',
        '按 executable 的设备顺序组织执行参数。'], 'IFRT PjRtLoadedExecutable Execute', 'PjRtArray', tag='消费 · 提取执行参数', color='data')
    n['call']=node('PJRT Execute', ['输入：每设备 Buffer 参数与执行选项。',
        '返回各设备输出 Buffer，可同时返回完成 Future。'], 'PJRT Execute', tag='外部接口 · provider 执行', color='data', link='../pjrt/buffer-centered-hub.svg')
    n['buffers']=node('输出 PjRtBuffer 列表', ['provider 返回设备结果存储的句柄。',
        '句柄可早于实际计算完成返回。'], 'IFRT PjRtLoadedExecutable Execute', tag='执行结果 · 设备分片', color='output')
    n['r0']=node('ExecuteResult.outputs', ['按输出 dtype、shape、sharding 包装为 ArrayRef 列表。',
        '上层继续包装为 Python 可见数组。'], 'IFRT PjRtLoadedExecutable Execute', 'PjRtArray Create', tag='输出对象返回', color='output')
    n['status']=node('ExecuteResult.status / Future', ['fill_status 启用时填充执行状态。',
        'Array 的 GetReadyFuture 表示其数据就绪；两类观察点分开。'], 'IFRT PjRtLoadedExecutable Execute', 'Array readiness', tag='完成状态 · 与输出对象分开', color='completion')
    section(s,'01 · 数组的产生与存储映射','IFRT / PJRT-backed 实现',['p0','core','p1'],['d2','extract'])
    section(s,'02 · 程序与逻辑数组汇合','IFRT',['loaded','c0'])
    section(s,'03 · 委派设备执行','外部 · PJRT',['call','buffers'],external=True)
    section(s,'04 · 输出包装与状态返回','IFRT',['p2','r0','status'])
    section(s,'05 · 按需改变分片、放置与所有权','IFRT',['t0','t1'],['t2','t3'],['after'])
    edge(s,'p0','core','ArrayRef · 主机数据入口','data'); edge(s,'p1','core','组装后的 ArrayRef','data')
    edge(s,'core','d2','PjRtArray · 实现映射','neutral')
    edge(s,'d2','extract','可寻址分片 Buffer','neutral')
    edge(s,'loaded','c0','已加载程序','data'); edge(s,'core','c0','ArrayRef 参数列表','data')
    edge(s,'c0','extract','Execute 内提取参数','data')
    chain(s,['extract','call','buffers','p2','r0'],['PjRtBuffer* 参数','设备结果句柄','按输出规格包装','ArrayRef 输出'],'output')
    s['edges'][-4]['kind']='data'
    edge(s,'call','status','可选执行 Future','completion')
    for i in range(4):
        edge(s,'core',f't{i}','Array + 操作参数','data',optional=True)
        edge(s,f't{i}','after','新的 Array / 分片列表','output')
    n['d2']['color']='neutral'
    return s


def pjrt():
    s=base('pjrt'); n=s['nodes']
    n['loaded']=node('PjRtLoadedExecutable', ['已编译并加载的程序，由具体 provider 实现。',
        'Execute 的数据参数是 Buffer，不是编译用 Module。'], 'PjRtLoadedExecutable', tag='执行前提 · 程序对象')
    n['provider']=node('设备 provider 执行', ['消费设备参数列表、执行选项与依赖。',
        'CPU / GPU / TPU 选择各自实现；CUDA 路径见运行时图。'], 'PJRT Execute', tag='外部实现 · 执行分派', color='data', link='../stream-executor/submission-centered-hub.svg')
    n['r0']=node('结果 PjRtBuffer 列表', ['返回按设备、按结果组织的存储句柄。',
        'IFRT 继续将这些句柄包装为逻辑 Array。'], 'PJRT Execute', tag='输出对象返回', color='output', link='../ifrt/array-centered-hub.svg')
    n['status']=node('执行 Future / 数据就绪', ['Execute 可返回对应设备的完成 Future。',
        'GetReadyFuture 单独观察 Buffer 的计算或传输就绪。'], 'PJRT Execute', 'GetReadyFuture', tag='完成状态', color='completion')
    n['invalid']=node('原句柄失效 / 所有权转移', ['Bitcast、donation、Delete / Release 各遵守接口契约。',
        '原句柄可失效；底层存储回收仍受异步使用和共享引用约束。'], 'Buffer bitcast','Buffer delete','Buffer release', tag='生命周期结果', color='neutral')
    section(s,'01 · 设备存储契约与输入产生','PJRT',['p0','core','d1'])
    section(s,'02 · 程序与输入数据汇合','PJRT',['loaded','c0'])
    section(s,'03 · 设备实现与异步结果','外部 · provider',['provider','r0','status'],external=True)
    section(s,'04 · 复制、捐赠与生命周期','PJRT',['t0','t1'],['t2','t3'],['after','invalid'])
    edge(s,'p0','core','PjRtBuffer · 可异步传输','data'); edge(s,'core','d1','client / device / memory_space','neutral')
    edge(s,'loaded','c0','调用 Execute','data'); edge(s,'core','c0','每设备输入 Buffer 列表','data')
    chain(s,['c0','provider','r0'],['provider 分派','结果 Buffer 句柄'],'data'); s['edges'][-1]['kind']='output'
    edge(s,'provider','status','可选执行 Future','completion')
    for i in range(4): edge(s,'core',f't{i}','Buffer + 操作参数','data',optional=True)
    for i in range(3): edge(s,f't{i}','after','新 Buffer / 新依赖','output')
    for i in (1,2,3): edge(s,f't{i}','invalid','原句柄 / 存储所有权变化','neutral')
    s['omit'].update({'p1','p2'})
    return s


def xla():
    s=base('xla'); n=s['nodes']
    n['plan']=node('调度与 BufferAssignment', ['位于目标后端内部，组织调度、内存规划和代码生成所需约定。',
        'BufferAssignment 描述存储分配，不是运行时 Buffer 数据。'], 'BufferAssignment','CpuCompiler RunBackend','GpuCompiler RunBackend', tag='消费准备 · 编译规划')
    n['hook']=node('PRE / POST_SCHEDULER 挂点', ['后端在 HLO 调度前后调用已注册的变换。',
        'CPU POST 位于 BufferAssignment 前；具体位置由后端决定。'], 'CPU PRE_SCHEDULER','CPU POST_SCHEDULER','GPU POST_SCHEDULER', tag='按需扩展 · 编译控制', color='transform')
    n['callback']=node('Python HLO callback', ['输入：序列化的 HloModuleProto bytes。',
        '返回 bytes 更新程序；返回 None 保留当前程序。',
        'CPU 直接注册；插件路径需要相应扩展支持。'], 'HLO transformation registration','CPU hook registration','PJRT hook registration', tag='外部回调 · D 接口', color='transform', owner='JAX 扩展接口', link='../overview/overview-software-stack-components-extended.svg#xla-interface-d')
    n['return']=node('后端 Executable → provider', ['CPU / GPU 后端返回程序代码与执行计划。',
        'provider 再包装为 PJRT 已加载可执行对象。'], 'CpuCompiler RunBackend','GpuCompiler RunBackend', tag='编译结果返回', color='result')
    section(s,'01 · 由编译输入产生 HLO','XLA 导入接口',['p0','core','p1'])
    section(s,'02 · 程序结构与 pass 调度','XLA HLO',['d0','d1'],['t0'],['t1','t2'],['after'])
    section(s,'03 · 按后端阶段调用扩展','XLA 编译控制',['hook','callback'])
    section(s,'04 · 后端消费、内部规划与返回','XLA',['c0','c1'],['plan','return'])
    edge(s,'p0','core','ConvertMlirHloToHloModule'); edge(s,'p1','core','HloModuleProto + config')
    edge(s,'core','d0','computations / entry','neutral'); edge(s,'d0','d1','instructions / root','neutral')
    edge(s,'core','t0','HloModule + pass 配置','transform')
    for i in (1,2):
        edge(s,'t0',f't{i}','已配置该 pass 时 · Run','transform',optional=True)
        edge(s,f't{i}','after','更新 HLO / changed','transform')
    edge(s,'t0','after','pipeline 结果','transform')
    edge(s,'after','hook','后端 PRE / POST 阶段关联','neutral',optional=True)
    edge(s,'hook','callback','HloModuleProto bytes','transform',optional=True)
    edge(s,'callback','hook','bytes / None · 继续编译','transform')
    edge(s,'hook','plan','挂点位置由目标后端决定','neutral',optional=True)
    for i in (0,1):
        edge(s,'after',f'c{i}','RunBackend · HloModule')
        edge(s,f'c{i}','plan','后端内部的调度 / 存储规划','neutral')
        edge(s,f'c{i}','return','Executable','result')
    s['paths'].append(['p0','core','t0','after','c0','return'])
    n['c0']['link']='cpu-llvm-ir-centered-hub.svg'; n['c1']['link']='gpu-ir-centered-hub.svg'
    s['omit'].update({'r0','r1'})
    return s


def cpu():
    s=base('cpu'); n=s['nodes']
    n['hlo']=node('HloModule + 目标与存储规划', ['RunBackend 进入 CPU 编译；提供 HLO、目标配置与内存规划。'], 'CpuCompiler RunBackend', tag='外部输入 · XLA', link='hlo-centered-hub.svg')
    n['thunks']=node('ThunkSequence / 库调用', ['描述 host kernel 与 oneDNN / Eigen 等调用的执行顺序。',
        '存储分配、常量和 kernel 符号共同约定运行时行为。'], 'CPU ThunkEmitter', tag='产生 · 执行计划')
    n['units']=node('LLVM 编译单元', ['从主模块及额外 kernel 模块收集编译单元。',
        '按条件 ExtractKernels / SplitModule；抽出或拆分片段整理无用符号。'], 'CPU extra kernels','CPU split module','CPU unused symbols', tag='变换 · 条件拆分', color='transform')
    n['symbols']=node('目标对象与已解析符号', ['LLVM 产生目标对象；模块编译器解析 kernel / comparator 符号。',
        'ASM 是可观察的代码表示，代码生成不要求单独落盘汇编文本。'], 'CPU machine code','CPU AddModule', tag='产生 · 目标代码', color='result')
    n['executable']=node('CpuExecutable', ['聚合代码、ThunkSequence、常量与内存计划。',
        '返回 provider，继续加载与 PJRT 包装。'], 'CpuCompiler CompileCpuExecutable', tag='编译结果返回', color='result', link='../pjrt/buffer-centered-hub.svg')
    section(s,'01 · 编译输入','外部 · XLA',['hlo'],external=True)
    section(s,'02 · LLVM 模块与执行计划的协作生成','XLA CPU 后端',['p0','p1','p2'],['core','thunks'])
    section(s,'03 · 编译单元与 LLVM 代码生成','XLA CPU / LLVM',['units','c1'],['t0','c0'],['symbols'])
    section(s,'04 · 聚合后端结果','XLA CPU 后端',['executable'])
    edge(s,'hlo','p0','HLO + target 配置'); edge(s,'p0','p1','LLVM Module + emitter 上下文')
    edge(s,'p1','p2','IrEmitter2 提供代码生成能力','neutral')
    edge(s,'p2','core','主模块 / 额外 host kernels'); edge(s,'p2','thunks','EmitEntryComputation → thunks')
    chain(s,['core','units','c1','t0','c0','symbols','executable'],['LLVM Module','AddModule / Compile','IrCompiler::RunIrPasses','优化后的 Module','目标对象字节','解析后的函数与代码'])
    s['edges'][-4]['kind']='transform'; s['edges'][-3]['kind']='transform'
    edge(s,'thunks','executable','ThunkSequence + 常量 / 内存计划','result')
    s['omit'].update({'after','r0','r1'})
    return s


def gpu():
    s=base('gpu'); n=s['nodes']
    s['summary']='GPU LLVM IR 是唯一核心概念；原生生成、Triton 编译与设备库路径分别展开。目标代码以 NVIDIA 路径为例。'
    n['core']=node('GPU LLVM IR', ['llvm::Module / Function / 指令描述待编译的 GPU 程序。',
        'target triple、data layout 与 kernel 标记参与目标代码生成。',
        '既可为主模块，也可为独立 kernel 编译单元。',
        'Triton 可以生成 LLVM IR；设备库调用另行进入执行计划。'], 'LLVM Module','GPU module emission', tag='定义 · 研究对象')
    n['hlo']=node('HLO + GPU 目标配置', ['后端为计算选择代码生成、Triton fusion 或设备库路径。',
        '不同路径可共同组成一个 GPU 可执行程序。'], 'GpuCompiler RunBackend', tag='外部输入 · XLA', link='hlo-centered-hub.svg')
    n['native']=node('原生 LLVM 代码生成', ['CompileModuleToLlvmIr 组织 emitter 与 kernel 生成。',
        '生成 LLVM 函数和相关常量，配合执行计划。'], 'GPU module emission', tag='产生 · 原生分支')
    n['library']=node('cuBLAS / cuDNN 等设备库', ['库调用作为执行计划的一部分。',
        '此分支不要求为每项操作现场生成 LLVM kernel。'], 'GPU module emission','GpuCompiler RunBackend', tag='消费 HLO · 库调用分支')
    n['thunks']=node('ThunkSequence + launch / 常量', ['组织生成的 kernel 与设备库调用。',
        '执行计划、内存约定与目标二进制共同构成程序。'], 'GPU module emission','GpuCompiler RunBackend', tag='产生 · 执行计划')
    n['triton_result']=node('TritonWrapperResult / LLVM 代码', ['Triton 编译返回包装结果，供 kernel 包装和后续代码编译使用。',
        '保留 launch、共享内存等元数据；不会把 Triton IR 等同于 LLVM IR。'], 'Triton compiler','Triton LLVM translation', tag='产生 · Triton 编译结果')
    n['t2']['refs']=('Triton to LLVM','Triton compiler')
    n['kernel_compile']=node('KernelCompiler::CompileToTargetBinary', ['Triton 分支对包装后的独立 LLVM kernel 提交目标编译。',
        'CUDA 实现经 CubinCustomKernelCompiler 产生二进制。'], 'GPU custom binary','Triton fusion', tag='消费 · 独立 kernel 路径')
    n['kernel_binary']=node('kernel 二进制 → CustomKernelThunk', ['将二进制、参数、launch 维度和 shared memory 组织为 custom kernel。',
        'CustomKernelThunk 进入执行计划；不是所有 kernel 都汇入同一个 LLVM 主模块。'], 'Triton custom thunk','GPU custom binary', tag='编译结果 · 独立 kernel', color='result')
    n['ptx']=node('PTX → compilation provider', ['NVIDIA 目标编译产生 PTX。',
        'compilation provider 继续编译 / 链接为设备二进制。'], 'NVPTX target binary', tag='消费 LLVM IR · NVIDIA 路径')
    n['binary']=node('BackendCompileResult / cubin', ['包含目标二进制及编译统计等信息。',
        'ROCm 使用另一条目标后端，不在此图展开。'], 'NVPTX target binary','GPU single module', tag='编译产物', color='result')
    n['executable']=node('GpuExecutable → provider', ['目标代码与 thunk、常量、存储计划共同封装。',
        '执行时再绑定 Buffer 与 stream，进入设备运行时。'], 'GpuCompiler RunBackend', tag='编译结果返回', color='result', link='../stream-executor/submission-centered-hub.svg')
    section(s,'01 · 目标分派与编译组织','XLA GPU 后端',['hlo','p0'])
    section(s,'02 · 三条互补路径','XLA GPU 代码生成',['native','p1','library'],['t2'],['t0','t1'],['triton_result'])
    section(s,'03 · 程序表示与执行计划','XLA GPU 后端',['core','thunks'])
    section(s,'04 · NVIDIA 目标编译','XLA GPU / LLVM',['c1','c0','kernel_compile'],['t3','ptx','kernel_binary'],['binary'])
    section(s,'05 · 聚合后端结果','XLA GPU 后端',['executable'])
    edge(s,'hlo','p0','HLO + target / buffer 信息')
    for target,label in [('native','原生 emitter'),('p1','选中 Triton fusion'),('library','设备库调用')]: edge(s,'p0',target,label,optional=True)
    edge(s,'native','core','LLVM Module / kernels')
    edge(s,'p1','t2','TritonKernelSource + 配置')
    edge(s,'t2','t0','编译控制 · XLA / Triton pass','transform')
    edge(s,'t0','t1','高层合法化后的模块','transform')
    edge(s,'t1','triton_result','后续翻译 / LLVM 代码','transform')
    edge(s,'triton_result','core','LLVM kernel 与包装代码')
    edge(s,'p0','thunks','生成执行计划','neutral'); edge(s,'library','thunks','设备库 thunk','neutral')
    chain(s,['core','c1','c0','t3','ptx','binary','executable'],['常规模块编译路径','CompileTargetBinary','链接 / LLVM 优化','目标代码生成','编译 / 链接 cubin','目标二进制'])
    s['edges'][-4]['kind']='transform';s['edges'][-3]['kind']='transform';s['edges'][-1]['kind']='result'
    edge(s,'thunks','executable','执行计划 + 常量 / 存储约定','result')
    edge(s,'core','kernel_compile','独立 kernel 编译路径',optional=True)
    edge(s,'kernel_compile','kernel_binary','设备二进制 + launch 信息','result')
    edge(s,'kernel_binary','thunks','CustomKernelThunk','result')
    s['omit'].update({'after','r0','r1'})
    return s


def tpu():
    s=base('tpu'); n=s['nodes']
    s['title']='TPU / libtpu · 公开接口与 LLO 证据边界'
    s['summary']='主路径只画公开插件的编译与执行契约；LLO 的内部结构、产生、变换和消费者单列为待取证内容。'
    n['loaded']=node('PJRT_LoadedExecutable', ['公开编译接口返回的不透明已加载程序句柄。',
        '句柄行为可见；内部 LLO 结构和完整编译流程未公开。'], 'PJRT compile ABI', tag='编译结果返回', color='result')
    n['input']=node('执行输入 Buffer + 选项', ['按设备提供输入存储句柄，并满足执行接口的生命周期约定。'], 'PJRT execute ABI', tag='执行输入', color='data')
    n['execute']=node('PJRT_LoadedExecutable_Execute', ['接收可执行句柄、输入 Buffer 和执行选项。',
        '公开结果包括输出 Buffer 与 device_complete_events。'], 'PJRT execute ABI', tag='消费 · 公开执行 ABI', color='data')
    n['outputs']=node('输出 Buffer', ['设备结果存储的句柄返回上层。',
        '输出包装与工作完成分开表达。'], 'PJRT execute ABI', tag='输出对象返回', color='output')
    n['events']=node('device_complete_events', ['公开事件用于观察对应设备的完成或错误。',
        '图示仅检查接口声明，不构成真实 TPU 执行证据。'], 'PJRT execute ABI', tag='完成状态', color='completion')
    n['p1']['color']='program'; n['p1'].pop('uncertain',None)
    n['p1']['tag']='消费 · 公开编译 ABI'
    n['core']['tag']='定义 · 内部证据边界';n['core']['uncertain']=True
    section(s,'01 · 获得插件 Client，提交程序','公开 · JAX / PJRT ABI',['p0','d1','p1'],external=True)
    section(s,'02 · 编译返回与运行时接口','公开 · PJRT ABI',['loaded','input'],['execute'],['outputs','events'],external=True)
    section(s,'03 · 私有编译内部的研究对象','libtpu · 待版本证据',['core'])
    edge(s,'p0','p1','Client + 插件编译入口')
    edge(s,'d1','p1','PJRT_Program + CompileOptions')
    edge(s,'p1','loaded','不透明 LoadedExecutable','result')
    edge(s,'loaded','execute','已加载程序句柄','data'); edge(s,'input','execute','Buffer 参数','data')
    edge(s,'execute','outputs','输出存储句柄','output'); edge(s,'execute','events','设备完成事件','completion')
    s['paths'].append(['p1','loaded','execute','outputs'])
    s['omit'].update({'after','c0','r0','c1','r1','r2'})
    n['c2']['title']='执行与序列化的不同契约'
    # Public ABI arrows never pretend to identify an LLO producer or consumer.
    return s


def runtime():
    s=base('runtime');n=s['nodes']
    n['args']=node('GetKernelAndArgs / 设备地址', ['加载的 kernel 与 BufferAllocations 提供函数和参数地址。',
        'launch 维度、shared memory 与 stream 共同约定本次启动。'], 'Kernel arguments', tag='产生 · 解析启动参数', color='data')
    n['submit_status']=node('提交 Status', ['CUDA 启动 / 传输 API 返回提交成功或错误。',
        '成功提交与设备工作完成是两个不同的观察时刻。'], 'CUDA driver launch','CUDA memcpy', tag='同步返回 · 提交结果', color='output')
    n['work']=node('队列中的设备工作', ['驱动安排 kernel 或传输，设备按 stream / event 依赖推进。',
        '公开源码确认驱动调用；驱动私有调度不在本图展开。'], 'CUDA launch','CUDA memcpy', tag='异步执行 · 设备侧', color='data')
    n['event']=node('RecordEvent / 完成点', ['在 stream 上记录事件，标记此前工作的完成点。',
        '事件可用于跨 stream 依赖或按需观察。'], 'Stream record', tag='完成状态 · 设备依赖', color='completion')
    n['wait']=node('WaitFor / 后续工作', ['后续 stream 按事件或另一条 stream 的依赖继续。',
        '主机无需在每次提交后阻塞等待。'], 'Stream wait', tag='消费 · 设备侧依赖', color='data')
    n['command_submit']=node('CommandBuffer::Submit', ['提交已记录或已更新的命令缓冲区。',
        '命令可以包含多个工作项；平台实现负责实际提交。'], 'Command buffer submission', tag='消费 · 命令缓冲区路径', color='data')
    n['core']['title']='执行提交'
    n['core']['rows']=['kernel / 传输请求、参数地址与 stream 在此汇合。',
        '具体接口由 Stream 与 CUDA 实现承担。', '提交返回、设备工作推进、完成观察分别表达。']
    section(s,'01 · 从执行计划产生具体工作','外部 · XLA GPU runtime',['p0','args','p1'],external=True)
    section(s,'02 · Stream 接口与平台实现','StreamExecutor / CUDA',['core','p2'],['c0','c1','command_submit'],['submit_status','work'])
    section(s,'03 · 异步依赖与完成观察','StreamExecutor / CUDA',['event','wait'],['c3','r3'])
    section(s,'04 · 按需记录和更新命令','XLA runtime / CUDA',['t1','t2'])
    edge(s,'p0','args','GetKernelAndArgs','data'); edge(s,'args','core','函数 / launch / 参数地址','data')
    edge(s,'p1','core','CommandBuffer::Submit','data',optional=True)
    edge(s,'p2','core','传输源 / 目标 / 字节数','data',optional=True)
    edge(s,'core','c0','kernel 启动分支','data'); edge(s,'core','c1','异步传输分支','data')
    edge(s,'core','command_submit','命令缓冲区分支','data')
    for key in ('c0','c1','command_submit'):
        edge(s,key,'submit_status','API 返回 Status','output')
        edge(s,key,'work','提交成功后排队','data')
    edge(s,'work','event','在 stream 上记录完成点','completion',optional=True)
    edge(s,'event','wait','event / stream 依赖','completion')
    edge(s,'work','c3','主机按需等待 stream','completion',optional=True)
    edge(s,'c3','r3','完成 / 错误','completion')
    edge(s,'t1','p1','capture 后的命令计划','data',optional=True)
    edge(s,'t2','p1','更新当前地址再提交','data',optional=True)
    s['paths'].extend([['p0','args','core','c0','submit_status'],['c0','work','c3','r3']])
    s['omit'].update({'after','r0','r1','c2','r2','t0','t3'})
    return s


def diagrams():
    from software_stack_compiler_views import apply_compiler_views
    from software_stack_runtime_views import apply_runtime_views
    from software_stack_tpu_views import apply_tpu_views

    result=[f() for f in (jax,jaxlib,ifrt,pjrt,xla,cpu,gpu,tpu,runtime)]
    result=[apply_tpu_views(apply_runtime_views(apply_compiler_views(s))) for s in result]
    # Colors describe the kind of flow / object, independently of the four views.
    observation_colors={
        'jax':dict(c0='data',c1='neutral',c2='neutral',r3='program'),
        'jaxlib':dict(c2='neutral',r2='neutral',c3='neutral',r3='neutral',r1='program'),
        'ifrt':dict(c2='neutral',r2='completion',r3='neutral'),
        'pjrt':dict(c2='neutral',r2='completion',c3='neutral',r3='neutral'),
        'xla':dict(c2='neutral',r2='neutral',c3='neutral',r3='neutral'),
        'cpu':dict(c2='neutral',r2='neutral',c3='neutral',r3='neutral'),
        'gpu':dict(c2='neutral',r2='neutral',c3='neutral',r3='neutral'),
        'runtime':dict(r3='completion'),
    }
    for s in result:
        used={key for part in s['sections'] for row in (*part['rows'], part.get('details',[])) for key in row if key}
        if len(used)!=sum(bool(key) for part in s['sections'] for row in (*part['rows'], part.get('details',[])) for key in row):
            raise ValueError('Repeated main node: '+s['key'])
        s['details']=[key for key in s['nodes'] if key not in used and key not in s['omit']]
        s['nodes']={key:n for key,n in s['nodes'].items() if key not in s['omit']}
        for n in s['nodes'].values(): n['owner']=n.get('owner') or LABELS[s['key']]
        for key,color in observation_colors.get(s['key'],{}).items():s['nodes'][key]['color']=color
    return result
