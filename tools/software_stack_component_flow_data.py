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
              cpu='CPU 后端', gpu='GPU 后端', tpu='TPU / libtpu', runtime='StreamExecutor / CUDA')


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
    s = dict(key='jax', path=PATHS['jax'], title='JAX · 定义与产生、变换、消费的调用链',
             summary='定义独立说明 Jaxpr；产生、变换、消费分块展开，以对象流与调用关系贯通。变换可按需跳过。',
             foot='定义区只解释抽象与结构。操作区展示产生 → 按需变换 → 消费，并保留直接消费支路。Pallas 重新绑定交给当前 Trace；只有记录外层程序时才展开到 DynamicJaxprTrace。灰色附区跟踪 Module、executable 和数组；设备完成另行观察。',
             nodes={}, sections=[], edges=[], paths=[], omit=set(), core_id='core',
             avoid_endpoint_captions=True, layout='connected_views', canvas_width=23900)
    n = s['nodes']
    n['core'] = node('Jaxpr', ['all_invars / invars / outvars 界定程序输入输出。',
        'eqns 保存 JaxprEqn；constvars / consts 表达已附常量。',
        'avals、effects、DebugInfo 与 is_high 约束语义和来源。'], 'Jaxpr', tag='定义 · 研究对象')
    for i, (tag, title, rows, refs) in enumerate(DEFINITIONS):
        n[f'd{i}'] = node(title, rows, *refs, tag='定义 · '+tag, color='neutral')
    for i, (tag, title, rows, refs) in enumerate(PRODUCERS):
        n[f'p{i}'] = node(title, rows, *refs, tag='产生 · '+tag)
    for i, (_, title, rows, refs) in enumerate(TRANSFORMS):
        n[f't{i}'] = node(title, rows, *refs, tag='变换 · 独立可选分支', color='transform')
    for i, (_, title, rows, result, result_rows, refs) in enumerate(CONSUMERS):
        n[f'c{i}'] = node(title, rows, *refs, tag='消费 · 解释器 / lowering')
        n[f'r{i}'] = node(result, result_rows, *refs, tag='消费结果', color='result')
    n['eqn'] = node('JaxprEqn', ['invars / outvars 连接 Var 或 Literal。',
        'primitive 标识操作；params 可以携带子 Jaxpr。', 'effects、source_info / ctx 保存方程语义和上下文。'], 'JaxprEqn', tag='定义 · 方程结构', color='neutral')
    n['var'] = node('Var / Literal', ['Var 保存抽象类型，并通过变量身份建立值依赖。',
        'Literal 携带字面值；它与常量变量的调用约定不同。'], 'Var', 'Literal', tag='定义 · 值与类型', color='neutral')
    n['primitive'] = node('Primitive 与解释规则', ['primitive.bind 根据当前 Trace 分派。',
        '抽象求值、batching、微分和 lowering 各有对应规则。'], 'Primitive', 'Primitive.bind', tag='定义 · 操作及其解释', color='neutral')
    n['after'] = node('变换后的 Jaxpr 与附加信息', ['返回一个或多个 Jaxpr，以及 used_inputs、out_axes、residual 等。',
        '变换产物仍受 Jaxpr 的类型、效果与良构性约束。'], 'dce_jaxpr', 'linearize_jaxpr', tag='变换结果', color='transform')
    n['lower_rules'] = node('jaxpr_subcomp / lowering rules', ['逐方程读取输入 IR 值，根据 primitive 查找 lowering 规则。',
        '规则返回 IR 结果值；写回方程输出的值环境。'], 'jaxpr_subcomp', 'register_lowering', tag='消费 · 原语规则')
    n['frontend_arrays'] = node('jax.Array / ArrayImpl', ['用户可见的数组值，带有 dtype、shape 与 sharding。',
        '实际数组与追踪中的 Tracer 分属不同对象。', '执行路径传递实际数组；追踪主要使用其抽象信息。'],
        'ArrayImpl', tag='定义 · 前端值与存储', color='data', link='#input_arrays')
    n['pytrees'] = node('PyTree / 函数展平', ['tree_flatten 将嵌套结构拆成 leaves 与 treedef。',
        'flatten_fun 组织展平函数的输入输出约定。', 'PyTree 描述结构，Jaxpr 记录展平后计算的值依赖。'],
        'tree_flatten', 'flatten_fun', tag='产生准备 · 调用结构', color='neutral', link='#p0')
    n['placement'] = node('Mesh / NamedSharding', ['Mesh 给设备轴命名；NamedSharding 组织数组分片。',
        '分片及布局参与 lowering 和运行时输入输出处理。'],
        'Mesh', 'NamedSharding', tag='定义 · 放置上下文', color='neutral', link='#c3')
    n['api_transforms'] = node('jit / grad / vmap', ['jit 组织暂存编译，grad 组织微分，vmap 组织批量化。',
        '函数变换按需要组合；各自使用对应解释规则。', 'API 调用并不固定对应一次追踪、一次 Jaxpr 或一次编译。'],
        'jit', 'grad', 'vmap', tag='变换入口 · 函数层协议', color='transform', link='#p0')
    n['trace_types'] = node('Trace / Tracer', ['Trace 解释 primitive；Tracer 表示追踪期间的值。',
        'DynamicJaxprTrace 将操作记录到 frame。', '最终 Jaxpr 不需要保留这些追踪实例。'],
        'Trace', 'Tracer', 'DynamicJaxprTracer', tag='产生机制 · 解释与抽象值', color='neutral', link='#p1')
    n['aval'] = node('aval / AbstractValue / ShapedArray', ['aval 定义值的抽象类型，不只是 shape。',
        '抽象求值根据输入 aval 推导输出 aval 与 effects。', 'lowering 则在平台上下文中生成 IR，两类规则分开。'],
        'AbstractValue', 'ShapedArray', 'default_process_primitive', tag='定义 · 类型与效果', color='neutral', link='#var')
    n['stages'] = node('Traced / Lowered / Compiled', ['Traced 表示追踪结果，可继续 lower。',
        'Lowered 暴露 lowering 结果，可继续 compile。',
        'Compiled 提供编译后调用接口；它不是“设备已执行”的证明。',
        '显式阶段 API 与普通 jit 调用、缓存复用按入口区分。'],
        'Traced', 'Lowered', 'Compiled', tag='消费接口 · 显式阶段对象', color='neutral', link='#compile_cache')
    n['partial_values'] = node('PartialVal / HiJAX 与 LoJAX', ['PartialVal 区分已知值与只有 aval 的未知值。',
        '高层类型与 primitive 可按协议展开为低层 Jaxpr。',
        '这种 JAX 内部转换不等于选择 StableHLO 或 Mosaic。'],
        'PartialVal', 'HiType', 'Primitive.bind_with_trace', tag='变换约束 · 已知值与表示层次', color='neutral', link='#t4')

    n['caller'] = node('Python 函数中的 pl.pallas_call', ['调用者传入数组，并接收结果数组。',
        'kernel、grid、BlockSpec 等描述所调用的计算。'], 'pallas_call', tag='产生 · Pallas 调用入口')
    n['kernel'] = node('kernel 函数 + GridMapping + avals', ['get_grid_mapping 整理网格、块映射与调用约定。',
        'kernel 使用 AbstractRef 等抽象输入参与追踪。'], 'get_grid_mapping', 'AbstractRef', tag='产生 · kernel 追踪输入')
    n['outer_jaxpr'] = node('外层 Jaxpr / pallas_call 方程', ['普通方程 → pallas_call → 后续方程。',
        'invars / outvars 连接外层数组值。', 'params.jaxpr 携带内层程序；GridMapping 等另行携带。',
        'pallas_call_p.bind 将调用与参数交给外层 Trace。'], 'JaxprEqn', 'pallas_call_p.bind', tag='定义 · Jaxpr 的调用者实例', link='#core')
    n['inner_jaxpr'] = node('内层 kernel Jaxpr', ['同样使用 core.Jaxpr，保存 kernel 方程与 effects。',
        '通过输入 / 输出 Ref 的读写表达计算。', 'kernel Python 函数返回 None，输出由 Ref 写入。',
        '追踪与 DCE 返回 kernel Jaxpr 和捕获的 consts。'], 'Jaxpr', '_trace_kernel_to_jaxpr', tag='定义 · Jaxpr 的 kernel 实例', link='#core')
    n['outer_lowering'] = node('jaxpr_subcomp → Pallas lowering', ['输入：外层 IR 实参、规则上下文与 eqn.params。',
        '_pallas_call_lowering 按平台和 interpret 模式分派。',
        '此处展开 interpret=False 的 Mosaic TPU 路径。'], 'jaxpr_subcomp', 'register Pallas lowering', '_pallas_call_lowering', tag='消费 · 外层方程规则')
    n['mosaic_module'] = node('Mosaic TPU MLIR Module', ['TPU 规则消费 kernel Jaxpr、GridMapping 和上下文。',
        'lower_jaxpr_to_pipelined_module 构造独立 ir.Module。',
        '该内层模块随后成为 custom_call 的序列化 payload。', 'Mosaic TPU MLIR 不是 LLO，也不是最终设备机器码。'],
        'pallas_call_tpu_lowering_rule', 'lower_jaxpr_to_pipelined_module', tag='消费 kernel · 产生专用模块')
    n['payload'] = node('CustomCallBackendConfig / payload', ['输入：消费区产生的 Mosaic TPU MLIR Module。',
        '克隆 Module，运行版本化 mosaic-serde pass。',
        'write_bytecode 写出模块字节；这里不是机器汇编。',
        'to_json 将字节码编码进 custom_call_config.body。',
        'payload 携带内层程序，与外层 operands 分别传递。'],
        '_lower_mosaic_module_to_asm', 'CustomCallBackendConfig.to_json', 'lower_module_to_custom_call', tag='后续 · Module 序列化与封装', color='transform', link='#mosaic_module')
    n['outer_module'] = node('外层 Module / stablehlo.custom_call', ['call_target_name = tpu_custom_call。',
        'backend_config 携带序列化 kernel 与配置。', 'operands / results 接入外层其他 StableHLO 操作。',
        'lowering 返回 IR 值；设备结果由之后的执行产生。'], 'emit tpu_custom_call', 'lower_jaxpr_to_module', tag='后续 · custom call 接入外层 IR')
    n['mapping'] = node('GridMapping 配合 kernel Jaxpr', ['grid / block_mappings / scratch_avals 描述调用映射。',
        'Jaxpr 定义 kernel 计算，GridMapping 组织其调用。',
        '两者是独立参数，由 TPU lowering 一起消费。'], 'GridMapping', 'GridSpec', 'BlockSpec', tag='定义 · kernel 的调用约定', color='neutral', link='#inner_jaxpr')
    n['transforms'] = node('Pallas 专用 JVP / batching 规则', ['JVP 消费 kernel Jaxpr 与切向信息，产生新程序。',
        '规则同时更新 GridMapping、Ref 顺序和输出约定。',
        'batching 也按专用规则处理调用；各自有适用约束。',
        '重建后的调用重新绑定到外层解释环境。'],
        '_pallas_call_jvp_rule', '_pallas_call_batching_rule', tag='变换 · 满足规则约束时应用', color='transform', link='#inner_jaxpr')

    n['compile_cache'] = node('compile_or_get_cached', ['输入：完整外层 Module、设备、编译选项与回调。',
        '持久缓存命中时恢复可执行对象；否则按策略编译。',
        '这里的恢复可包含反序列化与加载。'], 'compile_or_get_cached', tag='后续 · Module 编译缓存与策略')
    n['backend_compile'] = node('backend_compile_and_load', ['常规后端调用 Client.compile_and_load。',
        'CompileOnlyPyClient 有 compile 分支，不能视为执行。',
        '下列主线展开常规编译并加载路径。'], 'backend_compile_and_load', tag='后续 · Module 编译调用边界')
    n['compile'] = node('PyClient::CompileAndLoad', ['接收整个外层 Module 与编译选项。',
        '函数内部克隆 Module，包装为 IFRT HloProgram。',
        '将编译结果包装为 PyLoadedExecutable 返回。'], 'PyClient.CompileAndLoad', tag='外部接口 · 原生编译与加载', owner='jaxlib / IFRT', link='../jaxlib/mlir-module-centered-hub.svg#core')
    n['loaded_exec'] = node('已加载程序 / 执行包装', ['编译或缓存返回的 executable 由执行包装持有。',
        'ExecuteReplicated.xla_executable 供后续调用使用。',
        '已加载程序与本次输入数组在执行入口汇合。'], 'compile_or_get_cached', 'ExecuteReplicated', tag='编译结果返回', color='result')
    n['input_arrays'] = node('本次调用的数组输入', ['动态参数经 in_handler 组织为输入数组。',
        '按需加入 effects / tokens，满足执行约定。'], 'ExecuteReplicated', tag='执行 · 实际数组与依赖', color='data', link='../ifrt/array-centered-hub.svg#core')
    n['execute'] = node('ExecuteReplicated.__call__', ['调用 xla_executable.execute_sharded(input_bufs)。',
        '带 effects / callbacks 时使用 token 路径。',
        'jaxlib / IFRT / provider 负责后续执行提交。'], 'ExecuteReplicated', 'execute_sharded', tag='后续 · executable 与数组执行', color='data')
    n['result_arrays'] = node('Python 可见结果数组', ['consume_with_handlers 按输出规则包装结果。',
        '对象返回与设备数据就绪是不同观察时刻。'], 'ExecuteReplicated', tag='输出对象返回', color='output', link='../ifrt/array-centered-hub.svg#status')
    n['produced'] = node('构造结果：Jaxpr + constvals', [
        '输入与输出变量、方程、effects 汇总为程序对象。',
        '普通追踪与 kernel 追踪都产生同一 Jaxpr 抽象。',
        '调用方按约定附加常量，或将 constvals 单独传递。'],
        'Frame.to_jaxpr', '_trace_kernel_to_jaxpr', tag='产生结果 · 持久程序', link='#core')
    n['core']['rows'] = ['JAX 的显式程序表示：用方程与值依赖描述计算。',
        '输入与输出界定程序边界；avals 与 effects 约束语义。',
        '定义不依赖某一次 tracing、变换或后端编译。']
    n['kernel']['title'] = 'kernel 追踪：函数 + 映射 + avals'
    n['kernel']['rows'] = ['get_grid_mapping 整理网格、块映射与调用约定。',
        '_trace_kernel_to_jaxpr 用 AbstractRef 输入追踪 kernel。',
        '追踪及 DCE 返回 kernel Jaxpr 与捕获的 consts。']
    n['kernel']['refs'] += ('_trace_kernel_to_jaxpr',)
    n['kernel']['link'] = '#inner_jaxpr'
    n['caller']['rows'] = ['kernel、grid 与 BlockSpec 等描述被调用的计算。',
        '先追踪 kernel，得到内层 Jaxpr 与捕获常量。',
        'pallas_call_p.bind 携带内层程序、映射与数组实参，',
        '交给外层 Trace 收集 pallas_call 方程。']
    n['caller']['refs'] += ('pallas_call_p.bind',)
    n['caller']['link'] = '#outer_jaxpr'
    n['inner_jaxpr']['rows'] = ['同样使用 core.Jaxpr，保存 kernel 方程与 effects。',
        '通过输入 / 输出 Ref 的读写表达计算。', 'kernel Python 函数返回 None，输出由 Ref 写入。',
        '它与外层 Jaxpr 是同一抽象的不同实例。']
    n['t5']['tag'] = '共同实现机制 · 对象重建'
    n['t5']['color'] = 'neutral'
    n['t5']['rows'] = ['Jaxpr.replace / Jaxpr(...) 重建字段与程序对象。',
        '具体变换按自身规则选择要重建的签名、方程和元数据。',
        '这是构造机制，不与 DCE、微分等算法并列。']
    n['transforms']['rows'] = ['输入：kernel Jaxpr + GridMapping + 切向 / 批维信息。',
        'JVP 可重建 kernel；batching 按分支复用或改写程序 / 调用。',
        '按需更新映射、Ref 与输出约定，并重新绑定调用。',
        '输出程序与调用参数；各规则有适用约束。']
    for key in ('t0','t1','t2','t3','t4','transforms'):
        n[key]['link'] = '#core'
    n['after']['link'] = '#core'
    n['after']['title'] = '变换结果：Jaxpr 与附加信息'
    n['after']['rows'] = ['通常得到改写或分拆的 Jaxpr，以及 used_inputs、out_axes、residual 等。',
        'Pallas 专用分支也可复用 kernel，更新调用参数后重新绑定。',
        '产物仍受 Jaxpr 的类型、效果与良构性约束。']
    n['c0']['tag'] = '消费 · 解释 Jaxpr'
    n['c1']['tag'] = '消费 · 校验 Jaxpr'
    n['c2']['tag'] = '消费 · 打印 Jaxpr'
    n['c3']['tag'] = '消费 · 普通 JAX lowering 入口'
    n['c4']['title'] = 'lower_jaxpr_to_pipelined_module'
    n['c4']['rows'] = ['输入：kernel Jaxpr + GridMapping / Ref 上下文。',
        '按 Mosaic TPU 原语规则构造独立 MLIR Module。',
        '这是 Pallas kernel 消费分支；普通 JAX 不必经过它。']
    n['c4']['tag'] = '消费 · Pallas kernel lowering'
    n['c4']['link'] = '#inner_jaxpr'
    n['mosaic_module']['rows'] = ['结果：内层 kernel 的 Mosaic TPU MLIR Module。',
        '包含 TPU 专用操作及 kernel 映射语义。',
        '后续序列化为外层 custom_call 的 payload。',
        'Mosaic TPU MLIR 不是 LLO，也不是最终设备机器码。']
    n['mosaic_module']['tag'] = '消费结果 · kernel 的 MLIR Module'
    n['mosaic_module']['link'] = '#payload'
    n['r4']['title'] = '普通 lowering 与 Pallas 分支的关系'
    n['r4']['rows'] = ['普通方程由对应的 primitive lowering 规则处理。',
        '遇到 pallas_call 时才按模式与平台选择专用规则。',
        'TPU 规则消费内层 kernel，并将调用接回外层 IR。',
        '两类 Jaxpr 消费分别标出；不把所有调用串成必经路径。']
    n['r4']['tag'] = '消费说明 · 按方程与平台分支'
    n['r4']['link'] = '#outer_lowering'
    n['stages']['tag'] = '后续接口 · Traced / Lowered / Compiled'
    n['frontend_arrays']['tag'] = '相关输入 · 实际数组'
    n['placement']['tag'] = '相关上下文 · 放置与分片'
    n['api_transforms']['tag'] = '相关入口 · 函数层变换'
    for key in ('c0','c1','c2','c3'):
        n[key]['link'] = '#core'
    n['r3']['link'] = '#compile_cache'

    n['produced']['title'] = '调用者程序：Jaxpr + constvals'
    n['produced']['rows'] = ['由外层 frame 汇总输入、输出、方程与 effects。',
        '普通调用者与包含 pallas_call 的调用者都使用 Jaxpr。',
        '按调用约定附加常量，或将 constvals 单独传递。']
    n['produced']['refs'] = ('Frame.to_jaxpr', 'Jaxpr')
    n['outer_jaxpr']['tag'] = '产生结果 · 调用者程序实例'
    n['inner_jaxpr']['tag'] = '产生结果 · kernel 程序实例'
    n['mapping']['tag'] = '产生结果 · kernel 调用映射'
    n['mapping']['refs'] += ('get_grid_mapping',)
    n['caller']['rows'] = ['kernel、grid 与 BlockSpec 等描述被调用的计算。',
        '先追踪 kernel，得到内层 Jaxpr 与捕获常量。',
        '内层程序、映射和数组实参用于绑定 pallas_call。']
    n['outer_jaxpr']['rows'] = ['这是包含 pallas_call 方程的调用者 Jaxpr 实例。',
        'params.jaxpr 引用 kernel；params.grid_mapping 另存映射。',
        'invars / outvars 连接调用者的数组值。',
        '外层追踪收集该调用及前后方程，形成完整调用者程序。']
    n['transforms']['rows'][0] = '输入：kernel Jaxpr、映射及数组 / 切向 / 批维信息。'
    n['after']['rows'] = ['得到改写或分拆的 Jaxpr，以及 used_inputs、out_axes、residual 等。',
        '程序仍满足 Jaxpr 的类型、效果与良构性约束。',
        '调用方按各自协议选择需要解释或 lower 的程序。']
    n['transform_input'] = node('按需进入 Jaxpr 变换', [
        '输入：调用者 Jaxpr + 对应变换的参数。',
        '各分支是可选算法，不是依次执行的五个步骤。',
        '结果交回调用方，再按用途消费。'],
        'dce_jaxpr', 'linearize_jaxpr', tag='变换入口 · 程序实例', color='transform', link='#core')
    n['consume_input'] = node('待消费的调用者 Jaxpr', [
        '接收刚产生的程序，或由变换返回的程序。',
        '解释、校验、打印与 lowering 按需求独立选择。',
        '下列连线展开读取同一类程序的不同接口。'],
        'eval_jaxpr', 'check_jaxpr', 'lower_jaxpr_to_module',
        tag='消费入口 · 程序实例', link='#core')
    n['pallas_bind'] = node('pallas_call_p.bind → 当前 Trace', [
        '输入：kernel Jaxpr、GridMapping、常量与数组实参。',
        '原始调用或专用变换重建的调用按约定重新绑定。',
        '当前 Trace 决定解释方式；记录外层程序时收集方程。',
        '此图只在该记录场景展开到 DynamicJaxprTrace。'],
        'pallas_call_p.bind', 'Primitive.bind', tag='产生机制 · 外层调用绑定')
    n['kernel_after'] = node('专用规则内部的程序与调用参数', [
        '规则内部复用或改写 kernel Jaxpr，并组织调用参数。',
        '按需更新 GridMapping、Ref 顺序与输出约定。',
        '重新 bind 调用；规则面向调用方的结果仍是数组值。'],
        '_pallas_call_jvp_rule', '_pallas_call_batching_rule',
        tag='变换内部产物 · 再绑定所需信息', color='transform')
    n['execute']['rows'].insert(0,'输入：执行包装持有的 executable + 本次数组。')
    n['execute']['link']='#input_arrays'

    # Definition is a separate static block. Three operation blocks retain
    # cross-view object flow, including direct consumption without a transform.
    section(s, '定义 · Jaxpr 是什么？', 'JAX core',
            ['core'], ['eqn'], ['var'], ['primitive'],
            ['d0'], ['d1'], ['d2'], ['d3'], ['aval'])
    s['sections'][-1].update(id='view_definition',
        question='独立定义语义、结构和约束；不作为调用链中的执行阶段。')
    section(s, '产生 · 输入如何构成程序？', 'JAX tracing / Pallas tracing',
            ['p0','p1','p2'], ['caller','kernel','produced'],
            ['pallas_bind','inner_jaxpr','outer_jaxpr'],
            ['pytrees','trace_types','mapping'], ['frontend_arrays','placement'])
    s['sections'][-1].update(id='view_production',
        question='函数与抽象输入经 Trace / frame 构造调用者程序；kernel 先追踪，再绑定到当前解释环境。')
    section(s, '变换 · 按需改写或拆分程序', 'JAX interpreters / Pallas 专用规则',
            ['transform_input'], ['t0','t1','t2'], ['t3','t4','transforms'],
            ['after','kernel_after'], ['t5','partial_values','api_transforms'])
    s['sections'][-1].update(id='view_transformation',
        question='通用变换得到程序与元数据；Pallas 专用规则可复用或改写 kernel，再绑定调用。')
    section(s, '消费 · 谁读取程序，得到什么？', 'JAX 解释、校验、打印与 lowering',
            ['consume_input'], ['c0','c1','c2'], ['r0','r1','r2'],
            ['c3','lower_rules','r3'], ['outer_lowering','c4','mosaic_module'], ['r4'])
    s['sections'][-1].update(id='view_consumption',
        question='解释、校验、打印与 lowering 并列；Pallas 只在相关方程和平台分支中展开。')
    section(s, '消费结果的后续去向', 'MLIR Module → executable → 数组结果',
            ['payload','outer_module','compile_cache','backend_compile','compile'],
            ['stages','loaded_exec','input_arrays','execute','result_arrays'], external=True)
    s['sections'][-1].update(id='downstream', auxiliary=True,
        question='Pallas payload 接回完整外层 Module；只有完整模块交给编译。可执行对象与实际数组在执行入口汇合。')

    edge(s,'core','eqn','eqns · 持有方程','neutral')
    edge(s,'eqn','var','invars / outvars · 引用值','neutral')
    edge(s,'eqn','primitive','primitive / params','neutral')
    chain(s,['p0','p1','p2','produced'],['函数 / avals','追踪方程 / frame','Jaxpr(...) + constvals'])
    chain(s,['caller','kernel','inner_jaxpr','pallas_bind','p1'],
          ['kernel 与输入约定','追踪 + DCE','内层程序与常量','记录外层调用时'])
    s['edges'][-1]['optional']=True
    edge(s,'kernel','mapping','构造调用映射')
    edge(s,'mapping','pallas_bind','独立的映射参数')
    edge(s,'produced','outer_jaxpr','含 pallas_call 的程序实例','neutral')
    edge(s,'outer_jaxpr','inner_jaxpr','params.jaxpr · 内层程序','neutral')
    edge(s,'outer_jaxpr','mapping','params.grid_mapping','neutral')

    edge(s,'produced','consume_input','无需进一步变换 · 直接消费',optional=True)
    s['edges'][-1]['channel']='direct'
    edge(s,'outer_jaxpr','consume_input','调用者程序与常量',optional=True)
    edge(s,'produced','transform_input','调用者 Jaxpr + 变换参数','transform',optional=True)
    for key in ('t0','t1','t2','t3','t4'):
        edge(s,'transform_input',key,'按需选择算法','transform',optional=True)
        edge(s,key,'after','新 Jaxpr + 附加信息','transform')
    edge(s,'after','consume_input','选择待消费程序与常量')
    edge(s,'inner_jaxpr','transforms','kernel 与专用变换输入','transform',optional=True)
    edge(s,'mapping','transforms','对应的调用映射','transform',optional=True)
    chain(s,['transforms','kernel_after','pallas_bind'],
          ['组织程序与调用参数','按更新后的约定重新绑定'],'transform')
    for i in range(4):
        edge(s,'consume_input',f'c{i}',('解释程序','检查程序','打印程序','程序与平台上下文')[i],
             'program' if i in (0,3) else 'neutral',optional=True)
    for i in range(3):
        edge(s,f'c{i}',f'r{i}',('解释所得值','校验通过 / 类型错误','格式化文本')[i],
             'output' if i==0 else 'neutral')
    chain(s,['c3','lower_rules','r3'],['程序与 lowering 上下文','普通规则的 IR 值汇入模块'])
    edge(s,'lower_rules','outer_lowering','遇到 pallas_call 方程',optional=True)
    chain(s,['outer_lowering','c4','mosaic_module','payload','outer_module','r3'],
          ['非解释模式 · TPU kernel 规则','构造内层 MLIR Module',
           'kernel Module 序列化','payload 接入 custom call','IR 值接回完整外层 Module'])
    s['edges'][-5]['optional']=True
    s['edges'][-3]['kind']='transform'
    chain(s,['r3','compile_cache','backend_compile','compile','loaded_exec'],
          ['完整外层 Module + 编译选项','未命中 / 无缓存 · 按策略','常规后端调用','PyLoadedExecutable 返回'])
    s['edges'][-1]['kind']='result'
    edge(s,'compile_cache','loaded_exec','缓存命中 · 恢复 executable','result',optional=True)
    edge(s,'loaded_exec','execute','执行包装持有程序','neutral')
    edge(s,'input_arrays','execute','本次实参数组 / tokens','data')
    edge(s,'execute','result_arrays','输出处理与包装','output')

    # Checked graph paths make the cross-block call/object chain reviewable.
    for key in ('t0','t1','t2','t3','t4'):
        s['paths'].append(['produced','transform_input',key,'after','consume_input'])
    for key in ('c0','c1','c2','c3'):
        s['paths'].append(['produced','consume_input',key])
    s['paths'].extend([
        ['caller','kernel','inner_jaxpr','pallas_bind','p1','p2','produced','outer_jaxpr','consume_input'],
        ['inner_jaxpr','transforms','kernel_after','pallas_bind','p1'],
        ['consume_input','c3','lower_rules','outer_lowering','c4','mosaic_module','payload','outer_module','r3','compile_cache'],
        ['r3','compile_cache','backend_compile','compile','loaded_exec','execute','result_arrays'],
        ['input_arrays','execute','result_arrays']])

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
        'jax':dict(c0='data',r0='output',c1='neutral',r1='neutral',c2='neutral',r2='neutral',r3='program',r4='program'),
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
