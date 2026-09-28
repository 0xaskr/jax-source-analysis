#!/usr/bin/env python3
"""Render the pinned JAX v0.11.1 architecture and downstream API map."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/software-stack/jax/jax-internal-stack.svg"
W, H = 3600, 6280
svg: list[str] = []


def add(value: str) -> None:
    svg.append(value)


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str,
         radius: int = 18, dashed: bool = False) -> None:
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')


def label(x: int, y: int, value: str, kind: str = "body") -> None:
    add(f'<text x="{x}" y="{y}" class="{kind}">{escape(value)}</text>')


def link(x: int, y: int, title: str, href: str) -> None:
    add(f'<a href="{escape(href, quote=True)}">'
        f'<text x="{x}" y="{y}" class="source">{escape(title)}</text></a>')


def arrow(path: str, *, green: bool = False, dashed: bool = False) -> None:
    css = "green-arrow" if green else "arrow"
    dash = ' stroke-dasharray="11 7"' if dashed else ""
    add(f'<path d="{path}" class="{css}"{dash}/>')


def section(x: int, y: int, w: int, h: int, number: str, title: str,
            fill: str, stroke: str) -> None:
    rect(x, y, w, h, fill, stroke, 22)
    label(x + 30, y + 54, f"{number}  {title}", "section")


def card(x: int, y: int, w: int, h: int, title: str,
         rows: list[tuple[str, str]], *, fill: str = "#fff",
         stroke: str = "#a7bfd8", source: tuple[str, str] | None = None,
         line_step: int = 39) -> None:
    rect(x, y, w, h, fill, stroke, 15)
    label(x + 28, y + 50, title, "head")
    for i, (line, style) in enumerate(rows):
        label(x + 28, y + 94 + i * line_step, line, style)
    if source:
        link(x + 28, y + h - 25, source[0], source[1])


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
add('<title id="title">JAX v0.11.1 内部软件栈与 jaxlib、XLA 下游接口</title>')
add('<desc id="desc">整合 JAX 概念、DynamicJaxprTrace 到 Jaxpr 的六步流程，以及普通 matmul 和 Pallas TPU lowering。Jaxpr 区域给出二维矩阵乘法与 Pallas 外层调用/内层 Ref kernel 的精简结构示例。以三处跨界调用明确 jax 与 jaxlib 分界：MLIR Python 绑定、compile_and_load、execute_sharded。下游展示 jaxlib PyClient 与 PyLoadedExecutable、IFRT、PJRT 和 CPU/GPU/TPU provider 的 API、输入与输出。StableHLO、Shardy 是 IR 和编译组件而非独立服务；Mosaic TPU MLIR 不是 LLO。</desc>')
add('<defs><marker id="flow-head" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0 0L11 4.5L0 9Z" fill="#4b6d94"/></marker><marker id="green-head" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0 0L11 4.5L0 9Z" fill="#237c72"/></marker></defs>')
add('<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#20344d;font-size:25px}.title{font-size:47px;font-weight:700}.subtitle{font-size:22px;fill:#52677e}.section{font-size:34px;font-weight:700}.head{font-size:30px;font-weight:700}.body{font-size:24px}.small{font-size:21px;fill:#52677e}.mono{font-family:"DejaVu Sans Mono","Noto Sans CJK SC",monospace;font-size:22px}.source{font-size:21px;fill:#236b91;text-decoration:underline}.boundary{font-size:31px;font-weight:700;fill:#84552e}.arrow{fill:none;stroke:#4b6d94;stroke-width:3;marker-end:url(#flow-head)}.green-arrow{fill:none;stroke:#237c72;stroke-width:3;marker-end:url(#green-head)}</style>')
rect(0, 0, W, H, "#fff", "#fff", 0)
label(75, 78, "JAX 内部软件栈：从 Python 函数到下游执行", "title")
label(75, 123, "jax-v0.11.1 基线 2d66622450e2 · 本地源码注释提交 3d1d3e8963b4 · 蓝色链接直达当前检出源码", "subtitle")

section(70, 165, 3460, 475, "①", "用户侧输入与变换（jax Python 包）", "#f4f8fd", "#97b5d4")
card(100, 255, 815, 300, "jax.Array · PyTree", [
    ("输入：数组值与嵌套 Python 结构", "body"),
    ("数组：shape / dtype / sharding", "body"),
    ("PyTree：flatten → leaves + treedef", "mono"),
    ("输出：供追踪使用的扁平参数", "body"),
], source=("ArrayImpl · tree_flatten", "../../../upstream/jax/jax/_src/array.py#L185"))
card(960, 255, 815, 300, "Mesh · Sharding", [
    ("输入：设备 mesh、分片规格", "body"),
    ("作用：给逻辑数组和编译提供布局", "body"),
    ("不是 Primitive，也不是另一种 IR", "body"),
], source=("Mesh / NamedSharding", "../../../upstream/jax/jax/_src/named_sharding.py#L78"))
card(1820, 255, 815, 300, "jit · grad · vmap", [
    ("输入：Python 函数、静态/动态参数", "body"),
    ("作用：暂存编译、微分、批量化", "body"),
    ("输出：变换后的可调用对象/程序", "body"),
    ("规则由相应解释器处理", "body"),
], source=("api.jit / grad / vmap", "../../../upstream/jax/jax/_src/api.py#L208"))
card(2680, 255, 820, 300, "计算入口", [
    ("普通：jnp.matmul → lax.dot_general", "mono"),
    ("Pallas：kernel + Ref / GridSpec", "body"),
    ("Python API 可展开为多个 primitive", "body"),
], source=("matmul 的调用点", "../../../upstream/jax/jax/_src/numpy/tensor_contractions.py#L273"))
arrow("M1800 640V700")

section(70, 710, 3460, 1940, "②", "tracing：Python 运算如何变成 Jaxpr", "#f5f2fb", "#aa9bc7")
steps = [
    (805, "1 · 创建 Trace、frame 与输入 Tracer", [
        ("API：pe.trace_to_jaxpr / DynamicJaxprTrace.new_arg", "mono"),
        ("输入：函数 + 输入 avals（shape / dtype 等）", "body"),
        ("输出：JaxprStackFrame、invars、输入 DynamicJaxprTracer", "body"),
        ("new_arg：newvar(aval) → Tracer(trace, aval, var)", "mono"),
    ], ("partial_eval.py:1668 / 2048", "../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1668")),
    (1105, "2 · 设置当前 Trace，运行 Python 函数体", [
        ("API：core.set_current_trace(trace); fun(*tracers)", "mono"),
        ("输入：由 Tracer 组成的参数 PyTree", "body"),
        ("输出：函数返回的 Tracer / PyTree", "body"),
        ("普通 Python 语句也会运行；动态 bool 需显式控制流", "body"),
    ], ("partial_eval.py:2084", "../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L2084")),
    (1405, "3 · Primitive.bind 把操作交给 Trace", [
        ("API：primitive.bind(*args, **params)", "mono"),
        ("输入：Tracer / 常量、primitive 参数", "body"),
        ("分派：trace.process_primitive(primitive, args, params)", "mono"),
        ("例：lax.dot → dot_general_p.bind", "mono"),
    ], ("core.py:683 / 732", "../../../upstream/jax/jax/_src/core.py#L683")),
    (1705, "4 · 抽象求值：不用矩阵元素推导结果", [
        ("API：primitive.abstract_eval(*input_avals, **params)", "mono"),
        ("输入：avals + params；输出：out_avals + effects", "body"),
        ("例：float32[2,3] @ float32[3,4] → float32[2,4]", "mono"),
        ("这一步与把 primitive lower 到目标 IR 分开", "body"),
    ], ("partial_eval.py:1751", "../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1751")),
    (2005, "5 · 创建方程与结果 Tracer，继续 Python 运算", [
        ("API：make_eqn → frame.add_eqn", "mono"),
        ("输入：旧 Tracer、out_avals、primitive、effects", "body"),
        ("输出：TracingEqn、新 Var 与新 Tracer", "body"),
        ("例如后续 add 再重复步骤 3 → 5", "body"),
    ], ("partial_eval.py:1674", "../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1674")),
    (2305, "6 · Python 返回后封装 Jaxpr", [
        ("API：frame.to_jaxpr(trace, out_tracers, ...)", "mono"),
        ("输入：TracingEqn、输入/输出 Tracer、常量", "body"),
        ("输出：Jaxpr(invars, outvars, eqns, effects, consts)", "body"),
        ("方程中的值改用 Var / Literal；不保留本次 Tracer", "body"),
    ], ("partial_eval.py:1517", "../../../upstream/jax/jax/_src/interpreters/partial_eval.py#L1517")),
]
for y, title, rows, source in steps:
    card(100, y, 2240, 265, title, rows, fill="#fff", stroke="#aaa0cc",
         source=source, line_step=36)
for y in (1070, 1370, 1670, 1970, 2270):
    arrow(f"M1220 {y}V{y+35}")
card(2400, 805, 1100, 400, "对象关系：Trace / Tracer / aval", [
    ("Trace：解释/记录一次调用", "body"),
    ("Tracer：Python 函数中流动的值代理", "body"),
    ("aval：抽象类型，如 ShapedArray", "body"),
    ("Tracer._trace → Trace", "mono"),
    ("Tracer.aval → shape / dtype 等", "mono"),
    ("Tracer.val → 当前 Jaxpr 的 Var / Literal", "mono"),
], stroke="#aaa0cc", source=("core.Trace / Tracer", "../../../upstream/jax/jax/_src/core.py#L848"))
card(2400, 1240, 1100, 400, "同一 primitive 的不同规则", [
    ("impl：即时求值（EvalTrace）", "body"),
    ("abstract_eval：输出 aval / effects", "body"),
    ("JVP / transpose：自动微分", "body"),
    ("batching：vmap 的批量化解释", "body"),
    ("MLIR lowering：目标 IR 生成规则", "body"),
    ("规则常注册在不同表中", "body"),
], stroke="#aaa0cc", source=("Primitive / register_lowering", "../../../upstream/jax/jax/_src/interpreters/mlir.py#L1003"))
card(2400, 1675, 1100, 400, "Python 与缓存边界", [
    ("shape / dtype 可由 aval 提供", "body"),
    ("动态 Tracer 不能供 Python if 求 bool", "body"),
    ("jit 命中缓存时通常复用已编译结果", "body"),
    ("静态参数、shape、配置变化可重新追踪", "body"),
    ("普通 print 等副作用发生于追踪时", "body"),
], stroke="#aaa0cc", source=("Tracer.__bool__", "../../../upstream/jax/jax/_src/core.py#L1092"))
card(2400, 2110, 1100, 420, "同一版本的其他 JAX 抽象", [
    ("PartialVal：已知常量 / 未知 aval", "body"),
    ("HiJAX → LoJAX：高层 primitive 可展开", "body"),
    ("pallas_call_p 属于 HiPrimitive", "body"),
    ("这与 StableHLO / Mosaic 目标选择不同", "body"),
    ("嵌套 Trace 支持变换组合", "body"),
], stroke="#aaa0cc", source=("HiPrimitive", "../../../upstream/jax/jax/_src/hijax.py#L75"))
arrow("M1800 2650V2710")

section(70, 2720, 3460, 750, "③", "Jaxpr：同一套 IR 结构，普通计算与 Pallas 参数不同", "#eef8f5", "#82b8a9")
card(100, 2810, 1030, 575, "普通计算 · matmul Jaxpr", [
    ("二维 f32 matmul · 结构节选", "small"),
    ("{ lambda ; a:f32[2,3] b:f32[3,4]. let", "mono"),
    ("    c:f32[2,4] = dot_general[", "mono"),
    ("      dimension_numbers=", "mono"),
    ("        (([1], [0]), ([], []))", "mono"),
    ("      ...  # 其余静态参数省略", "mono"),
    ("    ] a b", "mono"),
    ("  in (c,) }", "mono"),
    ("输出是值；无内层 kernel Jaxpr", "small"),
], stroke="#8bb8a9", source=("matmul → dot_general", "../../../upstream/jax/jax/_src/numpy/tensor_contractions.py#L273"), line_step=38)
card(1210, 2810, 1180, 575, "共同结构 · core.Jaxpr / JaxprEqn", [
    ("Jaxpr：invars / outvars / consts / eqns", "mono"),
    ("Var：值标识及 aval；Literal：已知值", "body"),
    ("JaxprEqn：primitive + invars/outvars", "body"),
    ("             + params + effects + 源码信息", "mono"),
    ("方程参数可以携带另一段 Jaxpr", "body"),
    ("Ref 写回有 WriteEffect，不能按纯值丢弃", "body"),
], stroke="#8bb8a9", source=("core.Jaxpr / JaxprEqn", "../../../upstream/jax/jax/_src/core.py#L105"))
card(2470, 2810, 1030, 575, "Pallas · 外层 / kernel Jaxpr", [
    ("外层 Jaxpr · 结构节选", "small"),
    ("{ lambda ; a:f32[8,128] b:f32[8,128]. let", "mono"),
    ("  c:f32[8,128] = pallas_call[", "mono"),
    ("    jaxpr=kernel grid_mapping=...", "mono"),
    ("  ] a b", "mono"),
    ("  in (c,) }", "mono"),
    ("内层 kernel Jaxpr：", "small"),
    ("{ lambda ; x:Ref{f32[8,128]} y:Ref{f32[8,128]}", "mono"),
    ("           o:Ref{f32[8,128]}. let", "mono"),
    ("  u:f32[8,128] <- x[]", "mono"),
    ("  v:f32[8,128] <- y[]", "mono"),
    ("  w:f32[8,128] = add u v", "mono"),
    ("  o:Ref{f32[8,128]}[] <- w", "mono"),
    ("  in () }", "mono"),
], stroke="#8bb8a9", source=("pallas_call bind / Ref 方程", "../../../upstream/jax/jax/_src/pallas/pallas_call.py#L1112"), line_step=32)
arrow("M1800 3470V3530")

section(70, 3540, 3460, 1100, "④", "JAX Python lowering：普通与 Pallas 两条路径", "#fff8ed", "#c9a977")
card(100, 3635, 1640, 730, "普通 jnp.matmul → StableHLO", [
    ("API：mlir.lower_jaxpr_to_module(...)", "mono"),
    ("输入：Jaxpr / avals / platforms / sharding", "body"),
    ("       layout / donation / lowering 参数", "body"),
    ("规则：dot_general_p → hlo.dot_general", "mono"),
    ("借用 jaxlib.mlir.ir / stablehlo / sdy 绑定", "body"),
    ("输出：LoweringResult.module (ir.Module)", "mono"),
    ("内容：外层 stablehlo.dot_general", "body"),
    ("sdy 分片属性取决于 Shardy 配置", "body"),
], stroke="#d3b078", source=("lower_jaxpr_to_module", "../../../upstream/jax/jax/_src/interpreters/mlir.py#L1327"))
card(1860, 3635, 1640, 730, "Pallas TPU kernel → Mosaic payload", [
    ("API：_pallas_call_lowering(ctx, interpret, ...)", "mono"),
    ("interpret=True：转解释实现", "body"),
    ("CPU 非解释模式不支持；GPU 有独立后端", "body"),
    ("TPU + False：lower_jaxpr_to_pipelined_module", "mono"),
    ("输入：kernel Jaxpr / GridMapping / 参数", "body"),
    ("输出：内层 Mosaic TPU MLIR ir.Module", "body"),
    ("API：tpu_custom_call.lower_module_to_custom_call", "mono"),
    ("输出：外层 stablehlo.custom_call；配置", "body"),
    ("      的 body 携带 bytecode / Base64", "body"),
], stroke="#88b9a9", source=("TPU kernel lowering", "../../../upstream/jax/jax/_src/pallas/mosaic/pallas_call_registration.py#L435"))
card(100, 4390, 1030, 195, "① JAX 的 MLIR 绑定调用", [
    ("mlir.py 导入 jax._src.lib.mlir", "mono"),
    ("借用 jaxlib.mlir 构造 ir.Module", "body"),
], stroke="#d3b078", source=("mlir.py:54", "../../../upstream/jax/jax/_src/interpreters/mlir.py#L54"))
card(1210, 4390, 1180, 195, "② JAX 编译提交", [
    ("backend ← xla_bridge.get_backend(platform)", "mono"),
    ("backend_compile_and_load(module, ...) → executable", "mono"),
], stroke="#d3b078", source=("compiler.py:330", "../../../upstream/jax/jax/_src/compiler.py#L330"))
card(2470, 4390, 1030, 195, "③ JAX 运行与结果包装", [
    ("xla_executable.execute_sharded(input_bufs)", "mono"),
    ("out_handler → 用户可见的数组结果", "body"),
], stroke="#d3b078", source=("pxla.py:413", "../../../upstream/jax/jax/_src/interpreters/pxla.py#L413"))

# The boundary is ownership, not a single temporal point: MLIR bindings are
# already called while JAX executes the preceding lowering section.
add('<path d="M70 4700H3530" fill="none" stroke="#a47345" stroke-width="4" stroke-dasharray="14 9"/>')
label(85, 4675, "JAX Python（jax/_src）", "boundary")
label(2670, 4675, "jaxlib Python / native 绑定", "boundary")
label(85, 4740, "三处跨界：① lowering 期间使用 MLIR 绑定　　② 提交 compile_and_load　　③ execute_sharded / PyArray 包装", "small")
arrow("M625 4585V4795", dashed=True)
arrow("M1800 4585V4795", dashed=True)
arrow("M2950 4585V4795", dashed=True)

section(70, 4805, 3460, 570, "⑤", "jaxlib：MLIR 绑定、Python/native 客户端与数组包装", "#eef5fc", "#8cadd0")
card(100, 4900, 1030, 390, "① jaxlib.mlir.ir / dialects", [
    ("JAX 进口：ir、hlo(stablehlo)、sdy", "mono"),
    ("另有 Mosaic TPU 等方言绑定", "body"),
    ("输入：aval、MLIR operands、属性", "body"),
    ("API：ir.Module / hlo.dot_general", "mono"),
    ("      sdy sharding attrs / MLIR passes", "body"),
    ("输出：JAX 持有的 MLIR ir.Module", "body"),
    ("StableHLO / Shardy 是方言与编译组件", "body"),
], stroke="#9bb6d5", source=("jax._src.lib.mlir.dialects", "../../../upstream/jax/jax/_src/lib/mlir/dialects/__init__.py#L62"))
card(1210, 4900, 1180, 390, "② PyClient.compile_and_load", [
    ("JAX API：backend.compile_and_load(...)", "mono"),
    ("输入：mlir::ModuleOp / devices", "body"),
    ("      xla::CompileOptions / callbacks", "body"),
    ("jaxlib：PyClient 持有 ifrt::Client", "body"),
    ("包装为 IFRT HloProgram(MLIR)", "body"),
    ("输出：PyLoadedExecutable", "body"),
], stroke="#9bb6d5", source=("jaxlib/py_client.cc:475", "../../../upstream/jax/jaxlib/py_client.cc#L475"))
card(2470, 4900, 1030, 390, "③ PyLoadedExecutable / PyArray", [
    ("JAX API：execute_sharded(input_bufs)", "mono"),
    ("输入：PyArray[]（含 IFRT arrays）", "body"),
    ("jaxlib：转 ifrt::ArrayRef[] 后 Execute", "body"),
    ("输出：PyExecuteResults / PyArray", "body"),
    ("可含 token、状态 / ready future", "body"),
    ("JAX out_handler 包装用户结果", "body"),
], stroke="#9bb6d5", source=("jaxlib/py_executable.cc:424", "../../../upstream/jax/jaxlib/py_executable.cc#L424"))
arrow("M1800 5375V5435")

section(70, 5450, 3460, 600, "⑥", "XLA 下游：IFRT → PJRT → 后端 provider", "#f2f7f4", "#90b7a0")
card(100, 5550, 1030, 400, "IFRT：数组、程序、可执行对象", [
    ("API：GetDefaultCompiler()", "mono"),
    ("     → CompileAndLoad(program, options)", "mono"),
    ("输入：HloProgram(MLIR module)", "body"),
    ("      XlaCompileOptions / devices", "body"),
    ("输出：ifrt::LoadedExecutable", "body"),
    ("Execute(ArrayRef[]) → ArrayRef[]", "mono"),
], stroke="#9abd9e", source=("PyClient → IFRT compiler", "../../../upstream/jax/jaxlib/py_client.cc#L412"))
card(1210, 5550, 1180, 400, "PJRT：后端客户端与设备", [
    ("兼容实现：PjRtCompiler::CompileAndLoad", "mono"),
    ("输入：MLIR module + CompileOptions", "body"),
    ("调用：pjrt_client()->CompileAndLoad", "mono"),
    ("输出：PjRtLoadedExecutable", "body"),
    ("Execute：IFRT Array → PjRtBuffer", "body"),
    ("返回：设备数据与完成状态", "body"),
], stroke="#9abd9e", source=("pjrt_executable.cc:763", "../../../upstream/xla/xla/python/pjrt_ifrt/pjrt_executable.cc#L763"))
card(2470, 5550, 1030, 400, "CPU / GPU / TPU provider", [
    ("后端实现编译、加载、执行", "body"),
    ("输入：MLIR / 分片 / 编译选项", "body"),
    ("StableHLO / Shardy 参与后端处理", "body"),
    ("CPU/GPU：可用 LLVM、emitter、库调用", "body"),
    ("TPU 经 PJRT C API plugin / libtpu", "body"),
    ("输出：可执行对象、设备数组/状态", "body"),
    ("LLO/机器码需目标版本证据", "body"),
], stroke="#9abd9e", source=("TPU 插件接入", "../../../upstream/jax/jax/_src/xla_bridge.py#L205"))
arrow("M1130 5750H1210", green=True)
arrow("M2390 5750H2470", green=True)
rect(70, 6110, 3460, 115, "#fff7ec", "#caa06f", 16)
label(100, 6156, "证据边界：此图依据锁定源码绘制；源码检查、CPU 执行、TPU 模拟、离线 TPU 编译和真机执行须分别取证。", "body")
label(100, 6195, "Mosaic TPU MLIR 不是 LLO；单独的 Jaxpr / StableHLO / Mosaic IR 也不能证明后端编译或设备执行。", "small")
add('</svg>')

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
print(OUTPUT)
