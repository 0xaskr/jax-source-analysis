#!/usr/bin/env python3
"""Render a source-anchored Jaxpr-to-MLIR walkthrough for pinned JAX v0.11.1."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import subprocess
from unicodedata import east_asian_width
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/software-stack/jax/jaxpr-to-mlir.svg"
WIDTH, HEIGHT = 3600, 6650
EXPECTED_JAX_HEAD = "3d1d3e8963b4bc5074e8e7492424278ff64a0050"


@dataclass(frozen=True)
class Source:
    label: str
    path: str
    line: int

    @property
    def href(self) -> str:
        return f"../../../upstream/jax/{self.path}#L{self.line}"


svg: list[str] = []
source_refs: set[Source] = set()
text_bounds: list[tuple[str, int, int, int, int, str]] = []


def add(markup: str) -> None:
    svg.append(markup)


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str,
         radius: int = 18, stroke_width: int = 2) -> None:
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def estimated_width(value: str, size: int, mono: bool) -> int:
    units = sum(1.0 if east_asian_width(ch) in "WF" else
                (0.63 if mono else 0.58) for ch in value)
    return round(size * units)


def text(x: int, y: int, value: str, css: str = "body", *,
         right: int | None = None, size: int | None = None) -> None:
    add(f'<text x="{x}" y="{y}" class="{css}">{escape(value)}</text>')
    if right is not None:
        font_size = size or {
            "card-title": 27, "body": 23, "small": 20, "code": 21,
            "source": 18, "tiny": 18,
        }[css]
        text_bounds.append((value, x, y, right, font_size, css))


def source_link(x: int, y: int, source: Source, *, right: int) -> None:
    source_refs.add(source)
    add(f'<a href="{escape(source.href, quote=True)}">')
    add(f'<title>{escape(source.path)}:{source.line}</title>')
    text(x, y, f"↗ {source.label}", "source", right=right)
    add("</a>")


def arrow(path: str, kind: str = "main") -> None:
    add(f'<path d="{path}" class="arrow-{kind}"/>')


def section(x: int, y: int, w: int, h: int, number: str,
            title: str, tone: str) -> None:
    palette = {
        "blue": ("#edf5fb", "#9cbcd7"),
        "purple": ("#f4f0fa", "#b4a2d1"),
        "teal": ("#edf8f6", "#94c5bd"),
        "amber": ("#fff8ed", "#ddbd83"),
    }
    fill, stroke = palette[tone]
    rect(x, y, w, h, fill, stroke, 24)
    text(x + 28, y + 51, f"{number}  {title}", "section-title")


def card(x: int, y: int, w: int, h: int, title: str,
         rows: list[tuple[str, str]], *, tone: str = "blue",
         sources: tuple[Source, ...] = (),
         body_start: int = 83, body_step: int = 31) -> None:
    palette = {
        "blue": ("#ffffff", "#a4bfd6", "#2f668e"),
        "purple": ("#ffffff", "#b4a5d1", "#704b9c"),
        "teal": ("#ffffff", "#9bcbbf", "#167b70"),
        "amber": ("#fffdf9", "#dfc396", "#a96c22"),
    }
    fill, stroke, accent = palette[tone]
    rect(x, y, w, h, fill, stroke, 16)
    rect(x + 1, y + 1, 9, h - 2, accent, accent, 3, 0)
    text(x + 28, y + 47, title, "card-title", right=x + w - 22)
    for i, (value, css) in enumerate(rows):
        text(x + 28, y + body_start + i * body_step, value, css,
             right=x + w - 24)
    if sources:
        sy = y + h - 21
        sx = x + 28
        for source in sources:
            source_link(sx, sy, source, right=x + w - 24)
            sx += estimated_width(f"↗ {source.label}", 18, False) + 42


def code_card(x: int, y: int, w: int, h: int, title: str,
              rows: list[str], *, tone: str = "blue",
              sources: tuple[Source, ...] = ()) -> None:
    card(x, y, w, h, title, [(row, "code") for row in rows],
         tone=tone, sources=sources, body_start=93, body_step=39)


def validate_sources() -> None:
    actual_head = subprocess.run(
        ["git", "-C", str(ROOT / "upstream/jax"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    lock_entry = next(
        line for line in (ROOT / "upstream-sources.lock").read_text().splitlines()
        if line.startswith("core|upstream/jax|")
    )
    locked_head = lock_entry.split("|")[3]
    if actual_head != EXPECTED_JAX_HEAD or locked_head != EXPECTED_JAX_HEAD:
        raise ValueError(
            f"JAX source pin changed: checkout={actual_head}, lock={locked_head}"
        )
    for source in sorted(source_refs, key=lambda s: (s.path, s.line)):
        path = ROOT / "upstream/jax" / source.path
        if not path.is_file():
            raise FileNotFoundError(path)
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        if not 1 <= source.line <= line_count:
            raise ValueError(f"Invalid source anchor: {source.path}:{source.line}")


def validate_text() -> None:
    errors = []
    for value, x, y, right, size, css in text_bounds:
        width = estimated_width(value, size, css == "code")
        if x + width > right:
            errors.append(f"overflow at y={y}: {value} ({x + width} > {right})")
    if errors:
        raise ValueError("\n".join(errors))


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
    f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
    'role="img" aria-labelledby="title desc">'
)
add('<title id="title">以 Jaxpr 为轴：JAX 模型到 StableHLO MLIR</title>')
add(
    '<desc id="desc">固定 JAX v0.11.1 基线的源码走读：从 Python 模型、'
    'jit.trace、DynamicJaxprTrace 和 Jaxpr 对象，到 sharding、effects、'
    'HiJAX 到 LoJAX、pxla、MLIR lowering 规则和 StableHLO module。'
    '模型为 f32[2,3] 与 f32[3,4] 的矩阵乘法、偏置相加及 tanh；'
    '附本地 CPU host lowering 的实际 Jaxpr 与 StableHLO 摘录。'
    'MLIR 不是已编译可执行文件，未执行 TPU。'
    '</desc>'
)
add(
    '<defs>'
    '<marker id="head-blue" markerWidth="12" markerHeight="10" '
    'refX="11" refY="5" orient="auto"><path d="M0 0L12 5L0 10Z" '
    'fill="#4976a0"/></marker>'
    '<marker id="head-purple" markerWidth="12" markerHeight="10" '
    'refX="11" refY="5" orient="auto"><path d="M0 0L12 5L0 10Z" '
    'fill="#805ba8"/></marker>'
    '<marker id="head-teal" markerWidth="12" markerHeight="10" '
    'refX="11" refY="5" orient="auto"><path d="M0 0L12 5L0 10Z" '
    'fill="#238c80"/></marker>'
    '</defs>'
)
add(
    '<style>'
    'text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;'
    'fill:#20364d;font-size:23px}'
    '.title{font-size:51px;font-weight:700;fill:#15334f}'
    '.subtitle{font-size:21px;fill:#536a80}'
    '.section-title{font-size:32px;font-weight:700;fill:#203c59}'
    '.card-title{font-size:27px;font-weight:700;fill:#173b5c}'
    '.body{font-size:23px}.small{font-size:20px;fill:#4a6279}'
    '.tiny{font-size:18px;fill:#5a7185}'
    '.code{font-family:"DejaVu Sans Mono","Noto Sans CJK SC",monospace;'
    'font-size:21px;fill:#263e53}'
    '.source{font-size:18px;fill:#186b9a;text-decoration:underline}'
    '.badge{font-size:20px;font-weight:700;fill:#ffffff}'
    '.arrow-main{fill:none;stroke:#4976a0;stroke-width:4;'
    'marker-end:url(#head-blue)}'
    '.arrow-branch{fill:none;stroke:#805ba8;stroke-width:3;'
    'stroke-dasharray:12 8;marker-end:url(#head-purple)}'
    '.arrow-lower{fill:none;stroke:#238c80;stroke-width:4;'
    'marker-end:url(#head-teal)}'
    '</style>'
)
rect(0, 0, WIDTH, HEIGHT, "#f9fbfd", "#f9fbfd", 0, 0)
text(75, 78, "以 Jaxpr 为轴：JAX 模型如何变成 StableHLO MLIR", "title")
text(
    75, 119,
    "JAX v0.11.1 基线 · 本地源码注释提交 3d1d3e8963b4 · "
    "固定模型的 Python tracing 与 CPU host lowering 观测",
    "subtitle",
)
rect(75, 141, 236, 36, "#4976a0", "#4976a0", 18, 0)
text(98, 166, "蓝色：主调用链", "badge")
rect(328, 141, 268, 36, "#805ba8", "#805ba8", 18, 0)
text(350, 166, "紫色：Jaxpr 分支", "badge")
rect(612, 141, 250, 36, "#238c80", "#238c80", 18, 0)
text(635, 166, "绿色：MLIR 生成", "badge")
text(910, 165, "卡片底部蓝色文字可点击，跳转至本地锁定源码。", "subtitle")


# 00 — the concrete model and two actual, phase-separated observations.
section(70, 195, 3460, 750, "00", "固定模型与实际输出：四条 Jaxpr 方程，五条 StableHLO 操作", "blue")
code_card(100, 280, 800, 585, "Python 模型 / 输入规格", [
    "def model(x, w, b):",
    "    return jnp.tanh(x @ w + b)",
    "",
    "x : f32[2,3]",
    "w : f32[3,4]",
    "b : f32[4]",
    "",
    "traced = jax.jit(model).trace(...)",
    "lowered = traced.lower()",
], sources=(Source("stages.Traced", "jax/_src/stages.py", 412),))
code_card(945, 280, 1250, 585, "traced.jaxpr（真实摘录）", [
    "{ lambda ; a:f32[2,3] b:f32[3,4]",
    "           c:f32[4]. let",
    "  d:f32[2,4] = dot_general[...] a b",
    "  e:f32[1,4] = broadcast_in_dim c",
    "  f:f32[2,4] = add d e",
    "  g:f32[2,4] = tanh f",
    "  in (g,) }",
    "",
    "is_high=False · effects=∅",
], tone="purple",
    sources=(Source("core.Jaxpr", "jax/_src/core.py", 105),))
code_card(2240, 280, 1260, 585, "compiler_ir('stablehlo') 摘录", [
    "module @jit_model {",
    "  func.func public @main(...) {",
    "    %0 = stablehlo.dot_general ...",
    "    %1 = stablehlo.broadcast_in_dim ...",
    "    %2 = stablehlo.broadcast_in_dim %1",
    "    %3 = stablehlo.add %0, %2",
    "    %4 = stablehlo.tanh %3",
    "    return %4",
    "  }",
    "}",
], tone="teal",
    sources=(Source("Lowered.compiler_ir", "jax/_src/stages.py", 671),))
arrow("M903 565H940")
arrow("M2197 565H2235", "lower")
text(1590, 919, "上方是同一次固定形状的 CPU host lowering；未调用 compile() 或执行模型。", "small")


# 01 — creation of the Jaxpr.
section(70, 975, 3460, 1490, "01", "冷路径：Python 模型如何被追踪为 Jaxpr", "blue")
trace_steps = [
    (1060, "1  用户入口与 PyTree / 静态参数",
     [("jax.jit(model) 包装函数；trace(specs) 显式追踪", "body"),
      ("位置/关键字参数展平；静态参数留在追踪外", "body"),
      ("输入 aval 包含 shape、dtype、sharding 等", "body")],
     (Source("api.jit", "jax/_src/api.py", 208),
      Source("pjit.jit_trace", "jax/_src/pjit.py", 317))),
    (1275, "2  参数推断与 trace_to_jaxpr",
     [("jit_trace → _infer_params → _trace_for_jit", "code"),
      ("avals_ft + in_tree + DebugInfo + 编译参数", "code"),
      ("pe.trace_to_jaxpr(fun, avals_ft, dbg)", "code")],
     (Source("_trace_for_jit", "jax/_src/pjit.py", 491),)),
    (1490, "3  建立当前 Trace 和输入 Tracer",
     [("DynamicJaxprTrace(debug_info) + JaxprStackFrame", "code"),
      ("new_arg(aval) → Var(aval) + DynamicJaxprTracer", "code"),
      ("set_current_trace(trace)；fun(*tracers)", "code")],
     (Source("trace_to_jaxpr_nocache", "jax/_src/interpreters/partial_eval.py", 2048),)),
    (1705, "4  Python 运算进入 Primitive.bind",
     [("x @ w → dot_general_p；+b → broadcast + add", "body"),
      ("Primitive.bind → trace.process_primitive", "code"),
      ("Python 控制流仅可读取追踪期已知信息", "body")],
     (Source("Primitive.bind", "jax/_src/core.py", 690),)),
    (1920, "5  抽象求值和方程暂存",
     [("primitive.abstract_eval(avals, params) → avals + effects", "code"),
      ("make_eqn → TracingEqn + 输出 Var / Tracer", "code"),
      ("frame.add_eqn；常量折叠可省去某些方程", "body")],
     (Source("default_process_primitive", "jax/_src/interpreters/partial_eval.py", 1750),)),
    (2135, "6  返回时构造 Jaxpr",
     [("frame.get_eqns：TracingEqn → JaxprEqn", "code"),
      ("收集 constvars / invars / outvars / effects / is_high", "body"),
      ("Jaxpr(...) → stages.Traced.jaxpr", "code")],
     (Source("JaxprStackFrame.to_jaxpr", "jax/_src/interpreters/partial_eval.py", 1517),)),
]
for y, title, rows, sources in trace_steps:
    card(100, y, 2150, 195, title, rows, sources=sources)
for y in (1255, 1470, 1685, 1900, 2115):
    arrow(f"M1170 {y}V{y + 17}")

card(2310, 1060, 1190, 390, "变换改变追踪行为", [
    ("grad：JVP / VJP / transpose 规则决定导数计算", "body"),
    ("vmap：batching 规则决定批轴如何传播", "body"),
    ("jit：给编译阶段保留可暂存的程序", "body"),
    ("组合变换可能嵌套 Trace 或生成内层 Jaxpr", "body"),
    ("不要假设每个变换各产出一份独立 Jaxpr", "small"),
], tone="purple",
    sources=(Source("api.grad", "jax/_src/api.py", 393),
             Source("api.vmap", "jax/_src/api.py", 949)),
    body_step=47)
card(2310, 1490, 1190, 390, "缓存与特化的入口条件", [
    ("函数身份、静态参数、输入类型与配置参与缓存", "body"),
    ("命中可复用已有的追踪 / lowering / executable", "body"),
    ("本图主路径表示需要追踪与 lowering 的情况", "body"),
    ("静态 Python 分支由追踪时的已知条件决定", "body"),
    ("动态数组条件应由 lax.cond 等操作表达", "body"),
], tone="amber",
    sources=(Source("pjit._infer_params", "jax/_src/pjit.py", 635),),
    body_step=47)
card(2310, 1920, 1190, 410, "表示边界", [
    ("Tracer：追踪中流动的代理值", "body"),
    ("aval：抽象类型；Var：Jaxpr 中的值身份", "body"),
    ("JaxprEqn：primitive + 输入/输出 + params + effects", "body"),
    ("MLIR 尚未出现；此时得到的是 JAX Python IR", "body"),
    ("jit.trace 与 make_jaxpr 不意味着设备执行", "small"),
], tone="purple",
    sources=(Source("core.Tracer", "jax/_src/core.py", 981),
             Source("core.JaxprEqn", "jax/_src/core.py", 460)),
    body_step=48)
arrow("M1170 2465V2515")


# 02 — Jaxpr as the center of the figure.
section(70, 2525, 3460, 1120, "02", "Jaxpr 是交接契约：程序体、类型、效果与可选高层分支", "purple")
card(100, 2610, 2110, 295, "Jaxpr 对象：稳定的程序结构", [
    ("_all_invars / _consts：参数和附值的闭包常量", "body"),
    ("_outvars / _eqns：返回值与按顺序排列的方程", "body"),
    ("_effects：全程序效果；_debug_info：参数名与结果路径", "body"),
    ("_is_high：是否仍含高层 primitive 或抽象值", "body"),
], tone="purple",
    sources=(Source("core.Jaxpr", "jax/_src/core.py", 105),),
    body_step=43)
card(100, 2930, 2110, 295, "一条 JaxprEqn 的数据流", [
    ("invars: Var / Literal → primitive(params) → outvars: Var", "code"),
    ("每个 Var 带 aval；示例 d:f32[2,4] 的类型已确定", "body"),
    ("effects 记录外部可观察或状态相关的语义", "body"),
    ("源码位置和 eqn 上下文帮助诊断、lowering", "body"),
], tone="purple",
    sources=(Source("core.JaxprEqn", "jax/_src/core.py", 460),),
    body_step=43)
card(100, 3250, 2110, 295, "这个模型的 Jaxpr 状态", [
    ("4 条 eqn：dot_general → broadcast_in_dim → add → tanh", "code"),
    ("consts=[]，effects=∅，is_high=False", "code"),
    ("in_avals=(f32[2,3], f32[3,4], f32[4])", "code"),
    ("out_avals=(f32[2,4],)；还未转换为 MLIR", "code"),
], tone="purple",
    sources=(Source("Traced.jaxpr", "jax/_src/stages.py", 442),),
    body_step=43)
card(2260, 2610, 1240, 295, "HiJAX → LoJAX（条件分支）", [
    ("Traced.lojax 先检查 jaxpr.is_high", "body"),
    ("False：本例复用同一个 Jaxpr 对象", "body"),
    ("True：pe.lower_jaxpr 展开高层 eqn / HiType", "body"),
    ("LoJAX 仍是 Jaxpr，不是 MLIR", "body"),
], tone="purple",
    sources=(Source("Traced.lojax", "jax/_src/stages.py", 501),),
    body_step=43)
card(2260, 2930, 1240, 295, "effects 与 token 分支", [
    ("debug_print / io_callback / Ref 可给 eqn 附 effect", "body"),
    ("lowering 检查可降级 effect；有序效果传 token", "body"),
    ("本例无 effects，因此 IR 签名没有 token", "body"),
    ("纯 add / tanh 方程通常没有 effect", "body"),
], tone="purple",
    sources=(Source("make_jaxpr_effects", "jax/_src/interpreters/partial_eval.py", 1445),
             Source("mlir.TokenSet", "jax/_src/interpreters/mlir.py", 1611)),
    body_step=43)
card(2260, 3250, 1240, 295, "嵌套程序与特殊 primitive", [
    ("cond / scan / jit 可在参数中携带内层 Jaxpr", "body"),
    ("Pallas kernel 也是内层 Jaxpr；路径因平台而异", "body"),
    ("TPU Pallas 可产生 Mosaic TPU MLIR payload", "body"),
    ("这些路径不属于上方普通模型的实际输出", "small"),
], tone="amber",
    sources=(Source("pallas_call", "jax/_src/pallas/pallas_call.py", 72),
             Source("Mosaic TPU lowering",
                    "jax/_src/pallas/mosaic/pallas_call_registration.py", 393)),
    body_step=43)
arrow("M1170 3645V3695", "lower")


# 03 — the exact control path from Jaxpr to MLIR.
section(70, 3705, 3460, 1850, "03", "从 LoJAX Jaxpr 到 MLIR Module：控制链和逐方程 lowering", "teal")
lower_steps = [
    (3790, "1  Traced.lower()：进入 lowering 阶段",
     [("lo = traced.lojax；本例 lo.jaxpr is traced.jaxpr", "code"),
      ("调用 pjit._resolve_and_lower；返回 stages.Lowered", "body"),
      ("未调用 compile()，也未运行设备程序", "body")],
     (Source("Traced.lower", "jax/_src/stages.py", 538),)),
    (4005, "2  _resolve_and_lower：确定输入/输出约束",
     [("解析 in/out_shardings 与 in/out_layouts", "code"),
      ("携带 avals、donation、mesh、平台和编译选项", "body"),
      ("交给 _pjit_lower", "code")],
     (Source("pjit._resolve_and_lower", "jax/_src/pjit.py", 1062),)),
    (4220, "3  _pjit_lower → pxla.lower_sharding_computation",
     [("以 jit 路径建立 MeshComputation", "body"),
      ("传入 Jaxpr、分片、布局、donation、平台参数", "body"),
      ("此阶段仍处理 JAX 级程序和编译元数据", "body")],
     (Source("pjit._pjit_lower", "jax/_src/pjit.py", 1228),
      Source("pxla.lower_sharding_computation", "jax/_src/interpreters/pxla.py", 976))),
    (4435, "4  pxla 预处理 Jaxpr 与设备约束",
     [("DCE；Ref discharge；常量参数与输出布局处理", "body"),
      ("计算 device assignment、sharding 和平台", "body"),
      ("效果与 sharding 参与后续 MLIR 参数传递", "body")],
     (Source("pxla.lower_sharding_computation", "jax/_src/interpreters/pxla.py", 986),)),
    (4650, "5  _cached_lowering_to_hlo → MLIR 模块入口",
     [("构造 ShardingContext 与 ordered_effects", "code"),
      ("mlir.lower_jaxpr_to_module(jaxpr, avals, ...)", "code"),
      ("输出 LoweringResult.module: mlir.ir.Module", "body")],
     (Source("pxla._cached_lowering_to_hlo", "jax/_src/interpreters/pxla.py", 726),
      Source("mlir.lower_jaxpr_to_module", "jax/_src/interpreters/mlir.py", 1327))),
    (4865, "6  ModuleContext → func.func public @main",
     [("建 MLIR context/module；签名由 aval → tensor type", "body"),
      ("处理常量、动态维、分片、别名与 effect token", "body"),
      ("lower_jaxpr_to_fun 创建函数入口", "code")],
     (Source("mlir.ModuleContext", "jax/_src/interpreters/mlir.py", 804),
      Source("mlir.lower_jaxpr_to_fun", "jax/_src/interpreters/mlir.py", 1652))),
    (5080, "7  jaxpr_subcomp：逐条解释 JaxprEqn",
     [("Var → IR value；Literal → IR constant", "code"),
      ("按 primitive / 平台从 lowering 规则表选规则", "body"),
      ("LoweringRuleContext 传 avals、params、token", "body")],
     (Source("mlir.jaxpr_subcomp", "jax/_src/interpreters/mlir.py", 2129),
      Source("mlir.register_lowering", "jax/_src/interpreters/mlir.py", 1003))),
    (5295, "8  验证并暴露 StableHLO MLIR",
     [("各规则生成 ir.Value；func.func 返回结果", "body"),
      ("module.operation.verify()；MeshComputation 保存 module", "body"),
      ("Lowered.compiler_ir('stablehlo') 供检查", "code")],
     (Source("MLIR verify", "jax/_src/interpreters/mlir.py", 1455),
      Source("MeshComputation.stablehlo", "jax/_src/interpreters/pxla.py", 1202))),
]
for y, title, rows, sources in lower_steps:
    card(100, y, 2150, 195, title, rows, tone="teal", sources=sources)
for y in (3985, 4200, 4415, 4630, 4845, 5060, 5275):
    arrow(f"M1170 {y}V{y + 17}", "lower")

card(2310, 3790, 1190, 395, "模块级：ModuleContext", [
    ("backend / platforms / axis_context", "code"),
    ("MLIR ir.Context / ir.Module / SymbolTable", "code"),
    ("常量缓存、host callbacks、sharding attr 缓存", "body"),
    ("动态形状维变量与 channel id", "body"),
    ("只属于 lowering 阶段，不在 Jaxpr 中", "small"),
], tone="teal",
    sources=(Source("mlir.ModuleContext", "jax/_src/interpreters/mlir.py", 804),),
    body_step=48)
card(2310, 4210, 1190, 395, "方程级：LoweringRuleContext", [
    ("primitive / avals_in / avals_out / params", "code"),
    ("tokens_in/out、eqn 源码位置、name stack", "body"),
    ("平台专用规则优先于通用注册规则", "body"),
    ("lowering 缓存可复用相同 primitive 形态", "body"),
    ("一条方程可发出多条 MLIR operation", "small"),
], tone="teal",
    sources=(Source("LoweringRuleContext", "jax/_src/interpreters/mlir.py", 939),
             Source("rule selection", "jax/_src/interpreters/mlir.py", 2344)),
    body_step=48)
card(2310, 4630, 1190, 395, "jax 与 jaxlib 的构造边界", [
    ("JAX Python 决定何时调用哪条 lowering 规则", "body"),
    ("jaxlib.mlir.ir 提供 MLIR 对象绑定", "code"),
    ("jaxlib.mlir.dialects.stablehlo 提供 hlo.*", "code"),
    ("还可出现 func、sdy 等 MLIR 方言", "body"),
    ("它们不是独立网络服务", "small"),
], tone="teal",
    sources=(Source("JAX dialect imports", "jax/_src/lib/mlir/dialects/__init__.py", 62),),
    body_step=48)
card(2310, 5050, 1190, 440, "输出的语义边界", [
    ("ir.Module 已是编译器输入，不是 executable", "body"),
    ("StableHLO 是 MLIR 方言；普通路径的计算主体", "body"),
    ("sharding 可出现 sdy；特殊 primitive 可 custom_call", "body"),
    ("Pallas TPU payload 可为 Mosaic TPU MLIR", "body"),
    ("Mosaic TPU MLIR 不是 LLO；图中无 TPU 编译证明", "small"),
    ("CPU / GPU / TPU 后端进一步编译均在本图终点之后", "small"),
], tone="amber",
    sources=(Source("Lowered.compiler_ir", "jax/_src/stages.py", 671),),
    body_step=48)
arrow("M1170 5555V5625", "lower")


# 04 — equation-to-operation accounting and evidence boundary.
section(70, 5635, 3460, 925, "04", "逐方程核对：Jaxpr 操作如何变成这个 MLIR module", "amber")
card(100, 5720, 2180, 745, "固定模型的映射表（CPU host lowering 实际观测）", [
    ("Jaxpr ① dot_general a b    → stablehlo.dot_general %arg0, %arg1", "code"),
    ("Jaxpr ② broadcast_in_dim c → stablehlo.broadcast_in_dim %arg2", "code"),
    ("Jaxpr ③ add d e            → 新增 broadcast_in_dim %1", "code"),
    ("                               再发 stablehlo.add %0, %2", "code"),
    ("Jaxpr ④ tanh f             → stablehlo.tanh %3", "code"),
    ("Jaxpr 4 条方程；StableHLO 5 条计算操作。", "body"),
    ("额外广播来自 add 的逐元素 lowering 规则；并非逐行文本替换。", "body"),
    ("module 另有 func.func @main、return 和 mhlo.num_* 属性。", "body"),
    ("不同 shape、sharding、平台或版本可改变生成的 IR。", "small"),
], tone="amber",
    sources=(Source("dot_general lowering", "jax/_src/lax/lax.py", 6266),
             Source("add lowering", "jax/_src/lax/lax.py", 5053)),
    body_start=98, body_step=56)
card(2330, 5720, 1170, 745, "阅读图时的三个界限", [
    ("源码：锁定 JAX 提交 3d1d3e8963b4", "body"),
    ("运行：JAX 0.11.1.dev + jaxlib 0.11.1", "body"),
    ("VERSION-SKEW：jaxlib wheel 非本地注释提交构建", "small"),
    ("已验证：traced.jaxpr 与 host StableHLO", "body"),
    ("未验证：compile() / CPU 数值执行", "body"),
    ("未验证：TPU 离线编译 / 真机执行", "body"),
    ("MLIR 是表示与编译输入，不等于设备代码", "body"),
    ("图中示例没有 Pallas、FFI 或 ordered effect", "body"),
    ("其他路径请由对应 Jaxpr 和目标 IR 单独取证", "small"),
], tone="amber",
    sources=(Source("MLIR module return", "jax/_src/interpreters/mlir.py", 1482),),
    body_start=98, body_step=63)

text(75, 6610, "生成：tools/render_jaxpr_to_mlir.py  ·  链接与行号对应当前本地锁定源码  ·  SVG 可缩放、搜索文本并点击源码锚点", "subtitle")
add("</svg>")

validate_sources()
validate_text()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
ET.parse(OUTPUT)
print(f"Wrote {OUTPUT} ({WIDTH}x{HEIGHT}, {len(source_refs)} source anchors)")
