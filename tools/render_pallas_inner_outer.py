#!/usr/bin/env python3
"""Render the two Pallas program levels and their Mosaic TPU lowering boundary."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from render_overview_software_stack_components import Diagram, preview


ROOT = Path(__file__).resolve().parents[1]
JAX_PIN = "361c43e072cce92b7d3e9bdaf4dd16db26c49043"
PALLAS = "jax/_src/pallas/pallas_call.py"
CORE = "jax/_src/pallas/core.py"
MLIR = "jax/_src/interpreters/mlir.py"
TPU = "jax/_src/tpu_custom_call.py"
SOURCES = {
    "Jaxpr": ("jax/_src/core.py", "class Jaxpr:"),
    "JaxprEqn": ("jax/_src/core.py", "class JaxprEqn:"),
    "GridMapping": (CORE, "class GridMapping:"),
    "get_grid_mapping": (CORE, "def get_grid_mapping("),
    "pallas_call": (PALLAS, "def pallas_call("),
    "pallas_call_p.bind": (PALLAS, "      out_flat = pallas_call_p.bind("),
    "_trace_kernel_to_jaxpr": (PALLAS, "def _trace_kernel_to_jaxpr("),
    "_pallas_call_jvp_rule": (PALLAS, "def _pallas_call_jvp_rule("),
    "_pallas_call_batching_rule": (PALLAS, "def _pallas_call_batching_rule("),
    "_pallas_call_lowering": (PALLAS, "def _pallas_call_lowering("),
    "register Pallas lowering": (PALLAS, "mlir.register_lowering(pallas_call_p, _pallas_call_lowering)"),
    "jaxpr_subcomp": (MLIR, "def jaxpr_subcomp("),
    "lower_jaxpr_to_module": (MLIR, "def lower_jaxpr_to_module("),
    "pallas_call_tpu_lowering_rule": ("jax/_src/pallas/mosaic/pallas_call_registration.py", "def pallas_call_tpu_lowering_rule("),
    "lower_jaxpr_to_pipelined_module": ("jax/_src/pallas/mosaic/lowering.py", "def lower_jaxpr_to_pipelined_module("),
    "lower_module_to_custom_call": (TPU, "def lower_module_to_custom_call("),
    "_lower_mosaic_module_to_asm": (TPU, "def _lower_mosaic_module_to_asm("),
    "CustomCallBackendConfig.to_json": (TPU, "  def to_json(self) -> bytes:", 249),
    "emit tpu_custom_call": (TPU, "  call = mlir.custom_call("),
    "PyClient.CompileAndLoad": ("jaxlib/py_client.cc", "PyClient::CompileAndLoad(nb_class_ptr<PyClient> client, mlir::ModuleOp module,", 475),
}


def metadata(root):
    repo = root / "upstream/jax"
    if not (repo / ".git").exists():
        raise ValueError("The pinned JAX source checkout is required")
    entry = next(line.split("|") for line in (root / "upstream-sources.lock").read_text().splitlines()
                 if line.startswith("core|upstream/jax|"))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if entry[3] != JAX_PIN or head != JAX_PIN:
        raise ValueError("JAX revision changed; review the diagram before updating its source pin")
    files, anchors = {}, {}
    for name, (path, needle, *fixed) in SOURCES.items():
        if path not in files:
            committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{JAX_PIN}:{path}"], text=True)
            local = repo / path
            if local.exists() and local.read_text() != committed:
                raise ValueError("Source file modified: " + path)
            files[path] = committed.splitlines()
        if fixed:
            lines = [fixed[0]] if files[path][fixed[0] - 1].startswith(needle) else []
        else:
            lines = [i + 1 for i, line in enumerate(files[path]) if line.startswith(needle)]
        if len(lines) != 1:
            raise ValueError(f"Ambiguous source anchor {name}: {lines}")
        anchors[name] = {"repo": "jax", "path": path, "line": lines[0], "needle": needle,
                         "href": f"https://github.com/0xaskr/jax/blob/{JAX_PIN}/{path}#L{lines[0]}"}
    return {"source_pins": {"jax": JAX_PIN}, "source_anchors": anchors,
            "evidence_scope": "Pinned JAX source inspection only; no tracing, compilation, simulation, or device execution",
            "path": "Pallas Mosaic TPU lowering, interpret=False; platform dispatch may select other implementations"}


class PallasDiagram(Diagram):
    def __init__(self, meta):
        super().__init__(3200, 4020, "Pallas 的两层程序", "", meta)
        self.description = "左栏为调用者的外层 Jaxpr 与 MLIR Module，右栏为内层 kernel Jaxpr、GridMapping 与 Mosaic TPU 模块；展示参数携带、TPU lowering、版本化序列化和 custom_call 回接，并附定义、变换规则与固定源码链接。"
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.text_boxes, self.edge_boxes = [], []
        self.bg, self.card = "#fafbf8", "#ffffff"
        self.ink, self.muted = "#203344", "#52687b"
        self.relations = []
        self.text(90, 102, "Pallas 的两层程序", 58, bold=True)
        self.text(92, 163, "外层 pallas_call 携带内层 kernel Jaxpr；TPU lowering 将 kernel 封装回外层 Module。", 31, self.muted)
        self.text(92, 224, "↗ 返回 overview 的 Pallas 路径", 26, self.colors['program'], href='../overview/overview-software-stack-components-layered.svg#kernel')
        self.text(1230, 224, "↗ JAX / Jaxpr 详细图", 26, self.colors['program'], href='jaxpr-centered-hub.svg#core')
        self.text(2230, 224, "↗ MLIR Module 详细图", 26, self.colors['program'], href='../jaxlib/mlir-module-centered-hub.svg#core')
        for x, title, subtitle, fill, color in [
            (70, "外层 · 调用者程序", "张量输入输出、普通操作与 pallas_call 方程", "#edf3fb", "program"),
            (1710, "内层 · Pallas kernel", "Ref 程序、网格映射与独立 Mosaic 模块", "#edf3fb", "program"),
        ]:
            self.base.append(f'<rect x="{x}" y="280" width="1420" height="2530" rx="25" fill="{fill}" stroke="#b9c9d7" stroke-width="2"/>')
            self.text(x + 35, 338, title, 37, self.colors[color], bold=True)
            self.text(x + 37, 384, subtitle, 26, self.muted)

    def box(self, key, x, y, w, h, tag, title, rows, refs=(), color="program", core=False, links=()):
        self.boxes[key] = (x, y, w, h)
        stroke = self.colors[color]
        fill = ("#e1edfc" if color == "program" else "#ddf1e9") if core else self.card
        self.nodes.append(f'<g id="{key}" data-module="jax" data-core="{str(core).lower()}"><title>{escape(title)}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="19" fill="{fill}" stroke="{stroke}" stroke-width="{4 if core else 2}"/></g>')
        self.text(x + 28, y + 37, tag, 22, stroke, bold=True, owner=key)
        self.text(x + 28, y + 91, title, 38 if core else 34, bold=True, owner=key)
        baseline = y + 144
        for row in rows:
            for part in self.wrap(row, w - 58, 28):
                self.text(x + 28, baseline, part, 28, self.muted, owner=key)
                baseline += 40
        anchors=[(ref,self.ref(ref)) for ref in refs]+list(links)
        for i, (label, href) in enumerate(anchors):
            self.text(x + 28, y + h - 25 - 32 * (len(anchors) - i - 1), "↗ " + label,
                      23, stroke, href=href, owner=key)

    def flow(self, source, target, points, caption=None, label_at=None, color="program", dashed=False):
        assert source in self.boxes and target in self.boxes
        self.edge(points, color, dashed=dashed, width=4)
        if caption:
            self.label(*label_at, caption, color, size=24)
        self.relations.append({"source": source, "target": target, "label": caption or "", "kind": color})

    def save(self, path):
        required=[['caller','outer_jaxpr','outer_lowering','outer_module','compile'],
                  ['kernel','inner_jaxpr','mosaic_module','payload','outer_module'],
                  ['inner_jaxpr','outer_jaxpr']]
        pairs={(e['source'],e['target']) for e in self.relations}
        for chain in required:
            if any(pair not in pairs for pair in zip(chain,chain[1:])):
                raise ValueError('Incomplete Pallas workflow: '+str(chain))
        self.meta['checked_paths']=required
        super().save(path)
        root=ET.parse(path).getroot()
        ids=[e.get('id') for e in root.iter() if e.get('id')]
        if len(ids)!=len(set(ids)) or sum(e.get('data-core')=='true' for e in root.iter())!=1:
            raise ValueError('Expected unique SVG IDs and one Pallas core concept')
        for e in root.iter():
            href=e.get('href','')
            if href.startswith('#') and href[1:] not in ids:
                raise ValueError('Broken Pallas node link: '+href)


def render(meta):
    d = PallasDiagram(meta)
    left, right, width = 120, 1760, 1320
    d.box("caller", left, 440, width, 290, "产生 · 调用入口", "Python 函数中的 pl.pallas_call", [
        "调用者传入数组，并接收结果数组。",
        "kernel、grid、BlockSpec 等描述被调用的计算。",
    ], ("pallas_call",))
    d.box("kernel", right, 440, width, 290, "产生 · kernel 追踪输入", "kernel 函数 + GridMapping + avals", [
        "get_grid_mapping 整理网格、块映射与调用约定。",
        "kernel 的参数以 AbstractRef 等抽象类型参与追踪。",
    ], ("get_grid_mapping",))
    d.box("outer_jaxpr", left, 950, width, 450, "定义 · 外层程序中的一个方程", "外层 Jaxpr / pallas_call 方程", [
        "普通方程 → pallas_call → 后续方程。",
        "primitive = pallas_call；invars / outvars 连接外层值。",
        "params.jaxpr = 内层 kernel Jaxpr。",
        "params.grid_mapping / compiler_params 等另行携带。",
        "pallas_call_p.bind 将调用及其参数交给外层 Trace。",
    ], ("JaxprEqn", "pallas_call_p.bind"))
    d.box("inner_jaxpr", right, 950, width, 450, "核心概念 · Jaxpr 的 kernel 程序实例", "内层 kernel Jaxpr", [
        "同样使用 core.Jaxpr，描述 kernel 内部的方程与效果。",
        "通过输入 / 输出 Ref 的读写表达计算。",
        "kernel Python 函数返回 None；输出由 Ref 写入。",
        "GridMapping 是配套调用约定，独立于 Jaxpr 字段。",
        "追踪函数返回 kernel Jaxpr 与捕获的 consts。",
    ], ("Jaxpr", "_trace_kernel_to_jaxpr"), core=True)
    d.box("outer_lowering", left, 1670, width, 430, "消费 · 按外层方程选择 lowering 规则", "jaxpr_subcomp → Pallas lowering", [
        "输入：外层 IR 实参、LoweringRuleContext、eqn.params。",
        "_pallas_call_lowering 按平台与 interpret 模式分派。",
        "本图展开 interpret=False 的 Mosaic TPU 路径。",
        "输出：IR 结果值，写回该外层方程的输出变量。",
    ], ("jaxpr_subcomp", "register Pallas lowering", "_pallas_call_lowering"))
    d.box("mosaic_module", right, 1670, width, 430, "消费 kernel Jaxpr · 产生独立模块", "Mosaic TPU MLIR Module", [
        "输入：kernel Jaxpr + GridMapping + lowering 上下文。",
        "TPU 规则调用 lower_jaxpr_to_pipelined_module。",
        "构造独立 ir.Module，承载 kernel 的 Mosaic TPU IR。",
        "这个模块随后成为 custom_call 的序列化 payload。",
    ], ("pallas_call_tpu_lowering_rule", "lower_jaxpr_to_pipelined_module"))
    d.box("outer_module", left, 2310, width, 450, "产生 · 外层 Module 中的调用操作", "stablehlo.custom_call", [
        "call_target_name = tpu_custom_call。",
        "backend_config 携带序列化 kernel 与相关配置。",
        "operands / results 连接外层 IR 的输入与后续操作。",
        "与普通 StableHLO 等操作共存于外层 MLIR Module。",
        "lowering 返回 IR 值；这些值尚不是设备执行结果。",
    ], ("emit tpu_custom_call", "lower_jaxpr_to_module"))
    d.box("payload", right, 2310, width, 450, "变换与封装 · 内层模块成为外层配置", "CustomCallBackendConfig / payload", [
        "克隆 Mosaic 模块，运行版本化 mosaic-serde pass。",
        "write_bytecode 生成模块字节码。",
        "to_json 将字节码编码进 custom_call_config.body。",
        "lower_module_to_custom_call 构造外层调用操作。",
        "payload 保存内层程序；与外层 operands 分别传递。",
    ], ("_lower_mosaic_module_to_asm", "CustomCallBackendConfig.to_json", "lower_module_to_custom_call"), color="transform")
    d.box("compile", left, 2960, 2960, 240, "消费 · 编译提交边界", "外层 MLIR Module → PyClient.CompileAndLoad → IFRT HloProgram", [
        "绑定层克隆并包装整个外层 Module，继续提交编译；其中的 custom_call 携带 kernel payload。",
    ], ("PyClient.CompileAndLoad",))
    d.box("mapping", left, 3290, 1430, 440, "定义 · 两类对象共同约定调用", "GridMapping 配合 kernel Jaxpr", [
        "grid / block_mappings / scratch_avals 等描述映射。",
        "Jaxpr 定义 kernel 的计算；GridMapping 组织其调用。",
        "二者作为独立参数，由 TPU lowering 一起消费。",
    ], ("GridMapping",), color="neutral", links=(("定位 kernel Jaxpr ↑", "#inner_jaxpr"),("定位 TPU lowering ↑", "#mosaic_module")))
    d.box("transforms", 1650, 3290, 1430, 440, "变换 · 满足各规则约束时应用", "Pallas 专用 JVP / batching 规则", [
        "JVP 消费 kernel Jaxpr 与切向信息，产生新 Jaxpr。",
        "同步更新 GridMapping / 输出约定，再绑定调用。",
        "batching 也按 Pallas 专用规则处理调用与映射。",
    ], ("_pallas_call_jvp_rule", "_pallas_call_batching_rule"), color="transform", links=(("定位被变换的 kernel Jaxpr ↑", "#inner_jaxpr"),("定位重新绑定的外层调用 ↑", "#outer_jaxpr")))

    d.flow("caller", "kernel", [(1440, 590), (1760, 590)], "kernel 参数", (1473, 562), color='neutral', dashed=True)
    d.flow("caller", "outer_jaxpr", [(780, 730), (780, 950)], "外层 tracing 收集调用方程", (815, 846))
    d.flow("kernel", "inner_jaxpr", [(2420, 730), (2420, 950)])
    d.label(2460, 816, "_trace_kernel_to_jaxpr", "program", size=24)
    d.label(2460, 859, "trace_to_jaxpr + DCE", "program", size=24)
    d.flow("inner_jaxpr", "outer_jaxpr", [(1760, 1190), (1440, 1190)], "params.jaxpr", (1476, 1157), "neutral", True)
    d.text(1490, 1240, "+ GridMapping", 22, d.muted)
    d.flow("outer_jaxpr", "outer_lowering", [(780, 1400), (780, 1670)], "消费外层方程与 params", (820, 1540))
    d.flow("inner_jaxpr", "mosaic_module", [(2420, 1400), (2420, 1670)], "kernel Jaxpr + GridMapping", (2470, 1540))
    d.flow("outer_lowering", "mosaic_module", [(1440, 1850), (1760, 1850)], "TPU 分派", (1487, 1816))
    d.flow("outer_lowering", "outer_module", [(780, 2100), (780, 2310)], "在外层插入操作并接回 IR 结果", (820, 2200))
    d.flow("mosaic_module", "payload", [(2420, 2100), (2420, 2310)], "版本化序列化与配置封装", (2470, 2200), "transform")
    d.flow("payload", "outer_module", [(1760, 2530), (1440, 2530)], "backend_config", (1453, 2494))
    d.flow("outer_module", "compile", [(780, 2760), (780, 2960)], "提交整个外层模块", (825, 2880))
    d.text(92, 3800, "蓝色：编译程序与 lowering；紫色：程序变换；灰虚线：结构与参数持有。仅 kernel Jaxpr 使用核心概念填色。", 25, d.muted)
    d.text(92, 3870, "证据范围：固定 JAX 源码检查；本图未执行 tracing、编译、TPU 模拟或真实设备计算。Mosaic TPU MLIR ≠ LLO。", 25, d.muted)
    d.text(92, 3930, "固定提交：" + JAX_PIN + "   ·   点击框内链接查看类、函数及具体调用位置。", 24, d.muted)
    d.meta["relations"] = d.relations
    d.meta['core_concepts']={'jax':'inner_jaxpr'}
    d.meta['reference_associations']=[['mapping','inner_jaxpr'],['mapping','mosaic_module'],['transforms','inner_jaxpr'],['transforms','outer_jaxpr']]
    return d


def main():
    """Pallas is now a branch of the single maintained JAX diagram."""
    from render_software_stack_component_flows import main as render_selected
    render_selected(only='jax')


if __name__ == "__main__":
    main()
