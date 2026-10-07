#!/usr/bin/env python3
"""Render the selected layered overview of compilation, dispatch and results.

Uses pinned JAX Git objects and either pinned XLA Git objects or individually
hash-verified source files. --fetch-sources populates only the ignored source
cache; normal regeneration is offline and fails if evidence is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
import urllib.request

from render_overview_software_stack_components import Diagram, SOURCE_METADATA
from render_pallas_inner_outer import metadata as pallas_metadata
from software_stack_overview_flow_data import CORE_CONCEPTS, NODES


ROOT = Path(__file__).resolve().parents[1]
STEM = "overview-software-stack-components-layered"
XLA_PIN = "dcf304bc5dca1932b99f740b911dbd73631a1a69"
IFRT_IMPL = "xla/python/pjrt_ifrt/pjrt_executable.cc"
CPU_CLIENT = "xla/pjrt/cpu/cpu_client.cc"
PY_EXEC = "jaxlib/py_executable.cc"
EXTRA_ANCHORS = {
    "Var": ("jax", "jax/_src/core.py", "class Var:"),
    "PyArray": ("jax", "jaxlib/py_array.h", "class PyArray :"),
    "PyLoadedExecutable wrapping": ("jax", "jaxlib/py_client.cc", "return make_nb_class<PyLoadedExecutable>(std::move(client),", 430),
    "ExecuteReplicated": ("jax", "jax/_src/interpreters/pxla.py", "class ExecuteReplicated:"),
    "ExecuteSharded": ("jax", PY_EXEC, "absl::StatusOr<PyExecuteResults> PyLoadedExecutable::ExecuteSharded("),
    "ConsumeWithHandlers": ("jax", PY_EXEC, "std::vector<nb::object> PyExecuteResults::ConsumeWithHandlers("),
    "IFRT executable wrapping": ("xla", IFRT_IMPL, "return LoadedExecutableRef(new PjRtLoadedExecutable(", 798),
    "IFRT Execute": ("xla", IFRT_IMPL, "PjRtLoadedExecutable::Execute(absl::Span<ArrayRef> args,"),
    "IFRT output arrays": ("xla", IFRT_IMPL, "outputs.push_back(*PjRtArray::Create("),
    "IFRT ExecuteResult": ("xla", "xla/python/ifrt/executable.h", "struct ExecuteResult {"),
    "CPU LoadInternal": ("xla", CPU_CLIENT, "PjRtCpuClient::LoadInternal("),
    "CPU executable wrapping": ("xla", CPU_CLIENT, "auto executable = std::make_unique<PjRtCpuExecutable>("),
    "CPU JitCompile": ("xla", CPU_CLIENT, "static absl::StatusOr<std::unique_ptr<xla::Executable>> JitCompile("),
    "CPU Module compilation": ("xla", CPU_CLIENT, "PjRtCpuClient::CompileAndAssignDevices(MaybeOwningMlirModule module,"),
    "CPU runtime Execute": ("xla", CPU_CLIENT, "PjRtRawLoadedExecutable::RawExecuteResult CpuPjRtRawLoadedExecutable::Execute("),
    "PjRtLoadedExecutable Execute": ("xla", "xla/pjrt/pjrt_client.h", "Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,"),
    "XLA Executable": ("xla", "xla/service/executable.h", "class Executable {"),
    "MLIR to HLO": ("xla", "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc", "absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertMlirHloToHloModule("),
    "HloModule": ("xla", "xla/hlo/ir/hlo_module.h", "class HloModule {"),
    "BufferAssignment": ("xla", "xla/service/buffer_assignment.h", "class BufferAssignment {"),
    "HloPassPipeline RunImpl": ("xla", "xla/hlo/pass/hlo_pass_pipeline.cc", "absl::StatusOr<bool> HloPassPipeline::RunImpl(", 298),
    "CPU RunBackend": ("xla", "xla/service/cpu/cpu_compiler.cc", "absl::StatusOr<std::unique_ptr<Executable>> CpuCompiler::RunBackend("),
    "GPU RunBackend": ("xla", "xla/service/gpu/gpu_compiler.cc", "absl::StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend("),
    "PJRT Buffer readiness": ("xla", "xla/pjrt/pjrt_client.h", "virtual Future<> GetReadyFuture() = 0;"),
}


def metadata(root, fetch=False, *, extra_anchors=None):
    meta = pallas_metadata(root)
    lock = json.loads((root / "tools/overview_flow_sources.json").read_text())
    pin = next(line.split("|")[3] for line in (root / "upstream-sources.lock").read_text().splitlines()
               if line.startswith("core|upstream/xla|"))
    if pin != XLA_PIN or lock["revision"] != pin:
        raise ValueError("XLA pin changed; review the diagram and source hashes")
    meta["source_pins"]["xla"] = pin
    cache = root / "artifacts/environment/overview-sources" / pin
    repo = root / "upstream/xla"
    checkout = (repo / ".git").exists()
    if checkout and subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != pin:
        raise ValueError("XLA checkout does not match the source lock")
    files = {}
    for item in lock["files"]:
        path = item["path"]
        if checkout:
            data = subprocess.check_output(["git", "-C", str(repo), "show", f"{pin}:{path}"])
            if (repo / path).exists() and (repo / path).read_bytes() != data:
                raise ValueError("Source file modified: " + path)
        else:
            dest = cache / path
            if not dest.exists() and fetch:
                url = f"https://raw.githubusercontent.com/openxla/xla/{pin}/{path}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Downloaded source hash mismatch: " + path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            if not dest.exists():
                raise ValueError(f"Missing pinned source {path}; use --fetch-sources to populate the cache")
            data = dest.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"] or len(data) != item["size_bytes"]:
            raise ValueError("Pinned source checksum mismatch: " + path)
        files[("xla", path)] = data.decode().splitlines()
    specs = dict(EXTRA_ANCHORS)
    specs.update(extra_anchors or {})
    for name in ("trace_to_jaxpr_dynamic", "dce_jaxpr", "IFRT CompileAndLoad", "PJRT CompileAndLoad", "ifrt Array", "PjRtBuffer", "PjRtLoadedExecutable", "StableHLO binding"):
        anchor = SOURCE_METADATA["source_anchors"][name]
        specs[name] = (anchor["repo"], anchor["path"], anchor["needle"], anchor["line"])
    specs["Shardy module pass"] = ("jax", "jax/_src/interpreters/mlir.py", "'builtin.module(sdy-lift-inlined-meshes)')")
    for name, (kind, path, needle, *fixed) in specs.items():
        if (kind, path) not in files:
            repo_path = root / "upstream" / kind
            data = subprocess.check_output(["git", "-C", str(repo_path), "show", f"{meta['source_pins'][kind]}:{path}"])
            if (repo_path / path).exists() and (repo_path / path).read_bytes() != data:
                raise ValueError("Source file modified: " + path)
            files[(kind, path)] = data.decode().splitlines()
        lines = files[(kind, path)]
        hits = ([fixed[0]] if lines[fixed[0] - 1].strip().startswith(needle.strip()) else []) if fixed else [i + 1 for i, line in enumerate(lines) if line.strip().startswith(needle.strip())]
        if len(hits) != 1:
            raise ValueError(f"Ambiguous or changed anchor {name}: {hits}")
        host = "0xaskr/jax" if kind == "jax" else "openxla/xla"
        meta["source_anchors"][name] = {"repo": kind, "path": path, "line": hits[0], "needle": needle,
            "href": f"https://github.com/{host}/blob/{meta['source_pins'][kind]}/{path}#L{hits[0]}"}
    meta.pop("path", None)
    meta["evidence_scope"] = "Pinned JAX Git objects and checksum-verified pinned XLA source files; source inspection only, no compilation or device execution"
    meta["xla_source_files"] = lock["files"]
    meta["implementation_scope"] = "PJRT-backed IFRT, ordinary sharded array execution; CPU/GPU public compiler path and opaque TPU provider boundary"
    return meta


WIDTH, HEIGHT = 12840, 21800
CW, NW, NH, STEP = 4200, 1180, 520, 1360
FRAMES = {
    "jax": (4420, 480, CW, 3740),
    "jaxlib": (4420, 5220, CW, 1700),
    "ifrt": (1300, 8150, CW, 1850),
    "pjrt": (7540, 8150, CW, 1850),
    "xla": (1300, 11300, CW, 2200),
    "backend": (7540, 11300, CW, 2730),
    "asm": (1300, 15280, CW, 1000),
    "runtime": (7540, 15280, CW, 1920),
    "hardware": (4420, 18400, CW, 1040),
}
HEADINGS = {
    "jax": ("JAX · 函数追踪、程序变换与 lowering", "前端实现", "Jaxpr 描述计算与 effects；Pallas 的内层 kernel 程序随外层方程传递。", "../jax/jaxpr-centered-hub.svg"),
    "jaxlib": ("jaxlib · Python / 原生编译与运行时绑定", "原生绑定", "外层 MLIR Module 是编译输入；PyArray 和 PyLoadedExecutable 连接 Python 与 IFRT。", "../jaxlib/mlir-module-centered-hub.svg"),
    "ifrt": ("IFRT · 面向框架的逻辑数组与程序", "框架运行时接口", "主抽象 Array；此图展开 PJRT-backed 实现，编译、加载、执行与结果包装分别表达。", "../ifrt/array-centered-hub.svg"),
    "pjrt": ("PJRT · 设备、存储与可执行程序接口", "设备运行时接口", "主抽象 Buffer；Client、LoadedExecutable 与 Buffer 的行为由具体 provider 实现。", "../pjrt/buffer-centered-hub.svg"),
    "xla": ("XLA · HLO 优化与编译规划", "编译器子系统", "HloModule 经变换和规划交给目标后端；后端产物在 Executable 节点汇合并返回 provider。", "../xla/hlo-centered-hub.svg"),
    "backend": ("编译后端 · 按目标设备分流", "目标后端", "CPU / GPU 属于 XLA；TPU 经 provider / libtpu，公开接口与内部证据边界分开。", None),
    "asm": ("目标汇编表示 · ASM 与代码观察", "可观察表示", "汇编文本是代码表示或观察形式，不是每条编译 / 执行路径必经的软件组件。", None),
    "runtime": ("设备运行时与驱动 · 程序和数据汇合", "运行时实现", "已加载程序 × 输入 Buffer × 执行选项 / 依赖；输出对象和完成状态分别返回。", "../stream-executor/submission-centered-hub.svg"),
    "hardware": ("指令集接口与硬件 · ISA 约束执行", "硬件 / 指令契约", "设备执行目标工作、读写存储并报告状态；程序表示与硬件指令不作一一对应。", None),
}


def overlaps(a, b, padding=0):
    x, y, w, h = a
    u, v, s, t = b
    return min(x + w + padding, u + s) > max(x - padding, u) and min(y + h + padding, v + t) > max(y - padding, v)


def orthogonal_route(start, end, bounds, obstacles, occupied):
    """Route an internal relation through card gutters; forbid shared segments."""
    from functools import lru_cache
    import heapq
    bx, by, bw, bh = bounds
    xs, ys = {start[0], end[0], bx, bx + bw}, {start[1], end[1], by, by + bh}
    for x, y, w, h in obstacles:
        for gap in (35, 70, 105, 140):
            xs.update((x - gap, x + w + gap))
            ys.update((y - gap, y + h + gap))
    xs = sorted(x for x in xs if bx <= x <= bx + bw)
    ys = sorted(y for y in ys if by <= y <= by + bh)
    used_h, used_v = {}, {}
    for a, b in occupied:
        table, axis, other = (used_h, 1, 0) if a[1] == b[1] else (used_v, 0, 1)
        table.setdefault(a[axis], []).append(sorted((a[other], b[other])))

    @lru_cache(None)
    def blocked(a, b):
        vertical = a[0] == b[0]
        axis, along = (0, 1) if vertical else (1, 0)
        lo, hi = sorted((a[along], b[along]))
        for x, y, w, h in obstacles:
            low, high = ((x - 14, x + w + 14) if vertical else (y - 14, y + h + 14))
            start_o, end_o = ((y - 14, y + h + 14) if vertical else (x - 14, x + w + 14))
            if low < a[axis] < high and min(hi, end_o) > max(lo, start_o):
                return True
        for u, v in (used_v if vertical else used_h).get(a[axis], []):
            if min(hi, v) - max(lo, u) > 1:
                return True
        return False

    first = (xs.index(start[0]), ys.index(start[1]), -1)
    costs, previous, queue = {first: 0}, {}, [(0, 0, first)]
    found = None
    while queue:
        _, cost, state = heapq.heappop(queue)
        if costs.get(state) != cost:
            continue
        i, j, direction = state
        point = (xs[i], ys[j])
        if point == end:
            found = state
            break
        for ni, nj, nd in ((i - 1, j, 0), (i + 1, j, 0), (i, j - 1, 1), (i, j + 1, 1)):
            if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                continue
            target = (xs[ni], ys[nj])
            if blocked(point, target):
                continue
            nc = cost + abs(target[0] - point[0]) + abs(target[1] - point[1]) + (110 if direction not in (-1, nd) else 0)
            nxt = (ni, nj, nd)
            if nc < costs.get(nxt, float("inf")):
                costs[nxt], previous[nxt] = nc, state
                estimate = abs(target[0] - end[0]) + abs(target[1] - end[1])
                heapq.heappush(queue, (nc + estimate, nc, nxt))
    if found is None:
        raise ValueError(f"No internal route: {start} → {end}")
    points, state = [], found
    while state != first:
        points.append((xs[state[0]], ys[state[1]]))
        state = previous[state]
    return [start] + list(reversed(points))


def compact(points):
    result = []
    for point in points:
        if result and point == result[-1]:
            continue
        while len(result) > 1 and ((result[-2][0] == result[-1][0] == point[0]) or (result[-2][1] == result[-1][1] == point[1])):
            result.pop()
        result.append(point)
    return result


class ComponentFlowDiagram(Diagram):
    def __init__(self, meta):
        super().__init__(WIDTH, HEIGHT, "JAX 软件栈 / 分层架构 · 编译、执行与完整返回", "", meta)
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.text_boxes, self.edge_boxes = [], []
        self.bg, self.ink, self.muted = "#fafbf8", "#203344", "#506577"
        self.colors.update(output="#a34877", completion="#a34877")
        self.components, self.members, self.relations = dict(FRAMES), {}, []
        self.node_data = NODES
        self.concept_modules = {}
        self.port_usage, self.global_routes, self.sidebars, self.caption_boxes = {}, [], [], []
        self.description = "沿用浅色分层架构图的组件分布：JAX 与 jaxlib 居中，IFRT/PJRT、XLA/编译后端、ASM/设备运行时左右并列，硬件居下。每个模块只突出一个核心概念，CPU、GPU、TPU 后端分别计为一个模块。保留编译产物逐层返回、执行与输出对象往返、Pallas 内外层程序及命名接口。"
        self.text(180, 100, self.title, 68, bold=True)
        self.text(184, 170, "每个模块只突出一个核心概念，周围展开定义、产生、变换与消费；跨框端口标出接口、对象与方向。", 34, self.muted)
        for i, (label, color, dashed) in enumerate([
            ("编译 / 程序", "program", False), ("执行 / 输入数据", "data", False),
            ("程序变换", "transform", False), ("编译结果返回", "result", True),
            ("输出对象返回", "output", False), ("完成 / 就绪状态", "completion", True),
            ("结构 / 持有 / 观察", "neutral", True), ("TPU 内部证据边界", "unknown", True),
        ]):
            x = 185 + i * 1390
            self.edge([(x, 250), (x + 88, 250)], color, dashed=dashed, end=False, width=4)
            self.text(x + 111, 263, label, 32, self.muted)
        self.text(184, 339, "填色与粗边框只用于核心概念；CPU / GPU / TPU 各一个。线色区分流程，拱桥表示交叉，右上角标出组件角色。", 30, self.muted)
        self.text(184, 397, "范围：PJRT-backed IFRT · 常规分片数组执行 · Pallas 展开 Mosaic TPU 路径 · 固定源码检查。", 29, self.muted)

    def frame(self, name):
        x, y, w, h = self.components[name]
        title, role, subtitle, href = HEADINGS[name]
        color = "data" if name in {"ifrt", "pjrt", "runtime"} else "neutral" if name in {"asm", "hardware"} else "program"
        fill = "#f5f4ef" if name in {"asm", "hardware"} else "#f1f5f7"
        self.base.append(f'<g id="component_{name}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{fill}" stroke="#a8bbca" stroke-width="3"/></g>')
        self.base.append(f'<path d="M{x+30},{y+133}H{x+w-30}" stroke="#d1dce3" stroke-width="2"/>')
        self.text(x + 35, y + 63, title + (" ↗" if href else ""), 44, self.colors[color], bold=True, href=href)
        self.text(x + 38, y + 109, subtitle, 27, self.muted)
        rw = self.measure(role, 27, True)[1].width + 40
        self.base.append(f'<rect x="{x+w-rw-28}" y="{y+23}" width="{rw}" height="47" rx="11" fill="#e3e9ec"/>')
        self.text(x + w - rw - 8, y + 56, role, 27, self.colors[color], bold=True)

    def box(self, key, comp, col, row, *, h=NH):
        content = self.node_data[key]
        if key in self.boxes:
            raise ValueError("Node placed more than once: " + key)
        if "core" in content or "核心概念" in content["tag"]:
            raise ValueError("Core emphasis must come only from CORE_CONCEPTS: " + key)
        module = ("cpu", "gpu", "tpu")[col] if comp == "backend" else comp
        self.concept_modules[key] = module
        cx, cy, _, _ = self.components[comp]
        x = cx + 150 + STEP * col
        y = cy + (280 + row * 800 if comp == "backend" else 190 + row * 720)
        w = NW
        self.boxes[key], self.members[key] = (x, y, w, h), comp
        color, core = content.get("color", "program"), CORE_CONCEPTS[module] == key
        stroke = self.colors[color]
        fill = {"program": "#e6f0fd", "data": "#e0f4ed", "result": "#fff0e6", "output": "#fbeaf2", "unknown": "#fff1d9", "neutral": "#eef1f4"}.get(color, "#f1eafa") if core else "#ffffff"
        dash = ' stroke-dasharray="9 7"' if content.get("dashed") else ""
        self.nodes.append(f'<g id="{key}" data-module="{module}" data-core="{str(core).lower()}"><title>{escape(content["title"])}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="{4 if core else 2}"{dash}/></g>')
        tag = ("核心概念 · " if core else "") + content["tag"]
        self.text(x + 27, y + 38, tag, 25, stroke, bold=True, owner=key)
        size = 40 if core else 36
        while self.measure(content["title"], size, True)[1].width > w - 54:
            size -= 1
        self.text(x + 27, y + 93, content["title"], size, bold=True, owner=key)
        baseline = y + 149
        for row_text in content["rows"]:
            for part in self.wrap(row_text, w - 56, 29):
                self.text(x + 27, baseline, part, 29, self.muted, owner=key)
                baseline += 42
        links = [("源码 · " + ref, self.ref(ref)) for ref in content.get("refs", ())]
        if content.get("detail"):
            links.append(("进入组件详细图 ↗", content["detail"]))
        for i, (label, link) in enumerate(links):
            self.text(x + 27, y + h - 25 - (len(links) - i - 1) * 34, label, 23, stroke, owner=key, href=link)

    def note(self, comp, text):
        x, y, w, h = self.components[comp]
        self.text(x + 40, y + h - 54, text, 28, self.muted)

    def guide(self):
        x, w = 1300, 2280
        sections = [
            (650, 750, "阅读导航 · 点击定位组件", [
                ("前端：Jaxpr → 外层 MLIR Module", "#component_jax"),
                ("接口：IFRT Array / PJRT Buffer", "#component_ifrt"),
                ("编译：HLO → CPU / GPU / TPU 后端", "#component_xla"),
                ("执行：已加载程序 × 输入数据", "#component_runtime"),
            ]),
            (1570, 1040, "MLIR · 跨层基础设施", [
                ("对象：Module / Operation / Region / Type", None),
                ("JAX lowering 通过 jaxlib 绑定构造操作。", None),
                ("StableHLO、sdy、Mosaic 等方言规定语义。", None),
                ("模块可包含多个方言，并经过验证与 pass。", None),
                ("相应后端路径继续使用 MLIR 方言。", None),
                ("查看 Module / 方言详细图 ↗", "../jaxlib/mlir-module-centered-hub.svg"),
            ]),
            (2790, 1080, "往返路径 · 点击进入关键节点", [
                ("编译产物：Executable → PJRT → IFRT → Python", "#executable"),
                ("执行：ExecuteSharded → IFRT → PJRT → 后端", "#py_execute"),
                ("输出：PjRtBuffer → IFRT Array → PyArray", "#pjrt_outputs"),
                ("完成：事件 / Future / readiness 另行表达", "#runtime_completion"),
                ("JAX 图中的 Pallas 内外层程序 ↗", "../jax/jaxpr-centered-hub.svg#inner_jaxpr"),
            ]),
        ]
        for y, h, title, rows in sections:
            self.sidebars.append((x, y, w, h))
            self.base.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="#edf2f8"/>')
            self.text(x + 40, y + 63, title, 37, self.colors["neutral"], bold=True)
            for i, (line, link) in enumerate(rows):
                self.text(x + 45, y + 169 + i * 135, line, 33, self.colors["program"] if link else self.muted, href=link)

    def node_port(self, key, side):
        x, y, w, h = self.boxes[key]
        count = self.port_usage.get((key, side), 0)
        self.port_usage[(key, side)] = count + 1
        fractions = (.5, .26, .74, .38, .62, .14, .86)
        if count >= len(fractions):
            raise ValueError(f"Too many node ports: {key}, {side}")
        fraction = fractions[count]
        if side == "left": return (x, y + h * fraction), (-1, 0)
        if side == "right": return (x + w, y + h * fraction), (1, 0)
        if side == "top": return (x + w * fraction, y), (0, -1)
        return (x + w * fraction, y + h), (0, 1)

    def segments(self):
        return [pair for route in self.edge_boxes for pair in zip(route, route[1:])]

    def caption(self, points, lines, color, *, external=False, ref=None, at=None, bounds=None):
        if not lines:
            return
        sizes = [29 if external else 23] + [27 if external else 22] * (len(lines) - 1)
        width = max(self.measure(line, size)[1].width for line, size in zip(lines, sizes))
        line_step = 42 if external else 35
        height = (len(lines) - 1) * line_step + max(sizes) + 14
        obstacles = list(self.components.values()) + self.sidebars if external else list(self.boxes.values())
        candidates = [at] if at else []
        if not at:
            ordered = sorted(zip(points, points[1:]), key=lambda pair: (pair[0][1] != pair[1][1], -abs(pair[0][0]-pair[1][0])-abs(pair[0][1]-pair[1][1])))
            for a, b in ordered:
                for frac in (.5, .25, .75, .12, .88):
                    mx = a[0] + (b[0] - a[0]) * frac
                    my = a[1] + (b[1] - a[1]) * frac
                    if a[1] == b[1]:
                        candidates.extend(((mx, my - height + 15), (mx, my + 48)))
                    else:
                        candidates.extend(((mx + width / 2 + 36, my), (mx - width / 2 - 36, my)))
        for tx, ty in candidates:
            rect = (tx - width / 2 - 12, ty - max(sizes) - 9, width + 24, height)
            if rect[0] < 20 or rect[1] < 420 or rect[0] + rect[2] > self.w - 20 or rect[1] + rect[3] > self.h - 20:
                continue
            if bounds:
                bx, by, bw, bh = bounds
                if not (bx <= rect[0] and by <= rect[1] and rect[0] + rect[2] <= bx + bw and rect[1] + rect[3] <= by + bh):
                    continue
            if any(overlaps(rect, r, 5) for r in obstacles):
                continue
            if any(overlaps(rect, r[:4], 7) for r in self.text_boxes):
                continue
            for i, (line, size) in enumerate(zip(lines, sizes)):
                tw = self.measure(line, size)[1].width
                self.label(tx - tw / 2, ty + i * line_step, line, color, ref=ref if i == 0 else None, size=size)
            self.caption_boxes.append(rect)
            return
        raise ValueError(f"No readable caption: {lines}")

    def inside(self, source, target, kind="program", label=None, sides=None, ref=None):
        comp = self.members[source]
        assert self.members[target] == comp
        ax, ay, aw, ah = self.boxes[source]
        bx, by, bw, bh = self.boxes[target]
        if sides is None:
            if ax == bx:
                sides = ("bottom", "top") if ay < by else ("top", "bottom")
            else:
                sides = ("right", "left") if ax < bx else ("left", "right")
        p, pv = self.node_port(source, sides[0])
        q, qv = self.node_port(target, sides[1])
        start, end = (p[0] + 35 * pv[0], p[1] + 35 * pv[1]), (q[0] + 35 * qv[0], q[1] + 35 * qv[1])
        x, y, w, h = self.components[comp]
        obstacles = [box for key, box in self.boxes.items() if self.members[key] == comp]
        path = orthogonal_route(start, end, (x + 24, y + 148, w - 48, h - 280), obstacles, self.segments())
        points = compact([p] + path + [q])
        self.edge(points, kind, dashed=kind in {"neutral", "completion", "result", "unknown"}, bridge_over=self.segments(), width=3.5)
        if label:
            try:
                self.caption(points, (label,), kind, ref=ref, bounds=self.components[comp])
            except ValueError as exc:
                raise ValueError(f"{source} → {target}: {exc}") from exc
        self.relations.append({"source": source, "target": target, "kind": kind, "contract": [label] if label else [], "scope": "internal"})

    def frame_port(self, comp, side, fraction):
        x, y, w, h = self.components[comp]
        if side == "left": return (x, y + h * fraction), (-1, 0)
        if side == "right": return (x + w, y + h * fraction), (1, 0)
        if side == "top": return (x + w * fraction, y), (0, -1)
        return (x + w * fraction, y + h), (0, 1)

    def cross(self, code, source, target, source_port, target_port, kind, lines, *, via=(), bend_y=None, bend_x=None, at=None, ref=None):
        a, b = self.members[source], self.members[target]
        assert a != b
        p, pv = self.frame_port(a, *source_port)
        q, qv = self.frame_port(b, *target_port)
        start = (p[0] + 110 * pv[0], p[1] + 110 * pv[1])
        end = (q[0] + 110 * qv[0], q[1] + 110 * qv[1])
        middle = list(via)
        if bend_y is not None:
            middle = [(start[0], bend_y), (end[0], bend_y)]
        elif bend_x is not None:
            middle = [(bend_x, start[1]), (bend_x, end[1])]
        points = compact([p, start] + middle + [end, q])
        self.edge(points, kind, dashed=kind in {"result", "neutral", "completion"}, bridge_over=self.segments(), width=4.5)
        for point, normal, key in ((p, pv, source), (q, qv, target)):
            # Leave the complete arrowhead visible between the target badge
            # and the component boundary; a badge must not conceal direction.
            x, y = point[0] + 80 * normal[0], point[1] + 80 * normal[1]
            self.labels.append(f'<a href="#{key}"><title>{escape(code + " · " + self.node_data[key]["title"])}</title><circle cx="{x}" cy="{y}" r="29" fill="{self.bg}" stroke="{self.colors[kind]}" stroke-width="2.5"/></a>')
            self.text(x, y + 8, code, 22, self.colors[kind], bold=True, center=True, href="#" + key)
        self.caption(points, (code + " · " + lines[0],) + tuple(lines[1:]), kind, external=True, ref=ref, at=at)
        self.relations.append({"source": source, "target": target, "kind": kind, "contract": list(lines), "scope": "cross_component", "port": code})
        self.global_routes.append((a, b, points))

    def finish(self):
        self.text(1300, 19900, "跨框接口索引 · 端口与具体节点一一对应，点击节点名称定位", 43, bold=True)
        relations = [r for r in self.relations if r["scope"] == "cross_component"]
        for i, relation in enumerate(relations):
            col, row = divmod(i, 11)
            x, y = 1300 + col * 3780, 20000 + row * 122
            a, b = relation["source"], relation["target"]
            self.text(x, y, relation["port"] + " · " + self.node_data[a]["title"], 26, self.colors[relation["kind"]], href="#" + a)
            self.text(x + 45, y + 39, "→ " + self.node_data[b]["title"], 25, self.colors[relation["kind"]], href="#" + b)
        self.text(1300, 21300, "接口签名省略部分所有权模板、StatusOr 与可选参数；各层的 LoadedExecutable 是不同包装对象。", 29, self.muted)
        self.text(1300, 21365, "固定源码：JAX 361c43e072cc · XLA dcf304bc5dca。源码文件与接口锚点已核验；图示不构成设备执行证据。", 29, self.muted)
        self.text(1300, 21430, "Pallas 的 Mosaic payload 是外层 custom_call 的编译期数据；TPU 私有表示及转换阶段仍需独立版本证据。", 29, self.muted)

    def save(self, path):
        if set(self.concept_modules.values()) != set(CORE_CONCEPTS):
            raise ValueError("Every module must declare exactly one core concept")
        for module, key in CORE_CONCEPTS.items():
            if self.concept_modules.get(key) != module:
                raise ValueError(f"Missing or misplaced core concept for {module}: {key}")
        reserved_ids = {"title", "desc", *self.colors, *("component_" + name for name in self.components)}
        if collision := set(self.boxes) & reserved_ids:
            raise ValueError(f"Node IDs conflict with SVG markers / components: {collision}")
        required = [
            ["module", "binding_module", "py_compile", "ifrt_program", "ifrt_compile", "pjrt_compile", "provider", "import", "hlo", "passes", "plan", "cpu_lowering", "cpu_ir", "cpu", "executable", "pjrt_loaded", "ifrt_loaded", "py_loaded"],
            ["plan", "gpu_lowering", "gpu_ir", "gpu", "executable"],
            ["provider", "tpu", "tpu_loaded", "pjrt_loaded"],
            ["dispatch", "py_execute", "ifrt_execute", "pjrt_execute", "submit", "runtime", "hardware"],
            ["submit", "handles", "pjrt_outputs", "ifrt_outputs", "py_outputs", "outputs"],
            ["outer", "lower", "mosaic", "payload", "module"],
            ["kernel", "outer"], ["kernel", "mosaic"],
            ["inputs", "py_array", "ifrt_array", "pjrt_buffer"],
            ["hardware", "runtime_completion", "handles"],
        ]
        edges = {(r["source"], r["target"]) for r in self.relations}
        for chain in required:
            for edge in zip(chain, chain[1:]):
                if edge not in edges:
                    raise ValueError(f"Missing semantic path: {edge}")
        for key, (x, y, w, h) in self.boxes.items():
            cx, cy, cw, ch = self.components[self.members[key]]
            if not (cx < x and cy + 133 < y and x + w < cx + cw and y + h < cy + ch):
                raise ValueError("Node outside component: " + key)
        for a, b, points in self.global_routes:
            for p, q in zip(points, points[1:]):
                for comp, (x, y, w, h) in self.components.items():
                    if p[0] == q[0] and x + 1 < p[0] < x + w - 1 and min(max(p[1], q[1]), y + h - 1) > max(min(p[1], q[1]), y + 1):
                        raise ValueError(f"Cross-component edge {a} → {b} enters {comp}")
                    if p[1] == q[1] and y + 1 < p[1] < y + h - 1 and min(max(p[0], q[0]), x + w - 1) > max(min(p[0], q[0]), x + 1):
                        raise ValueError(f"Cross-component edge {a} → {b} enters {comp}")
        segments = [(i, a, b) for i, route in enumerate(self.edge_boxes) for a, b in zip(route, route[1:])]
        for n, (i, a, b) in enumerate(segments):
            if a[0] != b[0] and a[1] != b[1]:
                raise ValueError(f"Non-orthogonal edge: {a} → {b}")
            for j, u, v in segments[n + 1:]:
                if i == j:
                    continue
                for axis in (0, 1):
                    along = 1 - axis
                    if a[axis] == b[axis] == u[axis] == v[axis]:
                        overlap = min(max(a[along], b[along]), max(u[along], v[along])) - max(min(a[along], b[along]), min(u[along], v[along]))
                        if overlap > 1:
                            raise ValueError(f"Distinct routes share a segment: {i}, {j}")
        self.meta.update(relations=self.relations, checked_paths=required, component_layout=self.components,
                         core_concepts={module: {"node": key, "title": self.node_data[key]["title"]} for module, key in CORE_CONCEPTS.items()},
                         layout_reference="overview-software-stack-components-layered.svg",
                         port_semantics="Cross-component border ports link to actual source and target nodes; internal arrows connect node boundaries")
        super().save(path)


def render(meta, *, diagram_type=ComponentFlowDiagram):
    d = diagram_type(meta)
    for comp in FRAMES:
        d.frame(comp)
    d.guide()
    placement = {
        "jax": [("function", 0, 0), ("trace", 1, 0), ("inputs", 2, 0),
                ("values", 0, 1), ("outer", 1, 1), ("kernel", 2, 1),
                ("transforms", 0, 2), ("lower", 1, 2), ("mosaic", 2, 2),
                ("shardy", 0, 3), ("payload", 2, 3),
                ("dispatch", 0, 4), ("module", 1, 4), ("outputs", 2, 4)],
        "jaxlib": [("py_outputs", 0, 0), ("binding_module", 1, 0), ("py_array", 2, 0),
                   ("py_compile", 0, 1), ("py_loaded", 1, 1), ("py_execute", 2, 1)],
        "ifrt": [("ifrt_program", 0, 0), ("ifrt_compile", 1, 0), ("ifrt_array", 2, 0),
                 ("ifrt_loaded", 0, 1), ("ifrt_execute", 1, 1), ("ifrt_outputs", 2, 1)],
        "pjrt": [("pjrt_compile", 0, 0), ("pjrt_loaded", 1, 0), ("pjrt_buffer", 2, 0),
                 ("provider", 0, 1), ("pjrt_execute", 1, 1), ("pjrt_outputs", 2, 1)],
        "xla": [("import", 0, 0), ("hlo", 1, 0), ("hlo_structure", 2, 0),
                ("passes", 0, 1), ("plan", 1, 1), ("executable", 2, 1)],
        "backend": [("cpu_lowering", 0, 0), ("gpu_lowering", 1, 0), ("tpu", 2, 0),
                    ("cpu_ir", 0, 1), ("gpu_ir", 1, 1), ("tpu_boundary", 2, 1),
                    ("cpu", 0, 2), ("gpu", 1, 2), ("tpu_loaded", 2, 2)],
        "asm": [("asm_code", 0, 0), ("asm", 1, 0), ("asm_tools", 2, 0)],
        "runtime": [("runtime_program", 0, 0), ("submit", 1, 0), ("runtime_buffers", 2, 0),
                    ("runtime", 0, 1), ("runtime_completion", 1, 1), ("handles", 2, 1)],
        "hardware": [("isa", 0, 0), ("hardware", 1, 0), ("memory", 2, 0)],
    }
    bx, by, _, _ = FRAMES["backend"]
    for col, (label, href) in enumerate([
        ("XLA / CPU 后端 ↗", "../xla/cpu-llvm-ir-centered-hub.svg"),
        ("XLA / GPU 后端 ↗", "../xla/gpu-ir-centered-hub.svg"),
        ("provider / TPU 后端 ↗", "../libtpu/llo-boundary-centered-hub.svg"),
    ]):
        x = bx + 105 + col * STEP
        d.base.append(f'<rect x="{x}" y="{by+165}" width="1270" height="2435" rx="20" fill="#ffffff" stroke="#c4cfd7" stroke-width="2"/>')
        d.text(x + 28, by + 225, label, 35, d.colors["unknown" if col == 2 else "program"], bold=True, href=href)
    for comp, nodes in placement.items():
        for key, col, row in nodes:
            d.box(key, comp, col, row, h=620 if comp == "backend" else NH)
    for comp, note in {
        "jax": "Pallas kernel 的 Ref 程序经 Mosaic payload 回接外层 custom_call；外层 Module 与执行数组走独立通路。",
        "jaxlib": "Module 是程序表示，jaxlib 是实现组件；编译返回 PyLoadedExecutable，执行结果经 handler 包装回 PyArray。",
        "ifrt": "Array → Buffer 是实现映射；ExecuteResult.outputs 与按 fill_status 填充的 status 分开。",
        "pjrt": "Buffer 表示设备存储；PjRtLoadedExecutable 消费这些 Buffer，并返回输出 Buffer 与可请求的完成 future。",
        "xla": "BufferAssignment 是编译期存储规划；CPU/GPU 后端必须先产出 Executable，再由 provider 包装和加载。",
        "backend": "GPU 设备库保留旁路；TPU 内部证据节点仅作边界说明，不将其串成已确认的私有编译阶段。",
        "asm": "ASM 是表示，ISA 是语义契约；观察到目标代码或汇编文本不等于已验证实际设备执行。",
        "runtime": "输出句柄可以先返回；设备完成 / 错误随后更新 readiness。IFRT Array 与 PyArray 在上层分别包装。",
        "hardware": "源码结构图；未据此声称 CPU 执行、TPU 模拟、离线 TPU 编译或真实 TPU 执行。",
    }.items():
        d.note(comp, note)

    # Internal collaboration follows the reference's three-column card layout.
    links = [
        ("function", "trace", "program", "追踪"),
        ("trace", "outer", "program", "构建 Jaxpr"),
        ("outer", "values", "neutral", "引用"),
        ("kernel", "outer", "neutral", "eqn.params"),
        ("outer", "transforms", "transform", None),
        ("transforms", "outer", "transform", "重写程序"),
        ("outer", "lower", "program", "逐方程消费"),
        ("lower", "mosaic", "program", "TPU 规则"),
        ("kernel", "mosaic", "program", "kernel + 映射"),
        ("mosaic", "payload", "program", "序列化为 payload"),
        ("payload", "module", "program", "构造外层操作"),
        ("lower", "module", "program", "外层 Module"),
        ("module", "shardy", "transform", "按配置启用"),
        ("shardy", "module", "transform", "更新 Module"),
        ("binding_module", "py_compile", "program", "构造编译请求"),
        ("py_loaded", "py_execute", "neutral", "执行方法所属"),
        ("py_array", "py_execute", "data", "输入数组"),
        ("ifrt_program", "ifrt_compile", "program", "提交 Program"),
        ("ifrt_loaded", "ifrt_execute", "neutral", "执行方法所属"),
        ("ifrt_array", "ifrt_execute", "data", "Array 实参"),
        ("pjrt_compile", "provider", "program", "委托实现"),
        ("pjrt_loaded", "pjrt_execute", "neutral", "执行方法所属"),
        ("pjrt_buffer", "pjrt_execute", "data", "Buffer 实参"),
        ("import", "hlo", "program", "导入"),
        ("hlo", "hlo_structure", "neutral", "包含"),
        ("hlo", "passes", "transform", "变换"),
        ("passes", "hlo", "transform", None),
        ("passes", "plan", "program", "规划约束"),
        ("cpu_lowering", "cpu_ir", "program", "构造 / 优化 IR"),
        ("cpu_ir", "cpu", "program", "编译 / 链接"),
        ("gpu_lowering", "gpu_ir", "program", "代码生成路径"),
        ("gpu_ir", "gpu", "program", "目标代码"),
        ("gpu_lowering", "gpu", "program", "设备库旁路"),
        ("tpu_boundary", "tpu", "unknown", "边界说明"),
        ("tpu", "tpu_loaded", "program", "公开接口的输入 / 输出"),
        ("asm_code", "asm", "neutral", "观察"),
        ("asm", "asm_tools", "neutral", "读取"),
        ("runtime_program", "submit", "program", "程序"),
        ("runtime_buffers", "submit", "data", "输入 / 依赖"),
        ("submit", "runtime", "data", "提交工作"),
        ("submit", "handles", "output", "返回输出句柄"),
        ("runtime_completion", "handles", "completion", "就绪 / 错误"),
        ("isa", "hardware", "neutral", "约束"),
        ("hardware", "memory", "data", "读写"),
        ("memory", "hardware", "data", None),
    ]
    for source, target, kind, label in links:
        sides = ("right", "right") if (source, target) in {("gpu_lowering", "gpu"), ("tpu", "tpu_loaded")} else None
        if (source, target) == ("submit", "handles"):
            sides = ("bottom", "top")
        d.inside(source, target, kind, label, sides=sides)

    # Border ports intentionally match the reference architecture view. Every
    # port and index entry links to its actual source/target node, and the API
    # and payload are written beside the arrow rather than only in the index.
    d.cross("C1", "module", "binding_module", ("bottom", .5), ("top", .5), "program",
            ["CompileAndLoad(Module, options)", "外层 Module → jaxlib 编译输入"], at=(7140, 4500), ref="PyClient.CompileAndLoad")
    d.cross("D1", "inputs", "py_array", ("right", .11), ("right", .25), "data",
            ["输入数组放置 / 包装", "主机值或 jax.Array → PyArray"], bend_x=9300)
    d.cross("E1", "dispatch", "py_execute", ("right", .93), ("right", .80), "data",
            ["execute_sharded(input_bufs)", "PyArray[] + 执行选项"], bend_x=9740, ref="ExecuteReplicated")
    d.cross("O3", "py_outputs", "outputs", ("top", .16), ("bottom", .86), "output",
            ["consume_with_handlers(handlers)", "返回 Python 可见数组"], bend_y=4800, at=(7670, 4990), ref="ConsumeWithHandlers")
    d.cross("R3", "py_loaded", "dispatch", ("left", .72), ("left", .90), "neutral",
            ["ExecuteReplicated.xla_executable", "持有返回的 PyLoadedExecutable"], bend_x=4230)

    d.cross("C2", "py_compile", "ifrt_program", ("left", .60), ("top", .15), "program",
            ["Compiler::CompileAndLoad(program, options)", "HloProgram(Module) + IFRT CompileOptions"], bend_y=7270, at=(2800, 7170), ref="IFRT CompileAndLoad")
    d.cross("D2", "py_array", "ifrt_array", ("right", .32), ("top", .87), "neutral",
            ["PyArray 持有 IFRT Array", "逻辑数组引用 / Python 元信息"],
            via=[(9040, 5764), (9040, 7870), (4954, 7870)], at=(8200, 7750), ref="PyArray")
    d.cross("E2", "py_execute", "ifrt_execute", ("bottom", .73), ("top", .50), "data",
            ["Execute(args, options, devices)", "PyArray.ifrt_array() → ArrayRef[]"], bend_y=7560, at=(6220, 7430), ref="ExecuteSharded")
    d.cross("R2", "ifrt_loaded", "py_loaded", ("top", .36), ("bottom", .50), "result",
            ["编译 future → Await()", "LoadedExecutableRef → PyLoadedExecutable"], bend_y=7400, at=(4590, 7230), ref="PyLoadedExecutable wrapping")
    d.cross("O2", "ifrt_outputs", "py_outputs", ("top", .68), ("bottom", .17), "output",
            ["IFRT ExecuteResult", "ArrayRef[]；status 按 fill_status 填充"], bend_y=7770, at=(4680, 7980), ref="IFRT ExecuteResult")

    d.cross("C3", "ifrt_compile", "pjrt_compile", ("right", .22), ("left", .22), "program",
            ["PjRtClient::CompileAndLoad", "Module + xla::CompileOptions"], ref="PJRT CompileAndLoad")
    d.cross("S1", "ifrt_array", "pjrt_buffer", ("right", .40), ("left", .40), "neutral",
            ["PJRT-backed Array → Buffer 映射", "一个逻辑数组对应各设备上的分片存储"], ref="ifrt Array")
    d.cross("R1", "pjrt_loaded", "ifrt_loaded", ("left", .54), ("right", .54), "result",
            ["包装 provider 返回的可执行对象", "PjRtLoadedExecutable → IFRT LoadedExecutable"], ref="IFRT executable wrapping")
    d.cross("E3", "ifrt_execute", "pjrt_execute", ("right", .71), ("left", .71), "data",
            ["Execute(argument_handles, opts, futures)", "Array 分片 → PjRtBuffer*[][]"], ref="IFRT Execute")
    d.cross("O1", "pjrt_outputs", "ifrt_outputs", ("left", .91), ("right", .91), "output",
            ["PjRtArray::Create", "输出 Buffer 分片 + 元信息 → ArrayRef[]"], ref="IFRT output arrays")

    d.cross("C4", "provider", "import", ("left", .96), ("top", .18), "program",
            ["provider 调用公开 XLA 编译", "CPU/GPU：Module + 编译 / 设备配置"], bend_y=10700, at=(3750, 10570), ref="CPU Module compilation")
    d.cross("C4t", "provider", "tpu", ("bottom", .83), ("top", .83), "program",
            ["TPU provider 的编译 / 加载入口", "外层 Module + CompileOptions"], at=(10430, 10820))
    d.cross("Rt", "tpu_loaded", "pjrt_loaded", ("top", .96), ("bottom", .96), "result",
            ["TPU provider 返回 PJRT 对象", "PjRtLoadedExecutable"], at=(12070, 10820))
    for code, source, target, y, kind, text, ref in [
        ("C5a", "plan", "cpu_lowering", 11740, "program", ["CPU RunBackend", "目标 HLO + 编译规划"], "CPU RunBackend"),
        ("C5b", "plan", "gpu_lowering", 12110, "program", ["GPU RunBackend", "目标 HLO + 编译规划"], "GPU RunBackend"),
        ("R0a", "cpu", "executable", 12480, "result", ["CPU 编译产物", "CpuExecutable + 执行计划"], "CPU RunBackend"),
        ("R0b", "gpu", "executable", 12850, "result", ["GPU 编译产物", "GpuExecutable + 执行计划"], "GPU RunBackend"),
    ]:
        if kind == "program":
            a, b = ("right", (y - 11300) / 2200), ("left", (y - 11300) / 2730)
        else:
            a, b = ("left", (y - 11300) / 2730), ("right", (y - 11300) / 2200)
        d.cross(code, source, target, a, b, kind, text, ref=ref)
    d.cross("R0", "executable", "pjrt_loaded", ("right", .83), ("bottom", .27), "result",
            ["Executable → provider 包装 / 加载", "CPU：PjRtCpuExecutable → LoadInternal"],
            via=[(6300, 13126), (6300, 10330), (8674, 10330)], at=(7850, 10190), ref="CPU LoadInternal")

    d.cross("E4", "pjrt_execute", "submit", ("bottom", .50), ("top", .50), "data",
            ["provider 的 Execute 实现", "已加载程序 + 输入 Buffer + 执行选项"],
            via=[(9640, 10470), (12300, 10470), (12300, 15050), (9640, 15050)], at=(10400, 14850), ref="PjRtLoadedExecutable Execute")
    d.cross("P0", "pjrt_loaded", "runtime_program", ("left", .77), ("left", .25), "neutral",
            ["已加载程序 / 入口 / 执行计划", "provider 持有，执行实现消费"], bend_x=7070, at=(6550, 14450))
    d.cross("D0", "pjrt_buffer", "runtime_buffers", ("right", .34), ("right", .20), "data",
            ["输入 Buffer 与数据可用依赖", "provider 执行实现接收设备实参"], bend_x=12100, at=(11770, 14600))
    d.cross("O0", "handles", "pjrt_outputs", ("right", .79), ("right", .84), "output",
            ["Execute 返回输出 Buffer", "PjRtBuffer[][]；完成 future 按需请求"], bend_x=12540, at=(11770, 14270), ref="PjRtLoadedExecutable Execute")
    d.cross("N0", "executable", "asm_code", ("left", .88), ("top", .20), "neutral",
            ["可选 dump / 反汇编观察", "目标代码 → 可观察的汇编表示"],
            via=[(900, 13236), (900, 14770), (2140, 14770)], at=(2430, 14470))
    d.cross("E5", "runtime", "hardware", ("bottom", .22), ("top", .66), "data",
            ["后端提交目标工作", "CPU 任务 / GPU kernel 与库 / TPU provider"], bend_y=17870, at=(7560, 17730))
    d.cross("F1", "hardware", "runtime_completion", ("top", .88), ("bottom", .75), "completion",
            ["设备完成 / 输出可用 / 错误", "更新 Future / readiness"], bend_y=18100, at=(10300, 17950), ref="PJRT Buffer readiness")
    d.finish()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fetch-sources", action="store_true")
    args = parser.parse_args()
    meta = metadata(args.root, args.fetch_sources)
    output = args.output_dir or args.root / "research/software-stack/overview"
    output.mkdir(parents=True, exist_ok=True)
    render(meta).save(output / (STEM + ".svg"))
    print(f"Verified {len(meta['source_anchors'])} pinned source anchors; SVG only")


if __name__ == "__main__":
    main()
