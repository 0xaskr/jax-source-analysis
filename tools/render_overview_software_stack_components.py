#!/usr/bin/env python3
"""Shared SVG primitives and compatibility CLI for the selected JAX overview.

Includes Pango measurement, SVG primitives and fixed source anchors for standalone regeneration.
The CLI delegates to render_overview_software_stack_flows.py and outputs SVG only.
Install the locked drawing dependencies with tools/diagram_environment.py sync.
"""

from __future__ import annotations

import copy
import heapq
import json
from pathlib import Path
import subprocess

from html import escape
import xml.etree.ElementTree as ET

from diagram_environment import pango, preview_bindings

FONT_FAMILY = "Noto Sans CJK SC, Microsoft YaHei, sans-serif"


SOURCE_METADATA = {'source_pins': {'jax': '361c43e072cce92b7d3e9bdaf4dd16db26c49043',
                 'xla': 'dcf304bc5dca1932b99f740b911dbd73631a1a69'},
 'source_anchors': {'Jaxpr': {'repo': 'jax',
                              'path': 'jax/_src/core.py',
                              'line': 105,
                              'needle': 'class Jaxpr:',
                              'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L105'},
                    'trace_to_jaxpr_dynamic': {'repo': 'jax',
                                               'path': 'jax/_src/interpreters/partial_eval.py',
                                               'line': 2133,
                                               'needle': 'def trace_to_jaxpr_dynamic(',
                                               'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2133'},
                    'partial_eval_jaxpr_nounits': {'repo': 'jax',
                                                   'path': 'jax/_src/interpreters/partial_eval.py',
                                                   'line': 655,
                                                   'needle': 'def partial_eval_jaxpr_nounits(',
                                                   'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L655'},
                    'dce_jaxpr': {'repo': 'jax',
                                  'path': 'jax/_src/interpreters/partial_eval.py',
                                  'line': 1204,
                                  'needle': 'def dce_jaxpr(',
                                  'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1204'},
                    'eval_jaxpr': {'repo': 'jax',
                                   'path': 'jax/_src/core.py',
                                   'line': 807,
                                   'needle': 'def eval_jaxpr(',
                                   'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L807'},
                    'lower_jaxpr_to_module': {'repo': 'jax',
                                              'path': 'jax/_src/interpreters/mlir.py',
                                              'line': 1327,
                                              'needle': 'def lower_jaxpr_to_module(',
                                              'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327'},
                    'register_lowering': {'repo': 'jax',
                                          'path': 'jax/_src/interpreters/mlir.py',
                                          'line': 1003,
                                          'needle': 'def register_lowering(',
                                          'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1003'},
                    'StableHLO binding': {'repo': 'jax',
                                          'path': 'jax/_src/lib/mlir/dialects/__init__.py',
                                          'line': 62,
                                          'needle': 'from jaxlib.mlir.dialects import stablehlo as hlo',
                                          'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/lib/mlir/dialects/__init__.py#L62'},
                    'PrepareForExport': {'repo': 'xla',
                                         'path': 'xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc',
                                         'line': 6066,
                                         'needle': 'absl::Status PrepareForExport(',
                                         'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6066'},
                    'PyClient CompileAndLoad': {'repo': 'jax',
                                                'path': 'jaxlib/py_client.cc',
                                                'line': 475,
                                                'needle': 'PyClient::CompileAndLoad(nb_class_ptr<PyClient> '
                                                          'client, mlir::ModuleOp module,',
                                                'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475'},
                    'IFRT CompileAndLoad': {'repo': 'jax',
                                            'path': 'jaxlib/py_client.cc',
                                            'line': 412,
                                            'needle': 'client->ifrt_client_->GetDefaultCompiler()->CompileAndLoad(',
                                            'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L412'},
                    'PJRT CompileAndLoad': {'repo': 'xla',
                                            'path': 'xla/python/pjrt_ifrt/pjrt_executable.cc',
                                            'line': 763,
                                            'needle': 'client->pjrt_client()->CompileAndLoad(',
                                            'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L763'},
                    'ifrt Array': {'repo': 'xla',
                                   'path': 'xla/python/ifrt/array.h',
                                   'line': 66,
                                   'needle': 'class Array :',
                                   'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/array.h#L66'},
                    'MakeArrayFromHostBuffer': {'repo': 'xla',
                                                'path': 'xla/python/pjrt_ifrt/pjrt_client.cc',
                                                'line': 1038,
                                                'needle': 'PjRtClient::MakeArrayFromHostBuffer(',
                                                'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1038'},
                    'CopyArrays': {'repo': 'xla',
                                   'path': 'xla/python/pjrt_ifrt/pjrt_client.cc',
                                   'line': 1300,
                                   'needle': 'PjRtClient::CopyArrays(',
                                   'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1300'},
                    'RemapArrays': {'repo': 'xla',
                                    'path': 'xla/python/pjrt_ifrt/pjrt_client.cc',
                                    'line': 1750,
                                    'needle': 'PjRtClient::RemapArrays(',
                                    'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_client.cc#L1750'},
                    'IFRT PjRtLoadedExecutable Execute': {'repo': 'xla',
                                                          'path': 'xla/python/pjrt_ifrt/pjrt_executable.cc',
                                                          'line': 835,
                                                          'needle': 'PjRtLoadedExecutable::Execute(absl::Span<ArrayRef> '
                                                                    'args,',
                                                          'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L835'},
                    'PjRtBuffer': {'repo': 'xla',
                                   'path': 'xla/pjrt/pjrt_client.h',
                                   'line': 1108,
                                   'needle': 'class PjRtBuffer {',
                                   'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1108'},
                    'BufferFromHostBuffer': {'repo': 'xla',
                                             'path': 'xla/pjrt/pjrt_client.h',
                                             'line': 966,
                                             'needle': 'virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> '
                                                       'BufferFromHostBuffer(',
                                             'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L966'},
                    'CopyToMemorySpace': {'repo': 'xla',
                                          'path': 'xla/pjrt/pjrt_client.h',
                                          'line': 1306,
                                          'needle': 'virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> '
                                                    'CopyToMemorySpace(',
                                          'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1306'},
                    'GetReadyFuture': {'repo': 'xla',
                                       'path': 'xla/pjrt/pjrt_client.h',
                                       'line': 1380,
                                       'needle': 'virtual Future<> GetReadyFuture() = 0;',
                                       'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1380'},
                    'PjRtLoadedExecutable': {'repo': 'xla',
                                             'path': 'xla/pjrt/pjrt_client.h',
                                             'line': 1390,
                                             'needle': 'class PjRtLoadedExecutable {',
                                             'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1390'},
                    'HloModule': {'repo': 'xla',
                                  'path': 'xla/hlo/ir/hlo_module.h',
                                  'line': 95,
                                  'needle': 'class HloModule {',
                                  'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L95'},
                    'ConvertMlirHloToHloModule': {'repo': 'xla',
                                                  'path': 'xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc',
                                                  'line': 6232,
                                                  'needle': 'ConvertMlirHloToHloModule(',
                                                  'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6232'},
                    'HloPassPipeline RunImpl': {'repo': 'xla',
                                                'path': 'xla/hlo/pass/hlo_pass_pipeline.cc',
                                                'line': 298,
                                                'needle': 'HloPassPipeline::RunImpl(',
                                                'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/pass/hlo_pass_pipeline.cc#L298'},
                    'CpuCompiler CompileCpuExecutable': {'repo': 'xla',
                                                         'path': 'xla/service/cpu/cpu_compiler.cc',
                                                         'line': 1722,
                                                         'needle': 'CpuCompiler::CompileCpuExecutable(',
                                                         'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1722'},
                    'CpuCompiler RunBackend': {'repo': 'xla',
                                               'path': 'xla/service/cpu/cpu_compiler.cc',
                                               'line': 2139,
                                               'needle': 'CpuCompiler::RunBackend(',
                                               'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2139'},
                    'GpuCompiler RunBackend': {'repo': 'xla',
                                               'path': 'xla/service/gpu/gpu_compiler.cc',
                                               'line': 2943,
                                               'needle': 'GpuCompiler::RunBackend(',
                                               'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2943'},
                    'GPU LLVM verification': {'repo': 'xla',
                                              'path': 'xla/service/gpu/gpu_compiler.cc',
                                              'line': 2727,
                                              'needle': 'Invalid LLVM IR before optimizations:',
                                              'href': 'https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2727'},
                    'execute_sharded call': {'repo': 'jax',
                                             'path': 'jax/_src/interpreters/pxla.py',
                                             'line': 420,
                                             'needle': 'self.xla_executable.execute_sharded(input_bufs)',
                                             'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/pxla.py#L420'},
                    'PyLoadedExecutable ExecuteSharded': {'repo': 'jax',
                                                          'path': 'jaxlib/py_executable.cc',
                                                          'line': 484,
                                                          'needle': 'PyLoadedExecutable::ExecuteSharded(',
                                                          'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L484'},
                    'pallas_call_tpu_lowering_rule': {'repo': 'jax',
                                                      'path': 'jax/_src/pallas/mosaic/pallas_call_registration.py',
                                                      'line': 393,
                                                      'needle': 'def pallas_call_tpu_lowering_rule(',
                                                      'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393'},
                    'lower_jaxpr_to_pipelined_module': {'repo': 'jax',
                                                        'path': 'jax/_src/pallas/mosaic/lowering.py',
                                                        'line': 1008,
                                                        'needle': 'def lower_jaxpr_to_pipelined_module(',
                                                        'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008'},
                    'lower_module_to_custom_call': {'repo': 'jax',
                                                    'path': 'jax/_src/tpu_custom_call.py',
                                                    'line': 839,
                                                    'needle': 'def lower_module_to_custom_call(',
                                                    'href': 'https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839'}},
 'old_overview_sha256': 'd8bd03461dd2f5a3c4cbb89e808ad8040a0b497e2ec3570efcd1d42b5169caa5'}


class Diagram:
    def __init__(self, width, height, title, subtitle, metadata, dark=False):
        self.w, self.h, self.title = width, height, title
        self.meta = metadata
        self.dark = dark
        self.bg = "#101c2d" if dark else "#f5f7fb"
        self.card = "#192a40" if dark else "#ffffff"
        self.ink = "#eff5ff" if dark else "#17324d"
        self.muted = "#b4c4d9" if dark else "#536a80"
        self.colors = {"program": "#84b6ff" if dark else "#2866a4",
                       "data": "#65d9c4" if dark else "#087e78",
                       "transform": "#c9a7ff" if dark else "#8656ac",
                       "unknown": "#f1c37c" if dark else "#a66b21",
                       "neutral": "#99aabd" if dark else "#768899",
                       "result": "#be633c"}
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.boxes, self.text_boxes, self.edge_boxes = {}, [], []
        self.Pango, PangoCairo = pango()
        self.context = PangoCairo.FontMap.get_default().create_context()
        self.text(65, 88, title, 60, bold=True)
        self.text(68, 146, subtitle, 29, color=self.muted)
        x = 70
        for label, color in [("编译 / 程序", "program"), ("数据 / 执行", "data"),
                             ("变换", "transform"), ("编译结果返回", "result"),
                             ("待确认", "unknown"), ("结构 / 契约 / 观察", "neutral")]:
            dash = ' stroke-dasharray="8 7"' if color in {"result", "unknown", "neutral"} else ""
            self.labels.append(f'<path d="M{x} 211h48" stroke="{self.colors[color]}" stroke-width="4"{dash}/>')
            self.text(x + 64, 220, label, 25, color=self.muted)
            x += 340

    def measure(self, value, size, bold=False):
        Pango = self.Pango
        font = Pango.FontDescription()
        font.set_family(FONT_FAMILY)
        font.set_absolute_size(size * Pango.SCALE)
        font.set_weight(Pango.Weight.BOLD if bold else Pango.Weight.NORMAL)
        layout = Pango.Layout.new(self.context)
        layout.set_font_description(font)
        layout.set_text(value, -1)
        ink, logical = layout.get_pixel_extents()
        return ink, logical, layout.get_baseline() / Pango.SCALE

    def text(self, x, y, value, size=23, color=None, bold=False, href=None, owner=None, center=False):
        ink, logical, baseline = self.measure(value, size, bold)
        if center:
            x -= logical.width / 2
        text = (f'<text x="{x}" y="{y}" font-family="{FONT_FAMILY}" font-size="{size}" '
                f'font-weight="{700 if bold else 400}" fill="{color or self.ink}">{escape(value)}</text>')
        if href:
            target = '' if href.startswith('#') else ' target="_blank"'
            text = f'<a href="{escape(href, quote=True)}"{target}>{text}</a>'
        self.labels.append(text)
        self.text_boxes.append((x + ink.x, y - baseline + ink.y, ink.width, ink.height, value, owner))

    def wrap(self, value, width, size):
        lines, current = [], ""
        for char in value:
            if current and self.measure(current + char, size)[1].width > width:
                lines.append(current.rstrip())
                current = char.lstrip()
            else:
                current += char
        if current:
            lines.append(current)
        return lines

    def ref(self, name):
        return self.meta["source_anchors"][name]["href"]

    def panel(self, x, y, w, h, label, color="program"):
        fill = "#142337" if self.dark else "#edf2f8"
        self.base.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}"/>')
        self.text(x + 25, y + 43, label, 25, self.colors[color], bold=True)

    def node(self, key, x, y, w, h, tag, title, body=(), color="program", ref=None, link=None, dashed=False, pill=False):
        self.boxes[key] = (x, y, w, h)
        stroke = self.colors[color]
        dash = ' stroke-dasharray="9 7"' if dashed else ""
        self.nodes.append(f'<g id="{key}"><title>{escape(title)}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{h / 2 if pill else 18}" fill="{self.card}" stroke="{stroke}" stroke-width="2"{dash}/></g>')
        self.text(x + 24, y + 30, tag, 20, color=stroke, bold=True, owner=key)
        title_size = 36 if "核心概念" in tag else 32
        while self.measure(title, title_size, True)[1].width > w - 48:
            title_size -= 1
        self.text(x + 24, y + 71, title, title_size, bold=True, owner=key)
        baseline = y + 109
        for line in body:
            for part in self.wrap(line, w - 48, 24):
                self.text(x + 24, baseline, part, 24, color=self.muted, owner=key)
                baseline += 33
        if ref:
            self.text(x + 24, y + h - 20, link or ref, 18, color=stroke, href=self.ref(ref), owner=key)

    def edge(self, points, color="program", dashed=False, end=True, bridge_over=(), width=3):
        path = f"M{points[0][0]},{points[0][1]}"
        bridges = []
        for a, b in zip(points, points[1:]):
            crossings = sorted({crossing(a, b, u, v) for u, v in bridge_over} - {None},
                               key=lambda p: abs(p[0]-a[0])+abs(p[1]-a[1]))
            for cx, cy in crossings:
                if a[1] == b[1]:
                    dx = 12 if b[0] > a[0] else -12
                    path += f" L{cx-dx},{cy} Q{cx},{cy-24} {cx+dx},{cy}"
                    bridges.append(f"M{cx-dx},{cy} Q{cx},{cy-24} {cx+dx},{cy}")
                else:
                    dy = 12 if b[1] > a[1] else -12
                    path += f" L{cx},{cy-dy} Q{cx+24},{cy} {cx},{cy+dy}"
                    bridges.append(f"M{cx},{cy-dy} Q{cx+24},{cy} {cx},{cy+dy}")
            path += f" L{b[0]},{b[1]}"
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        marker = f' marker-end="url(#{color})"' if end else ""
        for bridge in bridges:
            self.edges.append(f'<path d="{bridge}" fill="none" stroke="{self.bg}" stroke-width="11" stroke-linecap="round"/>')
        self.edges.append(f'<path d="{path}" fill="none" stroke="{self.colors[color]}" '
                          f'stroke-width="{width}" stroke-linejoin="round"{dash}{marker}/>')
        self.edge_boxes.append(points)

    def label(self, x, y, value, color="program", ref=None, size=20):
        width = self.measure(value, size)[1].width
        self.labels.append(f'<rect x="{x-9}" y="{y-size-4}" width="{width+18}" height="{size+14}" rx="5" fill="{self.bg}"/>')
        self.text(x, y, value, size, self.colors[color], href=self.ref(ref) if ref else None)

    def save(self, path):
        errors = []
        for x, y, w, h, value, owner in self.text_boxes:
            if x < 0 or y < 0 or x + w > self.w or y + h > self.h:
                errors.append(f"outside canvas: {value}")
            if owner:
                bx, by, bw, bh = self.boxes[owner]
                if x < bx + 8 or y < by + 7 or x + w > bx + bw - 7 or y + h > by + bh - 7:
                    errors.append(f"outside {owner}: {value}")
        for i, a in enumerate(self.text_boxes):
            for b in self.text_boxes[i + 1:]:
                overlap_w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                overlap_h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if overlap_w > 2 and overlap_h > 2:
                    errors.append(f"text overlap: {a[4]} / {b[4]}")
        # Every orthogonal edge must remain outside unrelated node interiors.
        for points in self.edge_boxes:
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                for key, (x, y, w, h) in self.boxes.items():
                    if x1 == x2 and x + 3 < x1 < x + w - 3:
                        if min(max(y1, y2), y + h - 3) - max(min(y1, y2), y + 3) > 1:
                            errors.append(f"edge crosses node: {key}")
                    if y1 == y2 and y + 3 < y1 < y + h - 3:
                        if min(max(x1, x2), x + w - 3) - max(min(x1, x2), x + 3) > 1:
                            errors.append(f"edge crosses node: {key}")
        if errors:
            raise ValueError(str(path.name) + ":\n" + "\n".join(errors))
        defs = "".join(f'<marker id="{name}" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M1,1 L11,6 L1,11" fill="none" stroke="{color}" stroke-width="2"/></marker>' for name, color in self.colors.items())
        xml = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="title desc">'
               f'<title id="title">{escape(self.title)}</title><desc id="desc">{escape(getattr(self, "description", "JAX 到硬件的软件栈流程图。编译路径和数据路径在执行处汇合；包含定义、产生、变换、消费以及固定提交源码链接。"))}</desc>'
               f'<metadata>{escape(json.dumps(self.meta, ensure_ascii=False))}</metadata><defs>{defs}</defs>'
               '<style>a:hover text{text-decoration:underline}g:target rect{stroke:#c65028;stroke-width:6}</style>'
               f'<rect width="100%" height="100%" fill="{self.bg}"/>' + "".join(self.base + self.edges + self.nodes + self.labels) + '</svg>\n')
        path.write_text(xml)
        ET.parse(path)
        print(f"{path.name}: {len(self.boxes)} nodes, {len(self.text_boxes)} labels; layout checks passed")


W = 2140
HEIGHTS = {"jax": 1660, "jaxlib": 890, "ifrt": 890, "pjrt": 890,
           "xla": 890, "backend": 1160, "asm": 560, "runtime": 890,
           "hardware": 630}
TITLES = {
    "jax": "JAX · 函数追踪、程序变换与 lowering",
    "jaxlib": "jaxlib · Python 与原生编译 / 运行时绑定",
    "ifrt": "IFRT · 面向框架的逻辑数组与程序",
    "pjrt": "PJRT · 设备、存储与可执行程序接口",
    "xla": "XLA · HLO 优化与编译规划",
    "backend": "编译后端 · 按目标设备分流",
    "asm": "目标汇编表示 · ASM 与目标代码的观察关系",
    "runtime": "设备运行时与驱动 · 程序和数据在此汇合",
    "hardware": "指令集接口与硬件 · ISA 约束执行行为",
}

ROLES = {"jax": "前端实现", "jaxlib": "原生绑定", "ifrt": "框架运行时接口",
         "pjrt": "设备运行时接口", "xla": "编译器子系统", "backend": "目标后端",
         "asm": "可观察表示", "runtime": "运行时实现", "hardware": "硬件 / 指令契约"}
DETAILS = {"jax": "../jax/jaxpr-centered-hub.svg", "jaxlib": "../jaxlib/mlir-module-centered-hub.svg",
           "ifrt": "../ifrt/array-centered-hub.svg", "pjrt": "../pjrt/buffer-centered-hub.svg",
           "xla": "../xla/hlo-centered-hub.svg", "runtime": "../stream-executor/submission-centered-hub.svg"}


def crossing(a, b, u, v):
    """An interior perpendicular crossing; touching endpoints are not junctions."""
    if a[1] == b[1] and u[0] == v[0]:
        if min(a[0], b[0]) < u[0] < max(a[0], b[0]) and min(u[1], v[1]) < a[1] < max(u[1], v[1]):
            return u[0], a[1]
    if a[0] == b[0] and u[1] == v[1]:
        if min(a[1], b[1]) < u[1] < max(a[1], b[1]) and min(u[0], v[0]) < a[0] < max(u[0], v[0]):
            return a[0], u[1]
    return None


def source_metadata(root, allow_missing=False):
    meta = copy.deepcopy(SOURCE_METADATA)
    additions = [
        ("JaxprEqn", "jax", "jax/_src/core.py", 460, "class JaxprEqn:"),
        ("Var", "jax", "jax/_src/core.py", 539, "class Var:"),
        ("Primitive", "jax", "jax/_src/core.py", 670, "class Primitive:"),
        ("DynamicJaxprTrace", "jax", "jax/_src/interpreters/partial_eval.py", 1623, "class DynamicJaxprTrace("),
        ("PyArray", "jax", "jaxlib/py_array.h", 141, "class PyArray :"),
        ("PyLoadedExecutable", "jax", "jaxlib/py_executable.h", 185, "class PyLoadedExecutable {"),
        ("TPU custom_call payload", "jax", "jax/_src/tpu_custom_call.py", 461, "call = mlir.custom_call("),
        ("Shardy module pass", "jax", "jax/_src/interpreters/mlir.py", 1483, "'builtin.module(sdy-lift-inlined-meshes)'"),
        ("PjRtMemorySpace", "xla", "xla/pjrt/pjrt_client.h", 89, "class PjRtMemorySpace {"),
        ("PjRtDevice", "xla", "xla/pjrt/pjrt_client.h", 148, "class PjRtDevice {"),
        ("PjRtClient", "xla", "xla/pjrt/pjrt_client.h", 546, "class PjRtClient {"),
        ("BufferAssignment", "xla", "xla/service/buffer_assignment.h", 476, "class BufferAssignment {"),
    ]
    for name, repo, path, line, needle in additions:
        host = "0xaskr/jax" if repo == "jax" else "openxla/xla"
        meta["source_anchors"][name] = {
            "repo": repo, "path": path, "line": line, "needle": needle,
            "href": f'https://github.com/{host}/blob/{meta["source_pins"][repo]}/{path}#L{line}',
        }
    locks = {parts[1]: parts[3] for line in (root / "upstream-sources.lock").read_text().splitlines()
             if line and not line.startswith("#") and len(parts := line.split("|")) >= 4}
    checked = set()
    missing = []
    for repo, pin in meta["source_pins"].items():
        checkout = root / "upstream" / repo
        if locks.get(f"upstream/{repo}") != pin:
            raise ValueError(f"Source pin mismatch: {repo}")
        if not (checkout / ".git").exists():
            if not allow_missing:
                raise ValueError(f"Missing source checkout: {repo}; --allow-missing-sources retains its unverified pinned anchors")
            missing.append(repo)
            continue
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != pin:
            raise ValueError(f"Source pin mismatch: {repo}")
    for anchor in meta["source_anchors"].values():
        repo, path = anchor["repo"], anchor["path"]
        if repo in missing:
            continue
        checkout = root / "upstream" / repo
        committed = subprocess.check_output(["git", "-C", str(checkout), "show", f'{meta["source_pins"][repo]}:{path}'], text=True)
        if (repo, path) not in checked:
            local = checkout / path
            if local.exists() and committed != local.read_text():
                raise ValueError(f"Source file modified: {repo}/{path}")
            checked.add((repo, path))
        if anchor["needle"] not in committed.splitlines()[anchor["line"] - 1]:
            raise ValueError(f"Source anchor moved: {anchor}")
    meta["source_verification"] = {"checked_repositories": sorted(set(meta["source_pins"]) - set(missing)),
                                   "missing_repositories": missing}
    if missing:
        print("Source check incomplete; retained pinned anchors for missing checkouts: " + ", ".join(missing))
    meta["evidence_scope"] = "固定源码与图示生成；无 CPU/GPU/TPU 执行或性能验证"
    meta["component_model"] = "组件大框 + 核心抽象 + 内部关系 + 编译/数据/执行流程"
    return meta


class ComponentDiagram(Diagram):
    def __init__(self, width, height, title, subtitle, meta):
        super().__init__(width, height, title, subtitle, meta)
        self.bg = "#fafbf8"
        self.ink = "#203344"
        self.muted = "#506577"
        self.colors.update(program="#2866a4", data="#087e78", transform="#8656ac",
                           unknown="#a66b21", neutral="#768899", result="#be633c")
        self.components = {}
        self.members = {}
        self.connections = []
        self.text(65, 278, "实线表示流程；虚线关系按颜色与文字区分：按需分支、返回、观察或待确认。跨线拱桥表示互不连接。", 25, self.muted)
        self.text(65, 324, "大框右上标明角色；强调色节点是核心概念。点击组件标题进入详细图，点击编号端口定位框内节点。", 25, self.muted)

    def component(self, name, x, y):
        h = HEIGHTS[name]
        color = "data" if name in {"ifrt", "pjrt", "runtime"} else "neutral" if name in {"asm", "hardware"} else "program"
        self.components[name] = (x, y, W, h)
        fill = "#f5f4ef" if name in {"asm", "hardware"} else "#f1f5f7"
        self.base.append(f'<g id="component_{name}"><rect x="{x}" y="{y}" width="{W}" height="{h}" rx="24" fill="{fill}" stroke="#a8bbca" stroke-width="3"/></g>')
        self.base.append(f'<path d="M{x+26},{y+103}H{x+W-26}" stroke="#d1dce3" stroke-width="2"/>')
        self.text(x+32, y+57, TITLES[name] + (" ↗" if name in DETAILS else ""), 40,
                  self.colors[color], bold=True, href=DETAILS.get(name))
        role = ROLES[name]
        rw = self.measure(role, 23, True)[1].width + 32
        self.base.append(f'<rect x="{x+W-rw-28}" y="{y+21}" width="{rw}" height="43" rx="10" fill="#e3e9ec"/>')
        self.text(x+W-rw-12, y+51, role, 23, self.colors[color], bold=True)
        self.text(x+34, y+88, {
            "jax": "主抽象 Jaxpr · 描述计算与效果；运行时输入数据走独立通路",
            "jaxlib": "主表示 MLIR Module（StableHLO）· 绑定与包装不等于一个独立 IR 方言",
            "ifrt": "主抽象 Array · 一个逻辑数组可跨多个设备；编译、加载和执行是不同动作",
            "pjrt": "主抽象 Buffer · 本图使用 IFRT 的 PJRT-backed 实现；接口由具体 provider 实现",
            "xla": "主抽象 HLO · 模块、计算和指令组成程序图；passes 改写图及其编译约束",
            "backend": "归属：CPU / GPU 为 XLA 后端；TPU 经 provider / libtpu · 下方按目标展开",
            "asm": "主抽象 ASM · 汇编文本是代码表示 / 观察形式，不是所有编译调用的强制中间产物",
            "runtime": "主抽象 执行提交 · 已加载程序 + 输入 Buffer + 执行选项 / 依赖",
            "hardware": "主抽象 ISA · 指令和语义契约；设备按目标程序执行并读写存储",
        }[name], 23, self.muted)

    def item(self, comp, name, col, row, tag, title, body=(), color="program", ref=None, core=False, dashed=False, h=235):
        x, y, _, _ = self.components[comp]
        x += 60 + col * 700
        y += 155 + row * 370
        key = f"{comp}_{name}"
        self.node(key, x, y, 580, h, ("核心概念 · " if core else "") + tag, title, body, color, ref,
                  link="源码 · " + ref if ref else None, dashed=dashed)
        self.members[key] = comp
        if core:
            fills = {"program": "#e6f0fd", "data": "#e0f4ed", "unknown": "#fff1d9"}
            self.nodes[-1] = self.nodes[-1].replace('fill="#ffffff"', f'fill="{fills.get(color, "#f1eafa")}"').replace('stroke-width="2"', 'stroke-width="4"')
        return key

    def connect(self, comp, a, b, points, label=None, label_at=None, color="program", dashed=False):
        x, y, _, _ = self.components[comp]
        translated = [(x+px, y+py) for px, py in points]
        self.edge(translated, color, dashed)
        if label:
            lx, ly = label_at
            self.label(x+lx, y+ly, label, color, size=21)

    def note(self, comp, text, y=None):
        x, top, _, h = self.components[comp]
        self.text(x+34, top+(y or h-35), text, 23, self.muted)

    def reading_guide(self):
        self.panel(180, 500, 1030, 435, "阅读导航 · 点击定位组件", "neutral")
        for i, (label, target) in enumerate([
            ("前端：Jaxpr → 外层 MLIR Module", "jax"),
            ("接口：IFRT Array / PJRT Buffer", "ifrt"),
            ("编译：HLO → CPU / GPU / TPU 后端", "xla"),
            ("执行：已加载程序 × 输入数据", "runtime"),
        ]):
            self.text(216, 618+i*76, label, 30, self.colors["program"], href=f"#component_{target}")
        self.panel(180, 1030, 1030, 620, "MLIR · 跨层基础设施", "neutral")
        lines = ["对象：Module、Operation、Region、Type",
                 "JAX lowering 借助 jaxlib 绑定构造操作。",
                 "StableHLO、sdy、Mosaic 等方言规定语义。",
                 "模块可包含多个方言，并经过验证与 pass。",
                 "相应后端路径继续使用 MLIR 方言。"]
        for i, line in enumerate(lines):
            self.text(216, 1160+i*70, line, 27, self.muted)
        self.text(216, 1550, "作用范围跨越前端构造、绑定与编译处理。", 27, self.muted)
        self.text(216, 1610, "查看 Module / 方言详细图 ↗", 25, self.colors["program"], href=DETAILS["jaxlib"])

    def save(self, path):
        for key, comp in self.members.items():
            x, y, w, h = self.boxes[key]
            cx, cy, cw, ch = self.components[comp]
            if not (cx < x and cy+103 < y and x+w < cx+cw and y+h < cy+ch):
                raise ValueError(f"Node outside component: {key}")
        super().save(path)


def populate(d, positions):
    for comp, (x, y) in positions.items():
        d.component(comp, x, y)
    n, e = d.item, d.connect

    n("jax", "function", 0, 0, "输入", "Python 函数 + 抽象参数", ["jit / grad / vmap 可组合", "shape / dtype 等参与追踪"], ref="trace_to_jaxpr_dynamic")
    n("jax", "trace", 1, 0, "产生", "Trace / Tracer", ["Tracer 携带 aval；Trace 解释操作", "DynamicJaxprTrace 收集方程"], ref="DynamicJaxprTrace")
    n("jax", "input", 2, 0, "数据通路", "主机值 / jax.Array", ["本次执行使用的动态数组实参", "经放置 / 包装，或复用已有设备数组"], "data")
    n("jax", "values", 0, 1, "值与约束", "Var / Literal / aval", ["变量引用或字面值；抽象类型", "constvars 与附带的 consts 对应"], ref="Var")
    n("jax", "jaxpr", 1, 1, "程序图", "Jaxpr", ["invars / outvars / eqns / consts", "effects 与 DebugInfo 描述语义 / 来源"], ref="Jaxpr", core=True)
    n("jax", "eqn", 2, 1, "方程与操作语义", "JaxprEqn / Primitive", ["输入 → primitive(params) → 输出", "原语拥有 AD / batching / lowering 规则"], ref="JaxprEqn")
    n("jax", "transform", 0, 2, "变换", "Jaxpr → Jaxpr", ["部分求值、DCE 等重写程序", "消费原图，产生变换后的 Jaxpr"], "transform", "dce_jaxpr")
    n("jax", "lower", 1, 2, "消费", "MLIR lowering 规则", ["按 primitive 选择规则与上下文", "将方程序列生成为模块中的操作"], ref="lower_jaxpr_to_module")
    n("jax", "pallas", 2, 2, "可选 KERNEL 分支", "Pallas / Mosaic TPU MLIR", ["kernel Jaxpr → Mosaic TPU MLIR", "序列化后装入外层 custom_call"], ref="lower_module_to_custom_call", dashed=True)
    n("jax", "shardy", 0, 3, "共享分片系统 · 本阶段用法", "Shardy / sdy", ["此处：sdy-lift-inlined-meshes", "启用时整理 Module 内联 mesh"], "transform", "Shardy module pass", dashed=True)
    n("jax", "module", 1, 3, "构造产物 · 编译表示", "外层 MLIR Module", ["普通张量操作与 custom_call 共存", "lowering 返回，交由绑定层传递"], ref="lower_jaxpr_to_module")
    n("jax", "custom_call", 2, 3, "外层程序操作", "stablehlo.custom_call", ["target = tpu_custom_call", "backend_config 携带 Mosaic payload"], ref="TPU custom_call payload", dashed=True)
    e("jax", "function", "trace", [(640,270),(760,270)], "追踪", (658,243))
    e("jax", "trace", "jaxpr", [(1050,390),(1050,525)], "构建", (1080,465))
    e("jax", "jaxpr", "values", [(760,625),(640,625)], "引用", (658,600), "neutral", True)
    e("jax", "jaxpr", "eqn", [(1340,625),(1460,625)], "包含", (1358,600), "neutral", True)
    e("jax", "jaxpr", "lower", [(1050,760),(1050,895)], "逐方程消费", (1080,833))
    e("jax", "jaxpr", "transform", [(840,760),(840,815),(350,815),(350,895)], color="transform")
    e("jax", "transform", "jaxpr", [(640,1010),(700,1010),(700,720),(760,720)], color="transform")
    e("jax", "jaxpr", "pallas", [(1250,760),(1250,815),(1750,815),(1750,895)], "kernel Jaxpr（可选）", (1470,841), dashed=True)
    e("jax", "lower", "module", [(1050,1130),(1050,1265)], "构造", (1080,1200))
    e("jax", "pallas", "custom_call", [(1750,1130),(1750,1265)], "序列化为 payload", (1780,1200), dashed=True)
    e("jax", "custom_call", "module", [(1460,1380),(1340,1380)], "装入", (1360,1350), dashed=True)
    e("jax", "module", "shardy", [(760,1340),(640,1340)], "按需", (660,1310), "transform", True)
    e("jax", "shardy", "module", [(640,1450),(760,1450)], "更新", (660,1420), "transform", True)
    d.note("jax", "Pallas 分支在外层 Module 汇合：custom_call 携带 Mosaic TPU MLIR；Mosaic TPU MLIR ≠ LLO。", y=1568)
    d.note("jax", "Shardy 处理分片信息；此处展示 JAX lowering 中启用时运行的模块 pass。一个 primitive 不对应一条硬件指令。", y=1618)

    n("jaxlib", "mlir", 0, 0, "IR 基础设施", "Operation / Region / Type", ["MLIR 操作、区域、块与 SSA 值", "方言为操作和类型规定具体语义"], ref="register_lowering")
    n("jaxlib", "module", 1, 0, "编译输入表示", "MLIR Module（StableHLO）", ["承载张量计算及形状 / 类型", "也可含 sdy、func 或 custom_call"], ref="StableHLO binding", core=True)
    n("jaxlib", "array", 2, 0, "数据包装", "PyArray", ["把 Python 数组对象连接到 IFRT", "持有逻辑数组引用和相关元数据"], "data", "PyArray")
    n("jaxlib", "client", 0, 1, "编译绑定", "PyClient", ["克隆模块并构造 IFRT HloProgram", "CompileAndLoad 提交编译请求"], ref="PyClient CompileAndLoad")
    n("jaxlib", "executable", 1, 1, "返回的程序包装", "PyLoadedExecutable", ["包装 IFRT 已加载程序", "向 Python 提供执行与查询接口"], ref="PyLoadedExecutable")
    n("jaxlib", "execute", 2, 1, "执行绑定", "ExecuteSharded", ["解包输入数组并调用 IFRT Execute", "将输出包装回 Python 数组"], "data", "PyLoadedExecutable ExecuteSharded")
    e("jaxlib", "mlir", "module", [(640,270),(760,270)], "组成", (658,243), "neutral", True)
    e("jaxlib", "module", "client", [(1050,390),(1050,450),(350,450),(350,525)], "构造编译请求", (470,436))
    e("jaxlib", "client", "executable", [(640,650),(760,650)], "返回", (658,623), "result", True)
    e("jaxlib", "array", "execute", [(1750,390),(1750,525)], "输入数组", (1780,465), "data")
    e("jaxlib", "executable", "execute", [(1340,650),(1460,650)], "程序", (1358,623), "data")
    d.note("jaxlib", "Module 是绑定构造和传递的程序表示；StableHLO 是方言，jaxlib 是实现组件，两者不是同类软件层。")

    n("ifrt", "program", 0, 0, "编译输入", "HloProgram", ["本路径持有外层 MLIR Module", "名称不表示已经变成 HloModule"], ref="PyClient CompileAndLoad")
    n("ifrt", "compiler", 1, 0, "编译服务", "Client / Compiler", ["默认编译器接收 Program 与选项", "PJRT 实现向设备 provider 提交"], ref="IFRT CompileAndLoad")
    n("ifrt", "array", 2, 0, "逻辑数据", "Array", ["dtype / shape / sharding / layout", "一个逻辑数组可以跨多个设备", "产生入口：MakeArrayFromHostBuffer"], "data", "ifrt Array", core=True)
    n("ifrt", "executable", 0, 1, "消费数组的程序", "LoadedExecutable", ["Execute(ArrayRef...) 消费逻辑数组", "ExecuteResult：outputs + status"], "data", "IFRT PjRtLoadedExecutable Execute")
    n("ifrt", "sharding", 1, 1, "数组的放置约束", "Sharding / DeviceList", ["描述数组如何分布与放置", "PJRT-backed Array 管理多个 Buffer"], "data", "ifrt Array")
    n("ifrt", "copy", 2, 1, "按需变换", "CopyArrays / RemapArrays", ["复制数组或重映射已有分片", "不要求每次执行都经过这些操作"], "transform", "RemapArrays")
    e("ifrt", "program", "compiler", [(640,270),(760,270)], "提交", (658,243))
    e("ifrt", "compiler", "executable", [(1050,390),(1050,440),(350,440),(350,525)], "编译并加载后返回", (430,425), "result", True)
    e("ifrt", "sharding", "array", [(1250,525),(1250,475),(1650,475),(1650,390)], "约束放置", (1345,460), "neutral", True)
    e("ifrt", "array", "copy", [(1900,390),(1900,525)], "按需", (1930,465), "transform", True)
    e("ifrt", "copy", "array", [(2040,650),(2080,650),(2080,270),(2040,270)], color="transform", dashed=True)
    e("ifrt", "array", "executable", [(1460,340),(1395,340),(1395,805),(350,805),(350,760)], "Execute 消费 Array；输出仍为 Array", (560,830), "data")
    d.note("ifrt", "Array → Buffer 是实现与存储映射；执行结果对象与异步完成状态分开表达。", y=866)

    n("pjrt", "client", 0, 0, "设备后端入口", "PjRtClient", ["设备查询、编译 / 加载、数据创建", "接收 Module 与 CompileOptions"], ref="PjRtClient")
    n("pjrt", "executable", 1, 0, "消费 Buffer 的程序", "PjRtLoadedExecutable", ["Execute 接收各设备的 Buffer 列表", "返回输出 Buffer 与完成状态"], ref="PjRtLoadedExecutable")
    n("pjrt", "buffer", 2, 0, "设备数据", "PjRtBuffer", ["数据位于某设备 / 内存空间", "生命周期、所有权和就绪状态", "产生入口：BufferFromHostBuffer"], "data", "PjRtBuffer", core=True)
    n("pjrt", "provider", 0, 1, "具体实现 / 插件边界", "Backend provider", ["实现 CompileAndLoad / Execute", "可经 PJRT C API 接入设备后端"], ref="PJRT CompileAndLoad")
    n("pjrt", "memory", 1, 1, "放置与存储归属", "Device / MemorySpace", ["Buffer 关联设备和内存空间", "运行时分配 / 管理实际存储"], "data", "PjRtMemorySpace")
    n("pjrt", "copy", 2, 1, "按需复制与依赖", "Copy / Ready Future", ["CopyToMemorySpace 返回新 Buffer", "GetReadyFuture 表达数据就绪"], "transform", "CopyToMemorySpace")
    e("pjrt", "client", "provider", [(350,390),(350,525)], "委托实现", (380,465))
    e("pjrt", "provider", "executable", [(600,525),(600,445),(1050,445),(1050,390)], "编译 / 加载后返回", (720,430), "result", True)
    e("pjrt", "buffer", "executable", [(1460,270),(1340,270)], "实参", (1358,243), "data")
    e("pjrt", "memory", "buffer", [(1250,525),(1250,475),(1650,475),(1650,390)], "归属", (1360,460), "neutral", True)
    e("pjrt", "buffer", "copy", [(1900,390),(1900,525)], "按需", (1930,465), "transform", True)
    e("pjrt", "copy", "buffer", [(2040,650),(2080,650),(2080,270),(2040,270)], color="transform", dashed=True)
    d.note("pjrt", "BufferFromHostBuffer 创建设备数据；Execute 是执行调用，不是另一种编译 IR。")

    n("xla", "import", 0, 0, "产生", "MLIR → HLO 导入", ["准备 / 合法化编译输入", "ConvertMlirHloToHloModule"], ref="ConvertMlirHloToHloModule")
    n("xla", "hlo", 1, 0, "优化程序图", "HLO / HloModule", ["持有入口计算和嵌套 computations", "形状、布局、分片与配置约束"], ref="HloModule", core=True)
    n("xla", "structure", 2, 0, "组成", "Computation / Instruction", ["computation 包含指令及根结果", "operand / user 关系连接数据依赖"], ref="HloModule")
    n("xla", "passes", 0, 1, "变换", "HloPassPipeline", ["简化、融合、分片、布局等 passes", "具体启用项与顺序由后端配置决定"], "transform", "HloPassPipeline RunImpl")
    n("xla", "plan", 1, 1, "编译期规划", "Schedule / BufferAssignment", ["安排 HLO 次序与内存复用计划", "计划供代码生成与运行时使用"], ref="BufferAssignment")
    n("xla", "output", 2, 1, "后端结果", "Executable + 执行计划", ["目标代码、常量、内存 / 调度信息", "经 provider 加载为运行时程序"], ref="CpuCompiler CompileCpuExecutable")
    e("xla", "import", "hlo", [(640,270),(760,270)], "产生", (658,243))
    e("xla", "hlo", "structure", [(1340,270),(1460,270)], "包含", (1358,243), "neutral", True)
    e("xla", "hlo", "passes", [(850,390),(850,445),(350,445),(350,525)], "改写程序", (470,430), "transform")
    e("xla", "passes", "hlo", [(640,650),(700,650),(700,340),(760,340)], color="transform")
    e("xla", "hlo", "plan", [(1150,390),(1150,525)], "后端规划", (1180,465))
    e("xla", "plan", "output", [(1340,650),(1460,650)], "参与", (1358,623))
    d.note("xla", "HLO 调度 ≠ 最终机器指令调度；BufferAssignment 是编译期内存计划，实际分配由运行时完成。")

    bx, by, _, _ = d.components["backend"]
    for i, target in enumerate(["CPU", "GPU", "TPU"]):
        x = bx + 30 + i*700
        d.base.append(f'<rect x="{x}" y="{by+127}" width="680" height="974" rx="17" fill="#ffffff" stroke="#c4cfd7" stroke-width="2"/>')
        detail = {"CPU": "../xla/cpu-llvm-ir-centered-hub.svg", "GPU": "../xla/gpu-ir-centered-hub.svg",
                  "TPU": "../libtpu/llo-boundary-centered-hub.svg"}[target]
        heading = ("XLA / " if target != "TPU" else "provider / ") + target + " 后端 ↗"
        d.text(x+23, by+164, heading, 29, d.colors["unknown" if i == 2 else "program"], bold=True, href=detail)
    # Backend cards use a slightly lower row origin to leave a component heading.
    rows = [
        ("CPU", "HLO / emitter", ["HLO + 调度 / 内存计划", "IrEmitter 构造函数与 kernel"], "LLVM IR", ["Module / Function / BasicBlock", "Instruction 表达低层计算"], "CPU 目标代码", ["LLVM 优化与目标代码生成", "链接代码并集成执行计划"], "CpuCompiler CompileCpuExecutable"),
        ("GPU", "原生 / Triton / 库路径", ["融合计算选择适用的 emitter", "部分计算使用设备库调用"], "LLVM IR / Triton IR", ["不同代码生成路径，不强制串行", "优化后生成目标设备代码"], "GPU 代码 + 库调用", ["NVIDIA 示例：PTX → cubin", "kernel / 库调用纳入执行计划"], "GpuCompiler RunBackend"),
        ("TPU", "TPU provider / libtpu", ["处理外层程序与可选 kernel payload", "公开宿主代码可核对提交边界"], "LLO（待版本证据）", ["内部结构、passes 与转换位置", "不能从 Mosaic TPU MLIR 推定"], "TPU 设备程序", ["具体低层表示和代码细节待确认", "生成 / 加载可执行设备程序"], None),
    ]
    for col, (target, t0, b0, t1, b1, t2, b2, ref) in enumerate(rows):
        color = "unknown" if target == "TPU" else "program"
        for row, (title, body, core) in enumerate([(t0,b0,False),(t1,b1,True),(t2,b2,False)]):
            key = f"backend_{target}_{row}"
            x, y = bx+60+700*col, by+195+290*row
            d.node(key,x,y,580,225,"核心概念" if core else ("输入与选择" if row == 0 else "消费 / 产物"),title,body,color,ref,link="源码 · RunBackend / codegen" if ref else None,dashed=target=="TPU")
            d.members[key]="backend"
            if core:
                d.nodes[-1]=d.nodes[-1].replace('fill="#ffffff"', 'fill="#fff1d9"' if target=="TPU" else 'fill="#e6f0fd"').replace('stroke-width="2"','stroke-width="4"')
        for y in [420,710]:
            d.edge([(bx+350+700*col,by+y),(bx+350+700*col,by+y+65)],color,dashed=target=="TPU")
    d.edge([(bx+1340,by+300),(bx+1380,by+300),(bx+1380,by+885),(bx+1340,by+885)],"program",dashed=True)
    d.label(bx+815,by+1060,"GPU 设备库调用走右侧旁路",size=19)
    d.note("backend", "多操作可融合，单操作可分解；GPU 也可调用库；TPU 虚线框表示证据边界，不枚举未确认的内部阶段。")

    n("asm", "code", 0, 0, "已生成的代码", "目标代码 / 对象文件", ["CPU 目标码或 GPU cubin 等", "编译流程未必保留汇编文本"])
    n("asm", "asm", 1, 0, "可观察表示", "ASM", ["汇编助记符、寄存器与地址表达", "可由代码生成或反汇编获得"], "program", core=True)
    n("asm", "inspect", 2, 0, "消费", "汇编器 / 检查工具", ["文本可被汇编为目标代码", "也可供人工阅读和诊断"])
    e("asm", "code", "asm", [(640,270),(760,270)], "观察", (658,243), "neutral", True)
    e("asm", "asm", "inspect", [(1340,270),(1460,270)], "读取", (1358,243), "neutral", True)
    d.note("asm", "ASM 是表示层，不是独立运行时组件；ISA 是语义契约，不画成 ASM → ISA 的强制转换步骤。", y=461)
    d.note("asm", "NVIDIA 的 PTX 是虚拟指令集表示，SASS 是物理 GPU 指令的汇编表示；具体可见形式随后端变化。", y=506)

    n("runtime", "program", 0, 0, "编译通路交付", "已加载程序", ["代码 / 常量 / 内存与执行计划", "IFRT / PJRT LoadedExecutable"], ref="PjRtLoadedExecutable")
    n("runtime", "execute", 1, 0, "程序 × 数据", "执行提交", ["Execute + Buffer 参数 + 选项", "由 provider 实现具体执行行为"], "data", "IFRT PjRtLoadedExecutable Execute", core=True)
    n("runtime", "buffers", 2, 0, "数据通路交付", "输入 Buffer + 依赖", ["设备实参、存储与就绪条件", "可能复用已有数据或按需复制"], "data", "GetReadyFuture")
    n("runtime", "dispatch", 0, 1, "执行机制", "后端运行时 / 驱动", ["CPU 任务或设备 kernel / 库调用", "具体队列与调度机制随后端变化"], "data")
    n("runtime", "status", 1, 1, "完成与错误", "Future / 结果状态", ["表达任务或数据稍后就绪", "输出句柄可先于设备完成返回"], "data", "GetReadyFuture")
    n("runtime", "output", 2, 1, "输出", "Buffer → Array → PyArray", ["输出存储包装回逻辑数组", "调用者按需等待或取回主机"], "data", "IFRT PjRtLoadedExecutable Execute")
    e("runtime", "program", "execute", [(640,270),(760,270)], "程序", (658,243))
    e("runtime", "buffers", "execute", [(1460,270),(1340,270)], "实参", (1358,243), "data")
    e("runtime", "execute", "dispatch", [(900,390),(900,445),(350,445),(350,525)], "提交工作", (480,430), "data")
    e("runtime", "dispatch", "status", [(640,650),(760,650)], "报告", (658,623), "data")
    e("runtime", "execute", "output", [(1250,390),(1250,475),(1750,475),(1750,525)], "返回输出句柄", (1430,460), "data")
    e("runtime", "status", "output", [(1340,650),(1460,650)], "就绪", (1358,623), "data")
    d.note("runtime", "框内展开上层 Execute 的设备侧行为；返回数组句柄 ≠ 已完成设备计算，完成和错误通过状态传递。")

    n("hardware", "isa", 0, 0, "指令语义契约", "ISA", ["规定指令、操作数和执行语义", "目标程序必须符合设备契约"], "program", core=True)
    n("hardware", "units", 1, 0, "执行", "CPU / GPU / TPU 执行单元", ["执行目标程序中的设备指令", "完成计算、读写与通信"], "data")
    n("hardware", "memory", 2, 0, "状态变化", "寄存器 / 内存 / 互连", ["承载输入、临时值与输出数据", "设备结果由运行时报告完成"], "data")
    e("hardware", "isa", "units", [(640,270),(760,270)], "约束", (658,243), "neutral", True)
    e("hardware", "units", "memory", [(1340,235),(1460,235)], "读写", (1358,210), "data")
    e("hardware", "memory", "units", [(1460,340),(1340,340)], color="data")
    d.note("hardware", "此框表示软硬件接口与执行关系；不把一个 Jaxpr primitive、HLO 指令或 kernel 与一条硬件指令对应。", y=482)
    d.note("hardware", "证据范围：源码与接口关系图；源码复核范围见页脚。无 CPU/GPU/TPU 执行、数值或性能验证。", y=535)


def port(box, side, fraction):
    x,y,w,h=box
    if side=="top": return (x+w*fraction,y),(0,-1)
    if side=="bottom": return (x+w*fraction,y+h),(0,1)
    if side=="left": return (x,y+h*fraction),(-1,0)
    return (x+w,y+h*fraction),(1,0)


def route(d, source, target, color, label, used):
    """Route inter-component edges only through gutters, never through a component."""
    a,sa,fa=source; b,sb,fb=target
    p,va=port(d.components[a],sa,fa); q,vb=port(d.components[b],sb,fb)
    clearance = 110 if color == "result" else 50
    start=(p[0]+va[0]*clearance,p[1]+va[1]*clearance)
    end=(q[0]+vb[0]*clearance,q[1]+vb[1]*clearance)
    obstacles=list(d.components.values())
    xs={start[0],end[0],40,d.w-40}; ys={start[1],end[1],280}
    for x,y,w,h in obstacles:
        xs.update([x-50,x-110,x+w+50,x+w+110])
        ys.update([y-50,y-110,y+h+50,y+h+110])
    xs=sorted(v for v in xs if 20 <= v <= d.w-20)
    ys=sorted(v for v in ys if v >= 280)
    def blocked(a,b):
        for x,y,w,h in obstacles:
            if a[0]==b[0] and x-24<a[0]<x+w+24 and min(max(a[1],b[1]),y+h+24)>max(min(a[1],b[1]),y-24): return True
            if a[1]==b[1] and y-24<a[1]<y+h+24 and min(max(a[0],b[0]),x+w+24)>max(min(a[0],b[0]),x-24): return True
        return False
    si=(xs.index(start[0]),ys.index(start[1]),-1)
    queue=[(0,0,si)]; costs={si:0}; prev={}
    found=None
    while queue:
        _,cost,state=heapq.heappop(queue)
        if cost!=costs.get(state): continue
        i,j,di=state; cur=(xs[i],ys[j])
        if cur==end: found=state; break
        for ni,nj,nd in [(i-1,j,0),(i+1,j,0),(i,j-1,1),(i,j+1,1)]:
            if not(0<=ni<len(xs) and 0<=nj<len(ys)): continue
            nxt=(xs[ni],ys[nj])
            if blocked(cur,nxt): continue
            overlap=0
            for (u,v),count in used.items():
                if cur[0]==nxt[0]==u[0]==v[0]:
                    overlap += max(0,min(max(cur[1],nxt[1]),max(u[1],v[1]))-max(min(cur[1],nxt[1]),min(u[1],v[1])))*count
                if cur[1]==nxt[1]==u[1]==v[1]:
                    overlap += max(0,min(max(cur[0],nxt[0]),max(u[0],v[0]))-max(min(cur[0],nxt[0]),min(u[0],v[0])))*count
            nc=cost+abs(nxt[0]-cur[0])+abs(nxt[1]-cur[1])+(90 if di not in {-1,nd} else 0)+overlap*4
            ns=(ni,nj,nd)
            if nc<costs.get(ns,float("inf")):
                costs[ns]=nc; prev[ns]=state
                heapq.heappush(queue,(nc+abs(nxt[0]-end[0])+abs(nxt[1]-end[1]),nc,ns))
    if found is None: raise ValueError(f"No route {a} -> {b}")
    points=[]; state=found
    while state!=si:
        points.append((xs[state[0]],ys[state[1]])); state=prev[state]
    points.append(start); points.reverse()
    full=[p]+points+[q]; clean=[full[0]]
    for k in range(1,len(full)-1):
        if (full[k-1][0]==full[k][0]==full[k+1][0]) or (full[k-1][1]==full[k][1]==full[k+1][1]): continue
        clean.append(full[k])
    clean.append(q)
    number=len(d.connections)+1
    if number == 7:
        # Leave PJRT to the left before descending, avoiding the executable lane (12).
        gutter_x = d.components["ifrt"][0] + W + 110
        gutter_y = d.components["xla"][1] - 80
        clean = [p, (gutter_x, p[1]), (gutter_x, gutter_y), (q[0], gutter_y), q]
    d.edge(clean,color,dashed=color in {"result","neutral"},bridge_over=used,width=4)
    for u,v in zip(clean,clean[1:]):
        seg=tuple(sorted([u,v])); used[seg]=used.get(seg,0)+1
    d.connections.append((number,a,b,color,label))
    # Each endpoint links to its actual node, with a named tooltip and matching key entry.
    node_a, node_b = PORT_NODES[number-1]
    for (point,vector), node in zip([(p,va),(q,vb)], [node_a,node_b]):
        cx=point[0]+vector[0]*25; cy=point[1]+vector[1]*25
        d.labels.append(f'<a href="#{node}"><title>端口 {number:02d} · {escape(PORT_NAMES[node])}</title>'
                        f'<circle cx="{cx}" cy="{cy}" r="19" fill="{d.bg}" stroke="{d.colors[color]}" stroke-width="2"/></a>')
        d.text(cx,cy+7,str(number),20,d.colors[color],bold=True,center=True,href=f"#{node}")
    short = ["外层 Module → 绑定", "实参 → PyArray", "PyClient → HloProgram",
             "Array 包装", "CompileAndLoad", "Array → Buffer 映射",
             "provider → HLO 导入", "HLO → 目标后端", "编译 / 加载结果",
             "目标码观察", "输入 Buffer / 依赖", "已加载程序", "设备执行"][number-1]
    caption = f"{number:02d}  {short}"
    tw = d.measure(caption, 25)[1].width
    def overlaps(rect, other):
        x,y,w,h=rect; ox,oy,ow,oh=other
        return min(x+w,ox+ow)>max(x,ox) and min(y+h,oy+oh)>max(y,oy)
    # Prefer a horizontal gutter; a numbered key remains available for dense routes.
    segments=sorted(zip(clean,clean[1:]),key=lambda ab:(ab[0][1]!=ab[1][1],-abs(ab[0][0]-ab[1][0])-abs(ab[0][1]-ab[1][1])))
    placed=False
    for u,v in segments:
        for t in [.5,.3,.7]:
            mx=u[0]+(v[0]-u[0])*t; my=u[1]+(v[1]-u[1])*t
            candidates=[(mx-tw/2,my-16),(mx-tw/2,my+39)] if u[1]==v[1] else [(mx+23,my),(mx-tw-23,my)]
            for tx,ty in candidates:
                rect=(tx-12,ty-32,tw+24,46)
                if rect[0]<15 or rect[0]+rect[2]>d.w-15 or rect[1]<265: continue
                if any(overlaps(rect,o) for o in obstacles): continue
                if any(overlaps(rect,box[:4]) for box in d.text_boxes): continue
                d.label(tx,ty,caption,color,size=25)
                placed=True
                break
            if placed: break
        if placed: break
    if not placed:
        raise ValueError(f"No visible caption for flow {number:02d}: {caption}")


PORT_NODES = [
    ("jax_module", "jaxlib_module"), ("jax_input", "jaxlib_array"),
    ("jaxlib_client", "ifrt_program"), ("jaxlib_array", "ifrt_array"),
    ("ifrt_compiler", "pjrt_client"), ("ifrt_array", "pjrt_buffer"),
    ("pjrt_provider", "xla_import"), ("xla_plan", "component_backend"),
    ("component_backend", "pjrt_executable"), ("component_backend", "asm_code"),
    ("pjrt_buffer", "runtime_buffers"), ("pjrt_executable", "runtime_program"),
    ("runtime_dispatch", "hardware_units"),
]
PORT_NAMES = {
    "jax_module": "JAX · 外层 MLIR Module", "jaxlib_module": "jaxlib · MLIR Module",
    "jax_input": "JAX · 主机值 / jax.Array", "jaxlib_array": "jaxlib · PyArray",
    "jaxlib_client": "jaxlib · PyClient", "ifrt_program": "IFRT · HloProgram",
    "ifrt_array": "IFRT · Array", "ifrt_compiler": "IFRT · Compiler",
    "pjrt_client": "PJRT · PjRtClient", "pjrt_buffer": "PJRT · PjRtBuffer",
    "pjrt_provider": "PJRT · Backend provider", "xla_import": "XLA · MLIR → HLO 导入",
    "xla_plan": "XLA · HLO / 编译规划", "component_backend": "CPU / GPU / TPU 目标后端",
    "pjrt_executable": "PJRT · PjRtLoadedExecutable", "asm_code": "ASM 观察 · 目标代码",
    "runtime_buffers": "运行时 · 输入 Buffer / 依赖", "runtime_program": "运行时 · 已加载程序",
    "runtime_dispatch": "运行时 · 提交工作", "hardware_units": "硬件 · 执行单元",
}


LAYOUT = {
    "title": "JAX 软件栈 / 分层架构 · 组件与双通路",
    "subtitle": "从前端到设备逐层展开；同层并列组件通过编号箭头连接，框内展示抽象之间的作用关系。",
    "size": (5200,8630),
    "positions": {"jax":(1430,420),"jaxlib":(1430,2280),"ifrt":(180,3400),"pjrt":(2680,3400),"xla":(180,4630),"backend":(2680,4630),"asm":(180,6180),"runtime":(2680,6180),"hardware":(1430,7400)},
    "edges": [("jax","bottom",1050/W,"jaxlib","top",1050/W,"program","JAX lowering 构造外层 Module，经 jaxlib 绑定传递"),("jax","right",.2,"jaxlib","right",.2,"data","实际输入进入 PyArray / 运行时"),("jaxlib","left",.65,"ifrt","top",.25,"program","模块包装为 HloProgram，提交编译"),("jaxlib","right",.4,"ifrt","top",.8,"data","Python 数组对应 IFRT Array"),("ifrt","right",.3,"pjrt","left",.3,"program","IFRT 的 PJRT 实现调用 CompileAndLoad"),("ifrt","right",.68,"pjrt","left",.68,"data","逻辑 Array 映射到设备 Buffer"),("pjrt","left",.90,"xla","top",.3,"program","provider 将编译输入交给后端"),("xla","right",.45,"backend","left",.35,"program","优化后的 HLO / 规划按目标分流"),("backend","top",.75,"pjrt","bottom",.75,"result","编译产物返回并加载为可执行对象"),("backend","left",.85,"asm","top",.8,"neutral","目标代码可导出 / 反汇编为 ASM"),("pjrt","right",.85,"runtime","right",.25,"data","输入 Buffer 及其依赖进入执行"),("pjrt","bottom",.5,"runtime","top",.35,"program","已加载程序交给设备运行时执行"),("runtime","bottom",.3,"hardware","top",.7,"data","提交设备工作；硬件按 ISA 执行")],
    "key":(180,8150),
}


def render(layout, meta):
    width,height=layout["size"]
    d=ComponentDiagram(width,height,layout["title"],layout["subtitle"],meta)
    populate(d,layout["positions"])
    d.reading_guide()
    used={}
    for a,sa,fa,b,sb,fb,color,label in layout["edges"]:
        route(d,(a,sa,fa),(b,sb,fb),color,label,used)
    # Put the key in dedicated bottom space, expanding the canvas if necessary.
    x,y=layout["key"]
    d.text(x,y,"框间流程索引 · 点击端口名称定位节点；框内源码链接指向固定提交",32,d.ink,bold=True)
    for idx,(num,a,b,color,label) in enumerate(d.connections):
        col,row=divmod(idx,7)
        tx=x+col*2500; ty=y+62+row*86
        na,nb=PORT_NODES[idx]
        d.text(tx,ty,f"{num:02d}  {PORT_NAMES[na]}",24,d.colors[color],href=f"#{na}")
        first_width=d.measure(f"{num:02d}  {PORT_NAMES[na]}",24)[1].width
        d.text(tx+first_width+20,ty,f"→ {PORT_NAMES[nb]}",24,d.colors[color],href=f"#{nb}")
        d.text(tx+44,ty+33,label,22,d.muted)
    d.meta["flow_ports"] = [{"number":i+1,"source":a,"target":b} for i,(a,b) in enumerate(PORT_NODES)]
    footer=y+690
    pins=meta["source_pins"]
    d.text(x,footer,f'固定源码：JAX {pins["jax"][:12]} · XLA {pins["xla"][:12]} · 编译与数据在执行提交处汇合。',23,d.muted)
    verification=meta.get("source_verification",{})
    missing=verification.get("missing_repositories",[])
    checked=verification.get("checked_repositories",[])
    scope = "本次已复核图内固定源码锚点。"
    if missing:
        prefix = "本次复核 " + " / ".join(checked).upper() + " 源码；" if checked else ""
        scope = prefix + " / ".join(missing).upper() + " 检出缺失，沿用锁定锚点，未复核其源码。"
    d.text(x,footer+44,scope,23,d.muted)
    d.h=max(d.h,footer+100)
    return d


def preview(svg, png, width=1900):
    Rsvg, cairo = preview_bindings()
    import xml.etree.ElementTree as ET
    root=ET.parse(svg).getroot(); sw,sh=float(root.attrib["width"]),float(root.attrib["height"])
    height=round(sh*width/sw)
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,width,height)
    rect=Rsvg.Rectangle(); rect.x=rect.y=0; rect.width=width; rect.height=height
    Rsvg.Handle.new_from_file(str(svg)).render_document(cairo.Context(surface),rect)
    surface.write_to_png(str(png))


def main():
    from render_overview_software_stack_flows import main as render_selected
    render_selected()


if __name__=="__main__":
    main()
