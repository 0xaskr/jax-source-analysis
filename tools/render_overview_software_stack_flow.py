#!/usr/bin/env python3
"""Render the selected dual-path JAX stack flowchart.

Source revisions and anchors are embedded for reproducible regeneration. Requires the system
PyGObject/Pango packages; PNG previews additionally use Rsvg and cairo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import gi

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo


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
        self.colors = {"program": "#84b6ff" if dark else "#3971b7",
                       "data": "#65d9c4" if dark else "#168679",
                       "transform": "#c9a7ff" if dark else "#8963ad",
                       "unknown": "#f1c37c" if dark else "#af7935",
                       "neutral": "#99aabd" if dark else "#8293a6"}
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.boxes, self.text_boxes, self.edge_boxes = {}, [], []
        self.context = PangoCairo.FontMap.get_default().create_context()
        self.text(65, 82, title, 44, bold=True)
        self.text(68, 131, subtitle, 24, color=self.muted)
        x = 70
        for label, color in [("编译路径", "program"), ("数据与执行", "data"),
                             ("变换步骤", "transform"), ("待源码确认", "unknown"), ("契约 / 观察关系", "neutral")]:
            self.labels.append(f'<path d="M{x} 181h42" stroke="{self.colors[color]}" stroke-width="4"/>')
            self.text(x + 54, 189, label, 21, color=self.muted)
            x += 250

    def measure(self, value, size, bold=False):
        font = Pango.FontDescription()
        font.set_family("Noto Sans CJK SC")
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
        text = (f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC" font-size="{size}" '
                f'font-weight="{700 if bold else 400}" fill="{color or self.ink}">{escape(value)}</text>')
        if href:
            text = f'<a href="{escape(href, quote=True)}" target="_blank">{text}</a>'
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
        self.nodes.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{h / 2 if pill else 18}" fill="{self.card}" stroke="{stroke}" stroke-width="2"{dash}/></g>')
        self.text(x + 24, y + 30, tag, 18, color=stroke, bold=True, owner=key)
        title_size = 32
        if self.measure(title, title_size, True)[1].width > w - 48:
            title_size = 28
        self.text(x + 24, y + 71, title, title_size, bold=True, owner=key)
        baseline = y + 109
        for line in body:
            for part in self.wrap(line, w - 48, 23):
                self.text(x + 24, baseline, part, 23, color=self.muted, owner=key)
                baseline += 33
        if ref:
            self.text(x + 24, y + h - 20, link or ref, 18, color=stroke, href=self.ref(ref), owner=key)

    def edge(self, points, color="program", dashed=False, end=True):
        path = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        marker = f' marker-end="url(#{color})"' if end else ""
        self.edges.append(f'<path d="{path}" fill="none" stroke="{self.colors[color]}" '
                          f'stroke-width="3" stroke-linejoin="round"{dash}{marker}/>')
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
               f'<title id="title">{escape(self.title)}</title><desc id="desc">JAX 到硬件的软件栈流程图。编译路径和数据路径在执行处汇合；包含定义、产生、变换、消费以及固定提交源码链接。</desc>'
               f'<metadata>{escape(json.dumps(self.meta, ensure_ascii=False))}</metadata><defs>{defs}</defs>'
               f'<rect width="100%" height="100%" fill="{self.bg}"/>' + "".join(self.base + self.edges + self.nodes + self.labels) + '</svg>\n')
        path.write_text(xml)
        ET.parse(path)
        print(f"{path.name}: {len(self.boxes)} nodes, {len(self.text_boxes)} labels; layout checks passed")


def dual_path(meta):
    d = Diagram(2480, 2690, "JAX 软件栈 / 双通路汇合", "编译准备程序，运行时准备数据 · 两条通路在 Execute 处汇合", meta, dark=True)
    d.panel(60, 255, 1500, 1900, "01  编译程序", "program")
    d.panel(1620, 255, 800, 1900, "02  准备数据", "data")
    d.node("jaxpr", 130, 360, 610, 185, "JAX · TRACING 的产物", "Jaxpr", ["变量、方程、常量、effects", "由 Python 函数与抽象参数产生"], ref="Jaxpr")
    d.node("jax_transform", 910, 375, 570, 160, "变换", "Jaxpr → Jaxpr", ["部分求值 / DCE 等"], "transform", "dce_jaxpr", "partial_eval / dce_jaxpr")
    d.node("module", 130, 675, 610, 190, "JAX LOWERING · JAXLIB BINDINGS", "MLIR Module（StableHLO）", ["张量运算、类型与区域", "模块是后续编译入口的输入"], ref="lower_jaxpr_to_module")
    d.node("pallas", 910, 675, 570, 190, "可选的 KERNEL 分支", "Mosaic TPU MLIR", ["由 Pallas kernel Jaxpr lowering", "封装为模块内的 custom_call"], ref="lower_module_to_custom_call")
    d.node("bridge", 130, 980, 610, 170, "JAXLIB → IFRT → PJRT", "CompileAndLoad", ["跨运行时接口提交模块"], ref="PJRT CompileAndLoad", link="IFRT / PJRT 编译调用")
    d.node("hlo", 130, 1230, 610, 190, "XLA", "HLO", ["HloModule / Computation / Instruction", "后端编译与优化的计算图"], ref="HloModule")
    d.node("passes", 910, 1245, 570, 160, "变换", "HLO → HLO", ["优化 / 布局等 pass"], "transform", "HloPassPipeline RunImpl", "HloPassPipeline::RunImpl")
    for key, x, title, body, color, ref in [
        ("cpu", 100, "LLVM IR", ["CPU 代码生成"], "program", "CpuCompiler RunBackend"),
        ("gpu", 590, "LLVM IR / Triton IR", ["GPU kernel / 设备库路径"], "program", "GpuCompiler RunBackend"),
        ("tpu", 1080, "LLO", ["内部结构与 pass 待源码确认"], "unknown", None)]:
        d.node(key, x, 1580, 440, 170, key.upper() + " 后端", title, body, color, ref, link="RunBackend" if ref else None, dashed=key == "tpu")
    d.node("executable", 130, 1940, 660, 170, "目标代码经编译 / 链接 / 加载", "LoadedExecutable", ["持有可执行代码与执行信息"], ref="PjRtLoadedExecutable")
    d.node("input", 1700, 360, 640, 130, "数据输入", "输入数组", ["主机值或已有设备数组"], "data", pill=True)
    d.node("array", 1700, 675, 640, 190, "IFRT", "Array", ["逻辑数组：类型、形状、分片", "产生：由输入值构造逻辑数组"], "data", "ifrt Array", "ifrt::Array / MakeArrayFromHostBuffer")
    d.node("array_transform", 1700, 970, 640, 160, "按需变换", "复制 / 重映射", ["CopyArrays / RemapArrays"], "transform", "RemapArrays")
    d.node("buffer", 1700, 1275, 640, 190, "PJRT", "Buffer", ["设备存储、所有权与就绪状态", "由 PJRT-backed Array 管理设备存储"], "data", "PjRtBuffer", "PjRtBuffer / BufferFromHostBuffer")
    d.node("buffer_transform", 1700, 1610, 640, 170, "按需复制 / 依赖", "内存复制与就绪依赖", ["CopyToMemorySpace / GetReadyFuture"], "transform", "CopyToMemorySpace")
    d.node("asm", 920, 1940, 600, 170, "汇编层 · 目标代码的可观察形式", "ASM", ["汇编文本；具体形式随后端变化"], "neutral", dashed=True)
    d.node("execute", 890, 2250, 700, 170, "03  IFRT / PJRT → 设备运行时", "执行提交", ["已加载程序 × 输入 Buffer → 执行"], "data", "IFRT PjRtLoadedExecutable Execute", "ExecuteSharded → Execute")
    d.node("hardware", 890, 2500, 700, 130, "硬件执行", "CPU / GPU / TPU", ["结果经 Buffer → Array 返回"], "data", pill=True)
    d.node("isa", 1790, 2475, 590, 145, "指令集接口层", "ISA", ["规定设备指令及其语义"], "neutral", dashed=True)
    d.edge([(740, 415), (910, 415)], "transform")
    d.edge([(910, 500), (740, 500)], "transform")
    d.edge([(435, 545), (435, 675)])
    d.label(170, 614, "lower_jaxpr_to_module", ref="lower_jaxpr_to_module")
    d.edge([(650, 545), (650, 590), (1195, 590), (1195, 675)], dashed=True)
    d.label(965, 627, "Pallas kernel", size=19)
    d.edge([(910, 777), (740, 777)], dashed=True)
    d.label(758, 747, "custom_call", size=18)
    d.edge([(435, 865), (435, 980)])
    d.label(470, 932, "提交模块")
    d.edge([(435, 1150), (435, 1230)])
    d.label(470, 1190, "导入 HloModule", ref="ConvertMlirHloToHloModule")
    d.edge([(740, 1285), (910, 1285)], "transform")
    d.edge([(910, 1370), (740, 1370)], "transform")
    d.edge([(435, 1420), (435, 1490)], end=False)
    d.label(690, 1465, "按目标后端分流")
    for x in [320, 810, 1300]:
        d.edge([(435, 1490), (x, 1490), (x, 1580)])
        d.edge([(x, 1750), (x, 1830), (810, 1830)], end=False)
    d.edge([(810, 1830), (460, 1830), (460, 1940)])
    d.label(490, 1900, "生成目标代码")
    d.edge([(1300, 1830), (1300, 1940)], "neutral", dashed=True)
    d.edge([(2020, 490), (2020, 675)], "data")
    d.label(1775, 593, "MakeArrayFromHostBuffer", "data", "MakeArrayFromHostBuffer")
    d.edge([(2020, 865), (2020, 970)], "transform")
    d.edge([(1700, 770), (1660, 770), (1660, 1370), (1700, 1370)], "data")
    d.edge([(2020, 1130), (2020, 1275)], "data")
    d.label(1740, 1208, "PJRT-backed 实现映射设备存储", "data")
    d.edge([(2020, 1465), (2020, 1610)], "transform")
    d.edge([(2340, 1370), (2380, 1370), (2380, 2310), (1590, 2310)], "data")
    d.edge([(2340, 1695), (2380, 1695)], "data", end=False)
    d.label(1795, 2258, "输入 Buffer + 就绪依赖", "data")
    d.edge([(460, 2110), (460, 2190), (810, 2190), (810, 2310), (890, 2310)])
    d.label(505, 2180, "可执行程序")
    d.edge([(1240, 2420), (1240, 2500)], "data")
    d.label(1290, 2468, "运行时 / 驱动调度", "data")
    d.edge([(1790, 2555), (1590, 2555)], "neutral", dashed=True)
    d.label(1620, 2525, "遵循契约", "neutral", size=18)
    d.text(70, 2310, "读图方式", 24, d.muted, bold=True)
    d.text(70, 2355, "节点：定义 · 入边：产生", 21, d.muted)
    d.text(70, 2391, "紫色：变换 · 出边：消费", 21, d.muted)
    d.text(70, 2440, "Mosaic TPU MLIR ≠ LLO", 21, d.muted)
    d.text(70, 2476, "ASM 与 ISA 不表示强制编译阶段", 21, d.muted)
    pins = meta["source_pins"]
    d.text(70, 2661, f'源码快照：JAX {pins["jax"][:12]} · XLA {pins["xla"][:12]}  |  小字链接定位源码；虚线表示可选路径、观察关系或待源码确认部分。', 20, d.muted)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    original = args.root / "research/software-stack/overview/overview-software-stack.svg"
    preserved = {original: hashlib.sha256(original.read_bytes()).hexdigest()}
    metadata = SOURCE_METADATA
    for repo, pin in metadata["source_pins"].items():
        repo_path = args.root / "upstream" / repo
        actual = subprocess.check_output(["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True).strip()
        if actual != pin:
            raise SystemExit(f"Source pin changed for {repo}; update anchors before rendering")
    for anchor in metadata["source_anchors"].values():
        repo_path = args.root / "upstream" / anchor["repo"]
        local = (repo_path / anchor["path"]).read_text()
        if anchor["needle"] not in local.splitlines()[anchor["line"] - 1]:
            raise SystemExit(f"Source anchor moved: {anchor}")
        committed = subprocess.check_output(["git", "-C", str(repo_path), "show", f'{metadata["source_pins"][anchor["repo"]]}:{anchor["path"]}'], text=True)
        if local != committed:
            raise SystemExit(f"Source file has changes: {anchor['path']}")
    output = args.output_dir or original.parent
    output.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        gi.require_version("Rsvg", "2.0")
        from gi.repository import Rsvg
        import cairo
    path = output / "overview-software-stack-flow-c-dual-path.svg"
    diagram = dual_path(metadata)
    diagram.save(path)
    if args.preview_dir:
        scale = min(1, 1800 / diagram.w)
        width, height = round(diagram.w * scale), round(diagram.h * scale)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(surface)
        viewport = Rsvg.Rectangle()
        viewport.x, viewport.y = 0, 0
        viewport.width, viewport.height = width, height
        Rsvg.Handle.new_from_file(str(path)).render_document(context, viewport)
        surface.write_to_png(str(args.preview_dir / "c-dual-path.png"))
    for path, before in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before, f"Modified {path}"


if __name__ == "__main__":
    main()
