#!/usr/bin/env python3
"""Render a new layered overview with extension APIs and observation entrypoints.

Preserves the selected layered SVG. Adds optional API cards and an HLO callback
loop, while retaining exactly one core concept per module. Outputs SVG only.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from render_overview_software_stack_flows import (
    ROOT, ComponentFlowDiagram, metadata, render,
)


STEM = "overview-software-stack-components-extended"
EXTRA_ANCHORS = {
    "register_lowering": ("jax", "jax/_src/interpreters/mlir.py", "def register_lowering("),
    "custom_partitioning": ("jax", "jax/_src/custom_partitioning.py", "class custom_partitioning:"),
    "def_partition": ("jax", "jax/_src/custom_partitioning.py", "def def_partition("),
    "per-shard lowering": ("jax", "jax/_src/custom_partitioning.py", "def _custom_partitioning_partition("),
    "partitioning lowering": ("jax", "jax/_src/custom_partitioning.py", "def _custom_partitioning_lowering_rule("),
    "register_ffi_target": ("jax", "jax/_src/ffi.py", "def register_ffi_target("),
    "ffi_call_lowering": ("jax", "jax/_src/ffi.py", "def ffi_call_lowering("),
    "ffi_lowering": ("jax", "jax/_src/ffi.py", "def ffi_lowering("),
    "HLO transformation registration": ("jax", "jax/_src/xla_transform.py", "def register_hlo_module_transformation("),
    "HLO transformation clearing": ("jax", "jax/_src/xla_transform.py", "def clear_hlo_module_transformation("),
    "CPU hook registration": ("jax", "jax/_src/xla_transform.py", "_xla.register_xla_transform(name, stage.value, callback)"),
    "PJRT hook registration": ("jax", "jax/_src/xla_transform.py", "_xla.register_xla_transform_c_api(c_api, name, stage.value, callback)"),
    "PipelineStage": ("jax", "jax/_src/xla_transform.py", "class PipelineStage("),
    "CPU PRE_SCHEDULER": ("xla", "xla/service/cpu/cpu_compiler.cc", "HloXlaTransform::PipelineStage::kPreScheduler);", 1178),
    "CPU POST_SCHEDULER": ("xla", "xla/service/cpu/cpu_compiler.cc", "HloXlaTransform::PipelineStage::kPostScheduler);", 1833),
    "GPU PRE_SCHEDULER": ("xla", "xla/service/gpu/gpu_compiler.cc", "HloXlaTransform::PipelineStage::kPreScheduler);", 1026),
    "GPU POST_SCHEDULER": ("xla", "xla/service/gpu/gpu_compiler.cc", "HloXlaTransform::PipelineStage::kPostScheduler);", 3286),
    "HLO editing example": ("jax", "tests/xla_transform_test.py", "module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)", 344),
    "TPU hook test definition": ("jax", "tests/xla_transform_test.py", "def test_transformer_schedule_async_ops_sharded("),
    "Lowered.compiler_ir": ("jax", "jax/_src/stages.py", "def compiler_ir(self, dialect: str | None = None) -> Any | None:", 671),
    "Lowering.hlo export": ("jax", "jax/_src/stages.py", "def hlo(self) -> xc.XlaComputation:", 237),
    "MeshComputation.stablehlo": ("jax", "jax/_src/interpreters/pxla.py", "def stablehlo(self) -> ir.Module:", 1224),
    "Compiled analysis": ("jax", "jax/_src/stages.py", "class Compiled(Stage):"),
    "profiler.trace": ("jax", "jax/_src/profiler.py", "def trace(", 315),
    "profiler.start_trace": ("jax", "jax/_src/profiler.py", "def start_trace("),
    "TraceAnnotation": ("jax", "jax/_src/profiler.py", "class TraceAnnotation("),
}

HOOK_NODES = {
    "hlo_hook_stage": {
        "tag": "按需变换 · D 接口", "title": "PRE / POST_SCHEDULER",
        "rows": ["HLO 调度前 / 后的具体挂点。", "ApplyXlaTransforms 调用已注册变换。"],
        "refs": ("CPU PRE_SCHEDULER", "CPU POST_SCHEDULER"), "color": "transform",
    },
    "hlo_hook_callback": {
        "tag": "宿主回调 · D 接口", "title": "Python HLO callback",
        "rows": ["输入：序列化的 HloModuleProto。", "返回 bytes 更新程序；None 保留原程序。"],
        "refs": ("HLO transformation registration",), "color": "transform",
    },
}

INTERFACES = [
    {
        "code": "A", "id": "xla-interface-a", "module": "jax", "y": 910,
        "title": "自定义 primitive lowering", "scope": "JAX · 方程 → MLIR 规则", "color": "program",
        "sections": [
            ("注册入口", ["from jax.interpreters import mlir", "mlir.register_lowering(primitive, rule, platform=...)" ]),
            ("定义生成什么操作", ["Jaxpr 方程 → 查找平台专用 / 通用规则。", "rule(ctx, *operands, **params) → MLIR 结果值。", "生成的操作进入外层 Module，再沿编译通路提交。"]),
            ("规则职责", ["lowering 决定 primitive 如何生成目标平台 IR。", "抽象求值、自动微分和 batching 规则分别定义。"]),
        ],
        "targets": [("lower", "A · 方程 lowering：规则在此被选择并消费")],
        "refs": ("register_lowering", "jaxpr_subcomp"),
    },
    {
        "code": "B", "id": "xla-interface-b", "module": "jax", "y": 2950,
        "title": "custom_partitioning", "scope": "JAX 记录规则 · 分片编译使用", "color": "transform",
        "sections": [
            ("注册自定义分片行为", ["@custom_partitioning → f.def_partition(partition, ...)", "Shardy 启用时使用 sharding_rule；其他路径使用相应分片推导回调。"]),
            ("分片计算的契约", ["partition(mesh, arg_shapes, result_shape)", "→ mesh、lower_fn、输出 shardings、输入 shardings。", "lower_fn → 分片 Jaxpr → MLIR bytecode → 交回编译器。"]),
            ("作用对象", ["规则描述被装饰计算的分片与各 shard 的计算。", "它不接收完整 HloModuleProto；也不等同于 sdy-lift-inlined-meshes。"]),
        ],
        "targets": [("module", "B · 外层 Module：携带操作与分片表示"), ("passes", "B · 编译中的分片处理：消费相应规则")],
        "refs": ("custom_partitioning", "def_partition", "per-shard lowering", "partitioning lowering"),
    },
    {
        "code": "C", "id": "xla-interface-c", "module": "jax", "y": 5540,
        "title": "原生 FFI / custom_call", "scope": "编译期生成调用 · 执行期调用原生实现", "color": "program",
        "sections": [
            ("注册 target", ["jax.ffi.register_ffi_target(name, fn, platform=...)", "fn 持有原生函数指针 / stage handlers。", "注册经 xla_client.register_custom_call_target 连接后端。"]),
            ("建立并编译外部调用", ["jax.ffi.ffi_call(target_name, result_shape_dtypes, ...)", "ffi_call 方程 → ffi_call_lowering → StableHLO custom_call。", "目标名称、参数 / 结果签名、layout / alias 与 ABI 参与调用约定。"]),
            ("消费与返回", ["编译器形成后端调用，执行时进入匹配的原生 handler / kernel。", "结果继续进入输出对象返回通路。", "Pallas TPU custom_call 携带 Mosaic IR payload，是另一种用法。"]),
        ],
        "targets": [("module", "C · 编译输入：外层 custom_call 操作"), ("runtime", "C · 执行机制：调用对应后端的原生 target")],
        "refs": ("register_ffi_target", "ffi_call_lowering", "ffi_lowering"),
    },
    {
        "code": "D", "id": "xla-interface-d", "module": "xla", "y": 8610,
        "title": "编译期 HLO 变换", "scope": "宿主 Python 回调 · 更新编译中的程序", "color": "transform",
        "sections": [
            ("登记回调及平台", ["jax.extend.xla.register_hlo_module_transformation(", "    callback, name=..., stage=..., platforms=...)", "CPU → _xla.register_xla_transform（进程内直接注册）。", "其他平台 → get_pjrt_plugin → register_xla_transform_c_api。", "插件需要支持对应扩展；TPU 有测试定义，本图未执行该测试。"]),
            ("HLO pipeline 中按需调用", ["PRE_SCHEDULER：HLO 调度前；不是所有 HLO 优化之前。", "POST_SCHEDULER：已有 HloSchedule 后；不是机器指令调度之后。", "CPU 的 POST 挂点位于 CreateBufferAssignment 之前。", "CPU/GPU 源码可见两个挂点；注册支持仍需匹配具体后端。"]),
            ("输入、可选编辑与返回", ["HloModuleProto bytes → callback(bytes) → bytes / None。", "可解析为 HloModule，再修改 computation / instruction 或 schedule。", "返回修改后的序列化 bytes → 后端继续使用更新后的 HLO。", "返回 None → 保留原程序；同一 stage 的回调按注册顺序运行。", "处理的是外层 HLO；custom_call 内的 Mosaic payload 是另一套表示。"]),
            ("作用范围", ["单独修改导出的 HLO 副本不会自动更新已编译 executable。", "clear_hlo_module_transformation 按 name / stage / platform 清除注册。", "HLO 编辑对象和 Python 绑定需按目标版本核对。"]),
        ],
        "targets": [("hlo_hook_stage", "D · 图内挂点：ApplyXlaTransforms"), ("hlo_hook_callback", "D · 图内返回：bytes 更新 / None 保留")],
        "refs": ("HLO transformation registration", "CPU hook registration", "PJRT hook registration", "CPU PRE_SCHEDULER", "CPU POST_SCHEDULER", "GPU PRE_SCHEDULER", "GPU POST_SCHEDULER", "HLO editing example", "TPU hook test definition"),
    },
    {
        "code": "E", "id": "xla-interface-e", "module": "jax", "y": 13600,
        "title": "IR 检查与编译结果分析", "scope": "观察表示与元信息 · 按对象区分可变性", "color": "neutral",
        "sections": [
            ("lowered 阶段", ["lowered = jax.jit(f).lower(...)", "lowered.compiler_ir('stablehlo') → 当前实现持有的 MLIR Module。", "MeshComputation.stablehlo 返回 self._hlo；不是只读副本。", "lowered.compiler_ir('hlo') → 显式导出的 XlaComputation。", "lowered.as_text(...) → 调试文本。"]),
            ("compiled 阶段", ["compiled = lowered.compile()", "compiled.as_text() / cost_analysis() / memory_analysis()", "compiled.runtime_executable() → 底层可执行对象。", "可提供哪些表示与分析，取决于后端及版本。"]),
            ("修改与序列化边界", ["当前实现的 Module 原地修改可能影响尚未发生的编译。", "这不是稳定的编辑契约，也不会重写已经编译的 executable。", "compiler_ir / as_text 面向调试；可移植序列化使用 jax.export。"]),
        ],
        "targets": [("module", "E · 编译前：外层 Module / 显式 HLO 导出"), ("executable", "E · 编译后：可执行程序及分析元信息")],
        "refs": ("Lowered.compiler_ir", "MeshComputation.stablehlo", "Lowering.hlo export", "Compiled analysis"),
    },
    {
        "code": "F", "id": "runtime-observation", "module": "runtime", "y": 17320,
        "title": "运行观测与时间线", "scope": "采集与分析入口 · 源码检查未执行采集", "color": "neutral",
        "sections": [
            ("采集入口", ["jax.profiler.trace(...) / start_trace(...) / stop_trace()", "TraceAnnotation 标注宿主范围；后端决定可采集的设备事件。", "采集数据可交给 XProf 等工具进行时间线和性能分析。"]),
            ("异步执行的观察边界", ["提交、返回输出对象和设备完成是不同事件。", "若要覆盖计算完成，应在采集范围内等待结果就绪。", "宿主标记或编译产物不能替代真实设备执行证据。"]),
        ],
        "targets": [("runtime", "F · 观察执行提交与后端工作"), ("runtime_completion", "F · 结合完成事件 / Future 判断就绪")],
        "refs": ("profiler.trace", "profiler.start_trace", "TraceAnnotation", "PJRT Buffer readiness"),
    },
]


class ExtendedOverview(ComponentFlowDiagram):
    def __init__(self, meta):
        super().__init__(meta)
        self.w = 16440
        self.title += " · 扩展与观测接口"
        self.description += "右侧补充 A–E 扩展接口和 F 运行观测入口；XLA 框内展开可选 HLO 回调。接口均为辅助节点，不增加核心概念。"
        self.node_data = {**self.node_data, **HOOK_NODES,
                          **{i["id"]: {"title": i["title"]} for i in INTERFACES}}
        self.components["extensions"] = (13040, 480, 3200, 19900)
        self.text(13200, 101, "扩展与观测接口版", 59, self.colors["program"], bold=True)
        self.text(13200, 172, "A–F 为辅助接口，点击互相定位。", 34, self.muted)
        self.meta["variant"] = "layered overview with optional extension APIs and observation entrypoints"

    def interface_card(self, item):
        key, module = item["id"], item["module"]
        x, y, w = 13200, item["y"], 2880
        stroke = self.colors[item["color"]]
        self.text(x+36, y+49, "辅助接口 · " + item["scope"], 29, stroke, bold=True, owner=key)
        self.text(x+36, y+119, item["code"] + " · " + item["title"], 47, bold=True, owner=key)
        baseline = y + 215
        for title, rows in item["sections"]:
            self.text(x+36, baseline, title, 36, stroke, bold=True, owner=key)
            baseline += 65
            for row in rows:
                for line in self.wrap(row, w-82, 34):
                    self.text(x+42, baseline, line, 34, self.muted, owner=key)
                    baseline += 61
            baseline += 52
        self.text(x+36, baseline, "作用位置 · 点击定位主图", 32, stroke, bold=True, owner=key)
        for target, label in item["targets"]:
            baseline += 59
            self.text(x+42, baseline, label + " ↗", 31, self.colors["program"], href="#"+target, owner=key)
        baseline += 90
        for ref in item["refs"]:
            self.text(x+42, baseline, "源码 · " + ref, 27, stroke, href=self.ref(ref), owner=key)
            baseline += 46
        h = baseline - y + 25
        self.boxes[key], self.members[key] = (x, y, w, h), "extensions"
        self.concept_modules[key] = module
        self.nodes.append(f'<g id="{key}" data-module="{module}" data-core="false" data-interface="{item["code"]}"><title>{escape(item["title"])}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="#ffffff" stroke="{stroke}" stroke-width="2"/></g>')

    def interface_badges(self):
        targets = {}
        for item in INTERFACES:
            for key, _ in item["targets"]:
                targets.setdefault(key, []).append(item)
        for key, items in targets.items():
            x, y, w, _ = self.boxes[key]
            for i, item in enumerate(items):
                bx = x+w-160*(len(items)-i)-14
                color = self.colors[item["color"]]
                self.labels.append(f'<a href="#{item["id"]}"><rect x="{bx}" y="{y+10}" width="147" height="40" rx="8" fill="#ffffff" stroke="{color}" stroke-width="1.5"/></a>')
                self.text(bx+16, y+38, "接口 " + item["code"] + " ↗", 24, color, href="#"+item["id"], owner=key)

    def finish(self):
        self.box("hlo_hook_stage", "xla", 0, 2, h=390)
        self.box("hlo_hook_callback", "xla", 1, 2, h=390)
        self.inside("passes", "hlo_hook_stage", "transform", "按需调用")
        self.inside("hlo_hook_stage", "hlo_hook_callback", "transform", "bytes")
        self.inside("hlo_hook_callback", "passes", "transform", "返回", sides=("bottom", "bottom"))
        x, y, w, h = self.components["extensions"]
        self.base.append(f'<g id="component_extensions"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="#f1f5f7" stroke="#a8bbca" stroke-width="3"/></g>')
        self.text(x+40, y+69, "接口说明区 · 按实际作用位置阅读", 43, self.colors["program"], bold=True)
        self.text(x+40, y+139, "接口卡片不构成新的软件层，也不表示每次执行都经过。", 31, self.muted)
        self.text(x+40, y+205, "主图保留 11 个核心概念；卡片与主图通过字母按钮互相定位。", 31, self.muted)
        for item in INTERFACES:
            self.interface_card(item)
        self.interface_badges()
        self.meta["extension_interfaces"] = [{"code": i["code"], "node": i["id"], "module": i["module"],
            "targets": [key for key, _ in i["targets"]], "source_anchors": list(i["refs"])} for i in INTERFACES]
        self.meta["extension_paths"] = [["passes", "hlo_hook_stage", "hlo_hook_callback", "passes"]]
        super().finish()

    def save(self, path):
        edges = {(r["source"], r["target"]) for r in self.relations}
        for chain in self.meta["extension_paths"]:
            if any(edge not in edges for edge in zip(chain, chain[1:])):
                raise ValueError("Incomplete HLO callback loop")
        for item in self.meta["extension_interfaces"]:
            for key in item["targets"]:
                if key not in self.boxes:
                    raise ValueError("Missing extension target: " + key)
        cards = [self.boxes[i["id"]] for i in INTERFACES]
        if any(a[1]+a[3]+50 > b[1] for a, b in zip(cards, cards[1:])):
            raise ValueError("Extension cards overlap")
        super().save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fetch-sources", action="store_true")
    args = parser.parse_args()
    meta = metadata(args.root, args.fetch_sources, extra_anchors=EXTRA_ANCHORS)
    output = args.output_dir or args.root / "research/software-stack/overview"
    output.mkdir(parents=True, exist_ok=True)
    render(meta, diagram_type=ExtendedOverview).save(output / (STEM + ".svg"))
    print(f"Verified {len(meta['source_anchors'])} pinned anchors; 11 core concepts, 6 auxiliary interfaces; SVG only")


if __name__ == "__main__":
    main()
