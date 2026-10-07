#!/usr/bin/env python3
"""Render the source-locked JAX component architecture, without workload examples.

Uses the overview renderer's measured SVG primitives. System PyGObject/Pango is
required; PNG previews also require Rsvg and cairo. No JAX runtime is imported.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from render_overview_software_stack_components import Diagram, preview


JAX_PIN = "361c43e072cce92b7d3e9bdaf4dd16db26c49043"
ROOT = Path(__file__).resolve().parents[1]
CORE = "jax/_src/core.py"
PE = "jax/_src/interpreters/partial_eval.py"
AD = "jax/_src/interpreters/ad.py"
BATCH = "jax/_src/interpreters/batching.py"
MLIR = "jax/_src/interpreters/mlir.py"
STAGES = "jax/_src/stages.py"
PJIT = "jax/_src/pjit.py"
PALLAS = "jax/_src/pallas/pallas_call.py"

# Needles are resolved only after the checkout and file contents match this pin.
SOURCE_SPECS = {
    "jit": ("jax/_src/api.py", "def jit(", 208),
    "grad": ("jax/_src/api.py", "def grad("),
    "vmap": ("jax/_src/api.py", "def vmap["),
    "tree_flatten": ("jax/_src/tree_util.py", "def tree_flatten("),
    "flatten_fun": ("jax/_src/api_util.py", "def flatten_fun("),
    "ArrayImpl": ("jax/_src/array.py", "class ArrayImpl("),
    "Mesh": ("jax/_src/mesh.py", "class Mesh("),
    "NamedSharding": ("jax/_src/named_sharding.py", "class NamedSharding("),
    "jit_trace": (PJIT, "def jit_trace("),
    "_infer_params": (PJIT, "def _infer_params("),
    "_trace_for_jit": (PJIT, "def _trace_for_jit("),
    "_resolve_and_lower": (PJIT, "def _resolve_and_lower("),
    "Traced": (STAGES, "class Traced("),
    "Traced.lojax": (STAGES, "  def lojax(self)"),
    "Lowered": (STAGES, "class Lowered("),
    "Compiled": (STAGES, "class Compiled("),
    "Jaxpr": (CORE, "class Jaxpr:"),
    "ClosedJaxpr alias": (CORE, "ClosedJaxpr = Jaxpr"),
    "JaxprEqn": (CORE, "class JaxprEqn:"),
    "Var": (CORE, "class Var:"),
    "Literal": (CORE, "class Literal:"),
    "Primitive": (CORE, "class Primitive:"),
    "Primitive.bind": (CORE, "  def bind(self, *args, **params):"),
    "Primitive.bind_with_trace": (CORE, "  def bind_with_trace(self, trace, args, avals, params, /):"),
    "AbstractValue": (CORE, "class AbstractValue:"),
    "ShapedArray": (CORE, "class ShapedArray("),
    "Trace": (CORE, "class Trace:"),
    "Tracer": (CORE, "class Tracer["),
    "eval_jaxpr": (CORE, "def eval_jaxpr("),
    "jaxpr_as_fun": (CORE, "def jaxpr_as_fun("),
    "check_jaxpr": (CORE, "def check_jaxpr("),
    "pretty_print": (CORE, "  def pretty_print(self,", 248),
    "DebugInfo": ("jax/_src/linear_util.py", "class DebugInfo("),
    "Effect": ("jax/_src/effects.py", "class Effect:"),
    "effect_sets": ("jax/_src/effects.py", "ordered_effects: EffectTypeSet"),
    "PartialVal": (PE, "class PartialVal("),
    "DynamicJaxprTrace": (PE, "class DynamicJaxprTrace("),
    "DynamicJaxprTracer": (PE, "class DynamicJaxprTracer("),
    "JaxprStackFrame": (PE, "class JaxprStackFrame:"),
    "TracingEqn": (PE, "class TracingEqn:"),
    "trace_to_jaxpr_nocache": (PE, "def trace_to_jaxpr_nocache("),
    "new_arg": (PE, "  def new_arg(self, aval, source_info:"),
    "default_process_primitive": (PE, "  def default_process_primitive(", 1750),
    "make_eqn": (PE, "  def make_eqn("),
    "new_const": (PE, "  def _new_const("),
    "Frame.to_jaxpr": (PE, "  def to_jaxpr(", 1517),
    "make_jaxpr_effects": (PE, "def make_jaxpr_effects("),
    "partial_eval_jaxpr_nounits": (PE, "def partial_eval_jaxpr_nounits("),
    "dce_jaxpr": (PE, "def dce_jaxpr("),
    "has_effects": (PE, "def has_effects("),
    "lower_jaxpr": (PE, "def lower_jaxpr("),
    "JVPTrace": (AD, "class JVPTrace("),
    "LinearizeTrace": (AD, "class LinearizeTrace("),
    "linearize_jaxpr": (AD, "def linearize_jaxpr("),
    "backward_pass": (AD, "def backward_pass("),
    "primitive_jvps": (AD, "primitive_jvps :"),
    "primitive_transposes": (AD, "primitive_transposes:"),
    "BatchTrace": (BATCH, "class BatchTrace("),
    "batch_jaxpr2": (BATCH, "def batch_jaxpr2("),
    "fancy_primitive_batchers": (BATCH, "fancy_primitive_batchers:"),
    "HiPrimitive": ("jax/_src/hijax.py", "class HiPrimitive("),
    "HiType": ("jax/_src/hijax.py", "class HiType("),
    "ModuleContext": (MLIR, "class ModuleContext:"),
    "LoweringRuleContext": (MLIR, "class LoweringRuleContext:"),
    "register_lowering": (MLIR, "def register_lowering("),
    "lower_jaxpr_to_module": (MLIR, "def lower_jaxpr_to_module("),
    "lower_jaxpr_to_fun": (MLIR, "def lower_jaxpr_to_fun("),
    "jaxpr_subcomp": (MLIR, "def jaxpr_subcomp("),
    "LoweringResult": (MLIR, "class LoweringResult("),
    "TokenSet": (MLIR, "class TokenSet:"),
    "GridSpec": ("jax/_src/pallas/core.py", "class GridSpec:"),
    "BlockSpec": ("jax/_src/pallas/core.py", "class BlockSpec:"),
    "GridMapping": ("jax/_src/pallas/core.py", "class GridMapping:"),
    "AbstractRef": ("jax/_src/state/types.py", "class AbstractRef("),
    "ReadEffect": ("jax/_src/state/types.py", "class ReadEffect("),
    "WriteEffect": ("jax/_src/state/types.py", "class WriteEffect("),
    "pallas_call": (PALLAS, "def pallas_call("),
    "_trace_kernel_to_jaxpr": (PALLAS, "def _trace_kernel_to_jaxpr("),
    "pallas_call_tpu_lowering_rule": ("jax/_src/pallas/mosaic/pallas_call_registration.py", "def pallas_call_tpu_lowering_rule("),
    "lower_jaxpr_to_pipelined_module": ("jax/_src/pallas/mosaic/lowering.py", "def lower_jaxpr_to_pipelined_module("),
    "lower_module_to_custom_call": ("jax/_src/tpu_custom_call.py", "def lower_module_to_custom_call("),
    "mlir binding": ("jax/_src/lib/mlir/dialects/__init__.py", "from jaxlib.mlir.dialects import stablehlo as hlo"),
    "backend_compile_and_load": ("jax/_src/compiler.py", "def backend_compile_and_load("),
    "compile_or_get_cached": ("jax/_src/compiler.py", "def compile_or_get_cached("),
    "ExecuteReplicated": ("jax/_src/interpreters/pxla.py", "class ExecuteReplicated:"),
    "execute_sharded": ("jax/_src/interpreters/pxla.py", "        results = self.xla_executable.execute_sharded(input_bufs)"),
}


def metadata(root):
    repo = root / "upstream/jax"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    entry = next(line for line in (root / "upstream-sources.lock").read_text().splitlines()
                 if line.startswith("core|upstream/jax|"))
    if head != JAX_PIN or entry.split("|")[3] != JAX_PIN:
        raise ValueError("JAX revision changed; review sources and anchors before rendering")
    files, anchors = {}, {}
    for name, spec in SOURCE_SPECS.items():
        path, needle, *fixed = spec
        if path not in files:
            local = (repo / path).read_text()
            committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{JAX_PIN}:{path}"], text=True)
            if local != committed:
                raise ValueError(f"Source file modified: {path}")
            files[path] = local.splitlines()
        if fixed:
            line = fixed[0]
            if not files[path][line-1].startswith(needle):
                raise ValueError(f"Source anchor moved: {name}")
        else:
            matches = [i+1 for i, value in enumerate(files[path]) if value.startswith(needle)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous source anchor {name}: {matches}")
            line = matches[0]
        anchors[name] = {"repo": "jax", "path": path, "line": line, "needle": needle,
                         "href": f"https://github.com/0xaskr/jax/blob/{JAX_PIN}/{path}#L{line}"}
    return {"source_pins": {"jax": JAX_PIN}, "source_anchors": anchors,
            "scope": "JAX software-stack research: internal abstractions and their lifecycle; no workload examples",
            "evidence": "Fixed-source inspection and SVG generation only; no runtime execution"}


class JaxDiagram(Diagram):
    def __init__(self, meta):
        super().__init__(5000, 10940, "JAX 组件内部架构 / 以 Jaxpr 为中心", "", meta)
        self.base, self.edges, self.nodes, self.labels = [], [], [], []
        self.text_boxes, self.edge_boxes = [], []
        self.bg, self.card = "#fafbf8", "#ffffff"
        self.ink, self.muted = "#203344", "#506577"
        self.colors.update(program="#2866a4", data="#087e78", transform="#8656ac",
                           consume="#ac6430", neutral="#748795")
        self.sections, self.owners = {}, {}
        self.text(80, 95, self.title, 49, bold=True)
        self.text(82, 153, "系统架构 + 内部流程 + 组件协作 · 用对象、字段、规则与输入输出理解 JAX，不依赖具体模型或算子示例", 27, self.muted)
        for x, label, color in [(90,"结构 / 引用","program"),(870,"产生 / 追踪","data"),
                                 (1650,"变换 / 规则","transform"),(2430,"消费 / lowering","consume"),
                                 (3290,"约束 / 组件边界","neutral")]:
            self.labels.append(f'<path d="M{x},222h65" stroke="{self.colors[color]}" stroke-width="5"/>')
            self.text(x+83,231,label,25,self.colors[color])
        self.text(85,287,"A 入口与上下文 → B 核心定义 → C 产生机制 → D 可组合变换 → E 消费与 lowering → F Pallas 分支 → G 对外接口",23,self.muted)
        self.base.append('<rect x="45" y="330" width="4910" height="9610" rx="30" fill="#ffffff" stroke="#8fa7bb" stroke-width="4"/>')

    def section(self, name, y, height, title, subtitle, color="program"):
        self.sections[name] = (85,y,4830,height)
        self.base.append(f'<rect x="85" y="{y}" width="4830" height="{height}" rx="22" fill="#f1f5f7" stroke="#c6d2dd" stroke-width="2"/>')
        self.text(118,y+50,title,35,self.colors[color],bold=True)
        self.text(120,y+93,subtitle,24,self.muted)

    def card_at(self, section, key, col, y, height, tag, title, rows, refs=(), color="program", core=False, dashed=False):
        x = [180,1860,3540][col]
        self.node(key,x,y,1280,height,tag,title,rows,color,dashed=dashed)
        self.owners[key] = section
        if core:
            self.nodes[-1] = self.nodes[-1].replace('fill="#ffffff"','fill="#e4effc"').replace('stroke-width="2"','stroke-width="5"')
        sx=x+24
        for ref in refs:
            label = "↗ " + ref
            self.text(sx,y+height-22,label,19,self.colors[color],href=self.ref(ref),owner=key)
            sx += self.measure(label,19)[1].width+42

    def arrow(self, points, label=None, pos=None, color="program", dashed=False):
        self.edge(points,color,dashed)
        if label:
            self.label(*pos,label,color,size=23)

    def note(self, section, value, y):
        self.text(120,y,value,23,self.muted)

    def save(self, path):
        for key, owner in self.owners.items():
            x,y,w,h=self.boxes[key]
            sx,sy,sw,sh=self.sections[owner]
            if not (sx<x and sy+100<y and x+w<sx+sw and y+h<sy+sh):
                raise ValueError(f"Node outside its JAX section: {key}")
        super().save(path)
        xml=path.read_text()
        start=xml.index('<desc id="desc">')+len('<desc id="desc">')
        end=xml.index('</desc>',start)
        description="JAX 组件内部架构研究。围绕 Jaxpr 展开定义、产生、变换与消费，涵盖 Trace/Tracer、原语规则、抽象类型、effects、MLIR lowering、Pallas 和对外接口；不包含模型或算子实例。"
        path.write_text(xml[:start]+escape(description)+xml[end:])
        ET.parse(path)


def build(meta):
    d=JaxDiagram(meta)
    n,a=d.card_at,d.arrow

    d.section("A",360,970,"A  入口与上下文 · api / pjit / tree_util / stages",
              "函数、值、类型、结构与放置策略承担不同职责；本区给出追踪、变换和执行所需的边界信息。")
    n("A","input",0,510,285,"调用输入","Python 函数 / PyTree / 静态参数",
      ["函数定义计算；PyTree 定义参数与返回值的容器结构。", "动态参数参与追踪；静态参数参与特化与缓存区分。", "Python 对象结构与 Jaxpr 的扁平输入输出分开保存。"],("tree_flatten","jit"))
    n("A","api",1,510,285,"用户接口","jit · grad · vmap",
      ["jit 组织暂存、编译与复用；grad 构造导数计算；vmap 批量化。", "这些接口可组合；不同解释器调用 primitive 的相应规则。", "并非依次运行的三个编译阶段；具体变换关系见 D。"],("jit","grad","vmap"))
    n("A","placement",2,510,285,"数组与放置上下文","ArrayImpl / Mesh / NamedSharding",
      ["ArrayImpl 表示数组值；Mesh / Sharding 描述逻辑放置。", "shape、dtype、分片及布局参与参数推断与 lowering 配置。", "动态数组数据经运行时管理；这些对象并不是 Jaxpr 方程。"],("ArrayImpl","NamedSharding","Mesh"))
    n("A","flatten",0,1000,245,"组织参数","PyTree → leaves + tree definition",
      ["flatten_fun 等适配扁平函数边界；返回值结构单独记录。", "类型信息、静态参数和调用来源一起供追踪入口使用。"],("flatten_fun","tree_flatten"))
    n("A","jit_params",1,1000,245,"特化 / 缓存入口","_infer_params → _trace_for_jit",
      ["输入：函数、args/kwargs、JIT 配置、mesh 与 avals。", "输出：Jaxpr、常量、输入输出树及后续编译参数。"],("_infer_params","_trace_for_jit"))
    n("A","staged",2,1000,245,"阶段对象","Traced / Lowered / Compiled",
      ["Traced 持有 Jaxpr 与调用约定；lower() / compile() 推进阶段。", "阶段对象是程序的包装；得到它们不等于设备执行完成。"],("Traced","Lowered","Compiled"))
    a([(1460,645),(1860,645)],"调用 / 变换接口",(1500,615))
    a([(3540,700),(3140,700)],"类型与放置配置",(3190,670),"neutral")
    a([(820,795),(820,1000)],"展平与重建",(855,910))
    a([(2500,795),(2500,1000)],"JIT 追踪入口",(2535,910),"data")
    a([(1460,1120),(1860,1120)],"结构与参数",(1510,1090))
    a([(3140,1120),(3540,1120)],"jit_trace 包装",(3190,1090))
    d.note("A","阅读关系：A 组织调用边界；B 定义持久程序结构；C 解释追踪如何构造它。缓存命中时不要求再次经过完整追踪。",1300)

    d.section("B",1490,1560,"B  定义 · core.Jaxpr 是组件中心，其他抽象围绕它协作",
              "Jaxpr 保存程序结构和语义约束；Trace / Tracer 是产生或解释程序时的对象，不属于最终 Jaxpr 的结构字段。")
    n("B","consts",0,1650,285,"常量绑定","constvars / consts / all_invars",
      ["constvars = all_invars 的常量前缀；与 consts 中的值逐一对应。", "其余 all_invars 是调用者提供的 invars。", "当前版本 ClosedJaxpr = Jaxpr；.jaxpr 是兼容访问器。"],("Jaxpr","ClosedJaxpr alias"))
    n("B","signature",1,1650,285,"输入输出签名","invars: Var[] / outvars: Atom[]",
      ["输入变量声明调用边界；输出原子指定返回哪些值。", "Atom 为 Var 或 Literal；in_avals / out_avals 来自各原子的 aval。", "PyTree 外壳由阶段对象保存，不编码为这些列表的结构。"],("Jaxpr",))
    n("B","debug",2,1650,285,"诊断与来源","DebugInfo / SourceInfo / EqnContext",
      ["Jaxpr.debug_info：traced_for、函数来源、参数名与结果路径。", "eqn.source_info：方程的 traceback / name_stack。", "eqn.ctx：解释或 lowering 时恢复相应上下文。"],("DebugInfo","JaxprEqn"))
    n("B","atoms",0,2120,345,"值的引用与常量","Var / Literal",
      ["Var 以对象身份区分程序值，并携带 aval；不存放运行时数组。", "Literal 携带已知值 val 与 aval。", "JaxprEqn 的输入可引用 Var 或 Literal；输出绑定 Var。", "变量名由打印器临时分配，不是用户 Python 变量名。"],("Var","Literal"))
    n("B","jaxpr",1,2120,345,"首要抽象 · 持久程序表示","Jaxpr",
      ["边界：all_invars / outvars / consts。", "主体：eqns 记录 primitive 的调用与值依赖。", "语义：effects、is_high；来源：debug_info。", "产生见 C；变换见 D；解释、校验与 lowering 见 E。"],("Jaxpr",),core=True)
    n("B","eqn",2,2120,345,"组成 · 一次原语应用","JaxprEqn",
      ["invars / outvars：使用已有原子，绑定新的结果变量。", "primitive / params：操作定义与静态参数。", "params 可持有子 Jaxpr，表达嵌套计算或控制流主体。", "effects / source_info / ctx：效果、来源与解释上下文。"],("JaxprEqn",))
    n("B","aval",0,2650,285,"类型与约束","AbstractValue / ShapedArray",
      ["aval 描述抽象类型；ShapedArray 包含 shape、dtype 等信息。", "还可包含 sharding、memory_space 等上下文相关约束。", "abstract_eval 消费输入 aval，推导输出 aval 与 effects。"],("AbstractValue","ShapedArray"))
    n("B","effects",1,2650,285,"效果语义","Effect / EffectTypeSet",
      ["方程 effects 汇入 Jaxpr 时，还需处理输入 Ref 与局部分配关系。", "允许微分、控制流或 lowering 的效果由相应集合约束。", "有序效果在 lowering / 执行边界需要 token 依赖，见 E / G。"],("Effect","effect_sets","make_jaxpr_effects"))
    n("B","primitive",2,2650,285,"操作与解释协议","Primitive",
      ["bind 把操作交给当前 Trace；abstract_eval 定义抽象语义。", "impl、微分规则、batching 规则、lowering 规则各司其职。", "这些规则围绕同一个 primitive 注册，不是四种 Jaxpr 类。"],("Primitive","Primitive.bind_with_trace"))
    a([(1460,1790),(1860,1790)],"前缀与调用输入",(1500,1760))
    a([(2500,1935),(2500,2120)],"声明边界",(2535,2040))
    a([(4180,1935),(4180,2010),(2960,2010),(2960,2120)],"携带来源 / 上下文",(3220,1980),"neutral")
    a([(1860,2260),(1460,2260)],"引用值",(1560,2230))
    a([(3140,2260),(3540,2260)],"包含 eqns",(3220,2230))
    a([(820,2465),(820,2650)],".aval",(855,2565))
    a([(2500,2465),(2500,2650)],"effects 字段",(2535,2565),"neutral")
    a([(4180,2465),(4180,2650)],".primitive",(4215,2565))
    d.note("B","约束：变量先定义后使用、单次绑定、类型与原语语义兼容；check_jaxpr 验证这些关系，见 E。Jaxpr 的打印文本不是稳定交换格式。",3010)

    d.section("C",3220,1810,"C  产生 · DynamicJaxprTrace 如何把原语应用记录为 Jaxpr",
              "源码：interpreters/partial_eval.py 与 core.py。箭头给出对象创建和调用关系；函数体运行时可以嵌套其他 Trace。","data")
    n("C","trace_entry",0,3380,300,"追踪输入","trace_to_jaxpr_nocache",
      ["输入：函数、输入 avals / FlatTree、DebugInfo、requires_low。", "保存 parent_trace，创建 DynamicJaxprTrace 与 frame。", "返回值会被展平，随后转为输出 Tracer / aval 信息。"],("trace_to_jaxpr_nocache",),"data")
    n("C","trace",1,3380,300,"解释器对象","Trace / DynamicJaxprTrace",
      ["set_current_trace 设置作用域内的当前解释器。", "process_primitive 决定如何解释一次 primitive 调用。", "可分派 custom_staging_rules，或使用默认追踪逻辑。"],("Trace","DynamicJaxprTrace"),"data")
    n("C","bind",2,3380,300,"操作进入解释器","Primitive.bind → bind_with_trace",
      ["规范化值与类型，选取当前 Trace。", "若 Trace 要求低层表示，高层 primitive 先 to_lojax。", "否则将 primitive、输入值和 params 交给 process_primitive。"],("Primitive.bind","Primitive.bind_with_trace"),"data")
    n("C","tracer",0,3890,330,"追踪期间的值","DynamicJaxprTracer",
      ["new_arg：为输入 aval 创建 Var，再创建关联 Trace 的 Tracer。", "Tracer.aval 表示类型，Tracer.val 指向 Var 或 Literal。", "函数中的后续原语继续接收这些追踪值；它不是最终 IR 节点。", "基类 Tracer 定义解释器中的值协议，Var 定义 Jaxpr 的值引用。"],("new_arg","DynamicJaxprTracer","Tracer"),"data")
    n("C","abstract_eval",1,3890,330,"抽象语义与暂存决策","default_process_primitive",
      ["输入：primitive、输入 Tracer 的 avals、params。", "抽象求值得到 out_avals 与 effects；检查输出个数约定。", "可尝试常量折叠 / 转发；否则构造方程和输出 Tracer。", "抽象求值回答类型与效果，不负责生成 StableHLO。"],("default_process_primitive",),"data")
    n("C","frame",2,3890,330,"追踪状态","JaxprStackFrame",
      ["保存输入变量、常量映射和正在构造的方程。", "TracingEqn 可暂存输入 Tracer 与结果 Var。", "get_eqns 将记录转成持久 JaxprEqn；to_jaxpr 汇总程序。", "最终 Jaxpr 不需要保留这次 Trace / Tracer 的生命周期。"],("JaxprStackFrame","TracingEqn"),"data")
    n("C","capture",0,4410,330,"已知值处理","new_const / _new_const",
      ["可内联的值表示为 Literal；其余常量建立 Var 与值映射。", "constid_to_tracer 复用常量 Tracer，constvar_to_val 保存值。", "捕获常量与普通动态实参是两类来源，最终按调用约定传递。"],("new_const",),"data")
    n("C","emit",1,4410,330,"创建方程与结果","make_eqn → TracingEqn",
      ["输入：in_tracers、out_avals、primitive、params 与 effects。", "为各结果创建 Var，构造 TracingEqn 和输出 Tracer。", "frame 接收方程；后续操作引用这些新结果。"],("make_eqn","TracingEqn"),"data")
    n("C","seal",2,4410,330,"追踪结果","JaxprStackFrame.to_jaxpr",
      ["constvars + invars + outvars + eqns 形成程序结构。", "make_jaxpr_effects 汇总效果，并确定 is_high / debug_info。", "返回 Jaxpr 与常量值；调用方按接口附加或单独保存常量。"],("Frame.to_jaxpr","make_jaxpr_effects"),"data")
    a([(1460,3520),(1860,3520)],"创建与设置",(1510,3490),"data")
    a([(3140,3520),(3540,3520)],"当前 Trace 环境",(3190,3490),"neutral",True)
    a([(2250,3680),(2250,3760),(820,3760),(820,3890)],"new_arg",(1240,3730),"data")
    a([(4180,3680),(4180,3810),(2830,3810),(2830,3890)],"process_primitive 的默认分支",(3010,3780),"data")
    a([(1460,4045),(1860,4045)],"输入 Tracer",(1530,4015),"data")
    a([(2500,4220),(2500,4410)],"类型 / 效果 → 方程",(2535,4340),"data")
    a([(3140,4570),(3340,4570),(3340,4060),(3540,4060)],"暂存",(3370,4315),"data")
    a([(4180,4220),(4180,4410)],"函数返回后封装",(4215,4340),"data")
    a([(1460,4570),(1860,4570)],"Var / Literal 操作数",(1485,4540),"data")
    a([(4180,4740),(4180,4850),(2500,4850)],"结果对应 B 中的 Jaxpr；由 A 的阶段对象或后续解释器使用",(1880,4900),"data")
    d.note("C","追踪缓存、常量折叠、自定义 staging 和嵌套 Trace 会影响实际路径；这不是每次 JIT 调用都重新执行的固定逐步记录。",4990)

    d.section("D",5220,1650,"D  变换 · 同一程序结构由不同解释器与规则处理",
              "ad / batching / partial_eval / hijax 是协作模块。以下按机制分组，不表示必须从左到右或从上到下依次经过。","transform")
    n("D","jvp",0,5380,310,"前向微分","JVPTrace / JVPTracer",
      ["输入值被解释为 primal / tangent 对。", "process_primitive 查 primitive_jvps，产生 primal / tangent 输出。", "规则在 parent_trace 下运行，可与追踪或其他变换组合。"],("JVPTrace","primitive_jvps"),"transform")
    n("D","linearize",1,5380,310,"线性化","LinearizeTrace / linearize_jaxpr",
      ["区分前向计算、切向计算以及反向所需 residual。", "linearize_jaxpr 消费 Jaxpr 与切向非零标记。", "产生前向 / 线性程序及 residual、零值 / 转发等元数据。"],("LinearizeTrace","linearize_jaxpr"),"transform")
    n("D","transpose",2,5380,310,"反向传播","backward_pass / transpose rules",
      ["消费线性计算与所需前向信息，反向传播 cotangent。", "根据 primitive_transposes 调用相应的转置规则。", "输出输入侧 cotangent；并非反转 Jaxpr 方程就完成求导。"],("backward_pass","primitive_transposes"),"transform")
    a([(1460,5530),(1860,5530)],"JVP 规则可供线性化",(1480,5500),"transform",True)
    a([(3140,5530),(3540,5530)],"线性程序 / residual",(3170,5500),"transform")
    d.node("transform_input",1860,5735,1280,105,"共同输入 · 三条独立变换路径","B.Jaxpr + 各自的变换参数",color="program")
    d.owners["transform_input"]="D"
    d.edge([(2500,5840),(2500,5865)],"transform",end=False)
    for x in [820,2500,4180]:
        d.edge([(2500,5865),(x,5865),(x,5890)],"transform")
    n("D","batch",0,5890,320,"批量化","BatchTrace / BatchTracer",
      ["解释值与 batch_dim，结合 AxisData 处理映射轴。", "fancy_primitive_batchers 为原语提供批量规则。", "batch_jaxpr2 消费 Jaxpr + in_axes，再追踪出新 Jaxpr + out_axes。", "无映射参数可交给 parent_trace；不是无条件添加一层循环。"],("BatchTrace","batch_jaxpr2"),"transform")
    n("D","partial",1,5890,320,"部分求值","PartialVal / partial_eval_jaxpr_nounits",
      ["PartialVal 区分已知值与只有 aval 的未知值。", "输入 Jaxpr、unknowns、instantiate；依据数据依赖拆分程序。", "输出 known / unknown Jaxpr、未知输出标记及 residual 类型。", "residual 从已知部分流向未知部分，连接两段计算。"],("PartialVal","partial_eval_jaxpr_nounits"),"transform")
    n("D","dce",2,5890,320,"活性与效果约束","dce_jaxpr / dce_rules",
      ["输入 Jaxpr + used_outputs / instantiate；反向分析依赖。", "按 primitive 的 DCE 规则决定保留方程及其输入。", "返回 new_jaxpr 与 used_inputs；has_effects 过滤可消除效果。", "不能仅凭输出未用就删除操作，也不能把所有 effect 都当成屏障。"],("dce_jaxpr","has_effects"),"transform")
    n("D","hi_primitive",0,6440,300,"高层操作协议","HiPrimitive",
      ["定义 abstract_eval、to_lojax、JVP 与 transpose 等协议。", "bind_with_trace 在高层原语遇到 requires_low 的 Trace 时展开。", "高层操作的分解由原语定义，不是从 Jaxpr 直接选择目标 ISA。"],("HiPrimitive","Primitive.bind_with_trace"),"transform")
    n("D","hi_type",1,6440,300,"高层类型协议","HiType / AbstractValue",
      ["lo_ty 定义低层类型列表；lower_val / raise_val 转换值表示。", "类型协议还参与切向类型、批量轴等变换。", "HiJAX 与 LoJAX 共用 core.Jaxpr；is_high 标记是否仍含高层表示。"],("HiType","Jaxpr"),"transform")
    n("D","lojax",2,6440,300,"表示展开","Traced.lojax → pe.lower_jaxpr",
      ["消费高层 Jaxpr 与低层 avals，得到低层 Jaxpr 和输出类型结构。", "LoJax 阶段包装保存调用树、常量与低层参数约定。", "展开后的 Jaxpr 仍可被 B / C / E 中的通用机制处理。"],("Traced.lojax","lower_jaxpr"),"transform")
    a([(820,6440),(820,6320),(4180,6320),(4180,6440)],"高层原语展开规则",(1100,6300),"transform")
    a([(3140,6590),(3540,6590)],"低层类型 / 值协议",(3180,6560),"transform")
    d.note("D","组合关系：解释已有 Jaxpr 时，primitive.bind 仍进入当前 Trace；因此重解释与重新追踪能构造变换后的 Jaxpr，而不是直接修改原图才叫变换。",6830)

    d.section("E",7070,1610,"E  消费 · 解释、检查与 lowering 对同一 Jaxpr 有不同用途",
              "core 消费程序语义；interpreters/mlir 将程序降为编译输入。MLIR 的原生对象由 jaxlib 绑定提供，外部实现仅在 G 标出。","consume")
    n("E","eval",0,7230,300,"按程序语义解释","eval_jaxpr / jaxpr_as_fun",
      ["输入 Jaxpr、consts 与实参；建立 Var → 值的环境。", "逐方程读取 operands，primitive.bind 产生结果并写回环境。", "返回 outvars 指定的值；结果也可由当前 Trace 再解释。", "jaxpr_as_fun 将持有常量的 Jaxpr 适配为 callable。"],("eval_jaxpr","jaxpr_as_fun"),"consume")
    n("E","check",1,7230,300,"结构与类型检查","check_jaxpr",
      ["验证变量作用域、单次绑定、类型与原语语义等约束。", "输入合法时返回 None；不合法时抛出 JaxprTypeError。", "配置可在追踪或变换后触发检查；它不是设备执行。"],("check_jaxpr",),"consume")
    n("E","print",2,7230,300,"诊断与阅读","pretty_print / DebugInfo",
      ["消费 Jaxpr、打印选项、变量命名及来源信息。", "输出可读文本；effects、来源与类型可按选项展示。", "打印名和排版不属于持久 IR 的稳定外部解析协议。"],("pretty_print","DebugInfo"),"consume")
    n("E","module_context",0,7740,310,"模块级 lowering","lower_jaxpr_to_module / ModuleContext",
      ["输入 Jaxpr、avals、平台、分片 / 布局、效果与参数约定。", "创建模块上下文，组织函数、符号、常量及平台环境。", "lower_jaxpr_to_fun / jaxpr_subcomp 处理函数体与方程序列。"],("lower_jaxpr_to_module","ModuleContext"),"consume")
    n("E","subcomp",1,7740,310,"Jaxpr 值映射为 IR 值","jaxpr_subcomp",
      ["环境 env 将 Var 映射到 MLIR 值；Literal 走常量 lowering。", "逐方程读取 IR 输入，选择规则，写回结果 Var 对应的 IR 值。", "LoweringRuleContext 提供 avals、primitive、tokens 和来源上下文。"],("jaxpr_subcomp","LoweringRuleContext"),"consume")
    n("E","rules",2,7740,310,"按原语与平台分派","register_lowering / lowering rules",
      ["注册公共规则或平台专用规则；规则消费上下文、IR 输入与 params。", "产生 IR 结果及必要的 tokens；嵌套程序由相应规则继续处理。", "规则可以缓存 / 调用化，也可内联生成操作；不保证一方程一操作。"],("register_lowering","TokenSet"),"consume")
    n("E","module_result",0,8250,300,"消费结果","LoweringResult / MLIR Module",
      ["模块承载 StableHLO 等方言操作及编译所需属性。", "LoweringResult 同时保存 keepalive、host callback 等附加信息。", "这是后续编译的输入，不是已经加载的设备程序。"],("LoweringResult","mlir binding"),"consume")
    n("E","lowered",1,8250,300,"封装降级结果","Traced.lower → Lowered",
      ["_resolve_and_lower 组织类型、mesh 与 lowering 参数。", "Lowered 持有 lowering 及输入输出约定，支持 IR / 文本查询。", "compile() 再产生 Compiled；阶段推进与执行调用分开。"],("_resolve_and_lower","Lowered"),"consume")
    n("E","effects_tokens",2,8250,300,"语义约束跨越 lowering","有序效果 / TokenSet",
      ["TokenSet 将 effect 与相应 token IR 值关联。", "方程 lowering 消费 tokens_in，更新 tokens_out。", "所支持的效果需通过 lowerable_effects 等检查，执行端继续处理依赖。"],("effect_sets","TokenSet"),"neutral")
    a([(1460,7880),(1860,7880)],"函数体 / 上下文",(1505,7850),"consume")
    a([(3140,7840),(3540,7840)],"选择并调用",(3200,7810),"consume")
    a([(3540,7990),(3140,7990)],"结果 IR 值",(3210,7960),"consume")
    a([(820,8050),(820,8250)],"模块与附加信息",(855,8170),"consume")
    a([(1460,8400),(1860,8400)],"JAX 阶段包装",(1505,8370),"consume")
    a([(4180,8250),(4180,8050)],"token 输入 / 输出",(4215,8170),"neutral")
    d.note("E","接口区别：abstract_eval 产生 avals / effects；lowering 产生 IR 值 / 操作；compile 产生可执行程序；Execute 才提交执行。",8640)

    d.section("F",8870,1020,"F  Pallas / state · JAX 内部的可选 kernel 与可变引用分支",
              "Pallas 继续使用 Jaxpr、Primitive、aval 与 effects；它增加的是 kernel 调用约定、块映射、Ref 语义和平台 lowering。","transform")
    n("F","grid",0,9015,300,"调用与块映射","GridSpec / BlockSpec → GridMapping",
      ["GridSpec 描述网格与 I/O 规格；BlockSpec 描述块视图和映射。", "GridMapping 规范化动态 grid、index、inputs / outputs 与 scratch 约定。", "它为 kernel 和 index-map 的追踪提供参数类型与上下文。"],("GridSpec","BlockSpec","GridMapping"),"transform")
    n("F","refs",1,9015,300,"可变位置的抽象类型","AbstractRef / Ref effects",
      ["AbstractRef 包含 inner_aval、memory_space 与 kind。", "读取 / 写入由原语与 ReadEffect / WriteEffect 等语义表达。", "Ref 是可读写位置，不等于一个不可变数组值，也不是 PJRT Buffer。"],("AbstractRef","ReadEffect","WriteEffect"),"transform")
    n("F","kernel_jaxpr",2,9015,300,"产生内层程序","_trace_kernel_to_jaxpr",
      ["输入 kernel 函数、kernel avals、GridMapping 和变换信息。", "复用 partial_eval 追踪与 DCE，得到 kernel Jaxpr 与常量。", "内层 Jaxpr 描述 Ref 操作；与外层程序使用同一种结构。"],("_trace_kernel_to_jaxpr",),"data")
    n("F","pallas_call",0,9490,275,"外层调用","pallas_call primitive",
      ["外层方程的参数携带 kernel Jaxpr、GridMapping 等。", "普通程序通过这个调用节点连接内层 kernel 与输入输出约定。", "JAX 的变换机制继续通过相应原语规则处理该节点。"],("pallas_call",),"transform")
    n("F","platform",1,9490,275,"消费内层程序","平台 lowering / interpret 分支",
      ["实现由平台与模式选择；不是所有分支都生成 Mosaic TPU MLIR。", "TPU lowering 消费 kernel Jaxpr、grid mapping 和编译参数。", "其他路径保留各自的平台规则与解释语义。"],("pallas_call_tpu_lowering_rule",),"consume")
    n("F","mosaic",2,9490,275,"TPU 路径的表示交接","Mosaic TPU MLIR → custom_call",
      ["lower_jaxpr_to_pipelined_module 产生 kernel MLIR。", "lower_module_to_custom_call 将其封装回外层模块。", "Mosaic TPU MLIR ≠ LLO，也不是最终设备机器码。"],("lower_module_to_custom_call",),"consume",dashed=True)
    a([(1460,9150),(1860,9150)],"块类型 / 内存空间",(1490,9120),"transform")
    a([(3140,9150),(3540,9150)],"kernel 输入 avals",(3170,9120),"data")
    a([(4180,9315),(4180,9400),(820,9400),(820,9490)],"内层程序作为外层调用参数",(1870,9380),"transform")
    a([(1460,9620),(1860,9620)],"规则分派",(1550,9590),"consume")
    a([(3140,9620),(3540,9620)],"TPU 分支",(3210,9590),"consume",True)
    d.note("F","边界：本区只覆盖 JAX 侧 kernel 表示与封装；外层 custom_call 继续进入 E / G 的模块与编译接口，不在本图补造 TPU 私有后端阶段。",9848)

    d.section("G",10065,660,"G  JAX → jaxlib · 只标出对外接口，不展开下游组件内部",
              "JAX 负责解释、变换、构造编译输入与组织调用；原生绑定、设备运行时对象和后端编译器的内部关系留给后续组件图。","neutral")
    n("G","binding",0,10225,340,"构造编译输入的边界","jaxlib.mlir / StableHLO bindings",
      ["JAX 的 mlir.py 使用 jaxlib 提供的原生 MLIR / 方言绑定。", "JAX 中的 lowering 规则决定如何构造模块与操作。", "StableHLO 是 IR 方言；jaxlib 是提供绑定与运行时接口的组件。", "模块可包含多种方言或 custom_call；并非 jaxlib 自己定义一套 Jaxpr。"],("mlir binding","lower_jaxpr_to_module"),"neutral",dashed=True)
    n("G","compile",1,10225,340,"编译 / 加载调用边界","compile_or_get_cached → backend",
      ["compiler.py 接收 module、编译选项、设备等信息。", "常规 backend_compile_and_load 调用 compile_and_load。", "编译缓存与 compile-only 分支各有条件，不要求每次调用重新编译。", "返回的程序被 JAX 阶段 / 执行包装持有，内部实现由下游组件负责。"],("compile_or_get_cached","backend_compile_and_load"),"neutral",dashed=True)
    n("G","execute",2,10225,340,"数据与执行调用边界","ExecuteReplicated → execute_sharded",
      ["JAX 的 input handler 准备输入数组，处理保留实参与必要 token。", "已加载程序的 execute_sharded 提交计算；output handler 包装结果。", "调用含效果时处理对应 token 与运行时依赖。", "输入数组不是被 lowering 成 HLO；异步输出句柄不证明设备已完成。"],("ExecuteReplicated","execute_sharded"),"neutral",dashed=True)
    a([(1460,10400),(1860,10400)],"module + options",(1490,10370),"consume")
    a([(3140,10400),(3540,10400)],"可执行程序",(3200,10370),"consume")
    d.note("G","阅读终点：IFRT Array、PJRT Buffer、HLO 与 CPU/GPU/TPU 后端的定义和内部协作，分别在其所属组件图中继续展开。",10680)
    d.text(85,10805,f"源码：JAX {JAX_PIN} · 文件、符号和源码链接均按固定提交核对；复用仓库锁定源码，不改变环境。",23,d.muted)
    d.text(85,10854,"证据边界：本图为源码与架构研究；未运行模型、CPU/GPU/TPU 工作负载、离线 TPU 编译或性能测试。",23,d.muted)
    d.text(85,10900,"维护：tools/render_jax_internal_stack.py · 先输出临时 SVG / PNG 检查，再更新仓库中的组件图。",22,d.muted)
    return d


def main():
    """Keep the former command pointed at the single maintained JAX diagram."""
    from render_software_stack_component_flows import main as render_selected
    render_selected(only='jax')


if __name__=="__main__":
    main()
