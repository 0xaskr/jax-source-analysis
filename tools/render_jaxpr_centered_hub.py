#!/usr/bin/env python3
"""Render the selected Jaxpr-centered hub diagram from pinned sources."""

from __future__ import annotations

import argparse
import copy
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

from render_jax_internal_stack import ROOT, JAX_PIN, metadata
from render_overview_software_stack_components import Diagram, preview


DEFINITIONS = [
    ("边界与常量", "invars / outvars / consts", ["输入声明与输出引用界定一段程序。", "all_invars 的已附常量前缀对应 constvars / consts。", "ClosedJaxpr = Jaxpr；常量也可由调用约定单独传递。"], ("Jaxpr", "ClosedJaxpr alias")),
    ("程序主体", "eqns / JaxprEqn / Primitive", ["每个方程保存输入、输出、primitive 与 params。", "Var / Literal 连接值依赖；params 可包含子 Jaxpr。", "原语的抽象、微分、batching、lowering 规则解释操作。"], ("JaxprEqn", "Primitive")),
    ("语义与来源", "aval / effects / DebugInfo", ["aval 规定抽象类型；effects 表达可观察效果。", "is_high 标记高层表示；DebugInfo 记录函数来源。", "方程另带 source_info / ctx，用于诊断与解释上下文。"], ("AbstractValue", "Effect", "DebugInfo")),
    ("共同约束", "Jaxpr 的合法性与身份", ["变量先定义后使用、单次绑定、类型与原语语义兼容。", "Trace / Tracer 属于构造或解释机制，不是 Jaxpr 字段。", "变换产物仍遵守这套定义；check_jaxpr 检查其约束。"], ("check_jaxpr", "Jaxpr")),
]

PRODUCERS = [
    ("产生所需输入", "Python 函数 + avals + 上下文", ["函数提供计算；avals 提供输入的抽象类型。", "PyTree / 静态参数 / DebugInfo 组织追踪边界。", "目标产物是 Jaxpr，函数与 tracing 只是它的来源和机制。"], ("_trace_for_jit", "trace_to_jaxpr_nocache")),
    ("生成 Jaxpr 的机制", "DynamicJaxprTrace / Tracer", ["Primitive.bind 分派给当前 Trace。", "抽象求值产生 out_avals / effects；make_eqn 记录方程。", "frame 保存变量、常量映射和 TracingEqn。"], ("DynamicJaxprTrace", "make_eqn")),
    ("真正构造 Jaxpr 的位置", "JaxprStackFrame.to_jaxpr", ["get_eqns 把追踪记录转换为 JaxprEqn。", "汇总输入、输出、方程、effects、来源，调用 Jaxpr(...)。", "返回 Jaxpr + constvals；调用方附常量或单独保存。"], ("Frame.to_jaxpr", "Jaxpr")),
]

TRANSFORMS = [
    ("dce", "dce_jaxpr", ["输入：Jaxpr + used_outputs / instantiate。", "按依赖、效果与 dce_rules 保留所需方程。", "输出：新 Jaxpr + used_inputs。"], ("dce_jaxpr", "has_effects")),
    ("batch", "batch_jaxpr2", ["输入：Jaxpr + AxisData / in_axes。", "用 batching 规则重解释并再次追踪。", "输出：批量化 Jaxpr + out_axes。"], ("batch_jaxpr2", "BatchTrace")),
    ("pe", "partial_eval_jaxpr_nounits", ["输入：Jaxpr + unknowns / instantiate。", "依已知与未知依赖拆分，并引入 residual 连接。", "输出：known / unknown 两个 Jaxpr + 元数据。"], ("partial_eval_jaxpr_nounits",)),
    ("ad", "linearize_jaxpr", ["输入：Jaxpr + 切向非零标记等。", "线性化解释器组织前向、线性计算及 residual。", "输出：前向 / 线性 Jaxpr + 类型和标记信息。"], ("linearize_jaxpr", "LinearizeTrace")),
    ("lo", "pe.lower_jaxpr", ["输入：高层 Jaxpr + 低层 avals。", "按 HiPrimitive / HiType 协议展开表示。", "输出：低层 Jaxpr + 输出类型结构。"], ("lower_jaxpr", "HiPrimitive")),
    ("rebuild", "Jaxpr.replace / Jaxpr(...)", ["输入：已有程序结构及待替换的字段。", "由变换实现重建签名、方程与对应元数据。", "输出：同一 Jaxpr 抽象的新程序对象。"], ("Jaxpr", "dce_jaxpr")),
]

CONSUMERS = [
    ("eval", "eval_jaxpr", ["输入：Jaxpr + consts / 实参。", "逐方程读取值、调用 primitive.bind、读取 outvars。"], "值 / 当前 Trace 的解释结果", ["解释结果可以是普通值，也可处在其他 Trace 下。", "消费 Jaxpr 的语义，不要求在此产生另一种 IR。"], ("eval_jaxpr",)),
    ("check", "check_jaxpr", ["输入：Jaxpr。", "检查变量、类型与程序良构性。"], "通过 / JaxprTypeError", ["通过时返回 None；否则报告无效程序。", "这是程序校验，不是设备执行结果。"], ("check_jaxpr",)),
    ("print", "pretty_print", ["输入：Jaxpr + 打印选项。", "读取结构、类型、effects 和来源信息。"], "可读 Jaxpr 文本", ["供诊断与阅读；打印不改变程序语义。", "临时变量名与文本排版不是稳定交换协议。"], ("pretty_print",)),
    ("mlir", "lower_jaxpr_to_module", ["输入：Jaxpr + 平台 / 类型 / 分片等上下文。", "jaxpr_subcomp 按原语规则生成 IR 值与操作。"], "MLIR Module / LoweringResult", ["通常承载 StableHLO 等方言及附加编译信息。", "此处 Jaxpr 被消费为编译输入，后端编译另行发生。"], ("lower_jaxpr_to_module", "jaxpr_subcomp")),
    ("kernel", "Pallas / Mosaic TPU lowering", ["输入：kernel Jaxpr + GridMapping / Ref 上下文。", "满足 TPU 路径条件时，消费内层 kernel 程序。"], "Mosaic TPU MLIR / custom_call", ["kernel IR 封装回外层模块，由后端继续处理。", "Mosaic TPU MLIR 不是 LLO，也不是最终机器码。"], ("lower_jaxpr_to_pipelined_module",)),
]


class FocusDiagram(Diagram):
    def __init__(self, width, height, title, subtitle, meta):
        super().__init__(width,height,title,"",copy.deepcopy(meta))
        self.base,self.edges,self.nodes,self.labels=[],[],[],[]
        self.text_boxes,self.edge_boxes=[],[]
        self.bg,self.card="#fafbf8","#ffffff"
        self.ink,self.muted="#203344","#52687b"
        self.colors.update(program="#2866a4",data="#087e78",transform="#8656ac",consume="#ae622c",neutral="#788b9d")
        self.roles,self.relations={},[]
        self.text(80,95,title,49,bold=True)
        self.text(83,151,subtitle,26,self.muted)
        for x,label,color in [(90,"Jaxpr 对象","program"),(720,"产生 Jaxpr","data"),(1400,"Jaxpr → 新 Jaxpr","transform"),(2250,"消费 Jaxpr","consume"),(2940,"定义 / 约束","neutral")]:
            self.labels.append(f'<path d="M{x},222h58" stroke="{self.colors[color]}" stroke-width="5"/>')
            self.text(x+76,231,label,25,self.colors[color])
        self.text(85,287,"蓝色框始终是研究对象 Jaxpr；其他颜色表示对它的作用。构造机制、变换规则和消费者不是与 Jaxpr 并列的研究中心。",24,self.muted)

    def panel(self,key,x,y,w,h,title,subtitle,role):
        self.roles[key]=role
        self.base.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="#f0f4f7" stroke="#b1c3d2" stroke-width="3"/>')
        self.text(x+30,y+52,title,35,bold=True)
        for i,line in enumerate(self.wrap(subtitle,w-60,23)):
            self.text(x+32,y+94+i*32,line,23,self.muted)

    def box(self,key,x,y,w,h,tag,title,rows,refs=(),role="definition",color=None):
        colors={"jaxpr":"program","producer":"data","transform":"transform","consumer":"consume","result":"neutral","definition":"neutral"}
        color=color or colors[role]
        self.roles[key]=role
        self.node(key,x,y,w,h,tag,title,rows,color)
        if role=="jaxpr":
            self.nodes[-1]=self.nodes[-1].replace('fill="#ffffff"','fill="#e3effd"').replace('stroke-width="2"','stroke-width="5"')
        sx=x+24
        for ref in refs:
            label="↗ "+ref
            self.text(sx,y+h-22,label,18,self.colors[color],href=self.ref(ref),owner=key)
            sx+=self.measure(label,18)[1].width+35

    def ir(self,key,x,y,w,h,title="Jaxpr",rows=None):
        rows=rows or [
            "边界：all_invars / invars / outvars / consts。",
            "主体：eqns → JaxprEqn → primitive、params 与值依赖。",
            "语义：avals、effects、is_high；来源：debug_info。",
            "定义、产生、变换与消费都指向这个抽象。"]
        self.roles[key]="jaxpr"
        self.boxes[key]=(x,y,w,h)
        self.nodes.append(f'<rect id="{key}" x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="#e3effd" stroke="{self.colors["program"]}" stroke-width="5"/>')
        self.text(x+30,y+35,"研究对象 · core.Jaxpr",20,self.colors["program"],bold=True,owner=key)
        self.text(x+30,y+101,title,48,bold=True,owner=key)
        baseline=y+154
        for row in rows:
            for part in self.wrap(row,w-60,24):
                self.text(x+30,baseline,part,24,self.muted,owner=key)
                baseline+=37
        self.text(x+30,y+h-22,"↗ Jaxpr",19,self.colors["program"],href=self.ref("Jaxpr"),owner=key)

    def link(self,source,target,kind,points,label=None,pos=None):
        expected={"produce":("producer","jaxpr"),"transform_in":("jaxpr","transform"),
                  "transform_out":("transform","jaxpr"),"consume":("jaxpr","consumer"),
                  "define":("definition","jaxpr")}
        if kind in expected:
            if (self.roles[source],self.roles[target])!=expected[kind]:
                raise ValueError(f"Jaxpr is not the subject of {source} -> {target}: {kind}")
        color={"produce":"data","transform_in":"transform","transform_out":"transform",
               "consume":"consume","define":"neutral","internal":"neutral","repeat":"transform"}[kind]
        self.relations.append({"source":source,"target":target,"kind":kind})
        self.edge(points,color,dashed=kind in {"define","repeat"})
        if label: self.label(*pos,label,color,size=23)

    def foot(self,y):
        self.text(85,y,"范围：JAX 内部抽象研究；不含模型 / 算子实例或 call-to-LLO 跟踪。变换可按需组合，不把图中所有算子当作固定流水线。",23,self.muted)
        self.text(85,y+43,f"固定源码：JAX {JAX_PIN} · 小字链接定位源码；仅源码 / 图示校验，不构成设备执行证据。",23,self.muted)

    def save(self,path):
        kinds={r["kind"] for r in self.relations}
        assert {"produce","transform_in","transform_out","consume","define"}<=kinds
        self.meta["diagram_relations"]=self.relations
        self.meta["node_roles"]=self.roles
        super().save(path)
        xml=path.read_text(); start=xml.index('<desc id="desc">')+len('<desc id="desc">');end=xml.index('</desc>',start)
        description="以 Jaxpr 对象为核心的四视角架构图。定义说明它的结构；生成机制明确输出 Jaxpr；变换明确输入和输出 Jaxpr；消费者明确读取 Jaxpr 并产生相应结果。"
        path.write_text(xml[:start]+escape(description)+xml[end:]);ET.parse(path)


def producer_column(d,prefix,x,y,w,step=470,h=335):
    ids=[]
    for i,(tag,title,rows,refs) in enumerate(PRODUCERS):
        key=f"{prefix}_{i}"; ids.append(key)
        d.box(key,x,y+i*step,w,h,tag,title,rows,refs,"producer")
        if i:
            d.link(ids[i-1],key,"internal",[(x+w/2,y+(i-1)*step+h),(x+w/2,y+i*step)],"继续构造所需信息",(x+w/2+30,y+i*step-57))
    return ids


def hub(meta):
    d=FocusDiagram(8520,5980,"01 / 中心辐射 · 四个问题围绕一个 Jaxpr", "Jaxpr 是视觉中心：左侧产生它，上方定义它，下方改变它，右侧消费它。",meta)
    d.panel("definition",2600,350,2480,1140,"定义 · Jaxpr 是什么","结构和语义约束解释中央蓝色对象；灰色线不是计算执行顺序。","definition")
    for i,(tag,title,rows,refs) in enumerate(DEFINITIONS):
        col,row=i%2,i//2
        d.box("def_"+str(i),2690+1200*col,525+465*row,1100,330,tag,title,rows,refs)
    d.ir("jaxpr",2780,1900,2080,650,rows=["输入输出：all_invars / invars / outvars。","常量：constvars / consts，或由调用方单独保存。","程序体：eqns 中的 JaxprEqn 引用 Primitive 与 params。","值与约束：Var / Literal / aval / effects。","来源与层次：DebugInfo / is_high。","谁产生、谁变换、谁消费，都以这个对象为输入或输出。"])
    d.link("definition","jaxpr","define",[(3840,1490),(3840,1900)],"这些字段和约束定义 Jaxpr",(3890,1690))

    d.panel("generation",100,880,2200,2320,"产生 · 谁构造出 Jaxpr","框内只有输入和构造机制；最终生成箭头直接落到中央 Jaxpr。","producer")
    prod=producer_column(d,"producer",220,1110,1960,step=610,h=360)
    d.box("constructor",220,2940,1960,190,"持久表示的构造入口","core.Jaxpr(...)",["构造出的程序结构不依赖本次 Trace / Tracer 继续存活。"],("Jaxpr",),"producer")
    d.link(prod[-1],"constructor","internal",[(1200,2690),(1200,2940)],"汇总字段并构造",(1250,2830))
    d.link("constructor","jaxpr","produce",[(2180,3020),(2480,3020),(2480,2370),(2780,2370)],"生成 Jaxpr",(2505,2720))

    d.panel("consumers",5390,710,3030,2720,"消费 · 谁读取 Jaxpr，得到什么","橙色箭头的起点是中央 Jaxpr；结果类型在最右侧明确列出。","consumer")
    d.edge([(4860,2150),(5190,2150)],"consume",end=False)
    for i,(name,title,rows,out,body,refs) in enumerate(CONSUMERS):
        y=930+480*i; c="consumer_"+name; result="result_"+name
        d.box(c,5490,y,1280,300,"Jaxpr 的消费者",title,rows,refs,"consumer")
        d.box(result,7130,y,1170,300,"消费结果",out,body,(),"result")
        d.link("jaxpr",c,"consume",[(5190,2150),(5190,y+150),(5490,y+150)])
        d.link(c,result,"internal",[(6770,y+150),(7130,y+150)],"得到",(6870,y+120))

    d.panel("transforms",1970,3570,3990,2200,"变换 · Jaxpr 如何变成新的 Jaxpr","每条分支都消费 Jaxpr；可以产生一个或多个新 Jaxpr，并返回分析信息。各分支独立选用。","transform")
    for i,(name,title,rows,refs) in enumerate(TRANSFORMS):
        col,row=i%3,i//3; x=2080+1290*col; y=3790+560*row; key="transform_"+name
        d.box(key,x,y,1150,340,"输入 Jaxpr → 新程序",title,rows,refs,"transform")
        d.link("jaxpr",key,"transform_in",[(3580,2550),(3580,3410),(1870,3410),(1870,3730),(x-35,3730),(x-35,y+130),(x,y+130)])
    d.ir("new_jaxpr",2830,5200,2100,420,"新 Jaxpr / 多个 Jaxpr",["仍是 core.Jaxpr；结构和语义继续受上方定义约束。","可能是删减、批量化、线性化、拆分或低层展开后的程序。","连同 used_inputs、out_axes、residual 等附加信息返回。"])
    for i,(name,_,_,_) in enumerate(TRANSFORMS):
        col,row=i%3,i//3; x=2080+1290*col; y=3790+560*row
        d.link("transform_"+name,"new_jaxpr","transform_out",[(x+1150,y+220),(x+1185,y+220),(x+1185,5010),(3880,5010),(3880,5200)])
    d.link("new_jaxpr","jaxpr","repeat",[(2830,5400),(1750,5400),(1750,3350),(2590,3350),(2590,2500),(2780,2500)],"下一轮仍以 Jaxpr 为对象",(1810,3310))
    d.foot(5850)
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--preview-dir",type=Path)
    args=parser.parse_args()
    output=args.output_dir or args.root/"research/software-stack/jax"
    output.mkdir(parents=True,exist_ok=True)
    if args.preview_dir: args.preview_dir.mkdir(parents=True,exist_ok=True)
    meta=metadata(args.root)
    path=output/"jaxpr-centered-hub.svg"
    hub(meta).save(path)
    if args.preview_dir: preview(path,args.preview_dir/"hub.png",width=2100)


if __name__=="__main__":
    main()
