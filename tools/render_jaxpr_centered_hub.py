#!/usr/bin/env python3
"""Jaxpr four-view content and compatibility entrypoint for the selected SVG."""

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



def main():
    from render_software_stack_component_flows import main as render_selected
    render_selected(only='jax')


if __name__ == '__main__':
    main()
