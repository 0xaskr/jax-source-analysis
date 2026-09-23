#!/usr/bin/env python3
"""Locate reviewed API anchors in the kickoff's pinned source trees."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from runtime_chain_index import SITES as RUNTIME_SITES, EDGES as RUNTIME_EDGES, IDS as RUNTIME_IDS
from shardy_index import SITES as SHARDY_SITES, EDGES as SHARDY_EDGES, IDS as SHARDY_IDS


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# id, component, layer, path within component, anchor, minimum line, input,
# output, constraint. Descriptions are reviewed source claims, not runtime traces.
SITES = [
    ("jax.matmul", "jax", "Python API", "jax/_src/numpy/tensor_contractions.py", "def matmul(", 138,
     "数组 lhs/rhs、precision、preferred_element_type、out_sharding", "按广播与收缩维度定义的数组",
     "检查秩、batch 维度与 sharding；构造 dimension_numbers 后调用 lax.dot_general。"),
    ("jax.dot-general", "jax", "Primitive API", "jax/_src/lax/lax.py", "def dot_general(", 2528,
     "lhs/rhs 与收缩/batch 维度", "dot 运算的结果数组",
     "此 revision 的 dot_general 是 dot 的包装入口；不能按旧版本假设其内部直接 bind。"),
    ("jax.dot", "jax", "Primitive API", "jax/_src/lax/lax.py", "def dot(", 2544,
     "数组与 dimension_numbers、precision、结果类型/分片", "dot_general primitive 的结果",
     "precision 和 preferred_element_type 是不同参数；不把数学 dot 等同于某一种硬件指令。"),
    ("jax.jit", "jax", "Transformation", "jax/_src/api.py", "def jit(", 204,
     "Python callable 与静态参数、donation、sharding、编译选项", "支持 trace/lower/compile 的 JIT callable",
     "静态参数与输入抽象类型影响 specialization；调用接口不意味着每次都重编译。"),
    ("jax.grad", "jax", "Transformation", "jax/_src/api.py", "def grad(", 426,
     "标量结果函数与被求导参数位置", "求梯度的 callable",
     "通过 value_and_grad 构造结果；本实验使用实数标量 sum(square(matmul))。"),
    ("jax.vmap", "jax", "Transformation", "jax/_src/api.py", "def vmap[", 1003,
     "callable、in_axes、out_axes 与轴信息", "应用 batching 规则的 callable",
     "vmap 是程序变换；本次共同 rhs 的样本形成高秩 dot，不证明执行了 B 次独立 launch。"),
    ("jax.trace-for-jit", "jax", "Tracing", "jax/_src/pjit.py", "def _trace_for_jit(", 485,
     "函数、JIT 配置、mesh、输入 avals 与参数树", "PjitParams/Jaxpr 及输入输出结构",
     "Python tracing 与 native 编译分开记录；trace_count 不能单独证明编译次数。"),
    ("jax.pjit-lower", "jax", "Lowering", "jax/_src/pjit.py", "def _pjit_lower(", 1224,
     "Jaxpr、sharding/layout、donation、平台与编译参数", "pxla.MeshComputation",
     "把这些参数交给 lower_sharding_computation，不直接输出机器码。"),
    ("jax.lower-sharding", "jax", "Lowering", "jax/_src/interpreters/pxla.py", "def lower_sharding_computation(", 976,
     "Jaxpr、mesh、sharding/layout 与平台", "可继续 compile 的计算对象",
     "需要区分多设备/效果约束；单 CPU 示例不能证明多设备 sharding 策略。"),
    ("jax.cached-mlir-lowering", "jax", "Lowering", "jax/_src/interpreters/pxla.py", "def _cached_lowering_to_hlo(", 717,
     "Jaxpr、输入输出 avals、设备和别名信息", "lower_jaxpr_to_module 的 lowering 结果",
     "函数名含 hlo，但这里创建 MLIR module；不要据名称把它当成 XLA HloModule pass pipeline。"),
    ("jax.mlir-module", "jax", "Jaxpr to MLIR", "jax/_src/interpreters/mlir.py", "def lower_jaxpr_to_module(", 1324,
     "Jaxpr 与平台/设备、effects、alias/sharding 等 lowering 上下文", "LoweringResult，包含 MLIR module",
     "primitive lowering 使用平台注册规则；生成表示不同于执行后端编译。"),
    ("jax.dot-lowering", "jax", "Primitive lowering", "jax/_src/lax/lax.py", "def _dot_general_lower(", 6264,
     "lowering context、MLIR lhs/rhs、dimension_numbers 和 precision", "stablehlo.dot_general 及必要转换",
     "构造 DotDimensionNumbers 和 precision_config；输出仍是 MLIR value。"),
    ("jax.lowered-compile", "jax", "Staged API", "jax/_src/stages.py", "  def compile(", 643,
     "Lowered 与 compiler_options/device_assignment", "Compiled",
     "Lowered.compiler_ir('hlo') 是导出接口，不是逐 native pass 的执行记录。"),
    ("jax.backend-compile", "jax", "Python/native boundary", "jax/_src/compiler.py", "def backend_compile_and_load(", 335,
     "backend Client、MLIR module、设备、CompileOptions、host callbacks", "LoadedExecutable 或编译异常",
     "真实 backend 与 CompileOnlyPyClient 分支不同；普通路径调用 backend.compile_and_load。"),
    ("jaxlib.compile", "jax", "jaxlib binding", "jaxlib/py_client.cc", "PyClient::CompileAndLoad(", 475,
     "MLIR module、设备和编译参数", "PyLoadedExecutable",
     "克隆 module，包装 HloProgram/IFRT 编译选项；这是 JAX 仓库中的 jaxlib 源码。"),
    ("jaxlib.ifrt-call", "jax", "jaxlib binding", "jaxlib/py_client.cc", "PyClient::CompileAndLoadIfrtProgram(", 372,
     "IFRT Program 与 CompileOptions", "IFRT 已加载 executable 的 Python wrapper",
     "源码在调用 IFRT compiler 和等待 future 时释放 GIL；不据此推断旧挂起案例根因已修复。"),
    ("ifrt.compile", "xla", "IFRT", "xla/python/pjrt_ifrt/pjrt_compiler.cc", "tsl::Future<LoadedExecutableRef> PjRtCompiler::CompileAndLoad(", 91,
     "Program 与 IFRT CompileOptions", "LoadedExecutableRef future",
     "这个实现要求 HloProgram，翻译设备 ID，再创建 PjRtLoadedExecutable。"),
    ("ifrt.pjrt-create", "xla", "IFRT/PJRT", "xla/python/pjrt_ifrt/pjrt_executable.cc", "absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::Create(", 744,
     "MLIR module、PJRT client、编译选项与设备", "包装的 PJRT executable",
     "调用 client->pjrt_client()->CompileAndLoad；调用前读取捐赠、shape/layout 等 metadata。"),
    ("xla.mlir-to-hlo", "xla", "MLIR/HLO boundary", "xla/pjrt/mlir_to_hlo.cc", "absl::Status MlirToXlaComputation(", 99,
     "MLIR ModuleOp 与 tuple/shardy 编译选项", "XlaComputation/HLO",
     "先处理 CHLO、常量作用域及分片等转换，再导出；不是只更换文本格式。"),
    ("xla.stablehlo-export", "xla", "MLIR/HLO boundary", "xla/hlo/translate/stablehlo.cc", "absl::Status ConvertStablehloToHloProtoInternal(", 103,
     "MLIR module 与导出选项", "HloProto",
     "当前代码设置 direct_stablehlo_to_hlo=true；不能假设所有操作都先完整改写为 MHLO。"),
    ("xla.cpu-passes", "xla", "CPU compiler", "xla/service/cpu/cpu_compiler.cc", "absl::Status CpuCompiler::RunHloPasses(HloModule*", 1188,
     "HloModule、AOT 标志、目标特性与编译参数", "被优化的 HLO module",
     "布局分配前后是不同 pipeline；具体 pass 还受构建、平台与选项控制。"),
    ("xla.cpu-post-layout", "xla", "CPU compiler", "xla/service/cpu/cpu_compiler.cc", "absl::Status CpuCompiler::RunHloPassesAfterLayoutAssn(", 998,
     "已有布局的 HloModule 与目标特性", "规范化、库改写、fusion 等后的 HLO",
     "CpuInstructionFusion、FusionWrapper 与库专用改写不同；HLO 中出现 fusion 不等于业务两算子融合已完成。"),
    ("xla.pass-pipeline", "xla", "Pass infrastructure", "xla/hlo/pass/hlo_pass_pipeline.cc", "absl::StatusOr<bool> HloPassPipeline::RunPassesInternal(", 141,
     "HLO、DebugOptions、execution_threads", "changed 标志与被修改的 HLO",
     "字面量 dump regex .* 特判跳过未变化的 pass；.+ 可避免这项特判。"),
    ("xla.cpu-codegen", "xla", "CPU code generation", "xla/service/cpu/cpu_compiler.cc", "CpuCompiler::CompileCpuExecutable(", 1727,
     "优化后的 HloModule 与 IrCompiler/目标选项", "CpuExecutable",
     "创建 LLVMContext/Module；随后调度、buffer assignment、thunk/IR emission 和编译。"),
    ("xla.cpu-schedule", "xla", "CPU scheduling", "xla/service/cpu/cpu_compiler.cc", "absl::StatusOr<HloSchedule> CpuCompiler::CreateHloSchedule(", 2441,
     "HloModule", "HloSchedule",
     "源码按配置选择 DFS memory scheduler 或 BFS scheduler；不代表 TPU 调度实现。"),
    ("xla.cpu-buffers", "xla", "Buffer assignment", "xla/service/cpu/cpu_compiler.cc", "CpuCompiler::CreateBufferAssignment(", 2459,
     "带 schedule 的 HloModule", "BufferAssignment",
     "使用 SequentialHloOrdering、大小函数、alias 信息和对齐要求；估计不等于实测运行峰值。"),
    ("xla.dot-cost", "xla", "Cost analysis", "xla/service/hlo_cost_analysis.cc", "int64_t HloCostAnalysis::GetDotFlops(", 484,
     "lhs/result shape 与 DotDimensionNumbers", "按 FMA 口径计数的 FLOPs",
     "静态算量估计需要硬件带宽/吞吐和流量假设才能构成 roofline。"),
    ("jax.tpu-client", "jax", "TPU loader", "jax/_src/xla_bridge.py", "def make_tpu_client(", 198,
     "libtpu 路径与 client options", "TPU C API client",
     "加载 libtpu.so/PJRT plugin 并注册 profiler；公开接口不提供 libtpu 编译器内部源码。"),
    ("pjrt.c-api-compile", "xla", "PJRT C API", "xla/pjrt/c_api_client/pjrt_c_api_client.cc", "PjRtCApiClient::CompileAndLoad(MaybeOwningMlirModule", 763,
     "MLIR module 与 CompileOptions", "PjRtLoadedExecutable",
     "经 InitializeArgsAndCompile 到 PJRT_Client_Compile；plugin 内部实现另行取证。"),
    ("pallas.call", "jax", "Pallas API", "jax/_src/pallas/pallas_call.py", "def pallas_call(", 1135,
     "kernel、out_shape、grid、BlockSpec 与 compiler_params", "可调用的 kernel wrapper",
     "表达对 Ref/块的操作；CPU 仅有 interpret 路径，不能作为 TPU 编译或硬件验证。"),
    ("pallas.dispatch", "jax", "Pallas lowering", "jax/_src/pallas/pallas_call.py", "def _pallas_call_lowering(", 844,
     "kernel Jaxpr、grid 参数和 interpret 选项", "平台对应的 lowering 结果",
     "interpret 分支 lower 解释实现；TPU 分支进入 pallas_call_tpu_lowering_rule。"),
    ("pallas.tpu-lower", "jax", "Mosaic TPU", "jax/_src/pallas/mosaic/pallas_call_registration.py", "def pallas_call_tpu_lowering_rule(", 393,
     "kernel Jaxpr、GridMapping、CompilerParams、alias/effects", "外层 TPU custom call",
     "先创建内层 Mosaic module，再通过 _lower_to_custom_call 嵌入外层 module。"),
    ("mosaic.module", "jax", "Mosaic TPU", "jax/_src/pallas/mosaic/lowering.py", "def lower_jaxpr_to_pipelined_module(", 1051,
     "kernel Jaxpr、grid、mesh、dimension semantics", "Mosaic TPU MLIR Module",
     "被调用实现检查 libtpu 版本；该 MLIR 不是 LLO。"),
    ("mosaic.custom-call", "jax", "Mosaic payload boundary", "jax/_src/tpu_custom_call.py", "def _tpu_custom_call_lowering(", 409,
     "序列化 kernel config、输入节点、alias/layout 与结果信息", "target=tpu_custom_call 的外层 custom call",
     "backend_config 携带编译载荷；has_side_effect、alias、layout 是接口合同。"),
    ("stablehlo.dot-schema", "stablehlo", "StableHLO dialect", "stablehlo/dialect/StablehloOps.td", "def StableHLO_DotGeneralOp:", 2740,
     "StableHLO dot_general operands 和属性", "方言定义及验证约束",
     "描述操作语义与类型，不指定 CPU/TPU 的机器码或调度。"),
    ("shardy.propagation", "shardy", "Shardy", "shardy/dialect/sdy/transforms/propagation/propagation_pipeline.cc", "void addPropagationPipeline(", 62,
     "OpPassManager 与 PropagationOptions", "包含 import/propagation/export 的 pass pipeline",
     "本次单 CPU 只观测到空 mesh/分片表示；未执行多设备传播验收。"),
    ("jax.profile-events", "jax", "Profiling", "jax/_src/profiler.py", "class TraceAnnotation(", 347,
     "事件名称及 metadata/context 范围", "host TraceMe 事件",
     "构造时启动，__exit__ 停止；Python jit 函数体中的标记随 tracing 执行，不能代表每次设备运行。"),
    ("pallas.generic-interpret", "jax", "CPU interpreter", "jax/_src/pallas/hlo_interpreter.py", "def pallas_call_hlo_interpret(", 316,
     "Pallas kernel Jaxpr、grid 和输入数组", "状态消解后的 HLO 解释循环结果",
     "interpret=True 是通用解释路径；不能等同于 TPU 专用语义模拟。"),
    ("pallas.tpu-interpret-params", "jax", "TPU interpreter", "jax/_src/pallas/mosaic/interpret/params.py", "class InterpretParams(", 121,
     "随机种子、模拟核心数量、grid recorder 和检查选项", "TPU interpreter 配置",
     "本次仅验证单模拟核心和 6 个 grid 点；recorder 接收 token/coordinates/core_id 并返回 token。"),
    ("mosaic.scope-lowering", "jax", "Mosaic TPU lowering", "jax/_src/pallas/mosaic/lowering.py", "def jaxpr_subcomp(", 1746,
     "kernel Jaxpr、name stack、block shapes 与 MLIR values", "含 tpu.trace_start/stop 的 Mosaic MLIR",
     "name stack 边界产生 level=10 的标记；此开放源码阶段不能证明 libtpu/LLO/设备 trace 保留标记。"),
    ("mosaic.trace-start-op", "xla", "Mosaic TPU dialect", "xla/mosaic/dialect/tpu/tpu_ops.td", "def TPU_TraceStartOp", 1618,
     "message 字符串和 level i32 属性", "无 SSA 结果的 trace_start 操作",
     "方言定义不是设备时钟或采样实现；Mosaic TPU MLIR 不是 LLO。"),
    ("mosaic.trace-stop-op", "xla", "Mosaic TPU dialect", "xla/mosaic/dialect/tpu/tpu_ops.td", "def TPU_TraceStopOp", 1623,
     "无显式输入", "无 SSA 结果的 trace_stop 操作",
     "配对需要检查 IR 的嵌套和控制流；本次仅验证直线 kernel。"),
    ("xla.traceme-wrapper", "xla", "Python/native profiling", "xla/python/profiler.cc", "class TraceMeWrapper", 59,
     "Python name/kwargs", "拥有原生 tsl::profiler::TraceMe 的 wrapper",
     "这里是实际 nanobind 注册使用的 wrapper；构造启动，__enter__ 返回自身，__exit__ 调用 Stop。"),
    ("xla.pass-events", "xla", "Compiler profiling", "xla/hlo/pass/hlo_pass_pipeline.cc", "absl::StatusOr<bool> HloPassPipeline::RunPassesInternal(", 141,
     "HloModule、debug options、execution threads 和 pass 列表", "每个 pass 的 TraceMe 范围与 metadata/dump",
     "TraceMe 在过滤 gate 前创建，范围也包含 metadata/invariant/dump 开销；事件名称出现不单独证明 pass 改变了 IR。"),
    ("xla.trace-json", "xla", "Trace export", "xla/tsl/profiler/convert/trace_events_to_json.cc", "inline void AddTraceEvent(", 87,
     "TraceEvent 的 ps 时间、device/resource ID、名称和 args", "Chrome Trace Event JSON 的 X 事件",
     "ts/dur 转成微秒，raw duration=0 时先夹到 1 ps；displayTimeUnit=ns 不改变数值单位。嵌套时长不可直接当 wall time。"),
    ("xla.xplane-trace-events", "xla", "Trace export", "xla/tsl/profiler/convert/xplane_to_trace_events.cc", "void ConvertXPlaneToTraceEvents(", 62,
     "XPlaneVisitor 的事件、共享 metadata 与 occurrence stats", "TraceContainer 中的 TraceEvent 及可见 args",
     "过滤 internal context/program 字段；同名 stat 后值覆盖前值。raw XLine 的完整 ID 在这里转成 uint32 viewer resource ID。"),
    ("xla.internal-trace-stat", "xla", "Profiler schema", "xla/tsl/profiler/utils/xplane_schema.cc", "bool IsInternalStat(", 615,
     "可选 StatType", "该字段是否属于不向 trace viewer 导出的内部 stat",
     "_pt/_p/_ct/_c 与 program_id 均为 internal；未知自定义字段可保留。编译 Hack 另使用 research_program_id 供导出分组。"),
    ("jax.metadata-api", "jax", "Metadata API", "jax/_src/xla_metadata.py", "def set_xla_metadata(", 109,
     "可选数组值或 kwargs metadata", "标记 producer op 的 identity primitive 或 metadata context",
     "Python 值转成字符串（布尔小写）；属性能被传递不代表 backend 有消费逻辑。"),
    ("jax.metadata-call", "jax", "Metadata API", "jax/_src/xla_metadata.py", "def xla_metadata_call2(", 215,
     "函数、metadata、ad_metadata 策略", "带 metadata 的 staged call",
     "ad_metadata 控制线性化/转置派生调用；same/drop 的 matmul 反向调用已分别捕获。"),
    ("jax.metadata-call-lower", "jax", "Metadata lowering", "jax/_src/xla_metadata.py", "def _xla_metadata_call_lowering(", 316,
     "Jaxpr、operands、metadata", "带 mhlo.frontend_attributes 的 func.call",
     "属性最初在 call 上；后端内联和 fusion 可能改变属性所有者，不能按节点数量一一对应。"),
    ("xla.frontend-export", "xla", "Metadata export", "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc", "void CreateFrontendAttributes(mlir::ArrayRef", 970,
     "MLIR 命名属性数组", "XLA FrontendAttributes 字符串 map",
     "此重载接受 StringAttr 和 BoolAttr；直接写入 IntegerAttr 的自定义值未被导出。"),
    ("jaxlib.cost-binding", "jax", "Cost binding", "jaxlib/xla_compiler.cc", "void BuildXlaCompilerSubmodule(", 86,
     "hlo_module_cost_analysis 的 client 与 HloModule", "properties dict",
     "从该 client 的 PJRT backend 获取 analyzer 并遍历 entry computation；没有 Python rates 参数。"),
    ("xla.cost-preprocess", "xla", "Cost analysis", "xla/service/hlo_cost_analysis.cc", "absl::Status HloCostAnalysis::Preprocess(", 58,
     "HloInstruction 与 shapes", "默认输入/输出字节、utilization 等 properties",
     "这是静态估计，后续 handler 可覆盖，不等于实际 HBM/DRAM 流量。"),
    ("xla.cost-postprocess", "xla", "Cost analysis", "xla/service/hlo_cost_analysis.cc", "absl::Status HloCostAnalysis::Postprocess(", 91,
     "当前 properties、per_second_rates 与最小延迟选项", "逐指令 bottleneck time 与累计 properties",
     "最大资源时间需要配置 rates；默认空 rates 不产生目标硬件 roofline 或真实运行时间。"),
    ("xla.cost-custom-call", "xla", "Cost analysis", "xla/service/hlo_cost_analysis.cc", "absl::Status HloCostAnalysis::HandleCustomCall(", 1362,
     "未知 custom-call instruction", "未知 cost 的 -1 sentinel（内部 call markers 例外为 0）",
     "CPU analyzer 对本次 tpu_custom_call 返回 -1；这不是 libtpu backend cost 行为的验证。"),
    ("xprof.roofline-cli", "xprof", "Profiler CLI", "plugin/xprof/cli/tools/get_roofline_model_tool.py", "def get_roofline_model(", 24,
     "session_id、top_n、group_by、bypass_cache", "program/device/top_operations JSON 摘要",
     "此固定版 del group_by；获取 roofline_model.json 后 fallback roofline_model。缺失值部分转 0，必须结合原始数据判断。"),
    ("xprof.roofline-processor", "xprof", "Profiler conversion", "xprof/convert/roofline_model_processor.cc", "absl::Status RooflineModelProcessor::ProcessSession(", 39,
     "SessionSnapshot 和 ToolOptions", "合并 XSpace→OpStats→RooflineModel 的 JSON",
     "分别生成包含/排除 infeed/outfeed 的记录；没有在本轮执行 XProf native converter。"),
    ("xprof.roofline-record", "xprof", "Profiler analysis", "xprof/convert/op_stats_to_roofline_model.cc", "RooflineModelRecord ConvertOpMetricsToRooflineModelRecord(", 62,
     "OpMetrics、PerfEnv/RunEnvironment、record type 和总时间", "时间、强度、资源上限与效率记录",
     "利用率使用资源最大值；异步 copy 的估计可能超过 1，不应一律裁剪或解释为测量正确。"),
    ("xprof.roofline-db", "xprof", "Profiler analysis", "xprof/convert/op_stats_to_roofline_model.cc", "RooflineModelDatabase ConvertOpStatsToRooflineModel(", 272,
     "Combined OpStats 与 RooflineModelOptions", "profile/step 记录和 diagnostics",
     "输入必须已有运行环境及成本/时间数据；函数不是从单张 StableHLO 推导全部设备信息。"),
    ("xprof.roofline-program", "xprof", "Profiler aggregation", "xprof/convert/op_stats_to_roofline_model.cc", "RooflineModelRecord GenerateRooflineModelProgramRecord(", 145,
     "OpMetricsDb、OpStats、record type 和总时间", "聚合后的 program 记录",
     "跳过 MayHaveInnerOps 类别避免重复计数；infeed/outfeed 由选项控制。"),
    ("xprof.roofline-metrics", "xprof", "Profiler analysis", "xprof/convert/op_metrics_to_record.h", "inline void SetRooflineMetrics(", 180,
     "OpMetrics、PerfEnv、RunEnvironment 与 record", "吞吐、各级内存强度和 bottleneck",
     "measured_flop_rate 是 flops_v2/实测时间，不自动等于硬件 counter；无分层字节时此版按 HBM 处理。"),
    ("xprof.cost-wrapper", "xprof", "Profiler cost conversion", "xprof/utils/hlo_cost_analysis_wrapper.cc", "HloCostAnalysisWrapper::GeneratePerformanceInfo(", 60,
     "HloInstruction 与 backend cost analysis wrapper", "PerformanceInfo 的 FLOPs/字节和内存分解",
     "内存分解仅输出正值，跳过 0 与 -1；TPU/GPU 获取 cost 的路径需分别追踪。"),
    ("xprof.unknown-cost", "xprof", "Profiler cost conversion", "xprof/utils/cost_utils.h", "inline int64_t ValidHloCost(", 30,
     "cost 数值，包括 -1 未知 sentinel", "-1 映射为 0，其他值保持原样",
     "下游出现 0 不足以证明该算子没有计算/流量；需要追踪原始 cost 可用性。"),
    ("xla.cpu-cost-factory", "xla", "CPU cost factory", "xla/pjrt/cpu/cpu_client.cc", "PjRtCpuClient::GetHloCostAnalysis()", 470,
     "CPU PJRT client", "通用 HloCostAnalysis，使用 CPU ShapeSizeBytes",
     "没有在此 factory 填入硬件 per_second_rates；不能把 optimal_seconds 缺失当作零耗时。"),
    ("xla.hlo-metadata-setter", "xla", "Native HLO Python API", "xla/python/hlo.cc", "    void set_frontend_attribute(", 783,
     "HloInstruction wrapper、string key/value", "更新 instruction 的 FrontendAttributes map",
     "可修改 native HLO 属性；语义性图编辑仍受 shape/opcode/effect/alias 等验证约束。"),
    ("jax.donation-aliases", "jax", "Donation lowering", "jax/_src/interpreters/mlir.py", "def _set_up_aliases(", 1487,
     "donated arguments、输入输出 avals、memory kinds、layouts/shardings", "input-output aliases 与交给 XLA 的 donation 标记",
     "优先精确匹配，另有相同元素数量的 fallback；最终 backend 合法性、临时量和运行时复用仍需检查。"),
    ("xla.cpu-fusion-decision", "xla", "CPU fusion", "xla/service/cpu/cpu_instruction_fusion.cc", "FusionDecision CpuInstructionFusion::ShouldFuse(", 403,
     "consumer 与 producer operand index", "Allow/Forbid 及原因",
     "考虑可发射性、库调用边界、重复计算、concatenate/reduction 限制及公共 fusion 条件；不是数学可合并就必融合。"),
    ("xla.common-fusion-decision", "xla", "Fusion legality", "xla/service/instruction_fusion.cc", "FusionDecision InstructionFusion::ShouldFuse(HloInstruction*", 1133,
     "consumer 与 producer operand index", "公共 fusion 合法性/启发式判断",
     "CPU 的专用条件仍需额外满足；调用本层不能保证最终形成目标 fusion。"),
    ("xla.cpu-fusion-wrapper", "xla", "CPU codegen preparation", "xla/service/cpu/fusion_wrapper.cc", "bool FusionWrapper::MustWrapInstruction(", 29,
     "单个 HloInstruction 与 emitter 配置", "是否包装成 fusion computation",
     "单算子 wrapper 也有 kFusion；关闭 fusion pass 后仍可能有 wrapper 和库 fusion。"),
    ("xla.buffer-assignment-entry", "xla", "Buffer assignment", "xla/service/buffer_assignment.cc", "absl::StatusOr<std::unique_ptr<BufferAssignment>> BufferAssigner::Run(", 1835,
     "module、ordering、buffer sizes、alias info、alignment 与 options", "BufferAssignment",
     "分配依赖 schedule 和数据流/alias；静态 allocation 不是进程或设备实测峰值。"),
    ("xla.buffer-reuse", "xla", "Buffer reuse", "xla/service/buffer_assignment.cc", "absl::StatusOr<bool> BufferAssigner::MaybeAssignBuffer(", 1911,
     "候选 allocation 与 HloBuffer", "可复用判定及分配",
     "检查颜色/空间兼容、容量、只读、live-out、可复用性与存活区间；同一 allocation 的不同 offset 也可能只是打包。"),
    ("xla.buffer-interference", "xla", "Buffer liveness", "xla/service/buffer_assignment.cc", "bool BufferAssigner::LiveRangeInterferes(", 1845,
     "两个 HloValue 及各自 live ranges", "是否干涉",
     "端点相接不自动等于干涉或安全；还需验证 producer/user 能否共享 buffer，copy 有专门限制。"),
    ("xla.live-range", "xla", "Liveness analysis", "xla/hlo/utils/hlo_live_range.cc", "absl::StatusOr<std::unique_ptr<HloLiveRange>> HloLiveRange::Run(", 53,
     "HloSchedule、alias analysis、computation", "以 instruction order 为逻辑时钟的 live ranges",
     "逻辑时间不是微秒；输出和输入的范围、alias normalization 与 physical allocation 要分开解释。"),
    ("xla.live-range-peak", "xla", "Memory diagnostics", "xla/hlo/utils/hlo_live_range.cc", "int64_t HloLiveRange::ComputePeakMemoryMoment()", 325,
     "buffer live ranges 和 HLO shapes", "诊断 peak moment",
     "此算法同一时刻先处理 start 再处理 end+1；本次捕获的 reduce 诊断选点不是打印闭区间总量的最大值，需复验匹配 binary。"),
    ("xla.runtime-donation-hold", "xla", "PJRT ownership", "xla/pjrt/abstract_tracked_device_buffer.cc", "CommonPjRtBuffer::GetBufferForDonationHoldLocked()", 233,
     "buffer 状态与 usage/external/donation holds", "取得 donation storage 或返回错误",
     "固定公共实现拒绝已有外部引用的 donation；当前 CPU wheel 的具体 fallback 路径仍受 VERSION-SKEW 限制。"),
    ("xla.cpu-memory-stats", "xla", "CPU executable introspection", "xla/pjrt/cpu/cpu_client.cc", "absl::StatusOr<CompiledMemoryStats> PjRtCpuExecutable::GetCompiledMemoryStats()", 1050,
     "CPU executable 的 allocation/HLO proto", "CompiledMemoryStats 与静态 heap 信息",
     "不能用 alias_size 直接证明运行时指针复用；CPU 外部 NumPy view 对照已展示区别。"),
    ("jax.psum-start", "jax", "Private async collective API", "jax/_src/lax/parallel.py", "def psum_start(", 3120,
     "局部 array 与 collective axis", "AbstractFuture；具体分支可能返回普通 array",
     "不是 jax.lax 公开导出；关闭 check_vma 的 _psum 分支没有使用 is_async。"),
    ("jax.psum-dispatch", "jax", "Collective tracing", "jax/_src/lax/parallel.py", "def _psum(", 155,
     "array、axis、axis groups、is_async", "psum primitive 或 invariant async primitive",
     "check_vma=True 分支传播 is_async；False 分支直接绑定同步 psum_p。"),
    ("jax.async-collective-lower", "jax", "Collective lowering", "jax/_src/lax/parallel.py", "def _emit_async_start_custom_call(", 3055,
     "target、future aval、collective config", "携带 async_collective_config 的 custom call",
     "Future lowering 使用 inner_aval 的 tensor 类型；不同于 FFI 默认把 Future shape 当作空。"),
    ("jax.overlap-schedule", "jax", "Experimental scheduling", "jax/experimental/overlap.py", "def schedule(", 23,
     "按顺序排列的值", "相邻值的 control_dep custom calls",
     "只是控制提示；VMA、layout、backend 重写和合法性仍须逐阶段验证。"),
    ("jax.overlap-control", "jax", "Experimental scheduling", "jax/experimental/overlap.py", "def control_dep(", 18,
     "src 与 dst", "有副作用、无输出的 FFI control_dep",
     "当前 Future 在 VMA 或默认 layout 上存在已捕获失败；不能直接视为可用 async overlap API。"),
    ("jax.ffi-future-shape", "jax", "FFI layout", "jax/_src/ffi.py", "def _aval_shape(", 179,
     "FFI operand aval", "用于默认 layout 的 shape",
     "AbstractFuture 返回空 shape；本次 rank-2 future lowering 实际为 tensor，触发 MLIR layout 验证错误。"),
    ("xla.cpu-async-pipeline", "xla", "CPU collective pipeline", "xla/service/cpu/cpu_compiler.cc", "absl::Status CpuCompiler::RunHloPassesThroughLayoutAssn(", 596,
     "HloModule、AOT 与目标特性", "布局分配前后的优化 HLO",
     "async-collective pipeline 先重写 custom calls，再用全部为真的谓词将 async collectives 改回同步。"),
    ("xla.async-custom-rewriter", "xla", "Collective representation", "xla/service/async_collective_custom_call_rewriter.cc", "absl::StatusOr<bool> AsyncCollectiveCustomCallRewriter::RunImpl(", 467,
     "collective start/done custom calls", "原生 async HLO 表达",
     "识别匹配 start/done 与数据流路径；这是表示重写，不保证异步执行。"),
    ("xla.async-finish-rewrite", "xla", "Collective rewrite ownership", "xla/service/async_collective_custom_call_rewriter.cc", "absl::Status FinishRewrite(", 83,
     "旧 custom calls、新 async pair 与 forward path", "更新 uses/control edges 并清理旧指令",
     "固定源码删除 start 前检查 user_count==0；当前 wheel 的 live-start 删除错误不能据此直接归因于同版 C++。"),
    ("xla.async-to-sync", "xla", "CPU collective policy", "xla/hlo/transforms/collectives/async_collective_replacer.cc", "absl::StatusOr<bool> AsyncCollectiveReplacer::RunImpl(", 71,
     "HLO 与 collective predicates", "匹配的 async pair 变为同步 collective",
     "源码显式删除参与被替换 start/done 的 control_dep calls；不同于实际控制依赖边的保留。"),
    ("xla.control-dep-rewriter", "xla", "Control dependencies", "xla/service/control_dep_rewriter.cc", "absl::StatusOr<bool> ControlDepRewriter::RunImpl(", 32,
     "target=control_dep 的两个 operands", "src→dst 控制依赖并删除 custom call",
     "CPU sync-control capture 已在 pass 前后验证 dot→all-reduce 边；不证明 overlap。"),
    ("xla.async-creator", "xla", "Collective policy", "xla/hlo/transforms/collectives/async_collective_creator.h", "class AsyncCollectiveCreator", 31,
     "collective predicates 与阈值配置", "同步 collective 的 async start/done 表达",
     "各后端配置决定是否转换；generic 模式默认 false，不能把该声明当作 libtpu 实际配置。"),
    ("xla.lhs-config", "xla", "Latency hiding scheduling", "xla/service/latency_hiding_scheduler.h", "struct SchedulerConfig", 143,
     "overlap limits、memory limit、资源与策略选项", "scheduler 配置",
     "通用开源接口；本次 CPU 走 DFS/BFS，未证明目标 TPU 使用此配置。"),
    ("xla.lhs-cost", "xla", "Scheduling cost interface", "xla/service/latency_hiding_scheduler.h", "class LatencyEstimator", 198,
     "HLO node 与依赖边", "NodeCost、GetLatencyBetween 与时钟换算",
     "模型预测不是实测通信时间；latency_metadata 的两处非测试调用位于 GPU NodeCost 实现，不能推广为 TPU 或默认 cost_analysis 消费。"),
    ("xla.lhs-pass", "xla", "Latency hiding scheduling", "xla/service/latency_hiding_scheduler.h", "class LatencyHidingScheduler", 2176,
     "SchedulingContext、SchedulerCore 与 module", "调度与模型统计",
     "存在源码类不代表 CPU 或 TPU pipeline 调用了它；目标部署必须另行取证。"),
    ("xla.lhs-metadata-parser", "xla", "Latency metadata", "xla/service/latency_hiding_scheduler.cc", "LatencyEstimator::GetLatencyFromMetadata(", 376,
     "HLO frontend_attributes[latency_metadata]", "optional TimeCost：int64 ns × CyclesPerMicrosecond / 1000",
     "缺失或 SimpleAtoi 失败返回 nullopt；原生 CPU 单元测试已确认零/负数接受、int64 越界拒绝和配置的单位缩放，目标 GPU/TPU 消费另行取证。"),
    ("xla.gpu-node-cost", "xla", "Approximate GPU model", "xla/service/gpu/gpu_latency_hiding_scheduler.cc", "ApproximateLatencyEstimator::TimeCost GpuLatencyEstimator::NodeCost(", 870,
     "HloInstruction", "估算节点成本",
     "先处理 nop；只有 custom-call 分支读取 latency_metadata，其他 opcode 不经此读取。"),
    ("xla.sol-node-cost", "xla", "Unified GPU model", "xla/service/gpu/model/sol_latency_estimator.cc", "LatencyEstimator::TimeCost SolLatencyEstimator::NodeCost(", 521,
     "HloInstruction、GPU model/table config", "估算节点成本",
     "任何 opcode 都先尝试 latency_metadata，再判断 async、matmul、fusion 等分支；不能把该优先级套到其他 estimator。"),
    ("xla.analytical-node-cost", "xla", "Analytical GPU model", "xla/service/gpu/model/analytical_latency_estimator.cc", "LatencyEstimator::TimeCost AnalyticalLatencyEstimator::NodeCost(", 58,
     "HloInstruction 与 GPU performance model", "节点模型时间",
     "此 NodeCost 没有调用 metadata parser；async collective start/done 使用低成本，其他指令走 performance model。"),
    ("xla.pgle-node-cost", "xla", "Profile-guided model", "xla/service/profile_guided_latency_estimator.cc", "LatencyEstimator::TimeCost ProfileGuidedLatencyEstimator::NodeCost(", 137,
     "HloInstruction、instruction profile map、fallback estimator", "profile cost 或 fallback cost",
     "async collective start/done 的低成本优先，其后按指令名命中 profile；只有缺失时调用 fallback NodeCost，metadata 不是无条件最高优先级。"),
    ("xla.gpu-estimator-select", "xla", "GPU scheduling model selection", "xla/service/gpu/gpu_hlo_schedule.cc", "std::unique_ptr<LatencyEstimator> GetLatencyEstimator(", 491,
     "module、GPU description、fingerprint、config", "PGLE/analytical/SOL/approximate estimator",
     "profile 优先；其后 analytical flag、受支持的 SOL 与 fallback。选择依赖真实配置和设备，本轮只阅读源码。"),
    ("xla.gpu-lhs-pipeline", "xla", "GPU scheduling pipeline", "xla/service/gpu/gpu_hlo_schedule.cc", "absl::Status RunLatencyHidingSchedulerPasses(", 680,
     "HLO、GPU info、memory limit、config", "LHS 及相关标记/检查后的 module",
     "把选定 estimator 放入 SchedulingContext，并 AddPass<LatencyHidingScheduler>；不能据此声称本次 CPU 调用了该管线。"),
    ("xla.lhs-graph", "xla", "Scheduling graph", "xla/service/latency_hiding_scheduler.cc", "HloScheduleGraph::HloScheduleGraph(", 2937,
     "原 instruction order、SchedulingContext 与 reachability", "带 node cost、edge latency 和资源信息的图",
     "NodeCost 写入每个 HloGraphNode；这些是模型值，不是 trace 事件时间。"),
    ("xla.lhs-schedule-node", "xla", "Scheduling model clock", "xla/service/latency_hiding_scheduler.cc", "absl::StatusOr<HloGraphNode::TimeCost> DefaultSchedulerCore::ScheduleNode(", 2582,
     "候选 node 与调度状态", "更新模型 ready time 和当前时间",
     "schedule_time 受依赖 edge latency 与资源限制，current_time=schedule_time+node cost；不是直接按标签排序。"),
    ("xla.host-offload-attributes", "xla", "Attribute ownership", "xla/hlo/transforms/host_offloading_prepare.cc", "absl::StatusOr<bool> ConvertToCustomCall(", 102,
     "host async call 的 inner computation", "HostExecute custom call",
     "复制第一个带任意 frontend attributes 的 inner custom call 后 break；不是汇总多个 kernel 的 latency。"),
    ("xla.lhs-enable", "xla", "GPU scheduling gate", "xla/service/gpu/gpu_hlo_schedule.cc", "bool IsLHSEnabled(", 863,
     "module options、fingerprint 与 GPU description", "是否启用 LHS",
     "显式 flag 优先；还看 opt effort、SOL 支持与有效 PGLE profile。标记本身不绕过 gate。"),
    ("xla.lhs-metadata-test", "xla", "Upstream test specification", "xla/service/latency_hiding_scheduler_test.cc", "TEST(LatencyEstimatorTest, GetLatencyFromMetadata)", 158,
     "上游构造的 custom call：1000、invalid、missing", "测试规范中的 optional time 断言",
     "该固定测试已在 CPU 原生目标运行通过；独立测试补丁另补 14 个边界样本。测试配置的 cycles/us 不等于硬件频率。"),
    ("xla.llvm-dump-hooks", "xla", "LLVM observation", "xla/service/cpu/cpu_compiler.cc", "std::pair<LLVMCompiler::ModuleHook, LLVMCompiler::ModuleHook> GetIRModuleHooks(", 1245,
     "HLO module 与 user hooks", "LLVM 优化前后 hooks",
     "ir-no-opt 是本段 LLVM pipeline 前的观察点，不是原始 Jaxpr 或未经任何 lowering 的程序。"),
    ("xla.object-dump-hook", "xla", "Object observation", "xla/service/cpu/cpu_compiler.cc", "CreateOrcJITPostCompilationHook(", 1406,
     "LLVM module、ObjectFile 与 HLO module", "ObjFileProto 留存和可选 .o dump",
     "对象字节另存入 executable，dump 并不是通过系统链接器构造独立 .so。"),
    ("xla.ir-compile", "xla", "LLVM compiler adapter", "xla/backends/cpu/codegen/ir_compiler.cc", "llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> IrCompiler::operator()(", 289,
     "llvm::Module", "机器码对象 MemoryBuffer 或错误",
     "串联 target machine、IR hooks、RunIrPasses、EmitMachineCode；并发编译需独立 TargetMachine。"),
    ("xla.llvm-pass-manager", "xla", "LLVM optimization", "xla/backends/cpu/codegen/ir_compiler.cc", "llvm::Error IrCompiler::RunIrPasses(", 354,
     "LLVM module、TargetMachine 与 options", "优化后的 module",
     "使用新 pass manager，按 O0/其他选择 pipeline；IPO、vectorization 与代码生成不能混成单个 HLO pass。"),
    ("xla.emit-object", "xla", "Machine code emission", "xla/backends/cpu/codegen/ir_compiler.cc", "std::unique_ptr<llvm::MemoryBuffer> IrCompiler::EmitMachineCode(", 477,
     "LLVM module 与 target machine", "包含可重定位对象的 SmallVectorMemoryBuffer",
     "该阶段使用 legacy codegen PassManager 和 addPassesToEmitMC；与前面的 LLVM IR pass manager 不同。"),
    ("xla.jit-add-module", "xla", "ORC registration", "xla/backends/cpu/codegen/jit_compiler.cc", "absl::Status JitCompiler::AddModule(", 173,
     "ThreadSafeModule、dylib_index", "添加到 IRCompileLayer 的 module",
     "设置 target triple/data layout/xla_dylib_index；AddModule 不等于已完成全部符号 materialization。"),
    ("xla.jit-compile", "xla", "ORC materialization", "xla/backends/cpu/codegen/jit_compiler.cc", "absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompiler::Compile(", 193,
     "带类型标识的编译符号列表", "FunctionLibrary",
     "通过 ObjectLoader lookup 触发 materialization，并等待派发任务结束；源码中的旧 SimpleOrcJIT 注释不能代替当前调用。"),
    ("xla.object-lookup", "xla", "ORC symbol resolution", "xla/backends/cpu/codegen/object_loader.cc", "absl::StatusOr<llvm::orc::SymbolMap> ObjectLoader::LookupSymbols(", 172,
     "symbols 与 JITDylibs", "SymbolMap 或 unresolved-symbol 错误",
     "按 data layout 修饰名字，只查 exported symbols；不是任意字符串直接当函数地址。"),
    ("xla.compiled-function-library", "xla", "Runtime function binding", "xla/backends/cpu/codegen/object_loader.cc", "ObjectLoader::CreateFunctionLibrary(", 206,
     "请求的 symbols 与解析后的 SymbolMap", "持有 ExecutionEngine 与已解析地址的函数库",
     "函数库持有引擎生命周期；地址与 tensor BufferAssignment 是不同对象。"),
    ("xla.orc-object-layer", "xla", "JIT object linking", "xla/backends/cpu/codegen/execution_engine.cc", "CreateObjectLinkingLayer(", 37,
     "ExecutionSession", "RTDyldObjectLinkingLayer",
     "此 pin 的 XLA 使用 RTDyld 层和 ContiguousSectionMemoryManager，不能仅凭 ORC 名称称其使用 JITLink。"),
    ("xla.jit-memory-finalize", "xla", "JIT code memory", "xla/backends/cpu/codegen/contiguous_section_memory_manager.cc", "bool ContiguousSectionMemoryManager::finalizeMemory(", 157,
     "JIT code 与只读 data memory blocks", "代码页权限与指令缓存更新",
     "代码区设置 READ|EXEC，只读区设置 READ；这里不是 tensor 的 buffer allocation 或 TPU VMEM。"),
    ("xla.kernel-thunk-call", "xla", "CPU runtime invocation", "xla/backends/cpu/runtime/kernel_thunk.cc", "KernelThunk<num_arguments, num_results>::ExecuteInternal(", 166,
     "BufferAllocations、FunctionLibrary、threadpool", "Kernel 调用完成事件",
     "从 tensor slices 取得地址，按名字解析 kernel function；单 workgroup、线程池与同步循环为不同路径。"),
    ("xla.kernel-launch", "xla", "CPU kernel ABI", "xla/backends/cpu/runtime/kernel.cc", "absl::Status Kernel::Launch(const NumWorkGroups& num_workgroups,", 147,
     "workgroup grid 与 XLA_CPU_KernelArg 列表", "传入 KernelCallFrame 的实际函数调用",
     "同步循环通过 (*kernel_)(&call_frame) 调用；指令反汇编本身不能证明实测性能或 TPU 执行。"),
    ("xla.kernel-call-once", "xla", "CPU kernel ABI", "xla/backends/cpu/runtime/kernel.h", "inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::Status Kernel::CallOnce(", 118,
     "KernelArg 列表", "当前线程单 workgroup 调用",
     "小 kernel 的 fast path，与多 workgroup 的 Launch 分开定位。"),
    ("llvm.default-pipeline", "llvm", "LLVM optimization pipeline", "llvm/lib/Passes/PassBuilderPipelines.cpp", "PassBuilder::buildPerModuleDefaultPipeline(", 1755,
     "OptimizationLevel、LTO phase", "ModulePassManager pipeline",
     "具体 pass 序列取决于固定 LLVM 与参数；当前 wheel 的 .ll 前后差异不逐项证明这些固定源码 pass 已执行。"),
    ("llvm.mc-emission", "llvm", "LLVM machine code", "llvm/lib/CodeGen/CodeGenTargetMachineImpl.cpp", "bool CodeGenTargetMachineImpl::addPassesToEmitMC(", 262,
     "legacy PM、MCContext、目标输出流", "目标 codegen/MC emission passes",
     "生成对象代码的公开实现入口；机器码不是目标硬件微架构或性能的证明。"),
    ("llvm.orc-ir-emit", "llvm", "ORC compile layer", "llvm/lib/ExecutionEngine/Orc/IRCompileLayer.cpp", "void IRCompileLayer::emit(", 28,
     "MaterializationResponsibility 与 ThreadSafeModule", "提交到对象层的 compiled buffer",
     "XLA 配置的 Compile 为 IrCompiler；失败时 failMaterialization，而非返回可用 kernel。"),
    ("llvm.orc-lookup", "llvm", "ORC lookup", "llvm/lib/ExecutionEngine/Orc/Core.cpp", "ExecutionSession::lookup(const JITDylibSearchOrder &SearchOrder,", 1794,
     "Dylib 搜索顺序、符号集与 RequiredState", "等待解析完成的 SymbolMap",
     "公开 ORC lookup/materialization 入口，未在本轮直接绑定回调或测量加载时间。"),
    ("jax.serialize-executable", "jax", "Executable packaging", "jax/experimental/serialize_executable.py", "def serialize(", 27,
     "Compiled 对象", "含 PJRT executable 的 pickle 包与 pytree 信息",
     "离线审计只读原始字节，在包中定位完整 .o；没有 unpickle 或重新加载历史文件。"),
]


# Edges are limited to calls or dispatch interfaces read in the pinned source.

# Source-built matmul transition walkthrough.
SITES.extend([
    ('xla.dot-canonicalize', 'xla', 'CPU pass and dispatch', 'xla/hlo/transforms/expanders/dot_decomposer.cc', 'HloInstruction* CanonicalizeOperand(', 68, 'dot operand 与 batch/contracting 维度', 'transpose 和二维/三维 reshape', '合并非收缩维度；本例共享 W 的 batch 展平成 12 行。'),
    ('xla.identity-reshape', 'xla', 'CPU pass and dispatch', 'xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.cc', 'absl::StatusOr<bool> IdentityReshapeRemoving(HloInstruction* reshape) {', 158, 'reshape 与其 operand', 'operand use 重接', 'Shape::Equal 才转发；此函数没有立即删除原 reshape。'),
    ('xla.reshape-decompose', 'xla', 'CPU pass and dispatch', 'xla/hlo/transforms/expanders/reshape_decomposer.cc', '  absl::Status HandleReshape(HloInstruction* reshape) override {', 34, '带 layout 的 reshape', 'bitcast 或 copy/bitcast', '只有 layout 兼容才只需 bitcast；其他分支可插入一个或两个 copy。'),
    ('xla.algsimp-transpose', 'xla', 'CPU pass and dispatch', 'xla/hlo/transforms/simplifiers/algebraic_simplifier.cc', 'absl::Status AlgebraicSimplifierVisitor::HandleTranspose(', 9795, 'transpose 及前驱', '删除恒等转置或改写 dot 等', '本例还把 transpose(dot) 改写为调换操作数的 dot；不能将整个 algsimp 归结为单一规则。'),
    ('xla.reshape-mover', 'xla', 'CPU pass and dispatch', 'xla/hlo/transforms/simplifiers/reshape_mover.cc', 'absl::StatusOr<bool> ReshapeMover::RunImpl(', 399, '非 fusion computation 的候选', '移过 elementwise 的 reshape', '本例乘 2 从 [3,4,6] 改为 [12,6]；后续 algsimp 清理连锁 reshape。'),
    ('xla.transpose-folding', 'xla', 'CPU pass and dispatch', 'xla/service/transpose_folding.cc', 'absl::StatusOr<bool> TransposeFolding::RunImpl(', 221, 'dot/convolution 的转置操作数', '可由后端接受的维度改写', '先调用后端合法性回调；本例 W.T 折入 rhs contracting dims。'),
    ('xla.cpu-layout-constraints', 'xla', 'CPU pass and dispatch', 'xla/service/cpu/cpu_layout_assignment.cc', 'absl::Status CpuLayoutAssignment::AddBackendConstraints(', 133, 'CPU HLO 与 layout constraints', 'operand/result 布局约束', '与通用 layout assignment 配合；本例出现 copy，不等价于运行流量测量。'),
    ('xla.library-fusion-create', 'xla', 'CPU pass and dispatch', 'xla/backends/cpu/transforms/library_rewriter.cc', 'absl::StatusOr<HloFusionInstruction*> CreateLibraryFusion(', 66, '库 matcher 选中的 HLO 指令', 'kCustom fusion 与 backend_config', '写入库 fusion kind 并替换原指令；本例为 __ynn_fusion。'),
    ('xla.cpu-dot-thunk', 'xla', 'CPU pass and dispatch', 'xla/service/cpu/thunk_emitter.cc', 'absl::StatusOr<ThunkSequence> ThunkEmitter::EmitDotThunk(', 985, 'dot、buffer assignment、target features', 'KernelThunk 或 DotThunk', '由 implementation strategy 分支决定；不能仅凭没有 .o 推断具体分支。'),
    ('xla.cpu-dot-strategy', 'xla', 'CPU pass and dispatch', 'xla/service/cpu/dot_op_emitter.cc', 'DotImplementationStrategy GetDotImplementationStrategy(', 1413, 'HLO config、dot、target features', 'DotImplementationStrategy', 'batch 转 inner dot 后选择策略；形状、布局、类型、配置均影响分支。'),
    ('xla.cpu-ynn-thunk', 'xla', 'CPU pass and dispatch', 'xla/service/cpu/thunk_emitter.cc', 'absl::StatusOr<ThunkSequence> ThunkEmitter::EmitYnnFusionThunk(', 1410, 'YNN fusion 与 allocation slices', 'YNN subgraph builder / thunk', '收集参数与结果，常量单独捕获；源码路径不是本次 runtime sampling。'),
    ('xla.cpu-ynn-invoke', 'xla', 'CPU pass and dispatch', 'xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.cc', 'YnnFusionThunk::YnnExecutable::Invoke(', 90, '线程池、参数和结果地址', 'ynn_invoke_runtime 状态', '设置 external values 与线程池后调用 YNN runtime；不推导库内部机器码。'),
])


# CPU executable packaging and observed warm thunk dispatch.
SITES.extend([
    ('jax.pickle-exec', 'jax', 'CPU executable and runtime', 'jax/experimental/serialize_executable.py', '  def persistent_id(self, obj):', 93, 'Python 包装中的 executable 对象', 'exec persistent id 与序列化 bytes', '这里只读取该字节模式；没有调用 Unpickler 或 native loader。'),
    ('ifrt.executable-serialize', 'xla', 'CPU executable and runtime', 'xla/python/pjrt_ifrt/pjrt_executable.cc', 'absl::StatusOr<std::string> PjRtExecutable::CommonMetadata::Serialize(', 455, 'PJRT executable 与 IFRT common metadata', '长度前缀 metadata 加 PJRT opaque payload', 'IFRT metadata 与后端 executable 是相邻的两段，不能把整体当一个 CPU proto。'),
    ('xla.cpu-executable-serialize', 'xla', 'CPU executable and runtime', 'xla/pjrt/cpu/cpu_client.cc', 'absl::StatusOr<std::string> PjRtCpuExecutable::SerializeExecutable() const {', 524, 'PjRtCpuExecutable', 'ExecutableAndOptionsProto', 'CPU compiler Export 的结果再与 compile options 打包。'),
    ('xla.cpu-aot-create', 'xla', 'CPU executable and runtime', 'xla/service/cpu/cpu_aot_compilation_result.cc', 'CpuAotCompilationResult::Create(', 76, 'HLO、buffers、objects、symbols、thunks', 'CpuAotCompilationResult', '保存实际 thunk sequence 及对象；不是仅记录 HLO 图。'),
    ('xla.cpu-compilation-proto', 'xla', 'CPU executable and runtime', 'xla/service/cpu/executable.proto', 'message CompilationResultProto {', 45, 'CPU 编译产物', 'CompilationResultProto 字段合同', '本例 obj_files_kind=KERNELS，含 HLO/config、buffer assignment、thunks、symbols 与 objects。'),
    ('xla.cpu-thunk-proto', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk.proto', 'message ThunkProto {', 297, 'thunk kind/info/impl', 'oneof 实现与公共身份', 'kind 字符串与 oneof 必须一致；DotThunk 与 KernelThunk 分开。'),
    ('xla.cpu-dot-serdes', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk_serdes/dot_thunk_serdes.cc', 'absl::Status DotThunkToProto(const Thunk& thunk, ThunkProto& proto) {', 42, '实际 DotThunk', 'DotThunkProto', '序列化 dot dimensions、lhs/rhs/output shape 和 allocation slices。'),
    ('xla.cpu-ynn-serdes', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk_serdes/ynn_fusion_thunk_serdes.cc', 'absl::Status YnnFusionThunkToProto(const Thunk& thunk, ThunkProto& proto) {', 57, '实际 YnnFusionThunk', 'YnnFusionThunkProto', '记录 HLO instruction id 和参数/结果 slices；依赖对应 fusion computation。'),
    ('xla.cpu-thunk-sequence-proto', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk_proto_serdes.cc', 'absl::StatusOr<ThunkSequenceProto> ThunkSequenceSerDesProtobuf::ToProto(', 928, 'ThunkSequence 与资源关系', 'ThunkSequenceProto', '逐 thunk 序列化并收集资源使用者；列表顺序本身不能证明并发执行时间线。'),
    ('xla.cpu-thunk-traced-execute', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk_executor.cc', 'tsl::AsyncValueRef<Thunk::ExecuteEvent> ThunkExecutor::TracedExecute(', 222, 'thunk 与 ExecuteParams', 'ExecuteEvent 及 producer/consumer trace', 'TraceMeProducer 围绕 Execute 返回，完成回调生成 end 事件；producer duration 不能一般化为异步完整时长。'),
    ('xla.cpu-thunk-trace-fields', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/thunk.cc', 'std::string Thunk::TraceMeEncode(int64_t run_id, int64_t device_ordinal) const {', 181, 'op/module identity、run_id、device ordinal', 'TraceMe metadata', '导出 JSON 未保留 program_id；end 事件另只带名称，不能任意并发拼接。'),
    ('xla.cpu-dot-execute', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/dot_thunk.cc', 'tsl::AsyncValueRef<DotThunk::ExecuteEvent> DotThunk::Execute(', 74, 'dot shape/slices 与线程池', '矩阵乘完成事件', '从 allocation 取得地址，处理行列布局与 transpose，再调用 TypedMatMul。'),
    ('xla.cpu-eigen-typed', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/dot_lib.h', 'void TypedMatMul(const Eigen::ThreadPoolDevice* device, void* out, void* lhs,', 91, '类型化 lhs/rhs/output 指针与 m/n/k', '对应 alignment 的 MatMul', '按 16-byte 指针对齐选择模板分支；本轮未测实际指针对齐或内部 microkernel。'),
    ('xla.cpu-eigen-contract', 'xla', 'CPU executable and runtime', 'xla/backends/cpu/runtime/dot_lib.h', '          Eigen::AlignmentType alignment>', 53, '矩阵维度、transpose 与回调', 'Eigen contraction 赋值', '线程池路径与同步无 device 路径不同；不能推导目标 TPU 实现。'),
])


# Raw XSpace contexts and host start/completion correlation.
SITES.extend([
    ('xla.trace-context-producer', 'xla', 'XSpace contexts and export', 'third_party/tsl/tsl/profiler/lib/connected_traceme.h', 'class TraceMeProducer : public TraceMe {', 76, '事件名、context type 与可选 ID', '_pt/_p metadata 与 context id', '未提供 ID 时调用 NewActivityId；context type=0 是有效 Generic 值。'),
    ('xla.trace-context-consumer', 'xla', 'XSpace contexts and export', 'third_party/tsl/tsl/profiler/lib/connected_traceme.h', 'class TraceMeConsumer : public TraceMe {', 99, '事件名、与 producer 相同的 type/id', '_ct/_c metadata', '在本线程创建独立 consumer 事件；对应关系不依赖相同事件名。'),
    ('xla.trace-context-types', 'xla', 'XSpace contexts and export', 'third_party/tsl/tsl/profiler/lib/context_types.h', 'enum class ContextType : int {', 24, 'context type 枚举', 'Generic=0、ThreadpoolEvent=15 等', '本轮 matcher 只覆盖 Generic 与 ThreadpoolEvent；TPU launch 的特殊 PID 处理未外推。'),
    ('xla.trace-new-activity', 'xla', 'XSpace contexts and export', 'third_party/tsl/tsl/profiler/lib/traceme.h', '  static int64_t NewActivityId() {', 337, '新的 trace activity 请求', 'recorder 提供的 int64 activity id', '转发到 recorder，不是跨进程/跨文件的永久唯一标识。'),
    ('xla.trace-activity-id', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/backends/cpu/traceme_recorder.cc', '/*static*/ int64_t TraceMeRecorder::NewActivityId() {', 259, '线程局部和进程内计数器', '高 32 位线程、低 32 位事件的 ID', '计数器存在复用边界；审计另以 XSpace scope 和 process/type 作为命名空间。'),
    ('xla.context-group-stats', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/group_events.cc', 'GroupingEventStats::GroupingEventStats(const XEventVisitor& event) {', 103, '单个事件的 occurrence stats', 'producer/consumer type/id 与有效 PID', 'consumer 可通过 _pid 指定 PID；普通 producer 取 plane.process_id；平台 launch 还有特殊处理。'),
    ('xla.context-group-set', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/group_events.cc', 'void SetContextGroup(const GroupingEventStats& stats, EventNode* event,', 176, 'GroupingEventStats 与 EventNode', '按 type/id/PID 分组的两类节点', '使用 optional.has_value，不能因 ID 或 type 为零而丢弃。'),
    ('xla.context-group-connect', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/group_events.cc', 'void ConnectContextGroups(const ContextGroupMap& context_groups) {', 200, 'ContextGroupMap', 'producer 到 consumer 的有向关联', '原生实现支持组内多对多；本轮应用合同只接受一对一且不猜测歧义。'),
    ('xla.context-add-flows', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/xplane_utils.cc', 'void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme,', 497, 'host_id、connect_traceme 与 XPlane', '由 context/correlation 生成的 flow stats', '需要显式开启 connect_traceme；本轮自定义 JSON overlay 没有调用此原生方法。'),
    ('xprof.flow-arguments', 'xprof', 'XSpace contexts and export', 'xprof/convert/xplane_to_trace_container.cc', 'SpecialArguments ConvertXStatsToTraceEventArguments(', 98, 'XEvent stats 与 raw arguments', 'flow/group/is_async 等特殊字段', '由预处理后的 kFlow 创建 flow/async event；未实际运行此 converter。'),
    ('xprof.flow-json', 'xprof', 'XSpace contexts and export', 'xprof/convert/trace_viewer/trace_events_to_json.h', '  void WriteEvent(const TraceEvent& event) const {', 317, 'TraceEvent 与 flow 方向', 'FlowV2 bind_id/flow_in/flow_out 或 async JSON', '与本轮派生的 legacy s/f overlay 是不同导出路径；仅 SOURCE-ONLY。'),
    ('xprof.context-preprocess', 'xprof', 'XSpace contexts and export', 'xprof/convert/preprocess_single_host_xplane.cc', 'void PreprocessSingleHostXSpace(', 34, 'XSpace 与 step_grouping 等选项', '预处理、flow 和分组后的 XSpace', '仅 SOURCE-ONLY；有 step_grouping 且未分组时调用 AddFlowsToXplane，本轮没有构建/执行此 XProf 管线。'),
    ('xla.threadpool-record', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/backends/cpu/threadpool_listener.cc', 'void ThreadpoolEventCollector::RecordEvent(uint64_t arg) const {', 56, '调度事件 ID', 'ThreadpoolListener::Record 瞬时 producer', 'context type 为 ThreadpoolEvent；它不是任务执行区间。'),
    ('xla.threadpool-start', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/backends/cpu/threadpool_listener.cc', 'void ThreadpoolEventCollector::StartRegion(uint64_t arg) const {', 63, '与调度相同的 ID', 'StartRegion 瞬时 consumer', '本例九条关联跨线程；StopRegion 没有这个 context ID，不能混作同一终点。'),
    ('xla.profiled-future', 'xla', 'XSpace contexts and export', 'xla/pjrt/common_pjrt_client.cc', 'Future<> CommonPjRtClient::CreateProfiledFuture(PjRtMemorySpace* memory_space,', 817, 'Future 与 callee 名称', '带 block start/end profiling 的 Future', '两个短 TraceMe 事件可同名，关联 ID 经 ProfilingKeys 传递；producer dur 不等于整个等待时间。'),
    ('xla.xstat-wire-format', 'xla', 'XSpace contexts and export', 'third_party/tsl/tsl/profiler/protobuf/xplane.proto', 'message XStat {', 118, 'metadata_id 与 oneof value', 'int64/uint64/ref/string 等值', 'ref_value 指向同 plane 的 stat metadata；必须保留 uint64 精度。'),
    ('xla.xevent-time', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/xplane_visitor.h', '  int64_t TimestampPs() const {', 199, 'XLine timestamp_ns 与 event offset_ps', '同一坐标系的 timestamp_ps', '以整数计算 ns*1000+offset_ps，不把 line-relative offset 单独当全局时间。'),
    ('xla.xline-display-id', 'xla', 'XSpace contexts and export', 'xla/tsl/profiler/utils/xplane_visitor.h', '  int64_t DisplayId() const {', 332, 'line display_id / id', '显示资源 ID', 'display_id 非零时优先，否则回退完整 line id；JSON converter 再转为 uint32。'),
])

SITES.extend(RUNTIME_SITES)
SITES.extend(SHARDY_SITES)

EDGES = SHARDY_EDGES + RUNTIME_EDGES + [
    ('xla.trace-context-producer', 'xla.trace-new-activity', '                                           : TraceMe::NewActivityId()) {', 'direct', '调用者未提供 context_id 时。'),
    ('xla.trace-new-activity', 'xla.trace-activity-id', '    return TraceMeRecorder::NewActivityId();', 'direct', ''),
    ('xla.profiled-future', 'xla.trace-context-producer', '        tsl::profiler::TraceMeProducer traceme(', 'direct', 'on_block_start 回调。'),
    ('xla.profiled-future', 'xla.trace-context-consumer', '        tsl::profiler::TraceMeConsumer traceme(', 'direct', 'on_block_end 回调。'),
    ('xprof.context-preprocess', 'xla.context-add-flows', '      tsl::profiler::AddFlowsToXplane(host_id, /*is_host_plane=*/!is_device,', 'interface', 'step_grouping 且尚未分组；仅索引源码 API 关系，未验证匹配 XProf 构建。'),
    ('xla.xplane-trace-events', 'xla.internal-trace-stat', '            if (IsInternalStat(stat.Type())) return;', 'direct', 'metadata stats 和 occurrence stats 都经过此过滤。'),

    ('ifrt.executable-serialize', 'xla.cpu-executable-serialize', '                   pjrt_executable->SerializeExecutable());', 'interface', '底层 executable 为 PjRtCpuExecutable 时。'),
    ('xla.cpu-aot-create', 'xla.cpu-thunk-sequence-proto', '                   thunk_sequence_serdes.ToProto(thunks));', 'direct', ''),
    ('xla.cpu-thunk-traced-execute', 'xla.cpu-thunk-trace-fields', '      [&] { return thunk.TraceMeEncode(params.run_id, params.device_ordinal); },', 'direct', 'profiler active 分支。'),
    ('xla.cpu-thunk-traced-execute', 'xla.cpu-dot-execute', '  auto execute_event = thunk.Execute(params);', 'interface', 'thunk 的动态类型为 DotThunk；当前两组梯度的 serialized kind 与 trace 对应。'),
    ('xla.cpu-dot-execute', 'xla.cpu-eigen-typed', '      internal::TypedMatMul<LhsType, RhsType, OutType>(', 'direct', '类型分派后的 batch loop。'),
    ('xla.cpu-eigen-typed', 'xla.cpu-eigen-contract', '    MatMul<LhsType, RhsType, OutType, Eigen::Aligned16>(', 'direct', 'is_aligned=true 分支。'),
    ('xla.cpu-eigen-typed', 'xla.cpu-eigen-contract', '    MatMul<LhsType, RhsType, OutType, Eigen::Unaligned>(', 'direct', 'is_aligned=false 分支。'),

    ("xla.cpu-dot-thunk", "xla.cpu-dot-strategy", "  DotImplementationStrategy strategy = GetDotImplementationStrategy(", "direct", "EmitDotThunk passes allow_runtime_calls=true."),
    ("xla.xplane-trace-events", "xla.internal-trace-stat", "            if (IsInternalStat(stat.Type())) return;", "direct", "转换每个有值的 metadata/occurrence stat 时。"),
    ("xla.ir-compile", "xla.llvm-pass-manager", "          RunIrPasses(module, target_machine->get())) {", "direct", "目标机器创建成功后。"),
    ("xla.ir-compile", "xla.emit-object", "      EmitMachineCode(module, target_machine->get());", "direct", "LLVM IR passes 成功后。"),
    ("xla.llvm-pass-manager", "llvm.default-pipeline", "    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));", "direct", "优化等级不是 O0。"),
    ("xla.emit-object", "llvm.mc-emission", "  target_machine->addPassesToEmitMC(codegen_passes, mc_context, ostream);", "interface", "具体 TargetMachine 使用该继承实现时。"),
    ("xla.jit-compile", "xla.object-lookup", "  auto symbol_map = object_loader.LookupSymbols(symbols);", "direct", ""),
    ("xla.jit-compile", "xla.compiled-function-library", "      .CreateFunctionLibrary(std::move(symbols), *symbol_map);", "direct", "lookup 成功且派发任务已结束。"),
    ("xla.object-lookup", "llvm.orc-lookup", "  auto symbol_map = execution_session()->lookup(std::move(search_order),", "direct", ""),
    ("llvm.orc-ir-emit", "xla.ir-compile", "  if (auto Obj = TSM.withModuleDo(*Compile)) {", "interface", "XLA 的 IRCompileLayer 配置 Compile 为 IrCompiler。"),
    ("xla.kernel-thunk-call", "xla.kernel-call-once", "    ABSL_RETURN_IF_ERROR(kernel->CallOnce(kernel_args));", "direct", "call_once_ fast path。"),
    ("xla.kernel-thunk-call", "xla.kernel-launch", "  ABSL_RETURN_IF_ERROR(kernel->Launch(num_workgroups_, kernel_args));", "direct", "非 call_once 且没有 intra_op_threadpool。"),
    ("xla.gpu-node-cost", "xla.lhs-metadata-parser", "            GetLatencyFromMetadata(*instr)) {", "direct", "非 nop 且 opcode 为 custom-call。"),
    ("xla.sol-node-cost", "xla.lhs-metadata-parser", "  if (const std::optional<TimeCost> latency = GetLatencyFromMetadata(*instr)) {", "direct", "NodeCost 第一分支，所有 opcode。"),
    ("xla.gpu-lhs-pipeline", "xla.gpu-estimator-select", "      GetLatencyEstimator(*module, pointer_size, gpu_device_info, fingerprint,", "direct", "GPU LHS pipeline 被启用时。"),
    ("xla.gpu-lhs-pipeline", "xla.lhs-pass", "  pipeline.AddPass<LatencyHidingScheduler>(scheduling_context,", "pipeline", "选定 estimator 已放入 SchedulingContext。"),
    ("jax.overlap-schedule", "jax.overlap-control", "    control_dep(src, dst)", "direct", "相邻值逐对调用。"),
    ("xla.cpu-async-pipeline", "xla.async-custom-rewriter", "  async_collective_pipeline.AddPass<AsyncCollectiveCustomCallRewriter>(", "pipeline", "use_legacy_collectives=false。"),
    ("xla.cpu-async-pipeline", "xla.async-to-sync", "  async_collective_pipeline.AddPass<AsyncCollectiveReplacer>(acr_config);", "pipeline", "Config(HloPredicateTrue)。"),
    ("jax.matmul", "jax.dot-general", "  out = lax.dot_general(", "direct", ""),
    ("jax.dot-general", "jax.dot", "  return dot(lhs, rhs,", "direct", ""),
    ("jax.pjit-lower", "jax.lower-sharding", "  return pxla.lower_sharding_computation(", "direct", ""),
    ("jax.cached-mlir-lowering", "jax.mlir-module", "    lowering_result = mlir.lower_jaxpr_to_module(", "direct", ""),
    ("jax.backend-compile", "jaxlib.compile", "      return backend.compile_and_load(", "binding", "普通 backend 路径；CompileOnlyPyClient 分支单独处理。"),
    ("jaxlib.compile", "jaxlib.ifrt-call", "  return CompileAndLoadIfrtProgram(", "direct", ""),
    ("jaxlib.ifrt-call", "ifrt.compile", "        client->ifrt_client_->GetDefaultCompiler()->CompileAndLoad(", "interface", "选用 PjRtCompiler 实现时；IFRT 也有其他 compiler。"),
    ("ifrt.compile", "ifrt.pjrt-create", "  return PjRtLoadedExecutable::Create(", "direct", ""),
    ("ifrt.pjrt-create", "pjrt.c-api-compile", "      client->pjrt_client()->CompileAndLoad(", "interface", "PJRT client 为 C API adapter 时；CPU client 可有其他实现。"),
    ("pallas.dispatch", "pallas.tpu-lower", "    return mosaic_tpu_backend.pallas_call_tpu_lowering_rule(", "direct", "TPU backend，非 interpret 默认分支。"),
    ("pallas.tpu-lower", "mosaic.module", "    mosaic_module = lowering.lower_jaxpr_to_pipelined_module(", "direct", ""),
    ("xla.cpu-codegen", "xla.cpu-schedule", "  ABSL_ASSIGN_OR_RETURN(HloSchedule schedule, CreateHloSchedule(*module));", "direct", ""),
    ("xla.cpu-codegen", "xla.cpu-buffers", "                   CreateBufferAssignment(*module));", "direct", ""),
    ("jaxlib.cost-binding", "xla.cpu-cost-factory", "                            client->pjrt_client()->GetHloCostAnalysis());", "interface", "当传入 client 为 PjRtCpuClient 时。"),
    ("xprof.roofline-processor", "xprof.roofline-db", "  RooflineModelDatabase result = ConvertOpStatsToRooflineModel(", "direct", ""),
    ("xprof.roofline-program", "xprof.roofline-record", "  RooflineModelRecord program_record = ConvertOpMetricsToRooflineModelRecord(", "direct", ""),
    ("xprof.roofline-record", "xprof.roofline-metrics", "  SetRooflineMetrics(metrics, op_stats.perf_env(), op_stats.run_environment(),", "direct", ""),
    ("jax.mlir-module", "jax.donation-aliases", "    input_output_aliases, donated_args, xla_donated_args = _set_up_aliases(", "direct", "满足该 lowering 的 donation/alias 分支条件时。"),
    ("xla.cpu-fusion-decision", "xla.common-fusion-decision", "  RETURN_IF_NOT_FUSIBLE(InstructionFusion::ShouldFuse(consumer, operand_index));", "direct", "前面的 CPU 专用 early return 未结束判定时。"),
    ("xla.cpu-buffers", "xla.buffer-assignment-entry", "  return BufferAssigner::Run(", "direct", ""),
    ("xla.buffer-reuse", "xla.buffer-interference", "        if (LiveRangeInterferes(new_value, *cached_new_live_ranges[i],", "direct", "total_order_scheduled 分支；另有 partial ordering fallback。"),
]


def main():
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    sources = dict(baseline["repository"]["sources"])
    # This optional tooling source was already pinned in the lock, then materialized
    # for the roofline investigation. Do not change the five core baseline entries.
    matches = [line.split("|") for line in (ROOT / "upstream-sources.lock").read_text().splitlines()
               if line.startswith("tooling-reference|upstream/tooling/xprof|")]
    if len(matches) != 1:
        raise RuntimeError("Missing or duplicate pinned XProf source")
    _, path, url, revision, *_ = matches[0]
    sources["xprof"] = {"path": path, "git_commit": revision}
    origins = {"jax": "https://github.com/jax-ml/jax", "xla": "https://github.com/openxla/xla",
               "stablehlo": "https://github.com/openxla/stablehlo", "shardy": "https://github.com/openxla/shardy",
               "llvm": "https://github.com/llvm/llvm-project", "xprof": url.removesuffix(".git")}
    for source in sources.values():
        path = ROOT / source["path"]
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != source["git_commit"]:
            raise RuntimeError(f"Source revision changed: {path}")
        status = subprocess.check_output(["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"], text=True)
        if status:
            raise RuntimeError(f"Source is dirty: {path}")
    entries = []
    for entry_id, component, layer, relative, anchor, minimum, inputs, outputs, constraint in SITES:
        source = sources[component]
        path = ROOT / source["path"] / relative
        payload = path.read_bytes()
        text = payload.decode()
        lines = text.splitlines()
        matches = [i + 1 for i, line in enumerate(lines) if i + 1 >= minimum and line.startswith(anchor)]
        if not matches:
            raise RuntimeError(f"Missing source anchor: {entry_id}")
        line = matches[0]
        end = None
        if path.suffix == ".py":
            nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.lineno == line]
            if nodes:
                end = nodes[0].end_lineno
        symbol = anchor.strip().split("(")[0].removesuffix(":")
        if symbol.startswith(("def ", "class ")):
            symbol = symbol.split(" ", 1)[1].split("[")[0]
        else:
            symbol = symbol.split()[-1]
        if entry_id == "jax.lowered-compile":
            symbol = "Lowered.compile"
        entries.append({
            "id": entry_id, "component": component, "layer": layer,
            "symbol": symbol,
            "revision": source["git_commit"], "path": str(path.relative_to(ROOT)),
            "line": line, "end_line": end, "anchor": anchor,
            "source_url": f"{origins[component]}/blob/{source['git_commit']}/{relative}#L{line}",
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": inputs, "outputs": outputs, "constraint": constraint,
            "evidence_level": "SOURCE-ONLY",
            "runtime_boundary": "CPU captures with VERSION-SKEW do not prove execution of the indexed native source.",
        })
    by_id = {entry["id"]: entry for entry in entries}
    edges = []
    for caller, callee, anchor, kind, condition in EDGES:
        parent = by_id[caller]
        by_id[callee]
        lines = (ROOT / parent["path"]).read_text().splitlines()
        end = parent["end_line"] or min(len(lines), parent["line"] + 450)
        matches = [i + 1 for i in range(parent["line"] - 1, end)
                   if lines[i].startswith(anchor)]
        if not matches:
            raise RuntimeError(f"Missing call edge: {caller} -> {callee}")
        edges.append({"caller": caller, "callee": callee, "kind": kind,
                      "condition": condition, "path": parent["path"],
                      "line": matches[0], "anchor": anchor,
                      "evidence_level": "SOURCE-ONLY"})
    for entry in entries:
        entry["callers"] = sorted({edge["caller"] for edge in edges if edge["callee"] == entry["id"]})
        entry["callees"] = sorted({edge["callee"] for edge in edges if edge["caller"] == entry["id"]})
        if entry["id"] in {"xla.pass-events", "xla.xplane-trace-events", "xla.internal-trace-stat", "xla.trace-json"}:
            entry["related_experiments"] = ["research/software-stack/pass-event-patch.md",
                                            "research/software-stack/pass-hack-results.json"]
        if entry["id"] in {"xla.lhs-metadata-parser", "xla.lhs-metadata-test"}:
            entry["related_experiments"] = ["research/software-stack/latency-parser-native.md",
                                            "research/software-stack/latency-parser-results.json"]
        if entry["id"] in {'xla.cpu-ynn-invoke', 'xla.cpu-ynn-thunk', 'xla.identity-reshape', 'xla.library-fusion-create', 'xla.dot-canonicalize', 'xla.reshape-mover', 'xla.cpu-dot-strategy', 'xla.reshape-decompose', 'xla.cpu-layout-constraints', 'xla.cpu-dot-thunk', 'xla.algsimp-transpose', 'xla.transpose-folding'}:
            entry["related_experiments"] = ["research/call-to-llo/matmul-pass-walkthrough.md",
                                            "research/software-stack/pass-transition-results.json"]
            entry["runtime_boundary"] = "Source-built CPU dumps are separately bound to build 003; this entry does not claim runtime dispatch sampling or TPU execution."
        if entry["id"] in {'xla.cpu-compilation-proto', 'xla.cpu-eigen-typed', 'jax.pickle-exec', 'xla.cpu-executable-serialize', 'xla.cpu-thunk-sequence-proto', 'xla.cpu-thunk-traced-execute', 'xla.cpu-dot-serdes', 'xla.cpu-thunk-trace-fields', 'xla.cpu-dot-execute', 'xla.cpu-ynn-serdes', 'xla.cpu-aot-create', 'ifrt.executable-serialize', 'xla.cpu-eigen-contract', 'xla.cpu-thunk-proto'}:
            entry["related_experiments"] = ["research/software-stack/cpu-executable-and-trace.md", "research/software-stack/cpu-thunk-results.json"]
            entry["runtime_boundary"] = "Source-bound CPU serialized thunks and fresh CPU traces are verified separately; no TPU or internal microkernel claim."
        if entry["id"] in {'xprof.flow-arguments', 'xprof.flow-json', 'xla.xline-display-id', 'xla.internal-trace-stat', 'xla.threadpool-record', 'xla.xplane-trace-events', 'xla.profiled-future', 'xla.trace-new-activity', 'xla.context-group-connect', 'xla.context-group-set', 'xla.threadpool-start', 'xla.context-group-stats', 'xla.trace-context-producer', 'xprof.context-preprocess', 'xla.xevent-time', 'xla.context-add-flows', 'xla.trace-context-consumer', 'xla.xstat-wire-format', 'xla.trace-json', 'xla.trace-context-types', 'xla.trace-activity-id'}:
            entry.setdefault("related_experiments", []).extend(["research/software-stack/xspace-contexts.md", "research/software-stack/xspace-context-results.json"])
            entry["runtime_boundary"] = "Raw source-built CPU captures are audited separately. XProf preprocessing and viewer rendering are SOURCE-ONLY or unverified here."
        if entry["id"] in RUNTIME_IDS:
            entry["related_experiments"] = ["research/software-stack/tpu-runtime-boundary.md", "research/software-stack/runtime-chain-results.json"]
            entry["runtime_boundary"] = "SOURCE-ONLY public interfaces and reference adapters; no native dispatch sampling, plugin execution, TPU/LLO or timing evidence. The open-source C wrapper is not identified as libtpu implementation."
        if entry["id"] in SHARDY_IDS:
            entry["related_experiments"] = ["research/software-stack/shardy-round-trip.md", "research/software-stack/shardy-results.json"]
            entry["runtime_boundary"] = "Source-bound two-CPU Shardy/GSPMD evidence is recorded separately; mixed-IR fallback, V3, tuple/alias restoration and TPU execution are not runtime-verified here."
    index = {
        "schema_version": "1.0", "kickoff_revision": 51,
        "source_roots": {name: {"path": s["path"], "revision": s["git_commit"]} for name, s in sources.items()},
        "entries": entries,
        "edges": edges,
        "limits": ["Initial entry-point index; the complete caller/callee graph is still in progress.",
                   "libtpu private implementation, LLO and target hardware internals are not indexed.",
                   "Selected LLVM optimization/MC/ORC interfaces and CPU kernel invocation are indexed; the complete internal pass graph and hardware implementation remain outside current coverage."],
    }
    target = HERE / "source-index.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(f"Indexed {len(entries)} reviewed source anchors: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
