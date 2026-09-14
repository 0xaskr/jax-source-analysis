#!/usr/bin/env python3
"""Locate reviewed API anchors in the kickoff's pinned source trees."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


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
     "ts/dur 转成微秒；displayTimeUnit=ns 不改变这些数值的单位。嵌套时长不可直接求和当作 wall time。"),
]


# Edges are limited to calls or dispatch interfaces read in the pinned source.
EDGES = [
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
]


def main():
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    sources = baseline["repository"]["sources"]
    origins = {"jax": "https://github.com/jax-ml/jax", "xla": "https://github.com/openxla/xla",
               "stablehlo": "https://github.com/openxla/stablehlo", "shardy": "https://github.com/openxla/shardy",
               "llvm": "https://github.com/llvm/llvm-project"}
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
    index = {
        "schema_version": "1.0", "kickoff_revision": 51,
        "source_roots": {name: {"path": s["path"], "revision": s["git_commit"]} for name, s in sources.items()},
        "entries": entries,
        "edges": edges,
        "limits": ["Initial entry-point index; the complete caller/callee graph is still in progress.",
                   "libtpu private implementation, LLO and target hardware internals are not indexed.",
                   "LLVM/MLIR dependency and CPU compiler integration are identified; internal LLVM code-generation APIs remain to be indexed."],
    }
    target = HERE / "source-index.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(f"Indexed {len(entries)} reviewed source anchors: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
