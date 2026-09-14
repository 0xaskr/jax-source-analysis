#!/usr/bin/env python3
"""Trace metadata through StableHLO/HLO, test cost readers, and execute edited HLO."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

from matmul_probe import ROOT, fingerprint, write_json


def frontend_attributes(module):
    """Read dictionary attributes from actual operations, excluding locations."""
    from jax._src.lib.mlir import ir
    result=[]
    def visit(op):
        if "mhlo.frontend_attributes" in op.attributes:
            attrs=ir.DictAttr(op.attributes["mhlo.frontend_attributes"])
            result.append({"operation":op.name,"attributes":{a.name:str(a.attr) for a in attrs}})
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:visit(child.operation)
    visit(module.operation)
    return result


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.xla_metadata import set_xla_metadata, xla_metadata_call2
    from jax._src import compiler, xla_bridge
    from jax._src.interpreters import mlir
    from jax._src.lib import _jax
    from jax._src.lib.mlir import ir
    from jaxlib import _hlo

    jax.config.update("jax_enable_compilation_cache",False)
    origin=ROOT/"artifacts/jax-stack/cpu-matmul-003/inputs.npz"
    with np.load(origin,allow_pickle=False) as data:a,w=[data[k] for k in ("a","w")]
    b=np.linspace(-.2,.3,24,dtype=np.float32).reshape(4,6)
    np.savez(output/"inputs.npz",a=a,w=w,b=b)
    reference=a.astype(np.float64)@w.astype(np.float64)
    gradient=2*reference@w.astype(np.float64).T
    arguments=(jax.device_put(a),jax.device_put(w))
    metadata={"research_tag":"matmul","research_flops":999999,"research_boolean":True}

    def dot(a,w):return jnp.matmul(a,w,precision=jax.lax.Precision.HIGHEST)
    def context_dot(a,w):
        with set_xla_metadata(**metadata):return dot(a,w)
    def value_dot(a,w):return set_xla_metadata(dot(a,w),**metadata)
    tagged_call=xla_metadata_call2(dot,metadata=metadata)
    dropped_call=xla_metadata_call2(dot,metadata=metadata,ad_metadata="drop")
    cases=[("plain",dot,reference),("context",context_dot,reference),
           ("value",value_dot,reference),("call",tagged_call,reference),
           ("grad-call-same",jax.grad(lambda a,w:jnp.sum(tagged_call(a,w)**2)),gradient),
           ("grad-call-drop",jax.grad(lambda a,w:jnp.sum(dropped_call(a,w)**2)),gradient)]
    summaries=[]
    for name,function,expected in cases:
        directory=output/name;directory.mkdir()
        lowered=jax.jit(function).lower(*arguments)
        (directory/"jaxpr.txt").write_text(str(jax.make_jaxpr(function)(*arguments))+"\n")
        (directory/"stablehlo.mlir").write_text(lowered.as_text("stablehlo",debug_info=True))
        (directory/"exported-hlo.txt").write_text(lowered.compiler_ir("hlo").as_hlo_text())
        before=lowered.cost_analysis()
        compiled=lowered.compile();actual=np.asarray(compiled(*arguments))
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        np.save(directory/"output.npy",actual,allow_pickle=False)
        optimized=compiled.as_text();(directory/"optimized-hlo.txt").write_text(optimized)
        attrs=frontend_attributes(lowered.compiler_ir("stablehlo"))
        summary={"case":name,"stablehlo_frontend_attributes":attrs,
            "exported_hlo_attribute_lines":[line.strip() for line in (directory/"exported-hlo.txt").read_text().splitlines() if "frontend_attributes=" in line],
            "optimized_hlo_attribute_lines":[line.strip() for line in optimized.splitlines() if "frontend_attributes=" in line],
            "lowered_cost":before,"compiled_cost":compiled.cost_analysis(),
            "max_absolute_error":float(np.max(np.abs(actual-expected)))}
        write_json(directory/"summary.json",summary);summaries.append(summary)
    for summary in summaries[1:4]:
        assert summary["lowered_cost"]==summaries[0]["lowered_cost"]
        assert summary["compiled_cost"]==summaries[0]["compiled_cost"]
        assert summary["exported_hlo_attribute_lines"]
    assert summaries[0]["lowered_cost"]["flops"]==384
    assert summaries[0]["lowered_cost"]["bytes accessed"]==416

    # Mutate a parsed copy; leave the JAX Lowered object and its cache untouched.
    directory=output/"direct-ir-attributes";directory.mkdir()
    plain_text=(output/"plain/stablehlo.mlir").read_text()
    with mlir.make_ir_context():
        module=ir.Module.parse(plain_text)
        dot_op=next(op.operation for fn in module.body.operations
                    for op in fn.regions[0].blocks[0].operations if op.operation.name=="stablehlo.dot_general")
        dot_op.attributes["research.raw"]=ir.StringAttr.get("raw-visible")
        dot_op.attributes["mhlo.frontend_attributes"]=ir.DictAttr.get({
            "research_string":ir.StringAttr.get("string-visible"),
            "research_integer":ir.IntegerAttr.get(ir.IntegerType.get_signless(64),123),
            "research_bool":ir.BoolAttr.get(True)})
        assert module.operation.verify()
        (directory/"stablehlo.mlir").write_text(str(module)+"\n")
        computation=_jax.mlir.mlir_module_to_xla_computation(mlir.module_to_bytecode(module))
        hlo_text=computation.as_hlo_text();(directory/"exported-hlo.txt").write_text(hlo_text)
        assert 'research_string="string-visible"' in hlo_text and 'research_bool="true"' in hlo_text
        assert "research_integer" not in hlo_text and "research.raw" not in hlo_text
        roundtrip=_jax.mlir.xla_computation_to_mlir_module(computation)
        (directory/"roundtrip.mlir").write_text(roundtrip)
        direct={"preserved":["research_string","research_bool"],"dropped":["research_integer","research.raw"],
                "cost":_jax.hlo_module_cost_analysis(xla_bridge.get_backend("cpu"),computation.as_hlo_module())}
        assert direct["cost"]==summaries[0]["lowered_cost"]
        write_json(directory/"summary.json",direct)

    # The native HLO object API also exposes string metadata mutation.
    directory=output/"native-hlo-attributes";directory.mkdir()
    native=_hlo.hlo_module_from_text((output/"plain/exported-hlo.txt").read_text())
    dot_instructions=[i for c in native.computations() for i in c.instructions() if i.opcode==_hlo.HloOpcode.kDot]
    assert len(dot_instructions)==1
    dot_instruction=dot_instructions[0]
    assert dot_instruction.get_frontend_attribute("research_native_tag") is None
    dot_instruction.set_frontend_attribute("research_native_tag","edited-with-native-api")
    assert dot_instruction.get_frontend_attribute("research_native_tag")=="edited-with-native-api"
    (directory/"edited.hlo").write_text(native.to_string())
    native_cost=_jax.hlo_module_cost_analysis(xla_bridge.get_backend("cpu"),native)
    assert native_cost==summaries[0]["lowered_cost"]
    write_json(directory/"summary.json",{"tag":"edited-with-native-api","cost":native_cost})

    # Analyze the existing TPU-targeted outer custom call with the CPU analyzer.
    # This is an explicit reference-model limitation, not a claim about libtpu's model.
    directory=output/"opaque-custom-call-cost";directory.mkdir()
    mosaic_origin=ROOT/"artifacts/jax-stack/mosaic-events-001/named/stablehlo.mlir"
    shutil.copy2(mosaic_origin,directory/"input.mlir")
    opaque=_jax.mlir.mlir_module_to_xla_computation(mosaic_origin.read_text())
    (directory/"exported.hlo").write_text(opaque.as_hlo_text())
    opaque_cost=_jax.hlo_module_cost_analysis(xla_bridge.get_backend("cpu"),opaque.as_hlo_module())
    assert all(opaque_cost[key]==-1 for key in ("flops","bytes accessed","optimal_seconds"))
    opaque_summary={"analysis_backend":"cpu","input_target":"tpu_custom_call",
        "origin":{"path":str(mosaic_origin.relative_to(ROOT)),**fingerprint(mosaic_origin)},
        "cost":opaque_cost,"roofline_available":False,
        "reason":"Negative values denote unknown cost. They must not be treated as zero work or divided to produce an intensity.",
        "scope":"CPU reference analyzer applied to TPU-targeted IR; no libtpu cost analysis or device execution."}
    write_json(directory/"summary.json",opaque_summary)

    # Demonstrate a real semantic HLO edit and compile the edited representation.
    directory=output/"hlo-edit";directory.mkdir()
    initial=jax.jit(lambda a,w,b:dot(a,w)+b).lower(a,w,b).compiler_ir("hlo").as_hlo_text()
    edited,count=re.subn(r"\badd\(","subtract(",initial)
    assert count==1
    (directory/"before.hlo").write_text(initial);(directory/"after.hlo").write_text(edited)
    backend=xla_bridge.get_backend("cpu")
    native_args=[jax.device_put(v) for v in (a,w,b)]
    edit_results=[]
    for label,text,expected in [("before",initial,reference+b),("after",edited,reference-b)]:
        parsed=_hlo.hlo_module_from_text(text)
        computation=_hlo.XlaComputation(parsed.as_serialized_hlo_module_proto())
        module_text=_jax.mlir.xla_computation_to_mlir_module(computation)
        (directory/f"{label}-imported.mlir").write_text(module_text)
        executable=backend.compile_and_load(module_text,backend.devices(),compiler.get_compile_options(num_replicas=1,num_partitions=1))
        values=executable.execute(native_args)
        assert len(values)==1
        actual=np.asarray(values[0]);np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        np.save(directory/f"{label}-output.npy",actual,allow_pickle=False)
        hlo_modules=executable.hlo_modules();assert len(hlo_modules)==1
        (directory/f"{label}-optimized.hlo").write_text(hlo_modules[0].to_string())
        edit_results.append({"case":label,"max_absolute_error":float(np.max(np.abs(actual-expected)))})
    before=np.load(directory/"before-output.npy",allow_pickle=False)
    after=np.load(directory/"after-output.npy",allow_pickle=False)
    np.testing.assert_allclose(before-after,2*b,rtol=2e-5,atol=2e-5)
    malformed=initial.replace("add(","not_a_real_opcode(")
    (directory/"malformed.hlo").write_text(malformed)
    try:_hlo.hlo_module_from_text(malformed)
    except Exception as error:
        rejection={"type":type(error).__name__,"message":str(error),"stage":"HLO text parser"}
    else:raise AssertionError("Malformed HLO unexpectedly parsed")
    edit_summary={"edit":"one add opcode replaced by subtract; identical input/output types","results":edit_results,
                  "output_difference":"before - after equals 2*b","malformed_rejection":rejection,
                  "interface_scope":"Private jaxlib parser/import/compiler bindings; not a stable public JAX graph-editing API."}
    write_json(directory/"summary.json",edit_summary)

    summary={"outcome":"pass","evidence_level":"RUN-CPU","metadata_cases":summaries,
        "direct_ir_attributes":direct,"hlo_edit":edit_summary,"native_hlo_attribute_cost":native_cost,
        "opaque_custom_call":opaque_summary,
        "cost_conclusion":"The tested research_flops metadata reaches HLO but does not override this CPU dot cost model.",
        "limits":["Arbitrary metadata behavior is key-, pass-, backend- and version-dependent; this does not prove every attribute is ignored.",
                  "Compiled fusion can change instruction ownership and metadata propagation; no one-to-one IR mapping is assumed.",
                  "FLOPs and bytes are compiler estimates, not hardware counters or measured memory traffic.",
                  "No TPU runtime, LLO, hardware roofline or real inference workload was executed."]}
    write_json(output/"summary.json",summary)
    spec=importlib.util.spec_from_file_location("baseline_capture",ROOT/"tools/capture-baseline.py")
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    env=baseline.capture_baseline();assert all(not s["dirty"] for s in env["repository"]["sources"].values())
    write_json(output/"environment.json",env)
    return summary,env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output=parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT/"artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack directory and unset ambient XLA_FLAGS")
    os.environ.update(JAX_PLATFORMS="cpu",PYTHONDONTWRITEBYTECODE="1")
    output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,output/"producer.py")
    started=datetime.now(timezone.utc).isoformat()
    with (output/"run.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:summary,env=collect(output)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    write_json(output/"manifest.json",{"capture_id":output.name,"outcome":"pass","evidence_level":"RUN-CPU",
        "qualifiers":env["status"]["qualifiers"],"started_at":started,"finished_at":datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[".venv/bin/python","-B","research/jax-stack/attributes_cost_probe.py","--output",str(output.relative_to(ROOT))],
                    "source":str((output/"producer.py").relative_to(ROOT)),**fingerprint(output/"producer.py")},
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture":output.name,"metadata_cases":len(summary["metadata_cases"]),"hlo_edit":"executed","qualifiers":env["status"]["qualifiers"]}))


if __name__=="__main__":main()
