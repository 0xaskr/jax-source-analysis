#!/usr/bin/env python3
"""Observe named scopes in production Mosaic lowering without a TPU backend."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
from unittest import mock

from matmul_probe import ROOT, fingerprint, write_json


def make_matmul(*, named, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def scope(name):
        return jax.named_scope(name) if named else contextlib.nullcontext()

    def kernel(a_ref, w_ref, out_ref):
        with scope("research_kernel"):
            with scope("research_dot"):
                product = jnp.matmul(a_ref[...],w_ref[...],precision=jax.lax.Precision.HIGHEST)
            with scope("research_store"):
                out_ref[...] = product

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((16,128),jnp.float32),
        grid=(2,1), in_specs=[pl.BlockSpec((8,128),lambda i,j:(i,0)),
                             pl.BlockSpec((128,128),lambda i,j:(0,j))],
        out_specs=pl.BlockSpec((8,128),lambda i,j:(i,j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")),
        interpret=interpret, debug=True)


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src.pallas.mosaic import lowering

    shapes=(jax.ShapeDtypeStruct((16,128),jnp.float32),jax.ShapeDtypeStruct((128,128),jnp.float32))
    original=lowering.lower_jaxpr_to_pipelined_module
    summaries=[]
    for name,named in [("plain",False),("named",True)]:
        directory=output/name
        directory.mkdir()
        modules=[]

        def observe(*args,**kwargs):
            module=original(*args,**kwargs)
            assert module.operation.verify()
            modules.append(module.operation.get_asm(enable_debug_info=True))
            return module

        with mock.patch.object(lowering,"lower_jaxpr_to_pipelined_module",observe):
            traced=jax.jit(make_matmul(named=named)).trace(*shapes)
            lowered=traced.lower(lowering_platforms=("tpu",))
        assert lowering.lower_jaxpr_to_pipelined_module is original
        assert len(modules)==1
        (directory/"mosaic-raw.mlir").write_text(modules[0]+"\n")
        (directory/"jaxpr.txt").write_text(str(traced.jaxpr)+"\n")
        (directory/"stablehlo.mlir").write_text(lowered.as_text("stablehlo",debug_info=True))
        (directory/"exported-hlo.txt").write_text(lowered.compiler_ir("hlo").as_hlo_text())
        assert "tpu_custom_call" in (directory/"stablehlo.mlir").read_text()
        # Parse operations, not name occurrences in debug locations.
        from jax._src.interpreters import mlir
        from jax._src.lib.mlir import ir
        with mlir.make_ir_context(), ir.Location.unknown():
            module=ir.Module.parse(modules[0])
            starts=[]; stops=[]
            def visit(op):
                if op.name=="tpu.trace_start":
                    starts.append({"message":str(op.attributes["message"]),"level":str(op.attributes["level"])})
                if op.name=="tpu.trace_stop": stops.append(True)
                for region in op.regions:
                    for block in region.blocks:
                        for child in block.operations: visit(child.operation)
            visit(module.operation)
        assert len(starts)==len(stops)==(3 if named else 0)
        if named:
            assert [e["message"] for e in starts]==['"research_kernel"','"research_dot"','"research_store"']
            assert all(e["level"]=="10 : i32" for e in starts)
        summaries.append({"case":name,"trace_start":starts,"trace_stop_count":len(stops),"outer_custom_call":"tpu_custom_call"})

    rng=np.random.default_rng(20260914)
    a=(rng.standard_normal((16,128))*.02).astype(np.float32)
    w=(rng.standard_normal((128,128))*.02).astype(np.float32)
    np.savez(output/"inputs.npz",a=a,w=w)
    # Separate CPU semantic check; no .compile() is called on TPU-targeted IR.
    value=jax.jit(make_matmul(named=True,interpret=True))(a,w)
    value.block_until_ready()
    actual=np.asarray(value);np.save(output/"interpret-output.npy",actual,allow_pickle=False)
    reference=a.astype(np.float64)@w.astype(np.float64)
    np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=2e-5)
    summary={"outcome":"pass","evidence_level":"RUN-CPU","target_platform":"tpu",
        "phase":"open-source TPU lowering on a CPU host; no backend compilation",
        "cases":summaries,"interpret_max_absolute_error":float(np.max(np.abs(actual-reference))),
        "observer":"In-process wrapper calls the original production lowering and returns its unchanged module; finally restored by mock.patch.",
        "backend_compile":False,"tpu_execution":False,"llo_captured":False,
        "limitations":["Mosaic TPU MLIR is not LLO.","Balanced IR markers do not prove TPU timing, instruction preservation, trace visibility or hardware tile acceptance.",
                        "CPU interpreter numerical agreement does not verify TPU numerical behavior."]}
    write_json(output/"summary.json",summary)
    spec=importlib.util.spec_from_file_location("baseline_capture",ROOT/"tools/capture-baseline.py")
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    env=baseline.capture_baseline()
    assert all(not s["dirty"] for s in env["repository"]["sources"].values())
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
        "phase":summary["phase"],"qualifiers":env["status"]["qualifiers"],"started_at":started,
        "finished_at":datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[".venv/bin/python","-B","research/software-stack/mosaic_events_probe.py","--output",str(output.relative_to(ROOT))],"source":str((output/"producer.py").relative_to(ROOT)),**fingerprint(output/"producer.py")},
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture":output.name,"markers":[len(c["trace_start"]) for c in summary["cases"]],"backend_compile":False}))


if __name__=="__main__":main()
