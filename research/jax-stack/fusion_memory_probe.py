#!/usr/bin/env python3
"""Capture CPU fusion, donation, external aliases and static buffer reuse."""

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
import warnings

from matmul_probe import ROOT, fingerprint, write_json


def graph_summary(text):
    from jaxlib import _hlo
    module=_hlo.hlo_module_from_text(text)
    entry_name=re.search(r"^ENTRY %?([^ (]+)",text,re.M).group(1)
    entry=next(c for c in module.computations() if c.name==entry_name)
    schedule=module.schedule()
    return {
        "entry_name":entry_name,
        "entry_operations":[{"name":i.name,"opcode":i.opcode.name,"operands":[v.name for v in i.operands()]} for i in entry.instructions()],
        "entry_schedule":[i.name for i in schedule.sequence(entry)] if schedule is not None else None,
        "inner_computations":[{"name":c.name,"operations":[i.opcode.name for i in c.instructions()]} for c in module.computations() if c.name!=entry_name],
        "alias_header":text.splitlines()[0],
    }


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_compilation_cache",False)
    vector=np.arange(1024,dtype=np.float32)/np.float32(2048)-np.float32(.25)
    with np.load(ROOT/"artifacts/jax-stack/cpu-matmul-003/inputs.npz",allow_pickle=False) as data:
        a,w=[data[k].copy() for k in ("a","w")]
    bias=np.linspace(-.2,.3,24,dtype=np.float32).reshape(4,6)
    np.savez(output/"inputs.npz",vector=vector,a=a,w=w,bias=bias)
    v64=vector.astype(np.float64)
    expected_chain=np.tanh(np.cos(np.sin(v64)))
    expected_matmul=np.maximum(a.astype(np.float64)@w.astype(np.float64)+bias,0)
    # Produce owned CPU output storage; do not expose an input NumPy view except
    # in the explicitly named external-view case.
    @jax.jit
    def make_owned_vector():
        return jnp.arange(1024,dtype=jnp.float32)/np.float32(2048)-np.float32(.25)

    def chain(x):return jnp.tanh(jnp.cos(jnp.sin(x)))
    def reshape(x):return (x+np.float32(1)).reshape(32,32)
    def reduce(x):return jnp.sum(x*x)
    def identity(x):return x
    def matmul(a,w,b):return jnp.maximum(jnp.matmul(a,w,precision=jax.lax.Precision.HIGHEST)+b,0)
    cases=[
        ("chain-fused",chain,(),{},False,expected_chain),
        ("chain-donated",chain,(0,),{},False,expected_chain),
        ("chain-external-view",chain,(0,),{},True,expected_chain),
        ("chain-no-instruction-fusion",chain,(),{"xla_disable_hlo_passes":"fusion"},False,expected_chain),
        ("chain-no-fusion-no-wrapper",chain,(),{"xla_disable_hlo_passes":"fusion,fusion-wrapper"},False,expected_chain),
        ("reshape-not-donated",reshape,(),{},False,(v64+1).reshape(32,32)),
        ("reshape-donated",reshape,(0,),{},False,(v64+1).reshape(32,32)),
        ("reduce-donation-unused",reduce,(0,),{},False,np.sum(v64*v64)),
        ("identity",identity,(),{},False,v64),
        ("matmul-bias",matmul,(),{},False,expected_matmul),
        ("matmul-bias-no-fusion",matmul,(),{"xla_disable_hlo_passes":"fusion,fusion-wrapper"},False,expected_matmul),
    ]
    summaries=[]
    for name,function,donate,options,external,expected in cases:
        directory=output/name;directory.mkdir()
        def work(*args):return function(*args)
        work.__name__="memory_"+name.replace("-","_")
        arrays=([jax.device_put(v,may_alias=False) for v in (a,w,bias)] if name.startswith("matmul") else [make_owned_vector()])
        for v in arrays:v.block_until_ready()
        input_pointers=[v.unsafe_buffer_pointer() for v in arrays]
        input_sizes=[int(v.size*v.dtype.itemsize) for v in arrays]
        held_view=np.asarray(arrays[0]) if external else None
        if held_view is not None:
            np.testing.assert_array_equal(held_view,vector)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            lowered=jax.jit(work,donate_argnums=donate).lower(*arrays)
            (directory/"stablehlo.mlir").write_text(lowered.as_text("stablehlo",debug_info=True))
            (directory/"exported-hlo.txt").write_text(lowered.compiler_ir("hlo").as_hlo_text())
            compiled=lowered.compile(compiler_options=options or None)
            value=compiled(*arrays);value.block_until_ready()
        output_pointer=value.unsafe_buffer_pointer()
        deleted=[v.is_deleted() for v in arrays]
        output_size=int(value.size*value.dtype.itemsize)
        actual=np.asarray(value)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        np.save(directory/"output.npy",actual,allow_pickle=False)
        if held_view is not None:
            np.testing.assert_array_equal(held_view,vector)
            np.save(directory/"retained-view.npy",held_view,allow_pickle=False)
        optimized=compiled.as_text();(directory/"optimized-hlo.txt").write_text(optimized)
        memory=compiled.memory_analysis()
        fields=["argument_size_in_bytes","output_size_in_bytes","alias_size_in_bytes","temp_size_in_bytes","generated_code_size_in_bytes"]
        stats={f:getattr(memory,f) for f in fields}
        stats["accounted_bytes"]=stats["argument_size_in_bytes"]+stats["output_size_in_bytes"]-stats["alias_size_in_bytes"]+stats["temp_size_in_bytes"]
        visible={output_pointer:output_size}
        for pointer,size,is_deleted in zip(input_pointers,input_sizes,deleted):
            if not is_deleted:visible[pointer]=max(visible.get(pointer,0),size)
        # Addresses remain in ignored raw captures. Public results keep equality only.
        write_json(directory/"pointer-observations.json",{"input_pointers":input_pointers,"output_pointer":output_pointer,
            "input_sizes":input_sizes,"output_size":output_size,"input_deleted":deleted,"held_numpy_view":external,
            "visible_distinct_payload_bytes_after_call":sum(visible.values())})
        graph=graph_summary(optimized)
        summary={"case":name,"donate_argnums":list(donate),"compiler_options":options,
            "input_deleted":deleted,"output_pointer_matches_inputs":[output_pointer==p for p in input_pointers],
            "held_numpy_view":external,"retained_numpy_view_unchanged":True if external else None,
            "visible_distinct_payload_bytes_after_call":sum(visible.values()),
            "memory_analysis":stats,"cost_analysis":compiled.cost_analysis(),"graph":graph,
            "warnings":[str(w.message) for w in recorded],"max_absolute_error":float(np.max(np.abs(actual-expected)))}
        write_json(directory/"summary.json",summary);summaries.append(summary)
    by_name={s["case"]:s for s in summaries}
    assert by_name["chain-donated"]["input_deleted"]==[True]
    assert by_name["chain-donated"]["output_pointer_matches_inputs"]==[True]
    assert by_name["chain-fused"]["input_deleted"]==[False]
    assert by_name["chain-external-view"]["output_pointer_matches_inputs"]==[False]
    assert by_name["reduce-donation-unused"]["memory_analysis"]["alias_size_in_bytes"]==0
    assert any("not usable" in message for message in by_name["reduce-donation-unused"]["warnings"])
    for name,expected_fusions in [("chain-fused",1),("chain-no-instruction-fusion",3),("chain-no-fusion-no-wrapper",0)]:
        assert sum(op["opcode"]=="kFusion" for op in by_name[name]["graph"]["entry_operations"])==expected_fusions

    # Bind each case to the actual backend dumps using its unique compiled module header.
    dumps=output/"xla-dump"
    for summary in summaries:
        header=summary["graph"]["alias_header"]
        matches=[p for p in dumps.glob("*.cpu_after_optimizations.txt") if p.read_text().splitlines()[0]==header]
        assert len(matches)==1,(summary["case"],matches)
        prefix=matches[0].name.removesuffix(".cpu_after_optimizations.txt")
        selected=[]
        for suffix in ["buffer-assignment","buffer-assignment-values","live-range","memory-usage-report"]:
            path=dumps/f"{prefix}.cpu_after_optimizations-{suffix}.txt"
            assert path.is_file(),path
            selected.append(str(path.relative_to(ROOT)))
        summary["native_module_prefix"]=prefix
        summary["native_memory_artifacts"]=selected
        write_json(output/summary["case"]/"summary.json",summary)
    write_json(output/"summary.json",{"outcome":"pass","evidence_level":"RUN-CPU","cases":summaries,
        "limits":["Static accounted bytes and visible array payloads are not process/device peak memory measurements.",
                  "Donation permission, compiler alias plan, runtime invalidation and physical pointer reuse are separate observations.",
                  "Disabling fusion and fusion-wrapper does not disable every library-specific fusion pass.",
                  "This is a CPU mechanism probe, not the required real inference fusion/split or TPU memory acceptance."]})
    spec=importlib.util.spec_from_file_location("baseline_capture",ROOT/"tools/capture-baseline.py")
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    env=baseline.capture_baseline();assert all(not s["dirty"] for s in env["repository"]["sources"].values())
    write_json(output/"environment.json",env)
    return summaries,env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output=parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT/"artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack directory and unset ambient XLA_FLAGS")
    flags=[f"--xla_dump_to={output/'xla-dump'}","--xla_dump_hlo_as_text=true","--xla_dump_hlo_pass_re=.+","--xla_dump_emitter_re=llvm"]
    os.environ.update(JAX_PLATFORMS="cpu",PYTHONDONTWRITEBYTECODE="1",XLA_FLAGS=" ".join(flags))
    output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,output/"producer.py")
    started=datetime.now(timezone.utc).isoformat()
    with (output/"run.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:summaries,env=collect(output)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    write_json(output/"manifest.json",{"capture_id":output.name,"outcome":"pass","evidence_level":"RUN-CPU",
        "qualifiers":env["status"]["qualifiers"],"started_at":started,"finished_at":datetime.now(timezone.utc).isoformat(),
        "xla_flags":flags,"cases":[s["case"] for s in summaries],
        "producer":{"argv":[".venv/bin/python","-B","research/jax-stack/fusion_memory_probe.py","--output",str(output.relative_to(ROOT))],
                    "source":str((output/"producer.py").relative_to(ROOT)),**fingerprint(output/"producer.py")},
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture":output.name,"cases":len(summaries),"qualifiers":env["status"]["qualifiers"]}))


if __name__=="__main__":main()
