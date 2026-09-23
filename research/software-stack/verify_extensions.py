#!/usr/bin/env python3
"""Audit immutable Pallas/profiling captures and rederive their observations."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from verify_research import ROOT, HERE, check, local_path, read_json, sha256


def audit(capture, manifest=None):
    manifest = read_json(capture / "manifest.json") if manifest is None else manifest
    check(manifest["outcome"] == "pass", "capture failed")
    registered=set()
    for item in manifest["artifacts"]:
        path=local_path(item["path"])
        check(path.is_relative_to(capture) and path not in registered,"invalid artifact path")
        check(path.is_file() and path.stat().st_size==item["size_bytes"],"artifact missing or size differs")
        check(sha256(path)==item["sha256"],"artifact hash mismatch")
        registered.add(path)
    check(registered=={p for p in capture.rglob("*") if p.is_file() and p.name!="manifest.json"},"incomplete inventory")
    env=read_json(capture/"environment.json")
    if env["repository"]["sources"]["jax"]["git_commit"]!=env["runtime"]["jaxlib"]["build_revision"]:
        check("VERSION-SKEW" in manifest["qualifiers"],"missing VERSION-SKEW")
    check(env["runtime"]["device"]["default_platform"]=="cpu","unexpected runtime backend")
    check(all(not source["dirty"] for source in env["repository"]["sources"].values()),"capture sources dirty")
    return {"capture_id":capture.name,"manifest_path":str((capture/"manifest.json").relative_to(ROOT)),
            "manifest_sha256":sha256(capture/"manifest.json"),"artifact_count":len(registered),
            "qualifiers":manifest["qualifiers"],"evidence_level":"RUN-CPU"}


def events(capture):
    files=list((capture/"trace").rglob("perfetto_trace.json.gz"))
    check(len(files)==1,"wrong trace count")
    with gzip.open(files[0],"rt") as stream: raw=json.load(stream)
    return [e for e in raw["traceEvents"] if e.get("ph")=="X"]


def contains(parent,child):
    return (parent["pid"],parent["tid"])==(child["pid"],child["tid"]) and (
        parent["ts"]<=child["ts"] and parent["ts"]+parent["dur"]+1e-6>=child["ts"]+child["dur"])


def only(all_events,name):
    matches=[e for e in all_events if e["name"]==name]
    check(len(matches)==1,f"wrong event count: {name}")
    return matches[0]


def verify_pallas(capture):
    result=audit(capture)
    with np.load(capture/"inputs.npz",allow_pickle=False) as data:
        a,w,x=[data[k].astype(np.float64) for k in ("a","w","x")]
    cases=[]
    for name in ["regular","pallas-generic","pallas-tpu-interpret","pallas-generic-vmap"]:
        expected=(x if name.endswith("vmap") else a)@w
        actual=np.load(capture/name/"output.npy",allow_pickle=False)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        cases.append({**read_json(capture/name/"summary.json"),"rechecked_max_absolute_error":float(np.max(np.abs(actual-expected)))})
    points=read_json(capture/"grid-points.json")["observations"]
    check(len(points)==6 and {tuple(p["coordinates"]) for p in points}=={(i,j) for i in range(2) for j in range(3)},"grid mismatch")
    check(all(p["core"]==0 for p in points),"unexpected core")
    check("Only interpret mode is supported on CPU backend" in read_json(capture/"non-interpret-cpu.json")["message"],"expected failure absent")
    result.update(cases=cases,grid_points=points,evidence_levels=["RUN-CPU","SIM-TPU"])
    return result


def verify_host(capture):
    result=audit(capture);all_events=events(capture)
    selected=[e for e in all_events if e["name"].startswith("research_")]
    saved=read_json(capture/"selected-events.json")
    check(saved==[{k:e[k] for k in ("name","pid","tid","ts","dur","args") if k in e} for e in selected],"selected events differ from trace")
    counts=Counter(e["name"] for e in selected)
    expected={"research_step":3,"research_dispatch":3,"research_wait":3,"research_constructor_span":1,
              "research_before_enter":1,"research_after_enter":1,"research_manual_span":1,"research_manual_work":1,"research_exception_span":1}
    check(counts==expected,"host event counts differ")
    for parent,child in [("research_constructor_span","research_before_enter"),("research_constructor_span","research_after_enter"),("research_manual_span","research_manual_work")]:
        check(contains(only(selected,parent),only(selected,child)),"host containment failed")
    steps=[e for e in selected if e["name"]=="research_step"]
    for step in steps:
        for name in ["research_dispatch","research_wait"]:
            check(sum(contains(step,e) for e in selected if e["name"]==name)==1,"step membership failed")
    summary=read_json(capture/"summary.json")
    check(summary["event_counts"]==counts,"summary counts differ")
    for name in counts:
        durations=[e["dur"] for e in selected if e["name"]==name]
        computed={"count":len(durations),"inclusive_total_us":sum(durations),"min_us":min(durations),"max_us":max(durations)}
        check(computed==summary["stats"][name],"summary timing statistics differ")
    for step,reported in zip(sorted(steps,key=lambda e:e["ts"]),summary["step_scopes"],strict=True):
        intervals=sorted((e["ts"],e["ts"]+e["dur"]) for e in selected
                         if e["name"] in {"research_dispatch","research_wait"} and contains(step,e))
        # Exactly two children are established above; union removes their overlap.
        (a,b),(c,d)=intervals
        union=(b-a)+(d-c)-max(0,min(b,d)-max(a,c))
        check(abs(union-reported["selected_children_union_us"])<1e-6,"child union differs")
        check(abs(max(0,step["dur"]-union)-reported["remaining_scope_us"])<1e-6,"remaining scope differs")
    check(only(selected,"research_manual_span")["args"]["completed"]=="1","late metadata absent")
    result.update(event_counts=dict(counts),summary=summary)
    return result


def verify_compiler(capture):
    result=audit(capture);all_events=events(capture)
    compile=only(all_events,"research_cold_compile")
    children=[e for e in all_events if e!=compile and contains(compile,e)]
    check(children==read_json(capture/"compile-thread-events.json"),"compiler selection differs")
    check(contains(only(all_events,"research_cold_lower"),only(all_events,"research_python_inside_jit")),"Python span not inside lowering")
    warm=[e for e in all_events if e["name"]=="research_warm_execute"]
    check(len(warm)==3,"warm count differs")
    passes={"algsimp","constant_folding","layout-assignment"}
    check(passes<={e["name"] for e in children},"missing compiler events")
    check(not any(e["name"] in passes and contains(s,e) for e in all_events for s in warm),"warm compile occurred")
    with np.load(capture/"inputs.npz",allow_pickle=False) as data:
        reference=np.tanh(data["a"].astype(np.float64)@data["w"].astype(np.float64))
    with np.load(capture/"outputs.npz",allow_pickle=False) as data:
        check(set(data.files)=={f"output_{i}" for i in range(3)},"missing compiler outputs")
        for value in data.values(): np.testing.assert_allclose(value,reference,rtol=2e-5,atol=2e-5)
    result.update(summary=read_json(capture/"summary.json"))
    return result


def mosaic_markers(path):
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jax.experimental.mosaic.dialects import tpu  # Register TPU dialect.
    with mlir.make_ir_context() as ctx,ir.Location.unknown():
        tpu.register_dialect(ctx)
        module=ir.Module.parse(path.read_text())
        check(module.operation.verify(),"invalid Mosaic IR")
        starts=[];stop_count=0
        def walk(op):
            nonlocal stop_count
            for region in op.regions:
                for block in region.blocks:
                    depth=0
                    for child in block.operations:
                        child=child.operation
                        if child.name=="tpu.trace_start":
                            starts.append({"message":str(child.attributes["message"]),"level":str(child.attributes["level"])})
                            depth+=1
                        elif child.name=="tpu.trace_stop":
                            depth-=1;stop_count+=1
                            check(depth>=0,"unmatched trace stop")
                        walk(child)
                    check(depth==0,"unclosed trace scope")
        walk(module.operation)
    return starts,stop_count


def verify_mosaic(capture):
    result=audit(capture)
    for name,expected in [("plain",0),("named",3)]:
        starts,stops=mosaic_markers(capture/name/"mosaic-raw.mlir")
        check(len(starts)==stops==expected,"wrong marker count")
        if expected:
            check([v["message"] for v in starts]==['"research_kernel"','"research_dot"','"research_store"'],"wrong marker names")
            check(all(v["level"]=="10 : i32" for v in starts),"wrong trace level")
        check("tpu_custom_call" in (capture/name/"stablehlo.mlir").read_text(),"outer custom call absent")
    with np.load(capture/"inputs.npz",allow_pickle=False) as data:
        reference=data["a"].astype(np.float64)@data["w"].astype(np.float64)
    np.testing.assert_allclose(np.load(capture/"interpret-output.npy",allow_pickle=False),reference,rtol=2e-5,atol=2e-5)
    result.update(summary=read_json(capture/"summary.json"))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--selftest",action="store_true")
    args=parser.parse_args()
    base=ROOT/"artifacts/jax-stack"
    if args.selftest:
        capture=base/"pallas-matmul-002"; original=read_json(capture/"manifest.json")
        for kind in ["hash","inventory","qualifier"]:
            forged=copy.deepcopy(original)
            if kind=="hash":forged["artifacts"][0]["sha256"]="0"*64
            elif kind=="inventory":forged["artifacts"].pop()
            else:forged["qualifiers"]=[]
            try:audit(capture,forged)
            except ValueError:pass
            else:raise AssertionError(f"forged {kind} was accepted")
        print("selftest: 3 corrupted manifests rejected; immutable files untouched")
    results={"pallas":verify_pallas(base/"pallas-matmul-002"),"host":verify_host(base/"profile-events-001"),
             "compiler":verify_compiler(base/"compiler-events-001"),"mosaic":verify_mosaic(base/"mosaic-events-001")}
    if args.write:(HERE/"extension-results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({name:value["artifact_count"] for name,value in results.items()}))


if __name__=="__main__":main()
