#!/usr/bin/env python3
"""Audit CPU fusion/donation captures against graphs, storage and live ranges."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import re

import numpy as np

from verify_research import ROOT, HERE, check, read_json
from verify_extensions import audit


def allocations(text):
    result=[];current=None
    for line in text.splitlines():
        if line in {"", "BufferAssignment:"}:continue
        match=re.fullmatch(r"allocation (\d+): size (\d+), (.+):",line)
        if match:
            current={"index":int(match[1]),"size":int(match[2]),"flags":match[3],"values":[]}
            result.append(current);continue
        match=re.fullmatch(r" value: <(\d+) (.+?) @(\d+)> \(size=(\d+),offset=(\d+)\): (.+)",line)
        check(match is not None and current is not None,"unrecognized allocation line")
        value={"id":int(match[1]),"name":match[2],"version":int(match[3]),"size":int(match[4]),"offset":int(match[5]),"shape":match[6]}
        check(value["offset"]+value["size"]<=current["size"],"value exceeds allocation")
        current["values"].append(value)
    check([a["index"] for a in result]==list(range(len(result))),"allocation index gap")
    check(all(a["values"] for a in result),"empty allocation")
    return result


def live_ranges(text):
    sequence=[];ranges={};mode=None;peak=None;peak_values=[]
    maximum=int(re.search(r"HloLiveRange \(max (\d+)\)",text)[1])
    for line in text.splitlines():
        if line=="  InstructionSequence:":mode="sequence";continue
        if line=="  BufferLiveRange:":mode="range";continue
        match=re.fullmatch(r"  Live ranges at (\d+) \(peak\):",line)
        if match:peak=int(match[1]);mode="peak";continue
        if line.startswith("  Stack trace breakdown"):mode=None
        if mode=="sequence":
            match=re.fullmatch(r"    (\d+):(.+)",line);check(match is not None,"invalid schedule")
            check(int(match[1])==len(sequence),"schedule index gap");sequence.append(match[2])
        elif mode=="range":
            match=re.fullmatch(r"    (.+):(\d+)-(\d+)",line);check(match is not None,"invalid live range")
            check(match[1] not in ranges,"duplicate live range")
            ranges[match[1]]=(int(match[2]),int(match[3]))
        elif mode=="peak":
            match=re.fullmatch(r"    (.+): (\d+) bytes \(cumulative: (\d+) bytes\)",line)
            if match:peak_values.append({"name":match[1],"bytes":int(match[2]),"cumulative":int(match[3])})
    check(peak is not None and len(sequence)==maximum,"missing live-range summary")
    return {"schedule":sequence,"ranges":ranges,"max_time":maximum,"reported_peak_time":peak,
            "reported_peak_value_bytes":sum(v["bytes"] for v in peak_values)}


def storage_analysis(assigned,live):
    values={}
    for allocation in assigned:
        for value in allocation["values"]:
            key=value["name"] if value["name"].endswith("}") else value["name"]+"{}"
            values[key]=value
    check(set(live["ranges"])<=set(values),"unbound live range")
    timeline=[]
    for time in range(live["max_time"]+1):
        logical=sum(values[k]["size"] for k,(start,end) in live["ranges"].items() if start<=time<=end)
        timeline.append(logical)
    logical_peak=max(timeline)
    check(timeline[live["reported_peak_time"]]==live["reported_peak_value_bytes"],"reported list does not match printed ranges")
    shared=[]
    for allocation in assigned:
        for a,b in itertools.combinations(allocation["values"],2):
            overlap=max(0,min(a["offset"]+a["size"],b["offset"]+b["size"])-max(a["offset"],b["offset"]))
            if not overlap:continue
            ka=a["name"] if a["name"].endswith("}") else a["name"]+"{}"
            kb=b["name"] if b["name"].endswith("}") else b["name"]+"{}"
            if ka not in live["ranges"] or kb not in live["ranges"]:continue
            ra,rb=live["ranges"][ka],live["ranges"][kb]
            check(max(ra[0],rb[0])>=min(ra[1],rb[1]),"unexpected interior liveness overlap for shared storage")
            shared.append({"allocation":allocation["index"],"values":[ka,kb],"overlap_bytes":overlap,
                           "live_ranges":[ra,rb],"boundary_touch":max(ra[0],rb[0])==min(ra[1],rb[1])})
    accounted=sum(a["size"] for a in assigned if "constant" not in a["flags"] and "thread-local" not in a["flags"])
    return {"allocations":assigned,"shared_storage_pairs":shared,"live_range":live,
            "static_allocation_bytes":sum(a["size"] for a in assigned),"compiler_accounted_allocation_bytes":accounted,
            "inclusive_logical_value_bytes_by_time":timeline,"inclusive_logical_peak_bytes":logical_peak,
            "inclusive_logical_peak_first_time":timeline.index(logical_peak),
            "reported_peak_is_inclusive_logical_max":live["reported_peak_value_bytes"]==logical_peak,
            "scope":"Logical values use printed inclusive intervals. They may double-count reused storage and are not runtime peak memory."}


def entry_operations(text):
    from jaxlib import _hlo
    module=_hlo.hlo_module_from_text(text)
    entry_name=re.search(r"^ENTRY %?([^ (]+)",text,re.M)[1]
    entry=next(c for c in module.computations() if c.name==entry_name)
    return [{"name":i.name,"opcode":i.opcode.name,"operands":[v.name for v in i.operands()]} for i in entry.instructions()]


def verify(capture):
    result=audit(capture)
    manifest=read_json(capture/"manifest.json")
    from capture_runtime import verify_binding
    build_binding=verify_binding(capture,read_json(capture/"environment.json"),manifest)
    names={"chain-fused","chain-donated","chain-external-view","chain-no-instruction-fusion","chain-no-fusion-no-wrapper",
           "reshape-not-donated","reshape-donated","reduce-donation-unused","identity","matmul-bias","matmul-bias-no-fusion"}
    check(set(manifest["cases"])==names,"missing comparison case")
    with np.load(capture/"inputs.npz",allow_pickle=False) as data:
        vector,a,w,b=[data[k].astype(np.float64) for k in ("vector","a","w","bias")]
    chain=np.tanh(np.cos(np.sin(vector)));matmul=np.maximum(a@w+b,0)
    verified=[]
    for name in manifest["cases"]:
        directory=capture/name;summary=read_json(directory/"summary.json")
        expected=(chain if name.startswith("chain") else ((vector+1).reshape(32,32) if name.startswith("reshape")
                  else np.sum(vector*vector) if name.startswith("reduce") else vector if name=="identity" else matmul))
        np.testing.assert_allclose(np.load(directory/"output.npy",allow_pickle=False),expected,rtol=2e-5,atol=2e-5)
        observations=read_json(directory/"pointer-observations.json")
        matches=[p==observations["output_pointer"] for p in observations["input_pointers"]]
        check(matches==summary["output_pointer_matches_inputs"],"pointer equality mismatch")
        check(observations["input_deleted"]==summary["input_deleted"],"invalidation mismatch")
        visible={observations["output_pointer"]:observations["output_size"]}
        for p,size,deleted in zip(observations["input_pointers"],observations["input_sizes"],observations["input_deleted"],strict=True):
            if not deleted:visible[p]=max(visible.get(p,0),size)
        check(sum(visible.values())==summary["visible_distinct_payload_bytes_after_call"],"visible payload mismatch")
        if name=="chain-external-view":np.testing.assert_array_equal(np.load(directory/"retained-view.npy",allow_pickle=False),vector)
        optimized=(directory/"optimized-hlo.txt").read_text();ops=entry_operations(optimized)
        check(ops==summary["graph"]["entry_operations"],"entry graph mismatch")
        prefix=summary["native_module_prefix"];dump=capture/"xla-dump"
        check((dump/f"{prefix}.cpu_after_optimizations.txt").read_text().splitlines()[0]==optimized.splitlines()[0],"native module binding mismatch")
        assigned=allocations((dump/f"{prefix}.cpu_after_optimizations-buffer-assignment.txt").read_text())
        live=live_ranges((dump/f"{prefix}.cpu_after_optimizations-live-range.txt").read_text())
        check(live["schedule"]==summary["graph"]["entry_schedule"],"schedule mismatch")
        storage=storage_analysis(assigned,live);memory=summary["memory_analysis"]
        check(storage["compiler_accounted_allocation_bytes"]==memory["accounted_bytes"],"accounted allocation size mismatch")
        check(sum(a["size"] for a in assigned if "preallocated-temp" in a["flags"])==memory["temp_size_in_bytes"],"temp size mismatch")
        check(sum(a["size"] for a in assigned if "parameter " in a["flags"] and "maybe-live-out" in a["flags"])==memory["alias_size_in_bytes"],"alias size mismatch")
        pass_observations=[]
        for path in sorted(dump.glob(prefix+".*.txt")):
            if re.search(r"\.\d{4}\.",path.name) and any(x in path.name for x in ["before_fusion","after_fusion"]):
                pass_observations.append({"path":str(path.relative_to(ROOT)),"entry_operations":entry_operations(path.read_text())})
        verified.append({**summary,"storage_analysis":storage,"fusion_pass_observations":pass_observations})
    cases={c["case"]:c for c in verified}
    for name,count in [("chain-fused",1),("chain-no-instruction-fusion",3),("chain-no-fusion-no-wrapper",0)]:
        check(sum(op["opcode"]=="kFusion" for op in cases[name]["graph"]["entry_operations"])==count,"fusion control mismatch")
    check(cases["chain-donated"]["input_deleted"]==[True] and cases["chain-donated"]["output_pointer_matches_inputs"]==[True],"owned donation did not reuse")
    check(cases["chain-donated"]["memory_analysis"]==cases["chain-external-view"]["memory_analysis"],"external view changed compiler plan")
    check(cases["chain-external-view"]["visible_distinct_payload_bytes_after_call"]>cases["chain-external-view"]["memory_analysis"]["accounted_bytes"],"runtime/static distinction missing")
    check(cases["reshape-donated"]["memory_analysis"]["accounted_bytes"]==cases["reshape-not-donated"]["memory_analysis"]["accounted_bytes"],"reshape counterfactual differs")
    peak_difference=not cases["reduce-donation-unused"]["storage_analysis"]["reported_peak_is_inclusive_logical_max"]
    if build_binding is None:
        check(peak_difference,"expected historical diagnostic peak discrepancy absent")
    check(any("not usable" in w for w in cases["reduce-donation-unused"]["warnings"]),"missing unused donation warning")
    result.update(cases=verified,limits=read_json(capture/"summary.json")["limits"],build_binding=build_binding,
                  historical_peak_difference_present=peak_difference)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--capture",type=Path,default=ROOT/"artifacts/jax-stack/fusion-memory-002")
    args=parser.parse_args();capture=args.capture.resolve()
    if args.selftest:
        historical=ROOT/"artifacts/jax-stack/fusion-memory-002"
        original=read_json(historical/"manifest.json")
        for kind in ["hash","inventory","qualifier"]:
            fake=copy.deepcopy(original)
            if kind=="hash":fake["artifacts"][0]["sha256"]="0"*64
            elif kind=="inventory":fake["artifacts"].pop()
            else:fake["qualifiers"]=[]
            try:audit(historical,fake)
            except ValueError:pass
            else:raise AssertionError(f"bad manifest accepted: {kind}")
        try:allocations("BufferAssignment:\nallocation 0: size 4, page 0:\n value: <0 x @0> (size=8,offset=0): f32[2]\n")
        except ValueError:pass
        else:raise AssertionError("out-of-bounds allocation accepted")
        print("selftest: 3 corrupted manifests and out-of-bounds storage rejected")
    result=verify(capture)
    if args.write:(HERE/"fusion-memory-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"capture":capture.name,"cases":len(result["cases"]),"artifacts":result["artifact_count"],"qualifiers":result["qualifiers"]}))


if __name__=="__main__":main()
