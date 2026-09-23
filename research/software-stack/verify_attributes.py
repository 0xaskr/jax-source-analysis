#!/usr/bin/env python3
"""Independently reparse metadata/HLO and recheck numerical and cost evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from verify_research import ROOT, HERE, check, read_json
from verify_extensions import audit


def instructions(path):
    from jaxlib import _hlo
    module=_hlo.hlo_module_from_text(path.read_text())
    return module,[i for c in module.computations() for i in c.instructions()]


def check_dot_cost(cost):
    check(cost.get("flops")==2*4*8*6,"dot FLOPs differ")
    check(cost.get("bytes accessed")==4*(4*8+8*6+4*6),"dot byte estimate differs")


def verify(capture):
    from jax._src import xla_bridge
    from jax._src.lib import _jax
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jaxlib import _hlo

    result=audit(capture);backend=xla_bridge.get_backend("cpu")
    from capture_runtime import verify_binding, verify_current_reader
    binding=verify_binding(capture,read_json(capture/"environment.json"),read_json(capture/"manifest.json"))
    reader=verify_current_reader(binding)
    with np.load(capture/"inputs.npz",allow_pickle=False) as data:
        a,w,b=[data[k].astype(np.float64) for k in ("a","w","b")]
    reference=a@w;gradient=2*reference@w.T
    expected_counts={"plain":0,"context":1,"value":1,"call":1,"grad-call-same":2,"grad-call-drop":1}
    cases=[]
    for name,count in expected_counts.items():
        directory=capture/name
        expected=gradient if name.startswith("grad-") else reference
        actual=np.load(directory/"output.npy",allow_pickle=False)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        module,ops=instructions(directory/"exported-hlo.txt")
        tagged=[i for i in ops if i.get_frontend_attribute("research_tag") is not None]
        check(len(tagged)==count,"wrong native HLO metadata count")
        for op in tagged:
            check(op.get_frontend_attribute("research_tag")=="matmul","tag differs")
            check(op.get_frontend_attribute("research_boolean")=="true","bool normalization differs")
            check(op.get_frontend_attribute("research_flops")=="999999","integer normalization differs")
        cost=_jax.hlo_module_cost_analysis(backend,module)
        summary=read_json(directory/"summary.json")
        check(cost==summary["lowered_cost"],"current native cost differs from capture")
        if not name.startswith("grad-"):
            check_dot_cost(cost);check_dot_cost(summary["compiled_cost"])
        else:check(cost["flops"]==792,"gradient FLOPs differ")
        if name in {"call","grad-call-same","grad-call-drop"}:
            check(all(i.opcode==_hlo.HloOpcode.kCall for i in tagged),"tag not attached to call")
        cases.append({"case":name,"tagged_hlo_instructions":len(tagged),"cost":cost,
                      "max_absolute_error":float(np.max(np.abs(actual-expected)))})

    directory=capture/"direct-ir-attributes"
    with mlir.make_ir_context():
        module=ir.Module.parse((directory/"stablehlo.mlir").read_text())
        check(module.operation.verify(),"edited StableHLO invalid")
        dot=next(op.operation for fn in module.body.operations for op in fn.regions[0].blocks[0].operations
                 if op.operation.name=="stablehlo.dot_general")
        check(str(dot.attributes["research.raw"])=='"raw-visible"',"raw input attribute absent")
        attrs=ir.DictAttr(dot.attributes["mhlo.frontend_attributes"])
        check(str(attrs["research_integer"])=="123 : i64","typed integer absent")
    _,ops=instructions(directory/"exported-hlo.txt")
    dot=next(i for i in ops if i.opcode==_hlo.HloOpcode.kDot)
    check(dot.get_frontend_attribute("research_string")=="string-visible","string disappeared")
    check(dot.get_frontend_attribute("research_bool")=="true","bool disappeared")
    check(dot.get_frontend_attribute("research_integer") is None,"unexpected integer propagation")
    check(dot.get_frontend_attribute("research.raw") is None,"unexpected raw propagation")

    native,native_ops=instructions(capture/"native-hlo-attributes/edited.hlo")
    check(sum(i.get_frontend_attribute("research_native_tag")=="edited-with-native-api" for i in native_ops)==1,"native setter result missing")
    check_dot_cost(_jax.hlo_module_cost_analysis(backend,native))
    opaque,_=instructions(capture/"opaque-custom-call-cost/exported.hlo")
    unknown=_jax.hlo_module_cost_analysis(backend,opaque)
    check(all(unknown.get(k)==-1 for k in ("flops","bytes accessed","optimal_seconds")),"unknown cost sentinel differs")

    directory=capture/"hlo-edit"
    before_module,before_ops=instructions(directory/"before.hlo")
    after_module,after_ops=instructions(directory/"after.hlo")
    before_counts=Counter(i.opcode.name for i in before_ops)
    after_counts=Counter(i.opcode.name for i in after_ops)
    check(before_counts-after_counts=={"kAdd":1} and after_counts-before_counts=={"kSubtract":1},"semantic edit differs")
    before=np.load(directory/"before-output.npy",allow_pickle=False)
    after=np.load(directory/"after-output.npy",allow_pickle=False)
    np.testing.assert_allclose(before,reference+b,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(after,reference-b,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(before-after,2*b,rtol=2e-5,atol=2e-5)
    try:_hlo.hlo_module_from_text((directory/"malformed.hlo").read_text())
    except Exception as error:check("Unknown opcode" in str(error),"unexpected malformed HLO failure")
    else:raise ValueError("malformed HLO accepted")
    result.update(cases=cases,build_binding=binding,verification_runtime=reader,direct_attribute_transfer=read_json(capture/"direct-ir-attributes/summary.json"),
                  opaque_custom_call=read_json(capture/"opaque-custom-call-cost/summary.json"),
                  hlo_edit=read_json(capture/"hlo-edit/summary.json"),
                  validation_scope="Reparsed native HLO metadata/opcodes, current CPU cost reader, immutable artifact hashes, independent NumPy outputs. Executable rerun is in the producer capture.")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--capture",type=Path,default=ROOT/"artifacts/jax-stack/attributes-cost-002")
    args=parser.parse_args();capture=ROOT/"artifacts/jax-stack/attributes-cost-002"
    if args.selftest:
        original=read_json(capture/"manifest.json")
        for mode in ["hash","missing","qualifier"]:
            fake=copy.deepcopy(original)
            if mode=="hash":fake["artifacts"][0]["sha256"]="0"*64
            elif mode=="missing":fake["artifacts"].pop()
            else:fake["qualifiers"]=[]
            try:audit(capture,fake)
            except ValueError:pass
            else:raise AssertionError(f"bad manifest accepted: {mode}")
        try:check_dot_cost({"flops":999999,"bytes accessed":416})
        except ValueError:pass
        else:raise AssertionError("metadata-as-FLOPs accepted")
        print("selftest: 3 corrupted manifests and a metadata-as-FLOPs claim rejected")
    capture=args.capture.resolve()
    result=verify(capture)
    if args.write:(HERE/"attributes-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"capture":capture.name,"artifacts":result["artifact_count"],"cases":len(result["cases"]),"qualifiers":result["qualifiers"]}))


if __name__=="__main__":main()
