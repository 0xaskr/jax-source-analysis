#!/usr/bin/env python3
"""Verify latency labels and CPU cost evidence without simulating GPU estimators."""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from verify_extensions import audit
from verify_research import ROOT, HERE, check, local_path, read_json, sha256


EXPECTED = {"plain": None, "integer": "30000", "string": "30000", "zero": "0",
            "negative": "-1", "fractional": "30000.5", "invalid": "slow",
            "int64-overflow": "9223372036854775808"}


def check_cost(cost):
    check(cost["flops"] == 384 and cost["bytes accessed"] == 416, "latency label substituted for work/bytes")


def verify_source_contract(contract=None):
    contract = read_json(HERE / "latency-model-contract.json") if contract is None else contract
    check(contract["evidence_level"] == "SOURCE-ONLY", "source contract promoted to runtime evidence")
    baseline = read_json(ROOT / "manifests/baseline.json")
    check(contract["source_revision"] == baseline["repository"]["sources"]["xla"]["git_commit"], "source contract pin differs")
    entries = {e["id"]: e for e in read_json(HERE / "source-index.json")["entries"]}
    ids = [contract["parser"]["source_entry"], contract["selection_entry"], contract["gate_entry"],
           contract["attribute_transfer_entry"], contract["upstream_parser_test"]["source_entry"],
           *contract["scheduler_flow"], *[r["source_entry"] for r in contract["direct_readers"]],
           *[r["source_entry"] for r in contract["other_models"]]]
    for entry_id in ids:
        entry = entries[entry_id]
        path = local_path(entry["path"])
        check(entry["component"] == "xla" and entry["revision"] == contract["source_revision"], "source contract entry revision differs")
        check(sha256(path) == entry["source_sha256"] and path.read_text().splitlines()[entry["line"]-1].startswith(entry["anchor"]), "source contract anchor differs")
    # This is a bounded call-site inventory, not a C++ semantic model runner.
    expected = {entries[r["source_entry"]]["path"] for r in contract["direct_readers"]}
    found = set()
    source_root = ROOT / "upstream/xla/xla"
    for path in [*source_root.rglob("*.cc"), *source_root.rglob("*.h")]:
        if path.name.endswith("_test.cc"):
            continue
        if "GetLatencyFromMetadata(*instr)" in path.read_text():
            found.add(str(path.relative_to(ROOT)))
    check(found == expected and len(found) == 2, "non-test direct readers differ")
    check(contract["parser"]["input_unit"] == "nanoseconds", "wrong annotation unit")
    check(not contract["upstream_parser_test"]["executed"], "unexecuted upstream parser test counted as passed")
    check(not any(contract["runtime_evidence"][k] for k in ("gpu_estimator_executed", "tpu_estimator_executed", "scheduler_execution_from_this_capture")), "CPU capture promoted to estimator execution")
    return {"source_entries_checked": len(ids), "non_test_direct_reader_files": sorted(found),
            "contract_path": "research/jax-stack/latency-model-contract.json", "contract_sha256": sha256(HERE / "latency-model-contract.json")}


def verify(capture):
    from jax._src import xla_bridge
    from jax._src.lib import _jax
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jaxlib import _hlo

    result = audit(capture)
    source_contract = verify_source_contract()
    env = read_json(capture / "environment.json")
    for binary in env["runtime"]["jaxlib"]["native_binaries"]:
        check(sha256(local_path(binary["artifact_path"])) == binary["sha256"], "native identity changed")
    with np.load(capture / "inputs.npz", allow_pickle=False) as inputs:
        a, w = [inputs[k].astype(np.float64) for k in ("a", "w")]
    check(a.shape == (4, 8) and w.shape == (8, 6), "wrong input shapes")
    reference = a @ w
    backend = xla_bridge.get_backend("cpu")
    cases = []
    for name, expected in EXPECTED.items():
        directory = capture / name
        saved = read_json(directory / "summary.json")
        actual = np.load(directory / "output.npy", allow_pickle=False)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        with mlir.make_ir_context():
            module = ir.Module.parse((directory / "stablehlo.mlir").read_text())
            check(module.operation.verify(), "invalid StableHLO")
            dot = next(op.operation for fn in module.body.operations
                       for op in fn.regions[0].blocks[0].operations if op.operation.name == "stablehlo.dot_general")
            if expected is None:
                check("mhlo.frontend_attributes" not in dot.attributes, "plain dot unexpectedly tagged")
            else:
                attributes = ir.DictAttr(dot.attributes["mhlo.frontend_attributes"])
                check(ir.StringAttr(attributes["latency_metadata"]).value == expected, "StableHLO latency string differs")
        attrs = {}
        for phase, filename in [("exported", "exported-hlo.txt"), ("optimized", "optimized-hlo.txt")]:
            module = _hlo.hlo_module_from_text((directory / filename).read_text())
            tagged = [i for c in module.computations() for i in c.instructions()
                      if i.get_frontend_attribute("latency_metadata") is not None]
            check(len(tagged) == (0 if expected is None else (1 if phase == "exported" else 2)), "unexpected attribute multiplicity")
            if tagged:
                check(all(i.get_frontend_attribute("latency_metadata") == expected for i in tagged), "native latency string differs")
                check({i.opcode.name for i in tagged} == ({"kDot"} if phase == "exported" else {"kDot", "kFusion"}), "attribute owner differs")
            attrs[phase] = [{"name": i.name, "opcode": i.opcode.name, "value": i.get_frontend_attribute("latency_metadata")} for i in tagged]
            if phase == "exported":
                cost = _jax.hlo_module_cost_analysis(backend, module)
                check(cost == saved["lowered_cost"], "recomputed CPU cost differs")
        check(attrs == saved["attributes"], "recorded attributes differ")
        check_cost(saved["lowered_cost"])
        check_cost(saved["compiled_cost"])
        cases.append({"case": name, "annotation_input": saved["annotation_input"], "python_type": saved["python_type"],
                      "normalized_label": expected, "attribute_owners": attrs,
                      "cpu_lowered_cost": saved["lowered_cost"], "cpu_compiled_cost": saved["compiled_cost"],
                      "max_absolute_error": float(np.max(np.abs(actual-reference)))})
    for case in cases[1:]:
        check(case["cpu_lowered_cost"] == cases[0]["cpu_lowered_cost"] and case["cpu_compiled_cost"] == cases[0]["cpu_compiled_cost"], "CPU cost dictionaries differ")
    result.update(cases=cases, source_contract=source_contract, limits=read_json(capture / "summary.json")["limits"],
                  validation_scope="Artifact hashes and current native identity; independent NumPy reference; parsed MLIR/HLO labels and owners; recomputed CPU cost. GPU latency parsing/estimators and scheduling were not executed.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    capture = ROOT / "artifacts/jax-stack/latency-metadata-001"
    if args.selftest:
        for mode in ["hash", "missing", "qualifier"]:
            fake = copy.deepcopy(read_json(capture / "manifest.json"))
            if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
            elif mode == "missing": fake["artifacts"].pop()
            else: fake["qualifiers"] = []
            try: audit(capture, fake)
            except ValueError: pass
            else: raise AssertionError(f"invalid manifest accepted: {mode}")
        try: check_cost({"flops": 30000, "bytes accessed": 416})
        except ValueError: pass
        else: raise AssertionError("latency label treated as FLOPs")
        fake = copy.deepcopy(read_json(HERE / "latency-model-contract.json"))
        fake["upstream_parser_test"]["executed"] = True
        try: verify_source_contract(fake)
        except ValueError: pass
        else: raise AssertionError("source-only test reported as executed")
        print("selftest: 3 invalid manifests, latency-as-FLOPs and source-only-test-as-executed rejected")
    result = verify(capture)
    if args.write:
        (HERE / "latency-metadata-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"capture": capture.name, "artifacts": result["artifact_count"], "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
