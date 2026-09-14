#!/usr/bin/env python3
"""Reparse CPU collective/control IR and independently recompute outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import re

import numpy as np

from verify_extensions import audit
from verify_research import ROOT, HERE, check, local_path, read_json, sha256


def parse_hlo(path):
    from jaxlib import _hlo
    text = path.read_text()
    module = _hlo.hlo_module_from_text(text)
    opcodes = Counter(i.opcode.name for c in module.computations() for i in c.instructions())
    targets = Counter(re.findall(r'custom_call_target="([^"]+)"', text))
    return module, {"opcodes": dict(sorted(opcodes.items())), "custom_call_targets": dict(sorted(targets.items()))}


def mlir_targets(path):
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    with mlir.make_ir_context():
        module = ir.Module.parse(path.read_text())
        check(module.operation.verify(), "invalid StableHLO")
        def walk(op):
            yield op
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        yield from walk(child.operation)
        return dict(Counter(ir.StringAttr(op.attributes["call_target_name"]).value
                            for op in walk(module.operation) if op.name == "stablehlo.custom_call"))


def check_lowered_targets(targets, expected):
    selected = {k: v for k, v in targets.items() if k in {"all-reduce-start", "all-reduce-done", "control_dep"}}
    check(selected == expected, "lowered collective/control targets differ")


def control_edge(path):
    from jaxlib import _hlo
    module, _ = parse_hlo(path)
    edges = []
    for comp in module.computations():
        dots = {i.name for i in comp.instructions() if i.opcode == _hlo.HloOpcode.kDot}
        for op in comp.instructions():
            if op.opcode != _hlo.HloOpcode.kAllReduce:
                continue
            line = next(line for line in path.read_text().splitlines() if re.match(r"\s*(?:ROOT )?%?" + re.escape(op.name) + r" = ", line))
            match = re.search(r"control-predecessors=\{([^}]*)\}", line)
            if match:
                sources = {s.strip().removeprefix("%") for s in match[1].split(",")}
                edges.extend({"source": src, "target": op.name, "source_opcode": "kDot", "target_opcode": "kAllReduce"} for src in sources & dots)
    check(len(edges) == 1, "expected one dot-to-all-reduce control edge")
    return edges[0]


def verify(capture):
    result = audit(capture)
    env = read_json(capture / "environment.json")
    check(env["runtime"]["device"]["device_count"] == 2, "capture did not use two CPU devices")
    for binary in env["runtime"]["jaxlib"]["native_binaries"]:
        check(sha256(local_path(binary["artifact_path"])) == binary["sha256"], "current native binary changed")
    summary = read_json(capture / "summary.json")
    check(summary["topology"]["process_count"] == 1 and summary["topology"]["mesh_axes"] == {"d": 2}, "unexpected topology")
    with np.load(capture / "inputs.npz", allow_pickle=False) as inputs:
        x, w = [inputs[k].astype(np.float64) for k in ("x", "w")]
    check(x.shape == (128, 64) and w.shape == (64, 64), "unexpected input shapes")
    local_result = x[:64] + x[64:] + w @ w
    reference = np.concatenate((local_result, local_result), axis=0)
    cases = []
    for name in ["sync", "async", "sync-control"]:
        directory = capture / name
        actual = np.load(directory / "output.npy", allow_pickle=False)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        saved = read_json(directory / "summary.json")
        expected = {} if name == "sync" else ({"all-reduce-start": 1, "all-reduce-done": 1} if name == "async" else {"control_dep": 1})
        check_lowered_targets(mlir_targets(directory / "stablehlo.mlir"), expected)
        _, exported = parse_hlo(directory / "exported-hlo.txt")
        check(exported == saved["exported"], "exported counts differ")
        _, optimized = parse_hlo(directory / "optimized-hlo.txt")
        check(optimized == saved["optimized"], "optimized counts differ")
        check(optimized["opcodes"].get("kAllReduce") == 1, "missing final synchronous all-reduce")
        check(not any(optimized["opcodes"].get(op, 0) for op in ["kAsyncStart", "kAsyncDone", "kAllReduceStart", "kAllReduceDone"]), "final async instructions remain")
        check_lowered_targets(optimized["custom_call_targets"], {})
        header = (directory / "optimized-hlo.txt").read_text().splitlines()[0]
        matches = [p for p in (capture / "xla-dump").glob("*.cpu_after_optimizations.txt") if p.read_text().splitlines()[0] == header]
        check(len(matches) == 1, "ambiguous native module binding")
        prefix = matches[0].name.removesuffix(".cpu_after_optimizations.txt")
        check(prefix == saved["native_module_prefix"], "native prefix differs")
        paths = sorted(p for p in (capture / "xla-dump").glob(prefix + ".*.txt") if "async-collective" in p.name or "control-dep-rewriter" in p.name)
        rederived = [{"path": str(p.relative_to(ROOT)), **parse_hlo(p)[1]} for p in paths]
        check(rederived == saved["async_pipeline_boundaries"], "native pass boundaries differ")
        if name == "async":
            middle = next(b for b in rederived if ".after_async-collective-custom-call-rewriter." in b["path"])
            after = next(b for b in rederived if ".after_async-collective-replacer." in b["path"])
            check(middle["opcodes"].get("kAsyncStart") == middle["opcodes"].get("kAsyncDone") == 1, "generic async pair missing")
            check(after["opcodes"].get("kAsyncStart", 0) == after["opcodes"].get("kAsyncDone", 0) == 0, "replacer did not remove async pair")
        edge = None
        if name == "sync-control":
            before = next(b for b in rederived if ".before_control-dep-rewriter." in b["path"])
            after = next(b for b in rederived if ".after_control-dep-rewriter." in b["path"])
            check(before["custom_call_targets"].get("control_dep") == 1 and not after["custom_call_targets"].get("control_dep"), "control hint not consumed")
            edge = control_edge(local_path(after["path"]))
        cases.append({"case": name, "max_absolute_error": float(np.max(np.abs(actual-reference))),
                      "stablehlo_targets": expected, "optimized": optimized, "control_edge_after_rewriter": edge,
                      "native_module_prefix": prefix, "selected_boundaries": rederived})
    failures = []
    for name, fragment in [("scheduled-varying-checked", "requires varying manual axes to match"),
                           ("scheduled-varying-unchecked", "has no attribute done"),
                           ("scheduled-invariant-default-layout", "incorrect layout dense<>"),
                           ("explicit-layout-backend-error", "is live and cannot be removed")]:
        saved = read_json(capture / name / "expected-error.json")
        check(fragment in saved["message"], "expected error differs")
        failures.append({"case": name, "phase": saved["phase"], "error_type": saved["type"], "error_fragment": fragment})
    check_lowered_targets(mlir_targets(capture / "explicit-layout-backend-error/stablehlo.mlir"),
                          {"all-reduce-start": 1, "all-reduce-done": 1, "control_dep": 2})
    result.update(topology=summary["topology"], cases=cases, expected_failures=failures, limits=summary["limits"],
                  validation_scope="Immutable hashes, current native identity, independent NumPy reference, MLIR verifier/targets, native HLO opcodes and one control edge. Expected errors are captured producer observations, not rerun by this validator.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    capture = ROOT / "artifacts/jax-stack/overlap-cpu-005"
    if args.selftest:
        for mode in ["hash", "missing", "qualifier"]:
            fake = copy.deepcopy(read_json(capture / "manifest.json"))
            if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
            elif mode == "missing": fake["artifacts"].pop()
            else: fake["qualifiers"] = []
            try: audit(capture, fake)
            except ValueError: pass
            else: raise AssertionError(f"invalid manifest accepted: {mode}")
        try: check_lowered_targets({"all-reduce-start": 1}, {"all-reduce-start": 1, "all-reduce-done": 1})
        except ValueError: pass
        else: raise AssertionError("unpaired lowering accepted")
        paths = list((capture / "xla-dump").glob("*sync-control*.before_control-dep-rewriter.*"))
        check(len(paths) == 1, "unexpected negative control fixture")
        try: control_edge(paths[0])
        except ValueError: pass
        else: raise AssertionError("unconverted control hint counted as an edge")
        print("selftest: 3 bad manifests, unpaired lowering and hint-as-control-edge rejected")
    result = verify(capture)
    if args.write:
        (HERE / "overlap-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"capture": capture.name, "artifacts": result["artifact_count"], "cases": len(result["cases"]), "expected_failures": len(result["expected_failures"])}))


if __name__ == "__main__":
    main()
