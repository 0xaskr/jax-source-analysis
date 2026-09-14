#!/usr/bin/env python3
"""Audit pass-event captures, build binding and cold/warm/filter comparisons."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from pass_event_checks import EVENT, analyze, require
from pass_events_probe import bind_native, build_gate, read_events
from verify_extensions import audit
from verify_research import ROOT, HERE, local_path, read_json, sha256


def verify_capture(capture):
    result = audit(capture)
    manifest = read_json(capture / "manifest.json")
    require(manifest["runtime_acceptance"], "capture did not pass its runtime checks")
    summary = read_json(capture / "summary.json")
    environment = read_json(capture / "environment.json")
    saved_binding = read_json(capture / "build-binding.json")
    path = local_path(saved_binding["build_manifest_path"]) if saved_binding.get("source_build_verified") else None
    binding = bind_native(environment, path, manifest["expected_events"])
    require(binding == saved_binding, "recomputed build/native binding differs")
    qualifiers = list(environment["status"]["qualifiers"])
    if binding.get("source_patches"):
        qualifiers.append("SOURCE-PATCHED")
    require(manifest["qualifiers"] == qualifiers, "capture qualifiers differ from runtime/build identity")
    require(manifest["evidence_level"] == summary["evidence_level"] == "RUN-CPU", "incorrect runtime evidence level")
    for binary in environment["runtime"]["jaxlib"]["native_binaries"]:
        require(sha256(local_path(binary["artifact_path"])) == binary["sha256"], "captured native payload file changed")
    required_flags = ["--xla_cpu_enable_xprof_traceme=true"]
    if manifest["disabled_pass"] is not None:
        required_flags.append("--xla_disable_hlo_passes=" + manifest["disabled_pass"])
    require(manifest["xla_flags"] == required_flags, "capture flag set differs")
    module = re.match(r"HloModule ([^,\s]+)", (capture / "optimized-hlo.txt").read_text())[1]
    observations = analyze(read_events(capture), module, manifest["expected_events"], manifest["disabled_pass"])
    require(all(summary[k] == v for k, v in observations.items()), "raw trace and reported observations differ")
    accepted = manifest["expected_events"] == "present" and binding["source_build_verified"]
    require(summary["positive_patch_runtime_accepted"] == accepted, "positive patch acceptance lacks binding")
    require(summary["python_trace_count"] == 1, "unexpected trace count")
    with np.load(capture / "inputs.npz", allow_pickle=False) as inputs:
        a, w = [inputs[k].astype(np.float64) for k in ["a", "w"]]
    require(a.shape == (64, 128) and w.shape == (128, 64), "wrong workload shapes")
    reference = np.tanh(a @ w)
    errors = []
    with np.load(capture / "outputs.npz", allow_pickle=False) as outputs:
        require(set(outputs.files) == {"output_0", "output_1", "output_2"}, "wrong warm output set")
        for i in range(3):
            actual = outputs[f"output_{i}"]
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
            errors.append(float(np.max(np.abs(actual-reference))))
    require(errors == summary["max_absolute_errors"], "numerical summary differs")
    result.update(observations=observations, max_absolute_errors=errors, build_binding=binding,
                  positive_patch_runtime_accepted=accepted,
                  validation_scope="Artifact bytes, independent NumPy outputs, raw trace rules and canonical build/wheel-to-native binding. Producer and verifier share the documented event rules.")
    return result


def verify_pair(default, filtered):
    a, b = verify_capture(default), verify_capture(filtered)
    require(a["observations"]["disabled_pass"] is None and b["observations"]["disabled_pass"] == "algsimp", "not a default/algsimp-filter pair")
    require(a["observations"]["expected_custom_events"] == b["observations"]["expected_custom_events"], "pair expectations differ")
    for name in ["producer.py", "event-rules.py"]:
        require(sha256(default / name) == sha256(filtered / name), "pair producer/rule versions differ")
    def natives(capture):
        return {r["package_path"]: r["sha256"] for r in read_json(capture / "environment.json")["runtime"]["jaxlib"]["native_binaries"]}
    require(natives(default) == natives(filtered), "pair uses different native payloads")
    require(a["build_binding"] == b["build_binding"], "pair build origin differs")
    with np.load(default / "inputs.npz", allow_pickle=False) as x, np.load(filtered / "inputs.npz", allow_pickle=False) as y:
        for key in ["a", "w"]:
            np.testing.assert_array_equal(x[key], y[key])
    with np.load(default / "outputs.npz", allow_pickle=False) as x, np.load(filtered / "outputs.npz", allow_pickle=False) as y:
        for i in range(3):
            np.testing.assert_allclose(x[f"output_{i}"], y[f"output_{i}"], rtol=2e-5, atol=2e-5)
    return {"default": a, "filtered": b,
            "patched_runtime_pair_accepted": a["positive_patch_runtime_accepted"] and b["positive_patch_runtime_accepted"],
            "limits": ["An unbound absent pair is a current-wheel negative control, not a matching-source baseline.",
                       "Synthetic rule tests are not evidence of custom C++ event emission or of loading a built wheel.",
                       "No GPU/TPU or device-execution profiling is performed by these CPU controls.",
                       "Durations are instrumented host spans; no compiler speedup or device performance claim is made."]}


def verify_rejection(capture):
    manifest = read_json(capture / "manifest.json")
    require(manifest["outcome"] == "fail" and manifest["runtime_acceptance"] is False, "rejected build reported as accepted")
    require(manifest["evidence_level"] == "SOURCE-ONLY" and not (capture / "trace").exists(), "rejected gate promoted to runtime evidence")
    registered = set()
    for item in manifest["artifacts"]:
        path = local_path(item["path"])
        require(path.is_relative_to(capture) and path not in registered, "invalid rejected-capture path")
        require(path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"], "rejection artifact changed")
        registered.add(path)
    require(registered == {p for p in capture.rglob("*") if p.is_file() and p != capture / "manifest.json"}, "rejection inventory incomplete")
    observation = read_json(capture / "build-gate-observation.json")
    snapshot = local_path(observation["snapshot_path"])
    require(sha256(snapshot) == observation["snapshot_sha256"], "gate input snapshot differs")
    require(read_json(snapshot)["status"] == observation["status_at_read"] == "running", "wrong gate rejection state")
    require("not a successful real build" in read_json(capture / "failure.json")["message"], "wrong gate rejection error")
    return {"capture_id": capture.name, "manifest_sha256": sha256(capture / "manifest.json"),
            "artifact_count": len(registered), "evidence_level": "SOURCE-ONLY", "status_at_read": "running", "runtime_accepted": False}


def selftest():
    # This historical negative control stays fixed when --default selects a
    # future successful patched capture; synthetic checks never become evidence.
    default = ROOT / "artifacts/jax-stack/pass-events-unpatched-002"
    original = read_json(default / "manifest.json")
    for mode in ["hash", "missing", "qualifier"]:
        fake = copy.deepcopy(original)
        if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
        elif mode == "missing": fake["artifacts"].pop()
        else: fake["qualifiers"] = []
        try: audit(default, fake)
        except ValueError: pass
        else: raise AssertionError(f"invalid manifest accepted: {mode}")
    try: analyze(read_events(default), "jit_pass_event_workload", "present")
    except ValueError: pass
    else: raise AssertionError("generic native events accepted as the new custom event")
    try: build_gate(None, "present")
    except ValueError: pass
    else: raise AssertionError("positive event mode accepted without build provenance")

    def event(name, ts, dur, tid=1, args=None):
        return {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": tid, "args": args or {}}
    fixture = [event("pass_probe_lower", 0, 8), event("pass_probe_python_trace", 1, 2),
               event("pass_probe_compile", 10, 80), *[event("pass_probe_warm", 100+i*10, 5) for i in range(3)],
               event("algsimp", 20, 30, 9), event("algsimp", 24, 12, 9),
               event(EVENT, 21, 20, 9, {"pass": "algsimp", "pipeline": "simplification", "module": "fixture", "program_id": "1", "status": "OK", "changed": "true"}),
               event(EVENT, 25, 5, 9, {"pass": "algsimp", "pipeline": "simplification", "module": "fixture", "program_id": "1", "status": "OK", "changed": "false"})]
    stats = analyze(fixture, "fixture", "present")["groups"][0]
    require(stats["inclusive_total_us"] == 25 and stats["span_union_us"] == 20, "nested interval accounting differs")
    for mode in ["metadata", "warm", "warm-overlap", "outside", "parent", "status", "filtered", "driver"]:
        bad = copy.deepcopy(fixture)
        if mode == "metadata": del bad[-1]["args"]["program_id"]
        elif mode == "warm": bad[-1]["ts"] = 101
        elif mode == "warm-overlap": bad.append(event("constant_folding", 99, 3, 9))
        elif mode == "outside": bad[-1]["ts"] = 95
        elif mode == "parent": bad[-1]["tid"] = 11
        elif mode == "status": bad[-1]["args"]["changed"] = "unknown"
        elif mode == "driver": bad[2]["tid"] = 2
        try: analyze(bad, "fixture", "present", "algsimp" if mode == "filtered" else None)
        except ValueError: pass
        else: raise AssertionError(f"bad synthetic event accepted: {mode}")
    print("selftest: raw negative control and 8 invalid synthetic event variants rejected; cross-thread containment/nested union passed; no synthetic runtime evidence claimed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, default=ROOT / "artifacts/jax-stack/pass-events-unpatched-003")
    parser.add_argument("--filtered", type=Path, default=ROOT / "artifacts/jax-stack/pass-events-unpatched-filtered-003")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    result = verify_pair(args.default.resolve(), args.filtered.resolve())
    if args.write:
        result["running_build_rejection"] = verify_rejection(ROOT / "artifacts/jax-stack/pass-events-running-build-rejected-002")
        (HERE / "pass-events-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": 2, "custom_event_counts": [result[k]["observations"]["custom_event_count"] for k in ["default", "filtered"]],
                      "patched_runtime_pair_accepted": result["patched_runtime_pair_accepted"]}))


if __name__ == "__main__":
    main()
