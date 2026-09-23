#!/usr/bin/env python3
"""Verify the CPU compiler-marker build/load/rollback experiment in the fixed image."""

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import analyze, require
from pass_events_probe import read_events
from verify_pass_events import verify_pair

HERE = Path(__file__).resolve().parent
RAW = ROOT / "artifacts/jax-stack/pass-event-build-002"
PATCH = HERE / "xla-hlo-pass-events-exportable.patch"
CAPTURES = {
    "baseline": ROOT / "artifacts/jax-stack/source-runtime-002/baseline-env",
    "patched": ROOT / "artifacts/jax-stack/pass-patch-runtime-002/patched-env",
    "rollback": ROOT / "artifacts/jax-stack/pass-patch-rollback-001/rollback-env",
}
NEW_TESTS = {
    "ResearchLeafEventsDescribeChangesAndNestedPipeline",
    "ResearchFilteredLeafDoesNotEmitEvent",
    "ResearchErrorEventStopsAndPreservesStatus",
    "ResearchDisabledRecorderPreservesPassResult",
}


def audit_test_xml(xml):
    tests = xml.findall(".//testcase")
    require(len(tests) == 25 and xml.get("tests") == "25", "C++ test count differs")
    require(xml.get("errors") == xml.get("failures") == "0", "C++ test target failed")
    require(NEW_TESTS <= {t.get("name") for t in tests}, "new C++ tests missing")
    require(all(t.get("status") == "run" and t.get("result") == "completed"
                and len(t) == 0 for t in tests), "C++ test skipped or failed")
    return {"tests": len(tests), "new_tests": sorted(NEW_TESTS), "failures": 0}


def native_hashes(result):
    return {b["wheel_member"]: b["sha256"]
            for b in result["default"]["build_binding"]["matched_loaded_payloads"]}


def verify():
    xml = ET.parse(RAW / "unit-test.xml").getroot()
    cpp = audit_test_xml(xml)
    state = json.loads((RAW / "test-state.json").read_text())
    require(not state["Running"] and state["ExitCode"] == 0, "C++ container not successful")
    launch = json.loads((RAW / "test-launch.json").read_text())
    require(launch["patch_sha256"] == fingerprint(PATCH)["sha256"], "C++ tested another patch")
    patch_record = json.loads((RAW / "applied-patch.json").read_text())
    for name, hashes in patch_record["files"].items():
        require(fingerprint(RAW / "base" / name)["sha256"] == hashes["base_sha256"], "C++ base changed")
        require(fingerprint(RAW / "candidate" / name)["sha256"] == hashes["candidate_sha256"], "C++ candidate changed")
    pairs = {name: verify_pair(path / "default", path / "filtered")
             for name, path in CAPTURES.items()}
    require(pairs["patched"]["patched_runtime_pair_accepted"], "patched runtime not accepted")
    for name, pair in pairs.items():
        for mode in ("default", "filtered"):
            binding = pair[mode]["build_binding"]
            require(binding["source_build_verified"], "unbound runtime in round trip")
            if name == "patched":
                require(len(binding["source_patches"]) == 1 and
                        binding["source_patches"][0]["sha256"] == fingerprint(PATCH)["sha256"],
                        "runtime patch differs from C++ tested patch")
                require(pair[mode]["qualifiers"] == ["SOURCE-PATCHED"], "patched runtime qualifiers differ")
            else:
                require(not binding["source_patches"] and not pair[mode]["qualifiers"], "baseline or rollback is not pristine")
                require(pair[mode]["observations"]["custom_event_count"] == 0, "baseline or rollback has custom events")
    require(pairs["baseline"]["default"]["build_binding"]["build_id"] ==
            pairs["rollback"]["default"]["build_binding"]["build_id"], "rollback selected another baseline build")
    require(native_hashes(pairs["baseline"]) == native_hashes(pairs["rollback"]), "rollback native bytes differ")
    require(native_hashes(pairs["baseline"])["jaxlib/libjax_common.so"] !=
            native_hashes(pairs["patched"])["jaxlib/libjax_common.so"], "patched native payload unchanged")
    for item in ("inputs", "outputs"):
        with np.load(CAPTURES["baseline"] / "default" / f"{item}.npz", allow_pickle=False) as expected:
            for name in ("patched", "rollback"):
                with np.load(CAPTURES[name] / "default" / f"{item}.npz", allow_pickle=False) as actual:
                    require(set(expected.files) == set(actual.files), "cross-state array keys differ")
                    for key in expected.files:
                        if item == "inputs":
                            np.testing.assert_array_equal(actual[key], expected[key])
                        else:
                            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-5)
    rollback = json.loads((RAW / "rollback.json").read_text())
    require(rollback["patch_sha256"] == fingerprint(PATCH)["sha256"] and
            rollback["source_restored"] and rollback["git_status_after"] == "", "source rollback not verified")
    require(all(c["returncode"] == 0 for c in rollback["commands"]), "source rollback command failed")
    for name, hashes in patch_record["files"].items():
        expected = hashes["base_sha256"]
        require(rollback["restored_files"][name]["sha256"] == expected and
                fingerprint(RAW / "restored" / name)["sha256"] == expected, "restored source bytes differ")
    compact = {}
    for name, pair in pairs.items():
        compact[name] = {mode: {
            "capture": str((CAPTURES[name] / mode).relative_to(ROOT)),
            "manifest": fingerprint(CAPTURES[name] / mode / "manifest.json"),
            "qualifiers": pair[mode]["qualifiers"],
            "build_id": pair[mode]["build_binding"]["build_id"],
            "wheel": pair[mode]["build_binding"]["wheel"],
            "observations": pair[mode]["observations"],
            "max_absolute_errors": pair[mode]["max_absolute_errors"],
        } for mode in ("default", "filtered")}
    artifacts = [RAW / n for n in ("unit-test.xml", "test-state.json", "test-launch.json",
                                  "rollback.json", "program-id-loss.json", "xspace-decode.json")]
    return {"evidence_level": "RUN-CPU", "scope": "Host compiler-pass diagnostic Hack",
            "build_load_rollback_accepted": True, "cpp": cpp,
            "patch": {"path": str(PATCH.relative_to(ROOT)), **fingerprint(PATCH)},
            "states": compact, "source_restored": True,
            "artifacts": {str(p.relative_to(ROOT)): fingerprint(p) for p in artifacts},
            "limits": ["CPU compiler spans only; no TPU compilation, device trace, LLO or performance conclusion.",
                       "Rollback restores source and loads the retained verified baseline wheel in fresh processes; it does not rebuild a third wheel.",
                       "Test-only Googletest patches are recorded separately; production builds do not use that override."]}


def selftest():
    xml = ET.parse(RAW / "unit-test.xml").getroot()
    for mode in ("failed", "missing", "skipped"):
        bad = copy.deepcopy(xml)
        if mode == "failed":
            bad.set("failures", "1")
        else:
            case = next(t for t in bad.findall(".//testcase") if t.get("name") in NEW_TESTS)
            case.set("name" if mode == "missing" else "result", "missing" if mode == "missing" else "suppressed")
        try:
            audit_test_xml(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad C++ result accepted: {mode}")
    failed = ROOT / "artifacts/jax-stack/pass-patch-runtime-001/patched-env/default"
    events = read_events(failed)
    require(sum(e.get("name") == "research_hlo_pass_run" for e in events) == 127, "regression trace differs")
    try:
        analyze(events, "jit_pass_event_workload", "present")
    except ValueError as error:
        require("metadata missing" in str(error), "wrong regression rejection")
    else:
        raise AssertionError("export with missing program identity accepted")
    print("selftest: three invalid C++ outcomes and real 127-event metadata-loss capture rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    result = verify()
    if args.write:
        write_json(HERE / "pass-hack-results.json", result)
    print(json.dumps({"build_load_rollback_accepted": result["build_load_rollback_accepted"],
                      "cpp_tests": result["cpp"]["tests"],
                      "custom_counts": {n: [s[m]["observations"]["custom_event_count"]
                                            for m in ("default", "filtered")]
                                        for n, s in result["states"].items()}}))
