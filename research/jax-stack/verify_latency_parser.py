#!/usr/bin/env python3
"""Audit real C++ latency parser tests, observed boundary values and source rollback."""

import argparse
import copy
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import require

HERE = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/jax-stack/latency-parser-native-001"
BOUNDARY = ROOT / "artifacts/jax-stack/latency-parser-native-002"
PATCH = HERE / "xla-latency-parser-boundaries.patch"
EXPECTED = {
    "Missing": (None, 1, False), "Positive": ("30000", 1, True),
    "Zero": ("0", 1, True), "Negative": ("-1000", 1, True),
    "IntegerMaximum": ("9223372036854775807", 1, True),
    "IntegerMinimum": ("-9223372036854775808", 1, True),
    "PositiveOverflow": ("9223372036854775808", 1, False),
    "NegativeOverflow": ("-9223372036854775809", 1, False),
    "Fractional": ("30000.5", 1, False), "Empty": ("", 1, False),
    "Invalid": ("slow", 1, False), "Scientific": ("3e4", 1, False),
    "ClockScale": ("30000", 2, True), "SubMicrosecond": ("1", 1, True),
}


def inventory(capture, manifest=None):
    manifest = json.loads((capture / "manifest.json").read_text()) if manifest is None else manifest
    require(manifest["outcome"] == "pass" and manifest["evidence_level"] == "RUN-CPU"
            and manifest["execution_kind"] == "native-cpp-unit-test", "wrong C++ evidence kind")
    require(not manifest["production_source_changed"] and not manifest["gpu_execution"]
            and not manifest["tpu_execution"] and not manifest["target_hardware_clock_measured"], "C++ test scope promoted")
    expected_qualifiers = ["TEST-DEPENDENCY-PATCHED"] + (["TEST-SOURCE-PATCHED"] if capture == BOUNDARY else [])
    require(manifest["qualifiers"] == expected_qualifiers, "native test qualifiers differ")
    registered = set()
    for item in manifest["artifacts"]:
        path = (ROOT / item["path"]).resolve()
        require(path.is_relative_to(capture) and path not in registered, "invalid C++ artifact path")
        require(fingerprint(path) == {k: item[k] for k in ("sha256", "size_bytes")}, "C++ artifact fingerprint differs")
        registered.add(path)
    require(registered == {p for p in capture.rglob("*") if p.is_file() and p.name != "manifest.json"}, "C++ artifact inventory incomplete")
    producer = manifest["producer"]
    require(fingerprint(ROOT / producer["source"]) == {k: producer[k] for k in ("sha256", "size_bytes")}, "test source identity differs")
    return manifest


def audit_xml(xml, expected_count):
    cases = xml.findall(".//testcase")
    require(xml.get("tests") == str(expected_count) and len(cases) == expected_count, "C++ test count differs")
    require(xml.get("failures") == xml.get("errors") == "0", "C++ test suite failed")
    require(all(c.get("status") == "run" and c.get("result") == "completed"
                and c.find("failure") is None and c.find("skipped") is None for c in cases), "C++ test not completed")
    upstream = [c for c in cases if c.get("classname") == "LatencyEstimatorTest"
                and c.get("name") == "GetLatencyFromMetadata"]
    require(len(upstream) == 1, "pinned upstream parser test missing")
    observed = {}
    for case in cases:
        properties = {p.get("name"): p.get("value") for p in case.findall("./properties/property")}
        if not properties:
            continue
        label = properties["input_label"]
        require(label in EXPECTED and label not in observed, "unexpected or duplicate boundary case")
        text, rate, has_value = EXPECTED[label]
        require(properties["metadata_present"] == ("true" if text is not None else "false")
                and properties["metadata_text"] == (text if text is not None else "<missing>"), "metadata input differs")
        require(int(properties["cycles_per_microsecond"]) == rate, "test clock configuration differs")
        require(properties["has_value"] == ("true" if has_value else "false"), "parser acceptance differs")
        actual = None
        if has_value:
            actual = float(properties["parsed_cycles"])
            expected = float(Decimal(text) * Decimal(rate) / Decimal(1000))
            require(math.isfinite(actual) and actual == expected, "parsed value or ns-to-cycle conversion differs")
        else:
            require(properties["parsed_cycles"] == "nullopt", "rejected metadata has a value")
        observed[label] = {"input": text, "cycles_per_microsecond": rate,
                           "has_value": has_value, "observed_cycles": actual}
    require(set(observed) == (set(EXPECTED) if expected_count == 15 else set()), "boundary coverage differs")
    return observed


def verify():
    manifests = [inventory(BASE), inventory(BOUNDARY)]
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    for path, manifest, count in zip((BASE, BOUNDARY), manifests, (1, 15)):
        require(manifest["source_revision"] == baseline["repository"]["sources"]["xla"]["git_commit"], "native test source pin differs")
        state = json.loads((path / "state.json").read_text())
        require(not state["Running"] and state["ExitCode"] == 0, "native test container not successful")
        audit_xml(ET.parse(path / "test.xml").getroot(), count)
        binary = json.loads((path / "binary.json").read_text())
        require(fingerprint(path / "test-binary") == {k: binary[k] for k in ("sha256", "size_bytes")}, "test executable differs")
    prep = json.loads((BOUNDARY / "applied-patch.json").read_text())
    launch = json.loads((BOUNDARY / "launch.json").read_text())
    rollback = json.loads((BOUNDARY / "rollback.json").read_text())
    require(fingerprint(PATCH)["sha256"] == prep["patch_sha256"] == launch["test_patch_sha256"] == rollback["patch_sha256"], "native test patch identity differs")
    relative = prep["changed_path"]
    require(relative == "xla/service/latency_hiding_scheduler_test.cc", "production source included in patch")
    require(rollback["source_restored"] and rollback["clone_status"] == ""
            and all(c["returncode"] == 0 for c in rollback["commands"]), "test source rollback failed")
    require(fingerprint(BOUNDARY / "restored-test.cc")["sha256"] == prep["base_sha256"], "restored test bytes differ")
    source = json.loads((BASE / "source-snapshot.json").read_text())
    for name, record in source["files"].items():
        require(fingerprint(BASE / "sources" / name) == record and
                fingerprint(ROOT / "upstream/xla" / name) == record, "pinned source snapshot changed")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        target = directory / relative
        target.parent.mkdir(parents=True)
        target.write_bytes((BOUNDARY / "base" / relative).read_bytes())
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(directory.parent))
        for options in ([], ["--reverse"]):
            subprocess.run(["git", "apply", *options, "--check", str(PATCH)], cwd=directory, env=env, check=True, capture_output=True)
            subprocess.run(["git", "apply", *options, str(PATCH)], cwd=directory, env=env, check=True, capture_output=True)
            expected = prep["base_sha256"] if options else prep["candidate_sha256"]
            require(fingerprint(target)["sha256"] == expected, "test patch replay differs")
    observed = audit_xml(ET.parse(BOUNDARY / "test.xml").getroot(), 15)
    return {"evidence_level": "RUN-CPU", "execution_kind": "native-cpp-unit-test",
            "qualifiers": manifests[1]["qualifiers"], "baseline_qualifiers": manifests[0]["qualifiers"],
            "source_revision": prep["revision"], "production_source_changed": False,
            "baseline_tests_passed": 1, "boundary_run_tests_passed": 15,
            "new_boundary_cases": 14, "observations": dict(sorted(observed.items())),
            "patch": {"path": str(PATCH.relative_to(ROOT)), **fingerprint(PATCH)},
            "source_restored": True, "patch_replay_verified": True,
            "captures": [{"path": str(p.relative_to(ROOT)), "manifest": fingerprint(p / "manifest.json")}
                         for p in (BASE, BOUNDARY)],
            "source_contract_snapshot": {"path": "research/jax-stack/latency-model-contract.json",
                                         **fingerprint(HERE / "latency-model-contract.json")},
            "limits": ["Configured test clocks are 1 or 2 cycles/us; no device frequency or latency is measured.",
                       "The generic parser is executed; GPU NodeCost consumers, real scheduler behavior and TPU implementation are not.",
                       "The original SOURCE-ONLY contract is a retained earlier snapshot; these native runs are recorded separately."]}


def selftest():
    original = ET.parse(BOUNDARY / "test.xml").getroot()
    for mode in ("failure", "missing", "negative", "overflow", "clock"):
        xml = copy.deepcopy(original)
        if mode == "failure":
            xml.set("failures", "1")
        elif mode == "missing":
            xml.set("tests", "14")
        else:
            label = {"negative": "Negative", "overflow": "PositiveOverflow", "clock": "ClockScale"}[mode]
            case = next(c for c in xml.findall(".//testcase") if any(p.get("name") == "input_label" and p.get("value") == label for p in c.findall("./properties/property")))
            for prop in case.findall("./properties/property"):
                if mode == "overflow" and prop.get("name") == "has_value": prop.set("value", "true")
                if mode in ("negative", "clock") and prop.get("name") == "parsed_cycles": prop.set("value", "0")
        try: audit_xml(xml, 15)
        except ValueError: pass
        else: raise AssertionError(f"bad native test result accepted: {mode}")
    fake = json.loads((BOUNDARY / "manifest.json").read_text())
    fake["artifacts"][0]["sha256"] = "0" * 64
    try: inventory(BOUNDARY, fake)
    except ValueError: pass
    else: raise AssertionError("corrupted native artifact accepted")
    print("selftest: five invalid native outcomes and corrupted artifact rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    result = verify()
    if args.write:
        write_json(HERE / "latency-parser-results.json", result)
    print(json.dumps({k: result[k] for k in ("baseline_tests_passed", "boundary_run_tests_passed", "new_boundary_cases", "source_restored")}))
