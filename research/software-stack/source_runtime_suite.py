#!/usr/bin/env python3
"""Run and audit the CPU revalidation suite inside the fixed build image."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import require
from pass_events_probe import build_gate, load_baseline_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaxlib-build-manifest", type=Path, required=True)
    parser.add_argument("--suite", choices=["lowering", "metadata", "all"], default="lowering")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "artifacts/jax-stack"), "output must be repository-local artifacts")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    build = args.jaxlib_build_manifest.resolve()
    build_gate(build, "absent", output / "build-gate-observation.json")
    load_baseline_module()._load_build_validator().load_and_validate_manifest(build)
    from verify_research import verify as verify_matmul
    from verify_fusion_memory import verify as verify_memory
    from verify_overlap import verify as verify_overlap
    from verify_attributes import verify as verify_attributes
    from verify_latency_metadata import verify as verify_latency
    results = {}
    lowering = [
        ("matmul", "matmul_probe.py", verify_matmul, []),
        ("fusion-memory", "fusion_memory_probe.py", verify_memory, []),
        ("overlap", "overlap_probe.py", verify_overlap, ["--observe-prior-failures"]),
    ]
    metadata = [("attributes", "attributes_cost_probe.py", verify_attributes, []),
                ("latency", "latency_metadata_probe.py", verify_latency, [])]
    selected = {"lowering": lowering, "metadata": metadata, "all": lowering + metadata}[args.suite]
    for name, script, verifier, extra in selected:
        command = [sys.executable, "-B", str(ROOT / "research/software-stack" / script),
                   "--output", str(output / name), "--jaxlib-build-manifest", str(build), *extra]
        write_json(output / f"{name}.command.json", command)
        with (output / f"{name}.process.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = {"producer_exit_code": process.returncode, "outcome": "fail"}
        if process.returncode == 0:
            try:
                verified = verifier(output / name)
                require(verified["build_binding"]["source_build_verified"], "suite capture lacks source-build binding")
                require(not verified["qualifiers"], "unpatched matching-source suite has unexpected qualifiers")
                write_json(output / f"{name}.verified.json", verified)
                result.update(outcome="pass", verified_result=f"{name}.verified.json",
                              verified_result_fingerprint=fingerprint(output / f"{name}.verified.json"),
                              numerical_cases=len(verified["cases"]) + len(verified.get("hlo_edit", {}).get("results", [])))
            except Exception as error:
                result.update(error_type=type(error).__name__, message=str(error))
                (output / f"{name}.verification-error.log").write_text(traceback.format_exc())
        results[name] = result
        write_json(output / "suite-results.json", {"build_manifest": str(build.relative_to(ROOT)),
                   "build_manifest_fingerprint": fingerprint(build), "cases": results,
                   "suite": args.suite,
                   "outcome": "pass" if len(results) == len(selected) and all(r["outcome"] == "pass" for r in results.values()) else "incomplete-or-failed"})
        print(json.dumps({"case": name, **result}), flush=True)
    return 0 if all(r["outcome"] == "pass" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
