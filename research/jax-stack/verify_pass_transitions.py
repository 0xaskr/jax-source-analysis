#!/usr/bin/env python3
"""Reparse scoped HLO transitions and reject incomplete or overclaimed results."""

import argparse
import copy
import difflib
import json
from pathlib import Path

from audit_pass_transitions import inspect_case, pair_boundaries, semantic_checks
from capture_runtime import verify_current_reader
from verify_codegen_and_patch import inventory
from verify_research import ROOT, check, local_path, read_json, verify as verify_cpu


def verify(capture, selftest=False):
    result = inventory(capture, "REPLAY-OFFLINE")
    summary = read_json(capture / "summary.json")
    source = local_path(summary["source_capture"])
    original = verify_cpu(source)
    check(original.get("build_binding") is not None, "requires source-bound CPU capture")
    reader = verify_current_reader(original["build_binding"])
    check(summary["reader_identity"] == read_json(capture / "reader-identity.json") == reader,
          "producer and verifier native readers differ")
    check(summary["build_binding"] == original["build_binding"], "build binding differs")
    check(summary["source_manifest_sha256"] == original["manifest_sha256"], "source capture changed")
    check(summary["evidence_level"] == "REPLAY-OFFLINE" and summary["outcome"] == "pass" and
          summary["qualifiers"] == original["qualifiers"] == read_json(capture / "manifest.json")["qualifiers"] == [],
          "evidence level or qualifier changed")
    expected, native = [], {}
    for case in original["cases"]:
        actual, snapshots = inspect_case(source, case)
        directory = capture / case["name"]
        check(read_json(directory / "timeline.json") == actual, "timeline differs from reparsed source dumps")
        expected.append(actual)
        native[case["name"]] = snapshots
        changed = [p for p in actual["pairs"] if p["text_changed"]]
        total = {"case": case["name"], "native_module_prefix": case["native_module_prefix"],
                 "numbered_pipeline_boundaries": len(actual["boundaries"]),
                 "other_numbered_dumps": len(actual["other_numbered_dumps"]),
                 "paired_boundaries": len(actual["pairs"]),
                 "changed_leaf_boundaries": [p["sequence"] for p in changed if p["scope"] == "leaf-boundary"],
                 "changed_nested_aggregates": [p["sequence"] for p in changed if p["scope"] != "leaf-boundary"],
                 "final_entry": actual["final_entry"]}
        check(summary["cases"][len(expected) - 1] == total, "summary omits or reclassifies a boundary")
        for pair in changed:
            stem = f"{pair['sequence']:04d}-{pair['pass']}"
            a, b = [local_path(pair[k]["path"]) for k in ("before", "after")]
            delta = "".join(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                          fromfile=pair["before"]["path"], tofile=pair["after"]["path"]))
            check((directory / (stem + ".diff")).read_text() == delta, "pass diff changed")
            check(read_json(directory / (stem + ".native.json")) ==
                  {k: snapshots[pair[k]["path"]] for k in ("before", "after")}, "native nodes differ")
    check(len(summary["cases"]) == len(expected), "extra case summary")
    check(summary["semantic_checks"] == semantic_checks(expected, native), "semantic coverage differs")
    tests = []
    if selftest:
        def rejects(name, call):
            try:
                call()
            except ValueError:
                tests.append(name)
            else:
                raise AssertionError("invalid observation accepted: " + name)

        for mode in ["hash", "missing", "level"]:
            fake = copy.deepcopy(read_json(capture / "manifest.json"))
            if mode == "hash":
                fake["artifacts"][0]["sha256"] = "0" * 64
            elif mode == "missing":
                fake["artifacts"].pop()
            else:
                fake["evidence_level"] = "RUN-TPU"
            rejects(mode, lambda: inventory(capture, "REPLAY-OFFLINE", fake))
        rows = copy.deepcopy(expected[0]["boundaries"])
        rows[1]["after_pass"] = "not-the-next-pass"
        rejects("nonadjacent-pass", lambda: pair_boundaries(rows))
        bad = copy.deepcopy(expected)
        bad[2]["final_entry"]["remaining_entry_dots"] = 0
        rejects("all-dots-assumed-library-fused", lambda: semantic_checks(bad, native))
        bad_native = copy.deepcopy(native)
        row = next(p for p in expected[1]["pairs"] if p["pass"] == "dynamic-dimension-simplifier" and p["text_changed"])
        bad_native["vmap_matmul"][row["after"]["path"]] = bad_native["vmap_matmul"][row["before"]["path"]]
        rejects("operand-rewire-omitted", lambda: semantic_checks(expected, bad_native))
        aggregates = [p for c in expected for p in c["pairs"]
                      if p["text_changed"] and p["scope"] == "nested-pipeline-aggregate"]
        check(len(aggregates) == 3 and all(p["pass"] == "simplification" for p in aggregates),
              "nested simplification counted as a leaf pass")
    result.update(qualifiers=[], source_capture=summary["source_capture"],
                  source_manifest_sha256=summary["source_manifest_sha256"],
                  build_binding=original["build_binding"], verification_runtime=reader,
                  cases=summary["cases"], semantic_checks=summary["semantic_checks"],
                  negative_tests=tests, limits=summary["limits"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/source-lowering-audit-001/passes")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    result = verify(args.capture.resolve(), args.selftest)
    if args.result:
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "artifacts": result["artifact_count"],
                      "negative_tests": result["negative_tests"]}))


if __name__ == "__main__":
    main()
