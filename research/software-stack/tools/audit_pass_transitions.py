#!/usr/bin/env python3
"""Read source-bound matmul dumps as scoped pass boundaries, without executing HLO."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import shutil
import sys

from matmul_probe import ROOT, fingerprint, write_json
from verify_research import check, verify as verify_cpu
from capture_runtime import verify_current_reader


def record(path):
    return {"path": str(path.relative_to(ROOT)), **fingerprint(path)}


def native_snapshot(path):
    from jaxlib import _hlo
    text = path.read_text()
    module = _hlo.hlo_module_from_text(text)
    entry = re.search(r"^ENTRY %?([^\s(]+)", text, re.M)[1]
    nodes = {}
    for computation in module.computations():
        for op in computation.instructions():
            nodes[computation.name + "/" + op.name] = {
                "computation": computation.name, "name": op.name,
                "opcode": op.opcode.name, "operands": [p.name for p in op.operands()],
                "text": op.to_string(),
            }
    entry_nodes = {k: v for k, v in nodes.items() if v["computation"] == entry}
    return {"entry": entry, "nodes": nodes,
            "opcode_counts": dict(sorted(Counter(n["opcode"] for n in nodes.values()).items())),
            "entry_opcode_counts": dict(sorted(Counter(n["opcode"] for n in entry_nodes.values()).items()))}


def entry_ops(snapshot, opcode):
    return [n for n in snapshot["nodes"].values()
            if n["computation"] == snapshot["entry"] and n["opcode"] == opcode]


def boundary_records(source, prefix):
    rows, other = [], []
    pattern = re.compile(re.escape(prefix) + r"\.(\d+)\.(.+)\.after_(.+)\.before_(.+)\.txt")
    for path in sorted((source / "xla-dump").glob(prefix + ".*.txt")):
        if not re.match(re.escape(prefix) + r"\.\d+\.", path.name):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            other.append(record(path))
            continue
        sequence, pipeline, after, before = match.groups()
        rows.append({"sequence": int(sequence), "pipeline": pipeline,
                     "after_pass": after, "before_pass": before, "input": record(path)})
    rows.sort(key=lambda r: r["sequence"])
    check(len({r["sequence"] for r in rows}) == len(rows), "duplicate boundary sequence")
    return rows, other


def pair_boundaries(rows):
    previous, pairs = {}, []
    for row in rows:
        pipeline = row["pipeline"]
        old = previous.get(pipeline)
        if row["after_pass"] != "pipeline-start":
            check(old is not None and old["before_pass"] == row["after_pass"],
                  "non-adjacent scoped pass boundary")
            nested = any(old["sequence"] < child["sequence"] < row["sequence"] and
                         child["pipeline"] == row["after_pass"] and
                         child["after_pass"] == "pipeline-start" for child in rows)
            pairs.append({"sequence": row["sequence"], "pipeline": pipeline,
                          "pass": row["after_pass"],
                          "scope": "nested-pipeline-aggregate" if nested else "leaf-boundary",
                          "before": old["input"], "after": row["input"],
                          "text_changed": old["input"]["sha256"] != row["input"]["sha256"]})
        previous[pipeline] = row
    return pairs


def changes(before, after):
    a, b = before["nodes"], after["nodes"]
    return {"added": sorted(b.keys() - a.keys()), "removed": sorted(a.keys() - b.keys()),
            "modified": sorted(k for k in a.keys() & b.keys() if a[k] != b[k])}


def inspect_case(source, case):
    rows, other = boundary_records(source, case["native_module_prefix"])
    snapshots = {r["input"]["path"]: native_snapshot(ROOT / r["input"]["path"]) for r in rows}
    pairs = pair_boundaries(rows)
    for pair in pairs:
        before, after = [snapshots[pair[k]["path"]] for k in ("before", "after")]
        pair["node_changes"] = changes(before, after)
        pair["before_entry_opcodes"] = before["entry_opcode_counts"]
        pair["after_entry_opcodes"] = after["entry_opcode_counts"]
    final_path = source / case["name"] / "optimized-hlo.txt"
    final = native_snapshot(final_path)
    fusions = entry_ops(final, "kFusion")
    final_counts = {"ynn_custom_fusions": sum('__ynn_fusion' in n["text"] for n in fusions),
                    "loop_fusions": sum('kind=kLoop' in n["text"] for n in fusions),
                    "remaining_entry_dots": len(entry_ops(final, "kDot"))}
    return {"case": case["name"], "native_module_prefix": case["native_module_prefix"],
            "boundaries": rows, "other_numbered_dumps": other, "pairs": pairs,
            "final_hlo": record(final_path), "final_entry": final_counts}, snapshots


def semantic_checks(cases, snapshots):
    by_name = {c["case"]: c for c in cases}
    expected = {"matmul": (1, 0, 0), "vmap_matmul": (1, 0, 0),
                "grad_matmul": (2, 1, 1), "jit_grad_vmap_matmul": (2, 2, 1)}
    for name, counts in expected.items():
        actual = by_name[name]["final_entry"]
        check(tuple(actual[k] for k in ("ynn_custom_fusions", "loop_fusions", "remaining_entry_dots")) == counts,
              "final CPU dispatch structure changed")

    def select(name, pass_name, occurrence=0):
        pairs = [p for p in by_name[name]["pairs"] if p["pass"] == pass_name and p["text_changed"]
                 and p["scope"] == "leaf-boundary"]
        pair = pairs[occurrence]
        return pair, *(snapshots[name][pair[k]["path"]] for k in ("before", "after"))

    _, before, after = select("vmap_matmul", "dot_decomposer")
    check(any("f32[3,4,6]" in n["text"] for n in entry_ops(before, "kDot")) and
          any("f32[12,6]" in n["text"] for n in entry_ops(after, "kDot")), "vmap dot flattening changed")
    check(any("f32[12,8]" in n["text"] for n in entry_ops(after, "kReshape")), "flattened lhs absent")
    pair, before, after = select("vmap_matmul", "dynamic-dimension-simplifier")
    rewired = [k for k in pair["node_changes"]["modified"]
               if before["nodes"][k]["operands"] != after["nodes"][k]["operands"]]
    check(rewired and before["opcode_counts"] == after["opcode_counts"],
          "identity reshape use forwarding changed")
    _, before, after = select("vmap_matmul", "reshape-decomposer")
    check(len(entry_ops(before, "kReshape")) == 2 and not entry_ops(after, "kReshape") and
          len(entry_ops(after, "kBitcast")) == 2, "layout-aware reshape decomposition changed")
    _, before, after = select("jit_grad_vmap_matmul", "transpose-folding")
    check(len(entry_ops(before, "kTranspose")) == len(entry_ops(after, "kTranspose")) + 1 and
          any("rhs_contracting_dims={1}" in n["text"] for n in entry_ops(after, "kDot")),
          "transpose folding into dot dimensions changed")
    _, before, after = select("jit_grad_vmap_matmul", "layout-assignment")
    check(len(entry_ops(after, "kCopy")) == len(entry_ops(before, "kCopy")) + 1 and
          any("f32[6,3,4]{0,2,1}" in n["text"] for n in entry_ops(after, "kTranspose")),
          "layout copy insertion changed")
    for name in expected:
        _, before, after = select(name, "dot-library-rewriter")
        check(len(entry_ops(after, "kDot")) < len(entry_ops(before, "kDot")) and
              any('__ynn_fusion' in n["text"] for n in entry_ops(after, "kFusion")),
              "dot library rewrite absent")
    return ["four final entry dispatch structures", "vmap 3x4 to 12 dot flattening",
            "identity reshape operand forwarding without opcode-count change",
            "two layout-compatible reshapes become bitcasts", "rhs transpose folded into contracting dimension",
            "layout assignment inserts copy", "YNN custom fusion introduced at dot-library-rewriter"]


def collect(output, source):
    checked = verify_cpu(source)
    check(checked.get("build_binding") is not None and not checked["qualifiers"], "source build required")
    reader = verify_current_reader(checked["build_binding"])
    write_json(output / "reader-identity.json", reader)
    cases, snapshots, totals = [], {}, []
    for case in checked["cases"]:
        result, native = inspect_case(source, case)
        cases.append(result)
        snapshots[case["name"]] = native
        directory = output / case["name"]
        directory.mkdir()
        write_json(directory / "timeline.json", result)
        changed = [p for p in result["pairs"] if p["text_changed"]]
        for pair in changed:
            stem = f"{pair['sequence']:04d}-{pair['pass']}"
            a, b = [ROOT / pair[k]["path"] for k in ("before", "after")]
            delta = difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True),
                                        fromfile=pair["before"]["path"], tofile=pair["after"]["path"])
            (directory / (stem + ".diff")).write_text("".join(delta))
            write_json(directory / (stem + ".native.json"),
                       {k: native[pair[k]["path"]] for k in ("before", "after")})
        totals.append({"case": case["name"], "native_module_prefix": case["native_module_prefix"],
                       "numbered_pipeline_boundaries": len(result["boundaries"]),
                       "other_numbered_dumps": len(result["other_numbered_dumps"]),
                       "paired_boundaries": len(result["pairs"]),
                       "changed_leaf_boundaries": [p["sequence"] for p in changed if p["scope"] == "leaf-boundary"],
                       "changed_nested_aggregates": [p["sequence"] for p in changed if p["scope"] != "leaf-boundary"],
                       "final_entry": result["final_entry"]})
    summary = {"outcome": "pass", "evidence_level": "REPLAY-OFFLINE", "qualifiers": [],
               "source_capture": str(source.relative_to(ROOT)), "source_manifest_sha256": checked["manifest_sha256"],
               "build_binding": checked["build_binding"], "reader_identity": reader,
               "cases": totals, "semantic_checks": semantic_checks(cases, snapshots),
               "limits": ["Reads existing RUN-CPU dumps; no new compiler or executable run.",
                          "Paired scoped dumps show observed textual/structural changes, not pass return flags or timing.",
                          "Nested pipeline aggregates are reported separately from leaf boundaries.",
                          "Internal numbered dumps without before/after pairs are inventoried but not treated as pipeline passes.",
                          "No TPU, LLO, physical memory, performance, or runtime dispatch sampling evidence."]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, default=ROOT / "artifacts/jax-stack/source-runtime-002/suite/matmul")
    args = parser.parse_args()
    output = args.output.resolve()
    check(output.is_relative_to(ROOT / "artifacts/jax-stack"), "use a fresh artifacts/jax-stack directory")
    output.mkdir(parents=True, exist_ok=False)
    for name in [Path(__file__).name, "verify_research.py", "capture_runtime.py"]:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    started = datetime.now(timezone.utc).isoformat()
    summary = collect(output, args.source_capture.resolve())
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass",
        "evidence_level": "REPLAY-OFFLINE", "qualifiers": [], "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv],
                     "source": str((output / Path(__file__).name).relative_to(ROOT))},
        "artifacts": [record(p) for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps(summary["cases"]))


if __name__ == "__main__":
    main()
