#!/usr/bin/env python3
"""Verify source anchors, capture bytes and independent matmul references."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path leaves the repository: {relative}")
    return path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(capture):
    index = read_json(HERE / "source-index.json")
    check(len({e["id"] for e in index["entries"]}) == len(index["entries"]),
          "duplicate source ids")
    for name, source in index["source_roots"].items():
        path = local_path(source["path"])
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        check(head == source["revision"], f"source revision changed: {name}")
        state = subprocess.check_output(["git", "--no-optional-locks", "-C", str(path),
            "status", "--porcelain=v1", "--untracked-files=all"], text=True)
        check(not state, f"dirty source: {name}")
    for entry in index["entries"]:
        source = local_path(entry["path"])
        check(sha256(source) == entry["source_sha256"], f"source hash mismatch: {entry['id']}")
        check(source.read_text().splitlines()[entry["line"] - 1].startswith(entry["anchor"]),
              f"source anchor mismatch: {entry['id']}")
    ids = {e["id"] for e in index["entries"]}
    for edge in index["edges"]:
        check(edge["caller"] in ids and edge["callee"] in ids, "unknown call edge")
        check(local_path(edge["path"]).read_text().splitlines()[edge["line"] - 1].startswith(edge["anchor"]),
              "call edge source changed")
    origin = read_json(HERE / "kickoff-source.json")
    check(sha256(local_path(origin["body_artifact"])) == origin["body_sha256"],
          "kickoff snapshot changed")

    manifest_path = capture / "manifest.json"
    manifest = read_json(manifest_path)
    check(manifest["outcome"] == "pass", "capture did not pass")
    check(manifest["evidence_level"] == "RUN-CPU", "unexpected evidence level")
    expected_cases = {"matmul", "vmap_matmul", "grad_matmul", "jit_grad_vmap_matmul"}
    check(set(manifest["cases"]) == expected_cases, "missing or unexpected matmul cases")
    registered = set()
    for artifact in manifest["artifacts"]:
        path = local_path(artifact["path"])
        check(path.is_relative_to(capture), "artifact outside capture")
        check(path not in registered, "duplicate capture artifact")
        registered.add(path)
        check(path.is_file(), f"missing capture artifact: {path.name}")
        check(path.stat().st_size == artifact["size_bytes"], f"artifact size mismatch: {path.name}")
        check(sha256(path) == artifact["sha256"], f"artifact hash mismatch: {path.name}")
    actual_files = {path for path in capture.rglob("*") if path.is_file() and path != manifest_path}
    check(actual_files == registered, "capture inventory is incomplete")
    environment = read_json(capture / "environment.json")
    from capture_runtime import verify_binding
    build_binding = verify_binding(capture, environment, manifest)
    source_commit = environment["repository"]["sources"]["jax"]["git_commit"]
    binary_commit = environment["runtime"]["jaxlib"]["build_revision"]
    if source_commit != binary_commit:
        check("VERSION-SKEW" in manifest["qualifiers"], "missing VERSION-SKEW")
    check(environment["runtime"]["device"]["default_platform"] == "cpu", "not a CPU capture")
    # This is a live identity check, separate from the immutable capture audit.
    for binary in environment["runtime"]["jaxlib"]["native_binaries"]:
        check(sha256(local_path(binary["artifact_path"])) == binary["sha256"],
              f"current native binary differs from capture: {binary['package_path']}")

    with np.load(capture / "inputs.npz", allow_pickle=False) as inputs:
        a, w, x = [inputs[key].astype(np.float64) for key in ("a", "w", "x")]
    y, z = a @ w, x @ w
    references = {"matmul": (y,), "vmap_matmul": (z,),
        "grad_matmul": (2 * y @ w.T, 2 * a.T @ y),
        "jit_grad_vmap_matmul": (2 * z @ w.T, 2 * np.einsum("bmk,bmn->kn", x, z))}
    cases = []
    native_after = list((capture / "xla-dump").glob("*.cpu_after_optimizations.txt"))
    for name, expected in references.items():
        summary = read_json(capture / name / "summary.json")
        errors = []
        with np.load(capture / name / "outputs.npz", allow_pickle=False) as outputs:
            check(set(outputs.files) == {f"output_{i}" for i in range(len(expected))},
                  f"wrong output count: {name}")
            for i, reference in enumerate(expected):
                observed = outputs[f"output_{i}"]
                np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=2e-5)
                errors.append(float(np.max(np.abs(observed - reference))))
        check(summary["numerics_passed"], f"failed numerical comparison: {name}")
        check(summary["serialized_executable_reload"].startswith("pass;"),
              f"missing executable reload result: {name}")
        check((capture / name / "executable.bin").stat().st_size > 0, "empty serialized executable")
        header = (capture / name / "optimized-hlo.txt").read_text().splitlines()[0]
        matched = [p for p in native_after if p.read_text().splitlines()[0] == header]
        check(len(matched) == 1, f"cannot uniquely bind case to native dump: {name}")
        native_prefix = matched[0].name.removesuffix(".cpu_after_optimizations.txt")
        boundaries = sorted(p.name for p in (capture / "xla-dump").glob(f"{native_prefix}.*.txt")
                            if re.match(re.escape(native_prefix) + r"\.\d{4}\.", p.name))
        check(boundaries, f"no native pass boundaries: {name}")
        cases.append({**summary, "rechecked_max_absolute_errors": errors,
                      "native_module_prefix": native_prefix,
                      "native_pass_boundary_count": len(boundaries),
                      "native_pass_boundaries": boundaries})
    tracing = read_json(capture / "tracing.json")
    check([o["trace_count"] for o in tracing["observations"]] == [1, 1, 2], "wrong trace counts")
    log = (capture / "run.log").read_text()
    compile_events = [line for line in log.splitlines()
                      if line.startswith("Finished XLA compilation of jit(cache_matmul)")]
    check(len(compile_events) == 2, "expected two cache_matmul compilation completion events")

    objects = []
    for path in sorted((capture / "xla-dump").glob("*.o")):
        header = subprocess.check_output(["/usr/bin/readelf", "-hW", str(path)], text=True)
        properties = {}
        for line in header.splitlines():
            if ":" in line:
                key, value = line.strip().split(":", 1)
                if key in ("Class", "Type", "Machine"):
                    properties[key] = value.strip()
        check(properties.get("Type", "").startswith("REL"), "not a relocatable object")
        objects.append({"path": str(path.relative_to(ROOT)), "elf": properties})
    llvm_ir = list((capture / "xla-dump").glob("*.ll"))
    mlir_logs = list((capture / "xla-dump").glob("*.mlir-passes.log"))
    check(objects and llvm_ir and mlir_logs, "missing code-generation observations")
    return {
        "capture_id": manifest["capture_id"], "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path), "evidence_level": "RUN-CPU",
        "qualifiers": manifest["qualifiers"], "source_entries_checked": len(index["entries"]),
        "build_binding": build_binding,
        "source_edges_checked": len(index["edges"]), "artifact_count": len(registered),
        "artifact_bytes": sum(a["size_bytes"] for a in manifest["artifacts"]),
        "artifact_suffix_counts": dict(sorted(Counter(p.suffix for p in registered).items())),
        "cases": cases, "trace_counts": [1, 1, 2], "cache_compilation_completions": len(compile_events),
        "objects": objects, "readelf_sha256": sha256(Path("/usr/bin/readelf")),
        "limitations": manifest["limitations"] + [
            "Verification covers artifact identity, source anchors, native module header binding and NumPy numerical references; it is not a complete semantic IR parser.",
            "The executable reload claim is the producer's in-process round trip; this audit does not load serialized executable files.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/cpu-matmul-003")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    summary = verify(args.capture.resolve())
    if args.write_summary:
        (HERE / "cpu-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "capture_id", "source_entries_checked", "source_edges_checked", "artifact_count", "qualifiers")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
