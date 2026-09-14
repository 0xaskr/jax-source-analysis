#!/usr/bin/env python3
"""Audit raw context identities, cross-thread links and JSON field loss."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from capture_runtime import verify_binding
from cpu_executable_parser import SCHEMA as TOOL_SCHEMA, inventory, message
from cpu_thunk_probe import check_environment_transition
from host_context_probe import CONTEXT_IDS
from matmul_probe import ROOT, fingerprint, write_json
from verify_research import check, read_json, sha256
from xspace_contexts import (prepare_schema, load_schema, flatten, integer,
                             context_groups, unique_pairs, json_correspondence, flow_overlay)

THUNKS = ROOT / "artifacts/jax-stack/cpu-thunk-runtime-002/capture"
HOST = ROOT / "artifacts/jax-stack/host-context-runtime-001/capture"


def check_no_exported_flows(events):
    check(not any(e.get("ph") in {"s", "f", "t"} or
                  any(k in e for k in ("bind_id", "flow_in", "flow_out"))
                  for e in events), "original JSON already has flows; revisit export comparison")


def validate_runtime(source):
    checked = inventory(source, "RUN-CPU")
    manifest = read_json(source / "manifest.json")
    before, after = [read_json(source / name) for name in ("environment-before.json", "environment.json")]
    added = check_environment_transition(before, after)
    binding = verify_binding(source, after, manifest)
    from pass_events_probe import bind_native
    check(bind_native(before, ROOT / manifest["jaxlib_build_manifest"], "absent") ==
          read_json(source / "build-binding-before.json"), "initial native binding changed")
    check(read_json(source / "summary.json")["newly_loaded_native_libraries"] == added,
          "lazy native mapping record differs")
    check(all(not s["dirty"] for s in after["repository"]["sources"].values()), "capture contains dirty source")
    return checked, binding


def inspect_trace(directory, pool, label, mode, expected_thunks=None):
    files = list(directory.rglob("*.xplane.pb"))
    jsons = list(directory.rglob("perfetto_trace.json.gz"))
    check(len(files) == len(jsons) == 1, "trace does not have unique XSpace/JSON files")
    path = files[0]
    raw = path.read_bytes()
    space = message(pool, "tensorflow.profiler.XSpace", raw)
    rows = flatten(space)
    groups = context_groups(rows, sha256(path))
    if mode == "thunks":
        pairs = unique_pairs(rows, groups)
    else:
        pairs = unique_pairs(rows, groups, lambda r: r["name"] in {"research_link_send", "research_link_receive"})
    by_node = {r["node"]: r for r in rows}
    with gzip.open(jsons[0], "rt") as stream:
        exported = json.load(stream)
    matches = json_correspondence(rows, exported["traceEvents"], pairs)
    flows = flow_overlay(rows, pairs)
    check_no_exported_flows(exported["traceEvents"])
    result = {"label": label, "mode": mode,
              "xspace": {"path": str(path.relative_to(ROOT)), **fingerprint(path)},
              "json": {"path": str(jsons[0].relative_to(ROOT)), **fingerprint(jsons[0])},
              "row_count": len(rows), "context_groups": len(groups), "pairs": pairs,
              "cross_thread_pairs": sum(p["cross_thread"] for p in pairs), "json_matches": matches,
              "original_flow_events": 0, "derived_flow_events": len(flows)}
    if mode == "thunks":
        thunk_pairs = [p for p in pairs if "hlo_op" in by_node[p["producer"]]["stats"]]
        program = expected_thunks["views"]["fresh"]["projection"]
        names = {t["op_name"] for t in program["thunks"]}
        check(Counter(p["producer_name"] for p in thunk_pairs) == {name: 3 for name in names}, "raw thunk context counts differ")
        for pair in thunk_pairs:
            producer = by_node[pair["producer"]]
            check(pair["consumer_name"] == "end: " + pair["producer_name"] and
                  pair["context"]["context_type"] == 0 and
                  integer(producer["stats"]["program_id"]) == program["module_id"],
                  "raw thunk program or completion binding differs")
        counts = Counter(p["context"]["context_type"] for p in pairs)
        expected = {"matmul": (3, 0, 0), "vmap_matmul": (3, 3, 3),
                    "grad_matmul": (12, 3, 0), "jit_grad_vmap_matmul": (15, 3, 0)}[label]
        threadpool = [p for p in pairs if p["context"]["context_type"] == 15]
        await_pairs = [p for p in pairs if p["producer_name"] == "CommonPjRtBuffer::Await"]
        check((len(thunk_pairs), len(threadpool), len(await_pairs)) == expected, "native context family counts differ")
        check(all(p["producer_name"] == "ThreadpoolListener::Record" and
                  p["consumer_name"] == "ThreadpoolListener::StartRegion" and p["cross_thread"] for p in threadpool),
                  "threadpool context is not the observed cross-thread handoff")
        result.update(thunk_pairs=len(thunk_pairs), threadpool_pairs=len(threadpool), await_pairs=len(await_pairs),
                      context_type_counts={str(k): v for k, v in counts.items()})
    else:
        check(len(pairs) == 4 and {p["context"]["context_id"] for p in pairs} == {str(i) for i in CONTEXT_IDS},
              "explicit uint64 contexts lost, merged or duplicated")
        check(all(p["cross_thread"] and p["producer_name"] == "research_link_send" and
                  p["consumer_name"] == "research_link_receive" for p in pairs), "custom endpoints are not cross-thread markers")
        observations = []
        for pair in pairs:
            producer, consumer = [by_node[pair[k]] for k in ("producer", "consumer")]
            work = integer(producer["stats"]["work"])
            check(integer(consumer["stats"]["work"]) == work and
                  pair["context"]["context_id"] == str(CONTEXT_IDS[work]), "context paired to another work item")
            outcome = consumer["stats"]["outcome"]["value"]
            check(outcome == ("expected-error" if work == 3 else "success"), "completion outcome differs")
            observations.append({"work": work, "context_id": pair["context"]["context_id"],
                "producer_ps": producer["timestamp_ps"], "consumer_ps": consumer["timestamp_ps"],
                "producer_duration_ps": producer["duration_ps"], "outcome": outcome})
        check([o["work"] for o in sorted(observations, key=lambda o: o["producer_ps"])] == [0, 1, 2, 3] and
              [o["work"] for o in sorted(observations, key=lambda o: o["consumer_ps"])] == [3, 2, 1, 0], "controlled endpoint order differs")
        check(max(o["producer_ps"] for o in observations) < min(o["consumer_ps"] for o in observations), "logical operation intervals did not overlap")
        result["explicit_operations"] = sorted(observations, key=lambda o: o["work"])
    artifacts = {"rows.json": rows, "groups.json": groups, "pairs.json": pairs,
                 "json-matches.json": matches,
                 "derived-trace.json": {**exported, "traceEvents": [*exported["traceEvents"], *flows]}}
    return result, artifacts, raw


def collect(output=None, schema_capture=None, thunks=THUNKS, host=HOST):
    check(host.is_relative_to(ROOT / "artifacts/jax-stack"), "host capture must remain under artifacts/jax-stack")
    pool = prepare_schema(output) if output else load_schema(schema_capture)
    thunk_inventory, thunk_binding = validate_runtime(thunks)
    host_inventory, host_binding = validate_runtime(host)
    check(thunk_binding["build_manifest"] == host_binding["build_manifest"], "captures use different source builds")
    previous = read_json(ROOT / "research/jax-stack/cpu-thunk-results.json")
    check(previous["runtime_capture"] == thunk_inventory, "thunk witness does not cover this capture")
    cases = []
    for reference in previous["cases"]:
        name = reference["case"]
        cases.append((thunks / name / "trace", name, "thunks", reference))
    cases.append((host / "trace", "explicit_host", "host", None))
    records = []
    protoc = read_json(TOOL_SCHEMA / "generation.json")["argv"][0]
    schema_root = output or schema_capture
    for directory, label, mode, witness in cases:
        result, artifacts, raw = inspect_trace(directory, pool, label, mode, witness)
        argv = [protoc, f"--descriptor_set_in={schema_root}/xplane.descriptor.pb", "--decode=tensorflow.profiler.XSpace"]
        decoded = subprocess.run(argv, input=raw, capture_output=True, check=True).stdout
        encoded = subprocess.run([*argv[:-1], "--encode=tensorflow.profiler.XSpace"], input=decoded, capture_output=True, check=True).stdout
        check(message(pool, "tensorflow.profiler.XSpace", encoded) == message(pool, "tensorflow.profiler.XSpace", raw), "independent XSpace codec roundtrip differs")
        if output:
            dest = output / label
            dest.mkdir()
            for name, data in artifacts.items():
                write_json(dest / name, data)
            (dest / "xspace.txt").write_bytes(decoded)
            (dest / "protoc-roundtrip.pb").write_bytes(encoded)
        records.append(result)
    summary = read_json(host / "summary.json")
    check(summary["completion_order"] == [3, 2, 1, 0] and summary["context_ids"] == CONTEXT_IDS and
          summary["expected_errors"] == [{"work": 3, "error": "intentional error after device completion"}], "host producer summary differs")
    max_errors = []
    with np.load(host / "inputs.npz", allow_pickle=False) as data, np.load(host / "outputs.npz", allow_pickle=False) as outputs:
        check(set(outputs.files) == {f"output_{i}" for i in range(4)}, "missing host computation output")
        for i in range(4):
            reference = data[f"a_{i}"].astype(np.float64) @ data["weight"].astype(np.float64)
            np.testing.assert_allclose(outputs[f"output_{i}"], reference, rtol=2e-5, atol=2e-5)
            max_errors.append(float(np.max(np.abs(outputs[f"output_{i}"]-reference))))
    check(max_errors == summary["max_absolute_errors"], "host numerical summary differs")
    result = {"evidence_level": "REPLAY-OFFLINE", "qualifiers": [], "thunk_runtime": thunk_inventory,
              "host_runtime": host_inventory, "tool_schema": inventory(TOOL_SCHEMA, "SOURCE-ONLY"),
              "build_id": host_binding["build_id"], "build_manifest": host_binding["build_manifest"],
              "thunk_witness": {"path": "research/jax-stack/cpu-thunk-results.json", **fingerprint(ROOT / "research/jax-stack/cpu-thunk-results.json")},
              "cases": records, "host_numerical_max_errors": max_errors,
              "limits": ["Pure protobuf replay; runtime source/native identity comes from verified producer captures.",
                         "Keys are scoped to one XSpace, process, context type and exact uint64 id; multi-host and special launch contexts are outside this fixture.",
                         "Strict one-to-one application contract; XProf native grouping also supports many-to-many contexts.",
                         "Derived flow overlay is this audit's output, not a rerun of XProf preprocessing or proof of viewer rendering.",
                         "Controlled completion gates and profiling overhead are included; no kernel timing, performance or TPU execution claim."]}
    if output:
        write_json(output / "native-bindings.json", {"thunks": thunk_binding, "host": host_binding})
        write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-capture", type=Path, default=HOST)
    args = parser.parse_args()
    output = args.output.resolve()
    check(output.is_relative_to(ROOT / "artifacts/jax-stack"), "use a fresh artifacts/jax-stack output")
    output.mkdir(parents=True, exist_ok=False)
    for name in [Path(__file__).name, "xspace_contexts.py", "cpu_executable_parser.py", "capture_runtime.py"]:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    started = datetime.now(timezone.utc).isoformat()
    result = collect(output, host=args.host_capture.resolve())
    write_json(output / "manifest.json", {"capture_id": output.name, "evidence_level": "REPLAY-OFFLINE", "qualifiers": [],
        "outcome": "pass", "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv], "source": str((output / Path(__file__).name).relative_to(ROOT))},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"cases": len(result["cases"]), "pairs": sum(len(c["pairs"]) for c in result["cases"]),
                      "cross_thread": sum(c["cross_thread_pairs"] for c in result["cases"])}))


if __name__ == "__main__":
    main()
