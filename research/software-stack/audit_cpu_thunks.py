#!/usr/bin/env python3
"""Link serialized CPU thunks to HLO, object bytes and fresh warm trace events."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from matmul_probe import fingerprint, write_json
from verify_research import ROOT, check, read_json, verify as verify_matmul
from capture_runtime import verify_binding, verify_current_reader
from cpu_executable_parser import SCHEMA, inventory, schema_pool, decode_package, project, message
from cpu_thunk_probe import check_environment_transition

ORIGIN = ROOT / "artifacts/jax-stack/source-runtime-002/suite/matmul"
RUNTIME = ROOT / "artifacts/jax-stack/cpu-thunk-runtime-002/capture"


def verify_native_graph(cpu, path):
    from jaxlib import _hlo
    module = _hlo.hlo_module_from_text(path.read_text())
    actual = {(c.name, i.name): (i.opcode.name, tuple(o.name for o in i.operands()))
              for c in module.computations() for i in c.instructions()}
    expected = {}
    for c in cpu.hlo_module.hlo_module.computations:
        by_id = {i.id: i.name for i in c.instructions}
        for i in c.instructions:
            opcode = "k" + "".join(word[0].upper() + word[1:] for word in i.opcode.split("-"))
            expected[(c.name, i.name)] = (opcode, tuple(by_id[j] for j in i.operand_ids))
    check(actual == expected, "serialized HLO graph differs from captured optimized text")


def trace_checks(events, projection, case):
    names = {t["op_name"] for t in projection["thunks"]}
    parents = sorted([e for e in events if e.get("name") == "research_thunk_invocation"], key=lambda e: e["ts"])
    check(len(parents) == 3 and [p["args"]["iteration"] for p in parents] == ["0", "1", "2"] and
          all(p["args"]["case"] == case and p["ph"] == "X" for p in parents), "invalid serial invocation windows")
    check(all(a["ts"] + a["dur"] <= b["ts"] for a, b in zip(parents, parents[1:])), "invocation windows overlap")
    producers = [e for e in events if "hlo_op" in e.get("args", {})]
    consumers = [e for e in events if e.get("name", "").startswith("end: ")]
    check(Counter(e["name"] for e in producers) == {name: 3 for name in names}, "thunk producer counts differ")
    check(Counter(e["name"] for e in consumers) == {"end: " + name: 3 for name in names}, "completion counts differ")
    runs = []
    for parent in parents:
        def inside(e):
            return parent["pid"] == e["pid"] and parent["ts"] <= e["ts"] and e["ts"] + e["dur"] <= parent["ts"] + parent["dur"]
        starts = [e for e in producers if inside(e)]
        ends = [e for e in consumers if inside(e)]
        check(Counter(e["name"] for e in starts) == {name: 1 for name in names} and
              Counter(e["name"] for e in ends) == {"end: " + name: 1 for name in names}, "missing unique thunk within invocation")
        ids = {e["args"].get("run_id") for e in starts}
        check(len(ids) == 1 and None not in ids and all(e["ph"] == "X" and e["args"].get("hlo_op") == e["name"] and
              e["args"].get("hlo_module") == projection["module_name"] and e["args"].get("device_ordinal") == "0"
              for e in starts), "runtime thunk identity differs")
        for start in starts:
            end = next(e for e in ends if e["name"] == "end: " + start["name"])
            check(start["ts"] <= end["ts"], "completion precedes invocation")
        runs.append({"iteration": int(parent["args"]["iteration"]), "run_id": next(iter(ids)),
                     "producer_count": len(starts), "completion_count": len(ends),
                     "worker_thread_differs_from_python": any(e["tid"] != parent["tid"] for e in starts)})
    check(len({r["run_id"] for r in runs}) == 3, "run ids do not distinguish warm executions")
    selected_compile = [e["name"] for e in events if e.get("name") in
                        {"XlaCompile", "PjRtCpuClient::Compile", "algsimp", "dot-library-rewriter"}]
    check(not selected_compile, "selected compile/pass event inside warm trace")
    return {"invocations": runs, "producer_events": len(producers), "completion_events": len(consumers),
            "selected_compile_events": selected_compile,
            "limits": "End events have no exported run_id; association is valid for these three non-overlapping blocking windows only. Producer duration is not a general asynchronous operation duration."}


def collect(output=None, origin=ORIGIN, runtime=RUNTIME):
    check(origin.is_relative_to(ROOT / "artifacts/jax-stack") and runtime.is_relative_to(ROOT / "artifacts/jax-stack"),
          "capture inputs must remain under artifacts/jax-stack")
    pool = schema_pool()
    original = verify_matmul(origin)
    check(original.get("build_binding") is not None and not original["qualifiers"], "source-built origin capture required")
    reader = verify_current_reader(original["build_binding"])
    runtime_inventory = inventory(runtime, "RUN-CPU")
    manifest = read_json(runtime / "manifest.json")
    after = read_json(runtime / "environment.json")
    before = read_json(runtime / "environment-before.json")
    added = check_environment_transition(before, after)
    binding = verify_binding(runtime, after, manifest)
    from pass_events_probe import bind_native
    before_binding = bind_native(before, ROOT / manifest["jaxlib_build_manifest"], "absent")
    check(before_binding == read_json(runtime / "build-binding-before.json"), "initial native binding differs")
    check(binding["build_manifest"] == original["build_binding"]["build_manifest"], "runtime uses another source build")
    runtime_summary = read_json(runtime / "summary.json")
    check(runtime_summary["newly_loaded_native_libraries"] == added == ["mlir/_mlir_libs/_mlirHlo.so"], "unexpected lazy native loading")
    with np.load(runtime / "inputs.npz", allow_pickle=False) as inputs, np.load(origin / "inputs.npz", allow_pickle=False) as old:
        for key in ("a", "w", "x"):
            check(np.array_equal(inputs[key], old[key]), "runtime and original inputs differ")
        a, w, x = [inputs[key].astype(np.float64) for key in ("a", "w", "x")]
    y, z = a@w, x@w
    references = {"matmul": (y,), "vmap_matmul": (z,), "grad_matmul": (2*y@w.T, 2*a.T@y),
                  "jit_grad_vmap_matmul": (2*z@w.T, 2*np.einsum("bmk,bmn->kn", x, z))}
    expected_kinds = {"matmul": {"ynn-fusion": 1}, "vmap_matmul": {"ynn-fusion": 1},
                      "grad_matmul": {"ynn-fusion": 2, "kernel": 1, "dot": 1},
                      "jit_grad_vmap_matmul": {"ynn-fusion": 2, "kernel": 2, "dot": 1}}
    check(set(manifest["cases"]) == set(references), "runtime cases differ")
    cases = []
    protoc = read_json(SCHEMA / "generation.json")["argv"][0]
    for case, refs in references.items():
        views = {}
        for label, source in [("original", origin), ("fresh", runtime)]:
            package = (source / case / "executable.bin").read_bytes()
            cpu, meta, packaging, blobs = decode_package(package, pool)
            projection = project(cpu, package)
            verify_native_graph(cpu, source / case / "optimized-hlo.txt")
            check(Counter(t["kind"] for t in projection["thunks"]) == expected_kinds[case], "CPU thunk choice changed")
            argv = [protoc, f"--descriptor_set_in={SCHEMA}/cpu-executable.descriptor.pb", "--decode=xla.cpu.CompilationResultProto"]
            decoded = subprocess.run(argv, input=blobs["cpu-compilation.pb"], capture_output=True, check=True).stdout
            encoded = subprocess.run([*argv[:-1], "--encode=xla.cpu.CompilationResultProto"],
                                     input=decoded, capture_output=True, check=True).stdout
            check(message(pool, "xla.cpu.CompilationResultProto", encoded) == cpu, "independent protoc roundtrip differs")
            if output:
                directory = output / case / label
                directory.mkdir(parents=True)
                for name, data in blobs.items():
                    (directory / name).write_bytes(data)
                (directory / "cpu-compilation.txt").write_bytes(decoded)
                (directory / "protoc-roundtrip.pb").write_bytes(encoded)
                write_json(directory / "commands.json", {"decode": argv, "encode": [*argv[:-1], "--encode=xla.cpu.CompilationResultProto"]})
                write_json(directory / "projection.json", projection)
            views[label] = {"package": {"path": str((source / case / "executable.bin").relative_to(ROOT)),
                                        **fingerprint(source / case / "executable.bin")},
                            "packaging": packaging, "projection": projection}
        old_thunks, new_thunks = [views[k]["projection"]["thunks"] for k in ("original", "fresh")]
        check([(t["kind"], t["op_name"]) for t in old_thunks] == [(t["kind"], t["op_name"]) for t in new_thunks],
              "fresh and original thunk identities differ")
        directory = runtime / case
        trace_files = list((directory / "trace").rglob("perfetto_trace.json.gz"))
        check(len(trace_files) == 1, "missing unique trace")
        with gzip.open(trace_files[0], "rt") as stream:
            events = json.load(stream)["traceEvents"]
        selected = [e for e in events if e.get("name") == "research_thunk_invocation" or
                    "hlo_op" in e.get("args", {}) or e.get("name", "").startswith("end: ")]
        check(selected == read_json(directory / "selected-events.json"), "saved trace selection differs")
        trace = trace_checks(events, views["fresh"]["projection"], case)
        errors = []
        with np.load(directory / "outputs.npz", allow_pickle=False) as values:
            check(set(values.files) == {f"iteration_{iteration}_output_{i}" for iteration in range(3) for i in range(len(refs))},
                  "numerical output inventory differs")
            for iteration in range(3):
                for i, reference in enumerate(refs):
                    actual = values[f"iteration_{iteration}_output_{i}"]
                    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
                    errors.append(float(np.max(np.abs(actual-reference))))
        cases.append({"case": case, "views": views, "runtime_trace": trace, "max_absolute_error": max(errors)})
    result = {"evidence_level": "REPLAY-OFFLINE", "qualifiers": [], "schema": inventory(SCHEMA, "SOURCE-ONLY"),
              "original_capture": {"path": str(origin.relative_to(ROOT)), "manifest_sha256": original["manifest_sha256"]},
              "runtime_capture": runtime_inventory, "build_binding": binding, "verification_runtime": reader,
              "newly_loaded_native_libraries": added, "cases": cases,
              "limits": ["Source-built CPU serialized structure and separate fresh CPU executions; no TPU or LLO evidence.",
                         "Data-only pickle opcode/protobuf parsing; no package loading or native deserialization.",
                         "Protobuf 7.34.0 pure-Python and cached protoc 32.1 are fingerprinted analysis tools; schema definitions come from pinned sources.",
                         "Trace counts/identities prove observed CPU thunk dispatch, not internal Eigen/YNN microkernel selection or performance."]}
    if output:
        write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-capture", type=Path, default=ORIGIN)
    parser.add_argument("--runtime-capture", type=Path, default=RUNTIME)
    args = parser.parse_args()
    output = args.output.resolve()
    check(output.is_relative_to(ROOT / "artifacts/jax-stack"), "use fresh artifacts/jax-stack directory")
    output.mkdir(parents=True, exist_ok=False)
    for name in [Path(__file__).name, "cpu_executable_parser.py", "cpu_thunk_probe.py", "capture_runtime.py", "verify_research.py"]:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    started = datetime.now(timezone.utc).isoformat()
    result = collect(output, args.origin_capture.resolve(), args.runtime_capture.resolve())
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass",
        "evidence_level": "REPLAY-OFFLINE", "qualifiers": [], "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv]},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"cases": len(result["cases"]), "producer_events": sum(c["runtime_trace"]["producer_events"] for c in result["cases"])}))


if __name__ == "__main__":
    main()
