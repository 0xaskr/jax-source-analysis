#!/usr/bin/env python3
"""Recheck source-bound CPU thunk evidence and malformed/overclaimed counterexamples."""

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickletools
import subprocess

from audit_cpu_thunks import collect, trace_checks
from cpu_executable_parser import SCHEMA, inventory, schema_pool, decode_package, project, message, extract_exec, delimited
from cpu_thunk_probe import check_environment_transition
from verify_research import ROOT, check, read_json, local_path


def selftest(capture):
    saved = read_json(capture / "summary.json")
    origin = local_path(saved["original_capture"]["path"])
    runtime = local_path(saved["runtime_capture"]["path"]).parent
    passed = []
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError("invalid input accepted: " + name)

    for mode in ["hash", "missing", "level"]:
        fake = copy.deepcopy(read_json(capture / "manifest.json"))
        if mode == "hash":
            fake["artifacts"][0]["sha256"] = "0" * 64
        elif mode == "missing":
            fake["artifacts"].pop()
        else:
            fake["evidence_level"] = "RUN-TPU"
        rejects(mode, lambda: inventory(capture, "REPLAY-OFFLINE", fake))
    pool = schema_pool()
    package = (origin / "grad_matmul/executable.bin").read_bytes()
    cpu, _, _, blobs = decode_package(package, pool)
    marker = next(pos for op, arg, pos in pickletools.genops(package) if op.name == "SHORT_BINUNICODE" and arg == "exec")
    fake_package = package[:marker+2] + b"nope" + package[marker+6:]
    rejects("wrong-persistent-id", lambda: extract_exec(fake_package))
    rejects("ambiguous-byte-payload", lambda: extract_exec(package[:-1] + b"B\x01\x00\x00\x00z."))
    rejects("trailing-pickle-bytes", lambda: extract_exec(package + b"x"))
    rejects("invalid-metadata-varint", lambda: delimited(b"\x80" * 5))
    rejects("unknown-protobuf-field", lambda: message(pool, "xla.cpu.CompilationResultProto", blobs["cpu-compilation.pb"] + b"\xf8\xff\x03\x01"))
    bad = copy.deepcopy(cpu)
    bad.thunk_sequence.thunks[-1].kind = "ynn-fusion"
    rejects("kind-oneof-disagree", lambda: project(bad, package))
    bad = copy.deepcopy(cpu)
    bad.thunk_sequence.thunks[-1].dot_thunk.dot_dimensions.lhs_contracting_dimensions[0] = 1
    rejects("dot-dimensions-disagree", lambda: project(bad, package))
    bad = copy.deepcopy(cpu)
    bad.thunk_sequence.thunks[0].ynn_fusion_thunk.instruction_id += 1
    rejects("wrong-ynn-instruction", lambda: project(bad, package))
    bad = copy.deepcopy(cpu)
    bad.thunk_sequence.thunks[-1].dot_thunk.out_buffer_shape.slice.offset = 10**9
    rejects("slice-outside-allocation", lambda: project(bad, package))
    bad = copy.deepcopy(cpu)
    bad.thunk_sequence.thunks[1].kernel_thunk.kernel_name = "unregistered"
    rejects("wrong-kernel-symbol", lambda: project(bad, package))
    rejects("duplicate-object-bytes", lambda: project(cpu, package + cpu.object_files[0].contents))
    fresh = (runtime / "grad_matmul/executable.bin").read_bytes()
    fresh_cpu, *_ = decode_package(fresh, pool)
    projection = project(fresh_cpu, fresh)
    trace_file = next((runtime / "grad_matmul/trace").rglob("perfetto_trace.json.gz"))
    with gzip.open(trace_file, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    bad_events = [e for e in events if e.get("name") != "dot"]
    rejects("missing-dot-events", lambda: trace_checks(bad_events, projection, "grad_matmul"))
    bad_events = copy.deepcopy(events)
    for event in bad_events:
        if "hlo_op" in event.get("args", {}):
            event["args"]["run_id"] = "same"
    rejects("merged-run-identities", lambda: trace_checks(bad_events, projection, "grad_matmul"))
    bad_events = copy.deepcopy(events)
    next(e for e in bad_events if e.get("name") == "end: dot")["ts"] = -1000
    rejects("completion-outside-window", lambda: trace_checks(bad_events, projection, "grad_matmul"))
    before, after = [read_json(runtime / name) for name in ("environment-before.json", "environment.json")]
    bad_after = copy.deepcopy(after)
    bad_after["runtime"]["jaxlib"]["native_binaries"][0]["sha256"] = "0" * 64
    rejects("existing-native-payload-changed", lambda: check_environment_transition(before, bad_after))
    bad_after = copy.deepcopy(after)
    bad_after["repository"]["sources"]["xla"]["git_commit"] = "0" * 40
    rejects("source-changed-during-run", lambda: check_environment_transition(before, bad_after))
    return passed


def verify(capture, run_selftest=False):
    checked = inventory(capture, "REPLAY-OFFLINE")
    saved = read_json(capture / "summary.json")
    origin = local_path(saved["original_capture"]["path"])
    runtime = local_path(saved["runtime_capture"]["path"]).parent
    current = collect(origin=origin, runtime=runtime)
    check(saved == current, "saved thunk audit differs from full source/runtime recheck")
    pool = schema_pool()
    protoc = read_json(SCHEMA / "generation.json")["argv"][0]
    for case in current["cases"]:
        for label, source in [("original", origin), ("fresh", runtime)]:
            directory = capture / case["case"] / label
            package = (source / case["case"] / "executable.bin").read_bytes()
            cpu, _, _, blobs = decode_package(package, pool)
            for name, content in blobs.items():
                check((directory / name).read_bytes() == content, "extracted protobuf bytes differ")
            check(read_json(directory / "projection.json") == project(cpu, package), "saved projection differs")
            check(message(pool, "xla.cpu.CompilationResultProto", (directory / "protoc-roundtrip.pb").read_bytes()) == cpu,
                  "saved independent protoc roundtrip differs")
            argv = [protoc, f"--descriptor_set_in={SCHEMA}/cpu-executable.descriptor.pb"]
            commands = {"decode": [*argv, "--decode=xla.cpu.CompilationResultProto"],
                        "encode": [*argv, "--encode=xla.cpu.CompilationResultProto"]}
            check(read_json(directory / "commands.json") == commands, "recorded protoc command differs")
            encoded = subprocess.run(commands["encode"], input=(directory / "cpu-compilation.txt").read_bytes(),
                                     capture_output=True, check=True).stdout
            check(message(pool, "xla.cpu.CompilationResultProto", encoded) == cpu, "saved protobuf text differs semantically")
    negatives = selftest(capture) if run_selftest else []
    return {"audit_manifest": checked, **current, "negative_tests": negatives}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/cpu-thunk-audit-002/capture")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    result = verify(args.capture.resolve(), args.selftest)
    if args.result:
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "artifacts": result["audit_manifest"]["artifact_count"],
                      "negative_tests": result["negative_tests"]}))


if __name__ == "__main__":
    main()
