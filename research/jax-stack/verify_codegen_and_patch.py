#!/usr/bin/env python3
"""Verify offline object bindings and reversible, still-uncompiled pass patch."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import tempfile
from pathlib import Path

from verify_research import ROOT, HERE, check, local_path, read_json, sha256, verify as verify_cpu
from verify_extensions import events, verify_compiler


def inventory(capture, level, manifest=None):
    manifest = read_json(capture / "manifest.json") if manifest is None else manifest
    check(manifest["evidence_level"] == level, "wrong evidence level")
    check(manifest["outcome"] == ("pass" if level == "REPLAY-OFFLINE" else "patch-prepared-and-reversed"), "wrong preparation/audit outcome")
    registered = set()
    for record in manifest["artifacts"]:
        path = local_path(record["path"])
        check(path.is_relative_to(capture) and path not in registered, "invalid artifact path")
        check(path.is_file() and path.stat().st_size == record["size_bytes"] and sha256(path) == record["sha256"], "artifact identity differs")
        registered.add(path)
    check(registered == {p for p in capture.rglob("*") if p.is_file() and p != capture / "manifest.json"}, "incomplete inventory")
    return {"capture_id": capture.name, "manifest_path": str((capture / "manifest.json").relative_to(ROOT)),
            "manifest_sha256": sha256(capture / "manifest.json"), "artifact_count": len(registered), "evidence_level": level}


def embedded_object(package, obj, offset):
    check(obj.startswith(b"\x7fELF\x02\x01"), "not little-endian ELF64")
    check(int.from_bytes(obj[16:18], "little") == 1 and int.from_bytes(obj[18:20], "little") == 62, "not relocatable x86-64")
    check(package.count(obj) == 1 and package.find(obj) == offset, "object not uniquely embedded at the recorded offset")


def verify_codegen(capture):
    from jaxlib import _hlo
    result = inventory(capture, "REPLAY-OFFLINE")
    summary = read_json(capture / "summary.json")
    original = verify_cpu(local_path(summary["source_capture"]))
    from capture_runtime import verify_current_reader
    reader = verify_current_reader(original.get("build_binding"))
    check(summary.get("build_binding") == original.get("build_binding"), "codegen source build binding differs")
    if reader is not None:
        import hashlib
        import zipfile
        recorded = read_json(capture / "reader-identity.json")
        check(recorded == summary["reader_identity"] and recorded["source_build_verified"] and
              recorded["build_id"] == reader["build_id"] and
              recorded["native_payloads"] == reader["native_payloads"], "producer native reader identity differs")
        with zipfile.ZipFile(ROOT / original["build_binding"]["wheel"]["path"]) as wheel:
            for name, digest in recorded["native_payloads"].items():
                check(hashlib.sha256(wheel.read(name)).hexdigest() == digest, "producer reader payload differs from source wheel")
    check(original["manifest_sha256"] == summary["source_manifest_sha256"], "source capture changed")
    check(summary["qualifiers"] == original["qualifiers"] == read_json(capture / "manifest.json")["qualifiers"], "lost runtime qualifier")
    for item in read_json(capture / "input-records.json"):
        path = local_path(item["path"])
        check(sha256(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], "referenced input differs")
    tools = read_json(capture / "inspection-tools.json")
    from pathlib import Path
    for tool in tools.values():
        check(sha256(Path(tool["path"])) == tool["sha256"], "inspection tool changed")
    source = local_path(summary["source_capture"])
    expected_cases = [{"case": c["name"], "native_module_prefix": c["native_module_prefix"],
                       "object_count": len(list((source / "xla-dump").glob(c["native_module_prefix"] + ".*.o")))}
                      for c in original["cases"]]
    check(summary["cases"] == expected_cases, "case-to-native-module binding differs")
    expected_objects = {str(p.relative_to(ROOT)) for c in expected_cases
                        for p in (source / "xla-dump").glob(c["native_module_prefix"] + ".*.o")}
    recorded_objects = [r["inputs"]["object"] for r in summary["objects"]]
    check(len(recorded_objects) == len(set(recorded_objects)) and set(recorded_objects) == expected_objects, "distinct object inventory differs")
    check(len(summary["objects"]) == 3, "wrong object inventory")
    for record in summary["objects"]:
        paths = {key: local_path(value) for key, value in record["inputs"].items()}
        case = next(c for c in expected_cases if c["case"] == record["case"])
        prefix = case["native_module_prefix"]
        check(record["native_module_prefix"] == prefix and paths["object"].name.startswith(prefix + ".obj-file."), "object bound to another case")
        check(paths["hlo"] == source / record["case"] / "optimized-hlo.txt" and paths["serialized_executable"] == source / record["case"] / "executable.bin", "wrong case HLO/package")
        stem = paths["object"].name.removeprefix(prefix + ".obj-file.").removesuffix(".o")
        check(paths["ir_before"] == source / "xla-dump" / (prefix + "." + stem + ".ir-no-opt.ll") and paths["ir_after"] == source / "xla-dump" / (prefix + "." + stem + ".ir-with-opt.ll"), "wrong LLVM module pair")
        embedded_object(paths["serialized_executable"].read_bytes(), paths["object"].read_bytes(), record["serialized_package_byte_offset"])
        directory = local_path(record["inspection_directory"])
        actual_symbols = subprocess.check_output([tools["readelf"]["path"], "-sW", str(paths["object"])], text=True)
        check(actual_symbols == (directory / "elf-symbols.txt").read_text(), "symbol readback differs")
        definitions = re.findall(r"^define .*?@([A-Za-z0-9_.$-]+)\(", paths["ir_after"].read_text(), re.M)
        check(set(definitions) == {f["name"] for f in record["defined_functions"]}, "IR definitions do not match recorded ELF symbols")
        module = _hlo.hlo_module_from_text(paths["hlo"].read_text())
        ops = {i.name: i.opcode.name for c in module.computations() for i in c.instructions()}
        for function in record["defined_functions"]:
            name = function["name"]
            check(ops.get(name) == "kFusion", "kernel symbol not bound to its HLO fusion")
            lines = [line.split() for line in actual_symbols.splitlines() if line.split() and line.split()[-1] == name]
            check(len(lines) == 1 and lines[0][3:5] == ["FUNC", "GLOBAL"] and int(lines[0][2]) == function["size_bytes"], "ELF function identity/size differs")
            check(f"<{name}>:" in (directory / "disassembly.txt").read_text(), "disassembly function missing")
        for key in ["ir_before", "ir_after"]:
            check("; ModuleID = '" + record["llvm_module_id"] + "'" in paths[key].read_text(), "LLVM module identity differs")
        check(record["llvm_module_id"] in actual_symbols, "ELF source module identity differs")
    result.update(qualifiers=summary["qualifiers"], source_capture=summary["source_capture"],
                  build_binding=original.get("build_binding"), verification_runtime=reader,
                  cases=summary["cases"], objects=summary["objects"], limits=summary["limits"])
    return result


def check_patch_scope(summary):
    check(summary["evidence_level"] == "SOURCE-ONLY", "patch scope promoted to runtime")
    check(not any(summary[k] for k in ["native_compiled", "patched_binary_loaded", "custom_event_observed"]), "patch preparation counted as executed")


def verify_patch(capture):
    result = inventory(capture, "SOURCE-ONLY")
    summary = read_json(capture / "summary.json")
    check_patch_scope(summary)
    source = local_path(summary["source_path"])
    check(sha256(source) == summary["source"]["sha256"], "patch base changed")
    check(summary["source_revision"] == read_json(ROOT / "manifests/baseline.json")["repository"]["sources"]["xla"]["git_commit"], "patch revision differs")
    patch = HERE / "xla-hlo-pass-events.patch"
    check(sha256(patch) == summary["patch"]["sha256"] == sha256(local_path(summary["patch_path"])), "published patch differs from checked patch")
    candidate = capture / "patched-hlo_pass_pipeline.cc"
    check(sha256(candidate) == summary["patched_source"]["sha256"], "candidate identity differs")
    with tempfile.TemporaryDirectory(prefix="verify-pass-patch-", dir=ROOT / "artifacts/jax-stack") as temporary:
        from pathlib import Path
        work = Path(temporary)
        target = work / "xla/hlo/pass/hlo_pass_pipeline.cc"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        command = ["git", "apply", "--directory=" + str(work.relative_to(ROOT)), str(patch)]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        check(target.read_bytes() == candidate.read_bytes(), "published patch does not produce checked candidate")
        subprocess.run(["git", "apply", "--reverse", *command[2:]], cwd=ROOT, check=True, capture_output=True)
        check(target.read_bytes() == source.read_bytes(), "independent reverse apply differs")
    for name in ["base", "candidate"]:
        check((capture / name / "xla/hlo/pass/hlo_pass_pipeline.cc").read_bytes() == source.read_bytes(), "rollback bytes differ")
    check(all(summary["checks"].values()), "incomplete reversible patch checks")
    text = candidate.read_text()
    check(text.count('"research_hlo_pass_run"') == 1 and text.count("auto result = RunHelper<HloT>(pass, hlo, execution_threads);") == 1, "patch call or event multiplicity differs")
    check(text.index("continue;", text.index("// Run-time gate")) < text.index("std::optional<tsl::profiler::TraceMe> research_trace;"), "custom trace begins before filtering")
    check(text.index("return result;", text.index("research_trace;")) < text.index("if (auto status = status_or_changed.status()"), "custom scope includes outer diagnostics")
    control = ROOT / "artifacts/jax-stack/compiler-events-001"
    control_result = verify_compiler(control)
    observed = sum(e["name"] == summary["event_name"] for e in events(control))
    check(observed == 0, "custom patch event already present in the old wheel control")
    result.update(summary=summary, negative_control={"capture_id": control.name,
        "manifest_sha256": control_result["manifest_sha256"], "evidence_level": "RUN-CPU",
        "qualifiers": control_result["qualifiers"], "custom_event_count": observed,
        "scope": "Historical unpatched, version-skew wheel; does not replace the matching-source baseline required before patch application."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--codegen-capture", type=Path, default=ROOT / "artifacts/jax-stack/codegen-artifacts-001")
    parser.add_argument("--codegen-result", type=Path)
    parser.add_argument("--codegen-only", action="store_true", help="Do not replay the separate historical patch preparation")
    args = parser.parse_args()
    codegen = ROOT / "artifacts/jax-stack/codegen-artifacts-001"
    patch = ROOT / "artifacts/jax-stack/compiler-event-patch-002"
    if args.selftest:
        for capture, level in [(codegen, "REPLAY-OFFLINE"), (patch, "SOURCE-ONLY")]:
            for mode in ["hash", "missing", "level"]:
                fake = copy.deepcopy(read_json(capture / "manifest.json"))
                if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
                elif mode == "missing": fake["artifacts"].pop()
                else: fake["evidence_level"] = "RUN-TPU"
                try: inventory(capture, level, fake)
                except ValueError: pass
                else: raise AssertionError(f"invalid {capture.name} manifest accepted: {mode}")
        fake = copy.deepcopy(read_json(patch / "summary.json")); fake["native_compiled"] = True
        try: check_patch_scope(fake)
        except ValueError: pass
        else: raise AssertionError("prepared patch promoted to compiled")
        obj = local_path(read_json(codegen / "summary.json")["objects"][0]["inputs"]["object"]).read_bytes()
        try: embedded_object(obj + obj, obj, 0)
        except ValueError: pass
        else: raise AssertionError("ambiguous duplicate object accepted")
        print("selftest: 6 invalid manifests, uncompiled-as-compiled and duplicate object rejected")
    a = verify_codegen(args.codegen_capture.resolve())
    b = None if args.codegen_only else verify_patch(patch)
    if args.codegen_result:
        args.codegen_result.write_text(json.dumps(a, ensure_ascii=False, indent=2) + "\n")
    if args.write:
        for name, value in [("codegen-results.json", a), ("pass-event-patch-results.json", b)]:
            if value is not None:
                (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"objects": len(a["objects"]), "offline_artifacts": a["artifact_count"], "patch_artifacts": b["artifact_count"] if b else None}))


if __name__ == "__main__":
    main()
