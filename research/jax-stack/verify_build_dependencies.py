#!/usr/bin/env python3
"""Verify the bounded core dependency audit, without touching the live build."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from audit_build_dependencies import file_hash
from verify_research import ROOT, HERE, check, local_path, read_json


def identity(path, record):
    check(path.is_file() and path.stat().st_size == record["size_bytes"], "file size differs")
    check(file_hash(path) == record["sha256"], "file SHA-256 differs")


def inventory(capture, manifest):
    check(manifest["outcome"] == "pass" and manifest["evidence_level"] == "SOURCE-ONLY", "wrong evidence level")
    registered = set()
    for record in manifest["artifacts"]:
        path = local_path(record["path"])
        check(path.is_relative_to(capture) and path not in registered, "invalid capture path")
        identity(path, record)
        registered.add(path)
    check(registered == {p for p in capture.rglob("*") if p.is_file() and p != capture / "manifest.json"}, "incomplete inventory")
    identity(local_path(manifest["producer"]["source"]), manifest["producer"])
    return len(registered)


def verify_components(capture, summary):
    baseline = read_json(ROOT / "manifests/baseline.json")["repository"]["sources"]
    check([c["component"] for c in summary["components"]] == ["llvm", "stablehlo", "shardy"], "wrong core components")
    results = []
    for component in summary["components"]:
        name = component["component"]
        check(component["base_revision"] == baseline[name]["git_commit"], "base pin differs")
        source = ROOT / "upstream/xla/third_party" / name
        declaration = source / "workspace.bzl"
        identity(declaration, component["declaration"])
        text = declaration.read_text()
        declared_sha = re.search(name.upper() + r'_SHA256 = "([0-9a-f]{64})"', text)[1]
        declared_commit = re.search(name.upper() + r'_COMMIT = "([0-9a-f]{40})"', text)[1]
        check(declared_commit == component["base_revision"], "declaration revision differs")
        check(declared_sha == component["archive"]["sha256"], "archive differs from declaration")
        identity(local_path(component["archive"]["path"]), component["archive"])
        patch_names = re.findall(r'"//third_party/' + name + r':([^\"]+\.patch)"', text)
        check(patch_names == [Path(p["path"]).name for p in component["patches"]], "patch order differs")
        targets = set()
        for patch in component["patches"]:
            path = source / Path(patch["path"]).name
            identity(path, patch)
            check(patch["strip"] == 1, "patch strip differs")
            for line in path.read_text().splitlines():
                if line.startswith(("--- ", "+++ ")):
                    token = line[4:].split("\t")[0]
                    if token != "/dev/null":
                        check(".." not in Path(token).parts and not token.startswith("/"), "invalid patch target")
                        targets.add(token.split("/", 1)[1])
        recorded = [r["path"] for r in component["targets"]]
        check(len(recorded) == len(set(recorded)) == component["replayed_target_count"] and set(recorded) == targets, "patch target inventory differs")
        for target in component["targets"]:
            actual = local_path(str(Path(component["actual_source_root"]) / target["path"]))
            replay = capture / "replay" / name / target["path"]
            if target.get("deleted"):
                check(not actual.exists() and not replay.exists(), "deleted target exists")
            else:
                identity(actual, target["patched"])
                identity(replay, target["patched"])
                changed = (target.get("base") or {}).get("sha256") != target["patched"]["sha256"]
                check(changed == target["changed"], "target changed flag differs")
        results.append({"component": name, "base_revision": component["base_revision"],
                        "archive_sha256": declared_sha, "patch_count": len(patch_names),
                        "replayed_targets": len(targets),
                        "patches": [{"name": Path(p["path"]).name, "sha256": p["sha256"]} for p in component["patches"]]})
    return results


def verify(capture):
    manifest = read_json(capture / "manifest.json")
    count = inventory(capture, manifest)
    summary = read_json(capture / "summary.json")
    check(summary["outcome"] == "pass" and summary["evidence_level"] == "SOURCE-ONLY", "invalid summary")
    components = verify_components(capture, summary)
    cache = read_json(capture / "repository-cache.json")
    seen = set()
    for record in cache["verified_objects"]:
        check(record["sha256"] not in seen, "duplicate cached object")
        seen.add(record["sha256"])
        path = local_path(record["path"])
        check(path.parent.name == record["sha256"], "CAS key differs")
        identity(path, record)
    totals = {"verified_objects": len(seen), "verified_bytes": sum(r["size_bytes"] for r in cache["verified_objects"]), "incomplete_entries": cache["incomplete_entries"]}
    check(totals == summary["repository_cache"], "cache totals differ")
    overlay = summary["llvm_overlay"]
    sample = local_path(str(Path(overlay["path"]) / overlay["sample"]))
    check(sample == local_path(overlay["resolved_sample"]) and file_hash(sample) == overlay["sha256"], "LLVM overlay differs")

    receipt_path = ROOT / "artifacts/jax-stack/source-build-002/compiler-processes.json"
    preflight_path = ROOT / "artifacts/jax-stack/build-preflight-001/container.json"
    receipt, preflight = read_json(receipt_path), read_json(preflight_path)
    launch = read_json(ROOT / "artifacts/jax-stack/source-build-002/launch.json")
    check(receipt["container_id"] == launch["container_id"], "compiler receipt container differs")
    check(preflight["ok"] and receipt["compiler_processes"], "missing compiler observation")
    check(all(p["sha256"] == preflight["clang"]["sha256"] for p in receipt["compiler_processes"]), "live Clang differs from preflight")
    return {"capture_id": capture.name, "manifest_path": str((capture / "manifest.json").relative_to(ROOT)),
            "manifest_sha256": file_hash(capture / "manifest.json"), "artifact_count": count,
            "evidence_level": "SOURCE-ONLY", "build_id": summary["build_id"], "components": components,
            "patched_targets": sum(c["replayed_targets"] for c in components), "repository_cache": totals,
            "compiler_observation": {"observed_at": receipt["observed_at"], "process_count": len(receipt["compiler_processes"]),
                "clang_sha256": preflight["clang"]["sha256"], "clang_version": preflight["clang"]["parsed_version"],
                "receipt_path": str(receipt_path.relative_to(ROOT)), "receipt_sha256": file_hash(receipt_path),
                "preflight_path": str(preflight_path.relative_to(ROOT)), "preflight_sha256": file_hash(preflight_path)},
            "validation_scope": "Rechecked immutable capture inventory, pinned declarations/patches, archive/cache bytes and replay-to-live patch targets. The producer performed the patch replay; this validator does not rerun git apply.",
            "limits": summary["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/build-dependencies-002")
    args = parser.parse_args()
    capture = ROOT / "artifacts/jax-stack/build-dependencies-002"
    if args.selftest:
        original = read_json(capture / "manifest.json")
        for mode in ["hash", "missing", "level"]:
            fake = copy.deepcopy(original)
            if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
            elif mode == "missing": fake["artifacts"].pop()
            else: fake["evidence_level"] = "RUN-TPU"
            try: inventory(capture, fake)
            except ValueError: pass
            else: raise AssertionError(f"Accepted invalid manifest: {mode}")
        for mode in ["pin", "archive", "target"]:
            fake = copy.deepcopy(read_json(capture / "summary.json"))
            if mode == "pin": fake["components"][0]["base_revision"] = "0" * 40
            elif mode == "archive": fake["components"][0]["archive"]["sha256"] = "0" * 64
            else: fake["components"][0]["targets"][0]["patched"]["sha256"] = "0" * 64
            try: verify_components(capture, fake)
            except ValueError: pass
            else: raise AssertionError(f"Accepted invalid component: {mode}")
        print("selftest: 6 invalid evidence/identity records rejected")
    result = verify(args.capture.resolve())
    if args.write:
        (HERE / "build-dependency-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("capture_id", "artifact_count", "patched_targets", "repository_cache")}))


if __name__ == "__main__":
    main()
