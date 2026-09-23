#!/usr/bin/env python3
"""Audit the retained inherited-wheel environment; not source-build acceptance."""

import argparse
import copy
import json
import os
from pathlib import Path

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import require
from verify_pass_events import verify_pair

HERE = Path(__file__).resolve().parent
CAPTURE = ROOT / "artifacts/jax-stack/runtime-environment-001"


def check_metadata(index):
    require(set(index) == {p.name for p in CAPTURE.iterdir() if p.is_file()}, "metadata inventory differs")
    for name, expected in index.items():
        require(Path(name).name == name and fingerprint(CAPTURE / name) == expected, "metadata hash or path differs")


def audit():
    before = json.loads((CAPTURE / "dependency-files-before.json").read_text())
    after = json.loads((CAPTURE / "dependency-files-after.json").read_text())
    require(before == after, "before/after dependency inventories differ")
    source = ROOT / ".venv/lib/python3.12/site-packages"
    packages = CAPTURE / "venv/lib/python3.12/site-packages"
    for directory in [source, packages]:
        actual = set()
        for p in directory.rglob("*"):
            require(not p.is_symlink(), "package symlink found")
            if p.is_file() and "__pycache__" not in p.parts:
                actual.add(str(p.relative_to(directory)))
        require(actual == set(before), "live dependency inventory differs")
    for name, expected in before.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "dependency path escapes")
        for base in [source, packages]:
            require(fingerprint(base / name) == expected, "live dependency bytes differ")
        require(not os.path.samefile(source / name, packages / name), "dependency inode shared with host")
    python = CAPTURE / "venv/bin/python"
    require(not os.path.samefile(python, ROOT / ".venv/bin/python"), "Python inode shared with host")
    require(fingerprint(python) == fingerprint(ROOT / ".venv/bin/python"), "copied Python bytes differ")
    identity = json.loads((CAPTURE / "interpreter.json").read_text())
    require(Path(identity["prefix"]) == CAPTURE / "venv" and Path(identity["executable"]) == python, "interpreter identity differs")
    require(identity["user_site_enabled"] is False, "user site enabled")
    require(all(not Path(p).resolve().is_relative_to(ROOT / ".venv") for p in identity["sys_path"] if p), "host venv leaked into sys.path")
    for package in ["jaxlib", "numpy"]:
        require(Path(identity["imports"][package]).resolve().is_relative_to(packages), "package import escaped")
    require(Path(identity["imports"]["jax"]).resolve().is_relative_to(ROOT / "upstream/jax"), "editable JAX source differs")
    pair = verify_pair(CAPTURE / "default", CAPTURE / "filtered")
    require(not pair["patched_runtime_pair_accepted"], "inherited wheel promoted to patched runtime")
    preparation = json.loads((CAPTURE / "preparation.json").read_text())
    require(preparation["mode"] == "inherited-wheel-negative-control" and preparation["install"] is None, "unexpected bootstrap mode")
    require(preparation["copied_dependency_files"] == len(before), "dependency count differs")
    require(preparation["copied_dependency_bytes"] == sum(v["size_bytes"] for v in before.values()), "dependency bytes differ")
    rejections = []
    for kind, message in [("missing", "present events require a successful patched build manifest"),
                          ("running", "build manifest is not a successful real build")]:
        rejected = ROOT / f"artifacts/jax-stack/runtime-environment-{kind}-build-rejected-001"
        failure = json.loads((rejected / "failure.json").read_text())
        require(failure["runtime_accepted"] is False and failure["message"] == message, "wrong entry rejection")
        require(not (rejected / "venv").exists() and not (rejected / "default").exists(), "rejected entry created a runtime")
        if kind == "running":
            snapshot = json.loads((rejected / "build-gate-observation.input.json").read_text())
            observation = json.loads((rejected / "build-gate-observation.json").read_text())
            require(snapshot["status"] == observation["status_at_read"] == "running", "rejected build state differs")
            require(fingerprint(rejected / "build-gate-observation.input.json")["sha256"] == observation["snapshot_sha256"], "rejected input snapshot differs")
        rejections.append({"capture": str(rejected.relative_to(ROOT)), "evidence_level": "SOURCE-ONLY",
            "runtime_created": False, "error": message,
            "files": {p.name: fingerprint(p) for p in sorted(rejected.iterdir()) if p.is_file()}})
    return {"capture": str(CAPTURE.relative_to(ROOT)), "evidence_level": "RUN-CPU", "qualifiers": pair["default"]["qualifiers"],
        "mode": preparation["mode"], "metadata_files": {p.name: fingerprint(p) for p in sorted(CAPTURE.iterdir()) if p.is_file()},
        "copied_dependency_files": len(before), "copied_dependency_bytes": preparation["copied_dependency_bytes"],
        "python_executable_sha256": fingerprint(python)["sha256"], "no_shared_dependency_inodes": True,
        "host_package_files_unchanged": True, "runtime_sys_path_excludes_host_venv": True,
        "pair": {k: {"manifest_sha256": pair[k]["manifest_sha256"], "native_binding": pair[k]["build_binding"]["kind"],
                     "generic_known_pass_counts": pair[k]["observations"]["generic_known_pass_counts"],
                     "custom_events": pair[k]["observations"]["custom_event_count"], "max_absolute_errors": pair[k]["max_absolute_errors"]}
                 for k in ["default", "filtered"]},
        "source_built_wheel_installed": False, "patched_runtime_accepted": False, "entry_rejections": rejections,
        "limits": [preparation["boundary"], "The source-built wheel installation branch has not yet run; this environment retains the old wheel."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    path = HERE / "runtime-environment-results.json"
    saved = json.loads(path.read_text())
    check_metadata(saved["metadata_files"])
    if args.selftest:
        for mode in ["hash", "missing", "escape"]:
            index = copy.deepcopy(saved["metadata_files"])
            name = next(iter(index))
            if mode == "hash": index[name]["sha256"] = "0" * 64
            elif mode == "missing": del index[name]
            else: index["../escaped"] = index.pop(name)
            try: check_metadata(index)
            except ValueError: pass
            else: raise AssertionError(f"invalid metadata accepted: {mode}")
        print("selftest: 3 invalid metadata inventories rejected without changing evidence")
    observed = audit()
    if args.write: write_json(path, observed)
    else: require(saved == observed, "published environment observations differ")
    print(json.dumps({"files": observed["copied_dependency_files"], "bytes": observed["copied_dependency_bytes"],
                      "source_built_wheel_installed": False, "patched_runtime_accepted": False}))


if __name__ == "__main__":
    main()
