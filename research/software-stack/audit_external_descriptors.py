#!/usr/bin/env python3
"""Snapshot cached Bazel descriptors and identify the configured Python toolchain."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

from matmul_probe import ROOT, write_json
from audit_build_dependencies import file_hash

CACHE = ROOT / "artifacts/builds/kickoff-cpu-bazel-cache"
EXTERNAL = CACHE / "777193f2a9867a2e91177d2ba2b775c1/external"
PYTHON_REPO = "rules_python++python+python_3_12_x86_64-unknown-linux-gnu"
BUILD = ROOT / "manifests/build-fingerprints/kickoff-cpu-pass-events-002.json"
DESCRIPTORS = ("MODULE.bazel", "REPO.bazel", "WORKSPACE", "WORKSPACE.bazel", "BUILD", "BUILD.bazel")


def fp(path):
    return {"sha256": file_hash(path), "size_bytes": path.stat().st_size}


def copy_record(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert fp(source) == fp(target)
    return {"snapshot": str(target.relative_to(ROOT)), **fp(target)}


def collect(output):
    build = json.loads(BUILD.read_text())
    assert build["status"] == "build-succeeded"
    repositories = []
    for path in sorted(EXTERNAL.iterdir()):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        assert resolved.is_relative_to(ROOT), path
        descriptors = {name: copy_record(path / name, output / "descriptors" / path.name / name)
                       for name in DESCRIPTORS if (path / name).is_file()}
        module = re.search(r"\bmodule\((.*?)\)", (path / "MODULE.bazel").read_text(), re.S) if (path / "MODULE.bazel").is_file() else None
        def literal(key):
            found = re.search(r'\b' + key + r'\s*=\s*"([^\"]+)"', module[1]) if module else None
            return found[1] if found else None
        repositories.append({"repository": path.name, "symlink": path.is_symlink(),
                             "resolved_on_host": str(resolved.relative_to(ROOT)),
                             "declared_module": literal("name"), "declared_version": literal("version"),
                             "used_by_build": None, "descriptors": descriptors})
    write_json(output / "repositories.json", repositories)
    selected = {}
    for name, source in {
        "minor_mapping": EXTERNAL / "rules_python++python+pythons_hub/versions.bzl",
        "python_selection": EXTERNAL / "rules_ml_toolchain++python_version_ext+python_version_repo/py_version.bzl",
        "python_build": EXTERNAL / PYTHON_REPO / "BUILD.bazel",
        "runtime_manifest": EXTERNAL / "rules_python+/python/private/runtimes_manifest_workspace.bzl",
        "runtime_rule": EXTERNAL / "rules_python+/python/private/python_repository.bzl",
        "version_rules": EXTERNAL / "rules_python+/python/versions.bzl",
        "jax_module": ROOT / "upstream/jax/MODULE.bazel",
        "xla_module": ROOT / "upstream/xla/MODULE.bazel",
        "build_config": ROOT / build["result"]["generated_bazelrc"]["path"],
        "build_manifest": BUILD,
        "build_launch": ROOT / "artifacts/jax-stack/pass-event-build-002/wheel-build-launch.json",
    }.items():
        selected[name] = copy_record(source, output / "selection" / (name + source.suffix))
    archive_match = re.search(r"^([0-9a-f]{64})  (20260414/cpython-3\.12\.13\+20260414-x86_64-unknown-linux-gnu-install_only\.tar\.gz)$",
                              (ROOT / selected["runtime_manifest"]["snapshot"]).read_text(), re.M)
    assert archive_match
    archive = CACHE / "cache/repos/v1/content_addressable/sha256" / archive_match[1] / "file"
    assert file_hash(archive) == archive_match[1]
    members = []
    with tarfile.open(archive, "r:gz") as stream:
        for name in ("bin/python3.12", "lib/libpython3.12.so.1.0", "include/python3.12/patchlevel.h"):
            member = stream.getmember("python/" + name)
            assert member.isfile()
            digest = hashlib.sha256()
            with stream.extractfile(member) as data:
                while block := data.read(1024 * 1024):
                    digest.update(block)
            actual = EXTERNAL / PYTHON_REPO / name
            value = fp(actual)
            assert value == {"sha256": digest.hexdigest(), "size_bytes": member.size}
            members.append({"member": member.name, "actual_path": str(actual.relative_to(ROOT)), **value})
    command = [str(EXTERNAL / PYTHON_REPO / "bin/python3.12"), "-I", "-S", "-B", "-c",
               "import json,sys,sysconfig;print(json.dumps({'version':sys.version,'version_info':list(sys.version_info),'executable':sys.executable,'prefix':sys.prefix,'soabi':sysconfig.get_config_var('SOABI')}))"]
    process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    write_json(output / "interpreter-command.json", {"argv": command, "returncode": process.returncode,
                                                     "stdout": process.stdout, "stderr": process.stderr})
    process.check_returncode()
    runtime = json.loads(process.stdout)
    assert runtime["version_info"][:3] == [3, 12, 13]
    write_json(output / "interpreter.json", runtime)
    return {"outcome": "pass", "evidence_level": "SOURCE-ONLY", "build_id": build["build_id"],
            "build_manifest": {"path": str(BUILD.relative_to(ROOT)), **fp(BUILD)},
            "repository_count": len(repositories), "descriptor_count": sum(len(r["descriptors"]) for r in repositories),
            "cache_presence_proves_build_use": False, "complete_action_input_closure": False,
            "selection": selected, "python": {"requested_minor": "3.12", "configured_version": "3.12.13",
                "archive": {"path": str(archive.relative_to(ROOT)), **fp(archive)},
                "source_url": "https://github.com/astral-sh/python-build-standalone/releases/download/" + archive_match[2],
                "matched_archive_members": members, "local_interpreter_executed": True,
                "per_build_action_process_observed": False},
            "limits": ["Post-build host-resolved cache snapshot; repository presence and module declarations do not establish use by the wheel target.",
                       "Build-time container mounts are recorded separately; host symlink resolution is not the build's physical source mount mapping.",
                       "Python selection, archive identity and three member bytes are checked; no complete action input graph, offline rebuild or JAX workload is executed by this audit."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        parser.error("Use a new repository-local artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    summary = collect(output)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"outcome": "pass", "evidence_level": "SOURCE-ONLY",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv],
                     "source": str((output / "producer.py").relative_to(ROOT)), **fp(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fp(p)}
                      for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({k: summary[k] for k in ("repository_count", "descriptor_count", "python")}))
