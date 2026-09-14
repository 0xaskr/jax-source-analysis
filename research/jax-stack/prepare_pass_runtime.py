#!/usr/bin/env python3
"""Create a separate CPU runtime and exercise pass-event build/native gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import require
from pass_events_probe import build_gate, load_baseline_module
from verify_pass_events import verify_pair

HERE = Path(__file__).resolve().parent


def inventory(directory):
    records = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"package tree contains a symlink: {path}")
        if path.is_file() and "__pycache__" not in path.parts:
            records[str(path.relative_to(directory))] = fingerprint(path)
    return records


def child_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("UV_", "PYTHON", "JAX_")) and k != "VIRTUAL_ENV"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu")
    return env


def run(command, output, name, env):
    command = [str(v) for v in command]
    write_json(output / f"{name}.command.json", command)
    with (output / f"{name}.log").open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"{name} failed with exit {result.returncode}; inspect its log")


def prepare(output, build_path, expected):
    # Reject incomplete builds before creating or changing an environment.
    manifest = build_gate(build_path, expected, output / "build-gate-observation.json")
    if manifest is not None:
        manifest = load_baseline_module()._load_build_validator().load_and_validate_manifest(build_path)
    source = ROOT / ".venv/lib/python3.12/site-packages"
    before = inventory(source)
    write_json(output / "dependency-files-before.json", before)
    environment = output / "venv"
    env = child_environment()
    run([ROOT / ".venv/bin/python", "-B", "-m", "venv", "--without-pip", "--copies", environment], output, "create-venv", env)
    packages = environment / "lib/python3.12/site-packages"
    require(packages.is_dir(), "unexpected venv package path")
    shutil.copytree(source, packages, dirs_exist_ok=True, copy_function=shutil.copy2,
                    ignore=shutil.ignore_patterns("__pycache__"))
    require(inventory(packages) == before, "copied dependency bytes differ")
    for name in before:
        require(not os.path.samefile(source / name, packages / name), "dependency copy shares an inode with host")
    python = environment / "bin/python"
    require(not python.is_symlink() and not os.path.samefile(python, ROOT / ".venv/bin/python"), "Python was not copied")
    install = None
    if manifest is not None:
        uv = Path(shutil.which("uv")).resolve()
        baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
        pinned_uv = baseline["toolchain"]["uv"]
        require(fingerprint(uv) == {"sha256": pinned_uv["artifact_sha256"], "size_bytes": pinned_uv["artifact_size_bytes"]}, "installer identity differs")
        wheel = manifest["result"]["wheels"][0]
        run([uv, "--offline", "--no-config", "pip", "install", "--python", python,
             "--no-index", "--no-deps", "--no-build", "--no-cache", "--link-mode=copy",
             "--reinstall", ROOT / wheel["path"]], output, "install-wheel", env)
        install = {"build_id": manifest["build_id"], "build_manifest": fingerprint(build_path),
                   "wheel": wheel, "uv": fingerprint(uv)}
    identity_code = """import json, sys, site, jax, jaxlib, numpy
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'sys_path':sys.path,'user_site_enabled':site.ENABLE_USER_SITE,
 'imports':{'jax':jax.__file__,'jaxlib':jaxlib.__file__,'numpy':numpy.__file__}}))"""
    run([python, "-B", "-c", identity_code], output, "interpreter", env)
    identity = json.loads((output / "interpreter.log").read_text())
    require(Path(identity["prefix"]).resolve() == environment, "runtime selected another venv")
    require(identity["user_site_enabled"] is False, "user site packages are enabled")
    require(all(not Path(p).resolve().is_relative_to(ROOT / ".venv") for p in identity["sys_path"] if p), "host venv leaked into sys.path")
    for package in ["jaxlib", "numpy"]:
        require(Path(identity["imports"][package]).resolve().is_relative_to(packages), "import escaped copied packages")
    require(Path(identity["imports"]["jax"]).resolve().is_relative_to(ROOT / "upstream/jax"), "JAX is not the pinned editable checkout")
    write_json(output / "interpreter.json", identity)
    for name, extra in [("default", []), ("filtered", ["--disable-pass", "algsimp"])]:
        command = [python, "-B", HERE / "pass_events_probe.py", "--output", output / name,
                   "--expected-events", expected, *extra]
        if build_path is not None:
            command += ["--jaxlib-build-manifest", build_path]
        run(command, output, f"probe-{name}", env)
    result = verify_pair(output / "default", output / "filtered")
    for name in ["default", "filtered"]:
        captured = json.loads((output / name / "environment.json").read_text())
        for binary in captured["runtime"]["jaxlib"]["native_binaries"]:
            require((ROOT / binary["artifact_path"]).resolve().is_relative_to(packages), "loaded native file escaped runtime venv")
    after = inventory(source)
    require(before == after, "host package files changed")
    write_json(output / "dependency-files-after.json", after)
    write_json(output / "pair-results.json", result)
    write_json(output / "preparation.json", {
        "outcome": "pass", "evidence_level": "RUN-CPU", "qualifiers": result["default"]["qualifiers"],
        "mode": "source-built-wheel" if install else "inherited-wheel-negative-control",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "venv": str(environment.relative_to(ROOT)), "python": str(python.relative_to(ROOT)),
        "copied_dependency_files": len(before), "copied_dependency_bytes": sum(v["size_bytes"] for v in before.values()),
        "dependency_inventory": fingerprint(output / "dependency-files-before.json"),
        "host_package_files_unchanged": True, "no_shared_dependency_inodes": True,
        "install": install, "patched_runtime_pair_accepted": result["patched_runtime_pair_accepted"],
        "boundary": "Dependency/Python files are copied; OS libraries, standard library and the pinned editable JAX checkout are shared. This is not a hermetic system or TPU runtime."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jaxlib-build-manifest", type=Path)
    parser.add_argument("--expected-events", choices=["absent", "present"], default="absent")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or output == ROOT / "artifacts/jax-stack" or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack subdirectory and unset ambient XLA_FLAGS")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    path = args.jaxlib_build_manifest.resolve() if args.jaxlib_build_manifest else None
    try:
        result = prepare(output, path, args.expected_events)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "outcome": "fail", "runtime_accepted": False})
        traceback.print_exc()
        return 1
    print(json.dumps({"output": str(output.relative_to(ROOT)),
                      "patched_runtime_pair_accepted": result["patched_runtime_pair_accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
