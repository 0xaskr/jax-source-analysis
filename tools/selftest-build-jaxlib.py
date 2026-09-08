#!/usr/bin/env python3
"""Isolated contract tests for the jaxlib build wrapper; never runs build.py/Bazel."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPOSITORY_ROOT / "tools/build-jaxlib.py"
PREFLIGHT_PATH = REPOSITORY_ROOT / "tools/check-jaxlib-build-env.py"
SCHEMA_SOURCE = REPOSITORY_ROOT / "manifests/schema/build-jaxlib.schema.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load(WRAPPER_PATH, "jax_source_analysis_build_selftest")
CHECK = _load(PREFLIGHT_PATH, "jax_source_analysis_preflight_selftest")


def _run_git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["/usr/bin/git", "-c", "core.fsmonitor=false", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        },
    )
    return process.stdout.strip()


def _make_repository(path: Path, files: dict[str, bytes]) -> str:
    path.mkdir(parents=True)
    _run_git(path, "init", "-q")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _run_git(path, "add", ".")
    _run_git(
        path,
        "-c",
        "user.name=Build Selftest",
        "-c",
        "user.email=build-selftest@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return _run_git(path, "rev-parse", "HEAD")


def _make_analysis_repository(path: Path) -> str:
    _run_git(path, "init", "-q")
    (path / ".gitignore").write_text(
        ".venv/\nartifacts/\npatches/\ntoolchain/\nupstream/\n"
        "manifests/build-fingerprints/\n"
    )
    _run_git(path, "add", ".")
    _run_git(
        path,
        "-c",
        "user.name=Build Selftest",
        "-c",
        "user.email=build-selftest@example.invalid",
        "commit",
        "-q",
        "-m",
        "analysis fixture",
    )
    return _run_git(path, "rev-parse", "HEAD")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure(root: Path, commits: dict[str, str]) -> None:
    BUILD.ROOT = root
    BUILD.JAX_ROOT = root / "upstream/jax"
    BUILD.SOURCE_ROOTS = {
        "jax": BUILD.JAX_ROOT,
        "xla": root / "upstream/xla",
    }
    BUILD.SCHEMA_PATH = root / "manifests/schema/build-jaxlib.schema.json"
    BUILD.DEFAULT_MANIFEST_ROOT = root / "manifests/build-fingerprints"
    BUILD.DEFAULT_ARTIFACT_ROOT = root / "artifacts/builds"
    BUILD.GLOBAL_LOCK_PATH = BUILD.DEFAULT_ARTIFACT_ROOT / ".jaxlib-build.lock"
    BUILD.EXPECTED_PYTHON_PATH = root / ".venv/bin/python"
    BUILD.GENERATED_BAZELRC = BUILD.JAX_ROOT / ".jax_configure.bazelrc"
    BUILD.EXPECTED_SOURCE_COMMITS = commits
    BUILD.EXPECTED_TOOL_PATHS = {
        "bazel": BUILD.JAX_ROOT / "bazel-8.7.0-linux-x86_64",
        "clang": root / "toolchain/clang",
        "clangxx": root / "toolchain/clang++",
        "git": Path("/usr/bin/git"),
    }
    BUILD.EXPECTED_BAZEL_SHA256 = _sha256(BUILD.EXPECTED_TOOL_PATHS["bazel"])

    CHECK.ROOT = root
    CHECK.SOURCE_PATHS = BUILD.SOURCE_ROOTS
    CHECK.EXPECTED_SOURCE_COMMITS = commits
    CHECK.SOURCE_MARKERS = {"jax": "build/build.py", "xla": "xla"}


def _write_wheel(
    path: Path, *, metadata_version: str | None = None,
    metadata_name: str = "jaxlib", include_native: bool = True,
) -> None:
    version = str(BUILD.EXPECTED_JAXLIB_VERSION)
    dist_info = f"jaxlib-{version}.dist-info"
    members = {
        "jaxlib/__init__.py": b"from .version import __version__\n",
        "jaxlib/version.py": f"__version__ = {version!r}\n".encode(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {metadata_name}\n"
            f"Version: {metadata_version or version}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: selftest\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {BUILD.EXPECTED_WHEEL_TAG}\n\n"
        ).encode(),
    }
    if include_native:
        elf_header = bytearray(64)
        elf_header[:6] = b"\x7fELF\x02\x01"
        elf_header[18:20] = (62).to_bytes(2, "little")
        for name in BUILD.REQUIRED_JAXLIB_NATIVE_MEMBERS:
            members[name] = bytes(elf_header)
    rows: list[list[str]] = []
    for name, content in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append([name, f"sha256={digest}", str(len(content))])
    record_name = f"{dist_info}/RECORD"
    rows.append([record_name, "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    members[record_name] = output.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _tool(path: Path, version: str) -> dict[str, Any]:
    return {
        "path": BUILD._repo_path(path) if path.is_relative_to(BUILD.ROOT) else str(path),
        "resolved_path": str(path.resolve()),
        "version": version,
        "returncode": 0,
        "sha256": _sha256(path),
    }


def _fixture(root: Path) -> tuple[dict[str, Any], Path, Path]:
    schema = root / "manifests/schema/build-jaxlib.schema.json"
    schema.parent.mkdir(parents=True)
    shutil.copy2(SCHEMA_SOURCE, schema)
    for relative in (
        ".python-version",
        "pyproject.toml",
        "tools/build-jaxlib.py",
        "tools/check-jaxlib-build-env.py",
        "upstream-sources.lock",
        "uv.lock",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("3.12.3\n" if relative == ".python-version" else f"fixture {relative}\n")
    jax = root / "upstream/jax"
    xla = root / "upstream/xla"
    commits = {
        "jax": _make_repository(
            jax,
            {
                ".gitignore": b"bazel-*\n*.pyc\n*.egg-info\n.jax_configure.bazelrc\nMODULE.bazel.lock\n",
                ".bazelrc": b"build --color=no\n",
                ".bazelversion": b"8.7.0\n",
                "build/build.py": b"raise RuntimeError('selftest must never execute build.py')\n",
            },
        ),
        "xla": _make_repository(xla, {"xla/marker.txt": b"fixture\n"}),
    }
    (root / ".venv/bin").mkdir(parents=True)
    (root / "toolchain").mkdir()
    (root / ".venv/bin/python").symlink_to(BUILD.EXPECTED_BASE_PYTHON_PATH)
    (root / "toolchain/clang").write_bytes(b"fixture clang\n")
    (root / "toolchain/clang++").write_bytes(b"fixture clang++\n")
    (jax / "bazel-8.7.0-linux-x86_64").write_bytes(b"fixture bazel\n")
    _configure(root, commits)
    _make_analysis_repository(root)

    sources = {name: BUILD._source_observation(name) for name in ("jax", "xla")}
    tools = {
        "python": {
            "executable": str(BUILD.EXPECTED_PYTHON_PATH),
            "resolved_executable": str(BUILD.EXPECTED_BASE_PYTHON_PATH),
            "base_executable": str(BUILD.EXPECTED_BASE_PYTHON_PATH),
            "base_resolved_path": str(BUILD.EXPECTED_BASE_PYTHON_PATH),
            "base_size_bytes": BUILD.EXPECTED_BASE_PYTHON_SIZE_BYTES,
            "base_sha256": BUILD.EXPECTED_BASE_PYTHON_SHA256,
            "version": "3.12.3",
        },
        "bazel": _tool(BUILD.EXPECTED_TOOL_PATHS["bazel"], BUILD.EXPECTED_BAZEL_VERSION),
        "clang": _tool(BUILD.EXPECTED_TOOL_PATHS["clang"], "Ubuntu clang version 18.1.3"),
        "clangxx": _tool(BUILD.EXPECTED_TOOL_PATHS["clangxx"], "Ubuntu clang version 18.1.3"),
        "git": {
            "path": "/usr/bin/git",
            "resolved_path": str(Path("/usr/bin/git").resolve()),
            "version": BUILD.EXPECTED_GIT_VERSION,
            "returncode": 0,
            "sha256": BUILD.EXPECTED_GIT_SHA256,
        },
    }
    preflight_report = {"python": tools["python"], "sources": sources}
    config = BUILD._build_config(preflight_report, 4, [], [], False, {})
    preflight = {"ok": True, "blockers": [], "waived_blockers": []}
    inputs, fingerprint = BUILD._build_inputs(
        {"python": tools["python"], "sources": sources,
         "bazel": tools["bazel"], "clang": tools["clang"],
         "clangxx": tools["clangxx"], "git": tools["git"]},
        config,
        "fixture",
        [],
    )
    manifest_path = root / "manifests/build-fingerprints/fixture.json"
    artifact_dir = root / "artifacts/builds/fixture"
    manifest = BUILD._new_manifest(
        build_id="fixture",
        config=config,
        inputs=inputs,
        input_fingerprint=fingerprint,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        build_lock_path=artifact_dir / ".build-id.lock",
        preflight=preflight,
    )
    attempt_dir = artifact_dir / "attempts/0001"
    log = attempt_dir / "build.log"
    generated = attempt_dir / "generated.jax_configure.bazelrc"
    wheel = attempt_dir / (
        "wheels/jaxlib-0.11.2.dev0+selfbuilt-"
        "cp312-cp312-manylinux_2_27_x86_64.whl"
    )
    log.parent.mkdir(parents=True)
    log.write_text("[2026-09-08T00:00:01Z] pinned build command completed\n")
    generated.write_text("build --action_env=CC=/usr/bin/clang\n")
    _write_wheel(wheel)
    identity = {
        "pid": 123,
        "boot_id": "00000000-0000-0000-0000-000000000001",
        "start_time_ticks": 1,
        "cmdline_sha256": "0" * 64,
    }
    attempt = {
        "number": 1,
        "status": "build-succeeded",
        "started_at": "2026-09-08T00:00:01Z",
        "ended_at": "2026-09-08T00:00:02Z",
        "paths": {
            "attempt_dir": BUILD._repo_path(attempt_dir),
            "log": BUILD._repo_path(log),
            "wheel_dir": BUILD._repo_path(wheel.parent),
            "generated_bazelrc_archive": BUILD._repo_path(generated),
            "home_dir": BUILD._repo_path(attempt_dir / "home"),
            "tmp_dir": BUILD._repo_path(attempt_dir / "tmp"),
        },
        "command": {
            "cwd": "upstream/jax",
            "argv": BUILD._build_command(attempt_dir, config),
            "shell": "",
        },
        "environment": BUILD._attempt_environment_record(attempt_dir, config),
        "wrapper_process": identity,
        "child_process": identity,
        "exit_code": 0,
        "error": None,
        "log_artifact": BUILD._artifact_record(log, attempt_dir),
        "wheels": [BUILD._artifact_record(wheel, wheel.parent)],
        "generated_bazelrc": BUILD._artifact_record(generated, attempt_dir),
    }
    import shlex
    attempt["command"]["shell"] = shlex.join(attempt["command"]["argv"])
    manifest.update({
        "created_at": "2026-09-08T00:00:00Z",
        "updated_at": attempt["ended_at"],
        "status": "build-succeeded",
        "attempts": [attempt],
        "result": {
            "attempt_number": 1,
            "exit_code": 0,
            "error": None,
            "wheels": attempt["wheels"],
            "generated_bazelrc": attempt["generated_bazelrc"],
        },
    })
    manifest_path.parent.mkdir(parents=True)
    return manifest, manifest_path, wheel


def _expect_rejected(name: str, function: Callable[[], None], expected: str) -> None:
    try:
        function()
    except (BUILD.BuildError, OSError) as error:
        if expected not in str(error):
            raise AssertionError(f"{name}: expected {expected!r}, got {error}") from error
        print(f"OK: rejected {name}")
        return
    raise AssertionError(f"{name}: unexpectedly accepted")


def _validate(value: dict[str, Any], path: Path) -> None:
    BUILD._validate_manifest(value, path, verify_artifacts=True)


def _preflight_from_manifest(
    manifest: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tools = manifest["inputs"]["toolchain"]
    return {
        "python": tools["python"],
        "bazel": tools["bazel"],
        "clang": tools["clang"],
        "clangxx": tools["clangxx"],
        "git": tools["git"],
        "sources": sources,
    }


def _patched_fixture(root: Path) -> tuple[dict[str, Any], Path]:
    manifest, path, _ = _fixture(root)
    source = root / "upstream/jax"
    (source / ".bazelrc").write_text("build --color=yes\n")
    patch_path = root / "patches/jax-selftest.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_bytes(BUILD._source_diff(source))
    sources = {
        component: BUILD._source_observation(component)
        for component in ("jax", "xla")
    }
    patch_records = BUILD._validate_source_patches(
        {"jax": "patches/jax-selftest.patch"}, {"sources": sources}
    )
    inputs, fingerprint = BUILD._build_inputs(
        _preflight_from_manifest(manifest, sources),
        manifest["build_config"],
        manifest["build_id"],
        patch_records,
    )
    patch = patch_records[0]
    manifest["inputs"] = inputs
    manifest["input_fingerprint_sha256"] = fingerprint
    manifest["preflight"] = {
        "ok": True,
        "blockers": [],
        "waived_blockers": [{
            "code": "source-dirty:jax",
            "message": "selftest controlled source patch",
            "source_patch_sha256": patch["sha256"],
        }],
    }
    return manifest, path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="build-jaxlib-selftest-") as temporary:
        temporary_root = Path(temporary)

        root = temporary_root / "positive"
        manifest, path, _ = _fixture(root)
        _validate(manifest, path)
        serialized = json.dumps(manifest, sort_keys=True)
        assert str(root) not in serialized
        assert "/home/" not in serialized
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        BUILD.load_and_validate_manifest(path)
        print("OK: accepted valid isolated success manifest through public verifier")

        tampered = copy.deepcopy(manifest)
        tampered["inputs"]["files"]["pyproject.toml"]["sha256"] = "f" * 64
        tampered["input_fingerprint_sha256"] = BUILD._sha256_bytes(
            BUILD._canonical_bytes(BUILD._fingerprint_payload(tampered["inputs"]))
        )
        _expect_rejected(
            "self-consistent historical input-byte forgery",
            lambda: _validate(tampered, path),
            "recorded input file does not match committed reconstruction",
        )

        command = manifest["attempts"][0]["command"]["argv"]
        assert command[1] == "-S"
        assert manifest["attempts"][0]["environment"]["fixed"][
            "PYTHONDONTWRITEBYTECODE"
        ] == "1"
        customize = root / "site-fixture"
        customize.mkdir()
        marker = root / "sitecustomize-ran"
        (customize / "sitecustomize.py").write_text(
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\n"
        )
        subprocess.run(
            [str(BUILD.EXPECTED_PYTHON_PATH), "-S", "-c", "pass"],
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(customize)},
        )
        assert not marker.exists()
        print("OK: pinned -S child bypasses venv/PYTHONPATH site customization")

        root = temporary_root / "generated-transition"
        manifest, path, _ = _fixture(root)
        original_fingerprint = manifest["input_fingerprint_sha256"]
        (root / "upstream/jax/.jax_configure.bazelrc").write_text(
            "build --action_env=CC=/usr/bin/clang\n"
        )
        transitioned_sources = {
            component: BUILD._source_observation(component)
            for component in ("jax", "xla")
        }
        _, transitioned_fingerprint = BUILD._build_inputs(
            _preflight_from_manifest(manifest, transitioned_sources),
            manifest["build_config"],
            manifest["build_id"],
            [],
            manifest["inputs"]["files"],
        )
        assert transitioned_fingerprint == original_fingerprint
        _validate(manifest, path)
        print("OK: accepted controlled generated ignored-path transition")
        info_exclude = root / "upstream/jax/.git/info/exclude"
        info_exclude.write_text(info_exclude.read_text() + "*.unsafe\n")
        (root / "upstream/jax/late.unsafe").write_text("untrusted build input\n")
        unsafe_sources = {
            component: BUILD._source_observation(component)
            for component in ("jax", "xla")
        }
        _, unsafe_fingerprint = BUILD._build_inputs(
            _preflight_from_manifest(manifest, unsafe_sources),
            manifest["build_config"],
            manifest["build_id"],
            [],
            manifest["inputs"]["files"],
        )
        assert unsafe_fingerprint != original_fingerprint
        _expect_rejected(
            "non-allowlisted ignored-path transition",
            lambda: _validate(manifest, path),
            "differs from the current tree",
        )

        root = temporary_root / "patch-rollback"
        manifest, path = _patched_fixture(root)
        _validate(manifest, path)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        _run_git(root / "upstream/jax", "checkout", "--", ".bazelrc")
        BUILD.load_and_validate_manifest(path)
        _expect_rejected(
            "clean rollback as live patched input",
            lambda: BUILD._load_manifest(path, verify_current_inputs=True),
            "diff is empty",
        )
        print("OK: verified patch manifest after clean source rollback")

        fake_bin = temporary_root / "fake-bin"
        fake_bin.mkdir()
        fake_git = fake_bin / "git"
        fake_git.write_text("#!/bin/sh\necho forged\n")
        fake_git.chmod(0o755)
        previous_path = os.environ.get("PATH")
        previous_git_dir = os.environ.get("GIT_DIR")
        os.environ["PATH"] = f"{fake_bin}:{previous_path or ''}"
        os.environ["GIT_DIR"] = str(temporary_root / "forged.git")
        try:
            observed = BUILD._source_observation("jax")
            checked = CHECK._git_source("jax", root / "upstream/jax")
        finally:
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path
            if previous_git_dir is None:
                os.environ.pop("GIT_DIR", None)
            else:
                os.environ["GIT_DIR"] = previous_git_dir
        assert observed["commit"] == BUILD.EXPECTED_SOURCE_COMMITS["jax"]
        assert checked["commit"] == BUILD.EXPECTED_SOURCE_COMMITS["jax"]
        assert (root / checked["top_level"]).resolve() == (root / "upstream/jax").resolve()
        print("OK: ignored ambient PATH and GIT_DIR")

        root = temporary_root / "source-symlink"
        _fixture(root)
        outside_source = temporary_root / "outside-jax"
        (root / "upstream/jax").rename(outside_source)
        (root / "upstream/jax").symlink_to(outside_source, target_is_directory=True)
        _expect_rejected("source-root symlink", BUILD._validate_source_roots, "symlink")

        root = temporary_root / "lock-symlink"
        _fixture(root)
        outside_artifacts = temporary_root / "outside-artifacts"
        (root / "artifacts").rename(outside_artifacts)
        (root / "artifacts").symlink_to(outside_artifacts, target_is_directory=True)
        _expect_rejected(
            "artifact ancestor symlink",
            lambda: BUILD._exclusive_lock(BUILD.GLOBAL_LOCK_PATH, "selftest").__enter__(),
            "symlink",
        )
        assert not (outside_artifacts / "builds/.jaxlib-build.lock").exists()

        root = temporary_root / "lock-file-symlink"
        _fixture(root)
        lock_target = temporary_root / "outside.lock"
        lock_target.write_text("must remain untouched\n")
        BUILD.GLOBAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        BUILD.GLOBAL_LOCK_PATH.symlink_to(lock_target)
        _expect_rejected(
            "lock-file symlink",
            lambda: BUILD._exclusive_lock(BUILD.GLOBAL_LOCK_PATH, "selftest").__enter__(),
            "symlink",
        )
        assert lock_target.read_text() == "must remain untouched\n"

        root = temporary_root / "forged-source"
        manifest, path, _ = _fixture(root)
        manifest["inputs"]["sources"]["jax"]["commit"] = "f" * 40
        manifest["input_fingerprint_sha256"] = BUILD._sha256_bytes(
            BUILD._canonical_bytes(BUILD._fingerprint_payload(manifest["inputs"]))
        )
        _expect_rejected("self-consistent forged source", lambda: _validate(manifest, path), "not fixed")

        root = temporary_root / "bad-wheel"
        manifest, path, wheel = _fixture(root)
        wheel.write_bytes(b"not a wheel")
        record = BUILD._artifact_record(wheel, wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected("non-ZIP success wheel", lambda: _validate(manifest, path), "valid readable archive")

        root = temporary_root / "wrong-wheel-version"
        manifest, path, wheel = _fixture(root)
        _write_wheel(wheel, metadata_version="0.11.1")
        record = BUILD._artifact_record(wheel, wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "wheel METADATA version mismatch",
            lambda: _validate(manifest, path),
            "METADATA Version",
        )

        root = temporary_root / "wrong-wheel-metadata-name"
        manifest, path, wheel = _fixture(root)
        _write_wheel(wheel, metadata_name="not-jaxlib")
        record = BUILD._artifact_record(wheel, wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "wheel METADATA name mismatch",
            lambda: _validate(manifest, path),
            "METADATA Name",
        )

        root = temporary_root / "wrong-wheel-filename-version"
        manifest, path, wheel = _fixture(root)
        wrong_wheel = wheel.with_name(
            "jaxlib-0.11.1-cp312-cp312-manylinux_2_27_x86_64.whl"
        )
        wheel.replace(wrong_wheel)
        record = BUILD._artifact_record(wrong_wheel, wrong_wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "wheel filename version mismatch",
            lambda: _validate(manifest, path),
            "filename must be the pinned",
        )

        root = temporary_root / "wrong-wheel-project"
        manifest, path, wheel = _fixture(root)
        wrong_wheel = wheel.with_name(wheel.name.replace("jaxlib-", "other-", 1))
        wheel.replace(wrong_wheel)
        record = BUILD._artifact_record(wrong_wheel, wrong_wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "wheel filename project mismatch",
            lambda: _validate(manifest, path),
            "filename must be the pinned",
        )

        root = temporary_root / "wrong-wheel-tag"
        manifest, path, wheel = _fixture(root)
        wrong_wheel = wheel.with_name(
            wheel.name.replace("manylinux_2_27_x86_64", "manylinux_2_28_x86_64")
        )
        wheel.replace(wrong_wheel)
        record = BUILD._artifact_record(wrong_wheel, wrong_wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "wheel filename platform tag mismatch",
            lambda: _validate(manifest, path),
            "filename must be the pinned",
        )

        root = temporary_root / "pure-python-wheel"
        manifest, path, wheel = _fixture(root)
        _write_wheel(wheel, include_native=False)
        record = BUILD._artifact_record(wheel, wheel.parent)
        manifest["attempts"][0]["wheels"] = [record]
        manifest["result"]["wheels"] = [record]
        _expect_rejected(
            "pure-Python wheel as jaxlib build success",
            lambda: _validate(manifest, path),
            "native payload",
        )

        root = temporary_root / "bad-time"
        manifest, path, _ = _fixture(root)
        manifest["attempts"][0]["ended_at"] = "2026-09-07T23:59:59Z"
        manifest["updated_at"] = manifest["attempts"][0]["ended_at"]
        _expect_rejected("reversed attempt time", lambda: _validate(manifest, path), "ends before")

        root = temporary_root / "bad-result"
        manifest, path, _ = _fixture(root)
        manifest["status"] = "preflight-failed"
        manifest["attempts"] = []
        manifest["preflight"] = {
            "ok": False,
            "blockers": [{"code": "synthetic", "message": "synthetic blocker"}],
            "waived_blockers": [],
        }
        manifest["result"]["error"] = "synthetic blocker"
        _expect_rejected("preflight result carrying a wheel", lambda: _validate(manifest, path), "schema")

        root = temporary_root / "runtime"
        manifest, path, _ = _fixture(root)
        manifest["runtime_validation"] = {
            "status": "passed",
            "checked_at": "2026-09-08T00:00:03Z",
            "evidence": ["README.md"],
            "notes": "forged",
        }
        _expect_rejected("runtime validation before P1 contract", lambda: _validate(manifest, path), "schema")

        root = temporary_root / "hidden-index"
        _fixture(root)
        source = root / "upstream/jax"
        _run_git(source, "update-index", "--assume-unchanged", ".bazelrc")
        (source / ".bazelrc").write_text("build --color=yes\n")
        _expect_rejected(
            "assume-unchanged source mutation",
            lambda: BUILD._reject_untracked_or_submodule_dirty("jax", source),
            "assume-unchanged",
        )

        root = temporary_root / "ignored-bytecode"
        _fixture(root)
        source = root / "upstream/jax"
        bytecode = source / "build/tools/__pycache__/helper.cpython-312.pyc"
        bytecode.parent.mkdir(parents=True)
        bytecode.write_bytes(b"unchecked bytecode")
        assert not CHECK._ignored_source_path_allowed(
            "jax", "build/tools/__pycache__/helper.cpython-312.pyc"
        )
        _expect_rejected(
            "ignored Python bytecode",
            lambda: BUILD._reject_untracked_or_submodule_dirty("jax", source),
            "non-allowlisted ignored",
        )

    print("OK: build wrapper isolated selftest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
