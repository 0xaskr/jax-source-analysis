#!/usr/bin/env python3
"""Exercise semantic evidence validation with isolated negative fixtures."""

from __future__ import annotations

import copy
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "tools/validate-evidence.py"
CAPTURE_BASELINE_PATH = REPOSITORY_ROOT / "tools/capture-baseline.py"
TOPIC_RELATIVE = Path("docs/contributing/examples/03-selftest/topic.json")
CAPTURE_RELATIVE = Path(
    "docs/contributing/examples/03-selftest/captures/cpu-run/manifest.json"
)
NATIVE_RELATIVE = CAPTURE_RELATIVE.parent / "native-binaries.json"


def _load_validator_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "jax_source_analysis_validate_evidence", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_baseline_capture_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "jax_source_analysis_capture_baseline_selftest", CAPTURE_BASELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CAPTURE_BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator_module()


def _run(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return result.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, artifact_id: str, kind: str, format_: str) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "path": path.as_posix(),
        "sha256": "",
        "size_bytes": 0,
        "kind": kind,
        "stage": "execute",
        "format": format_,
        "generated": True,
        "retrieval": "Regenerate with the capture producer command.",
    }


def _create_valid_fixture(root: Path) -> None:
    schema_directory = root / "manifests/schema"
    schema_directory.mkdir(parents=True)
    for relative in VALIDATOR.SCHEMAS.values():
        source = REPOSITORY_ROOT / relative
        shutil.copy2(source, schema_directory / source.name)
    (root / "tools").mkdir()
    shutil.copy2(REPOSITORY_ROOT / "tools/build-jaxlib.py", root / "tools/build-jaxlib.py")

    _run("git", "init", "--quiet", cwd=root)
    _run("git", "config", "user.email", "selftest@example.invalid", cwd=root)
    _run("git", "config", "user.name", "Evidence Selftest", cwd=root)
    (root / ".gitignore").write_text(
        ".venv/\ndocs/\nartifacts/\nmanifests/build-fingerprints/\n"
        "runtime/\ntoolchain/\nupstream/\n"
    )
    (root / "probe.py").write_text("print('base')\n")
    (root / "alias-probe.py").symlink_to("probe.py")
    (root / "unchanged.txt").write_text("unchanged\n")
    _run(
        "git",
        "add",
        ".gitignore",
        "probe.py",
        "alias-probe.py",
        "unchanged.txt",
        cwd=root,
    )
    _run("git", "commit", "--quiet", "-m", "fixture base", cwd=root)
    analysis_revision = _run("git", "rev-parse", "HEAD", cwd=root)
    (root / "probe.py").write_text("print('captured')\n")

    patch_relative = CAPTURE_RELATIVE.parent / "analysis-source.patch"
    patch_path = root / patch_relative
    patch_path.parent.mkdir(parents=True)
    patch = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            analysis_revision,
            "--",
            "probe.py",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    patch_path.write_bytes(patch)

    source_root = root / "upstream/jax"
    source_root.mkdir(parents=True)
    _run("git", "init", "--quiet", cwd=source_root)
    _run("git", "config", "user.email", "selftest@example.invalid", cwd=source_root)
    _run("git", "config", "user.name", "Evidence Selftest", cwd=source_root)
    (source_root / ".gitignore").write_text(
        "bazel-*\n*.pyc\n*.egg-info\n.jax_configure.bazelrc\nMODULE.bazel.lock\n"
    )
    (source_root / ".bazelrc").write_text("build --color=no\n")
    (source_root / ".bazelversion").write_text("8.7.0\n")
    (source_root / "build").mkdir()
    (source_root / "build/build.py").write_text(
        "raise RuntimeError('evidence selftest must never execute build.py')\n"
    )
    (source_root / "api.py").write_text("def jit(function):\n    return function\n")
    (source_root / "alias.py").symlink_to("api.py")
    _run("git", "add", ".", cwd=source_root)
    _run("git", "commit", "--quiet", "-m", "source fixture", cwd=source_root)
    source_revision = _run("git", "rev-parse", "HEAD", cwd=source_root)

    xla_root = root / "upstream/xla"
    xla_root.mkdir(parents=True)
    _run("git", "init", "--quiet", cwd=xla_root)
    _run("git", "config", "user.email", "selftest@example.invalid", cwd=xla_root)
    _run("git", "config", "user.name", "Evidence Selftest", cwd=xla_root)
    (xla_root / "xla").mkdir()
    (xla_root / "xla/marker.txt").write_text("fixture\n")
    _run("git", "add", ".", cwd=xla_root)
    _run("git", "commit", "--quiet", "-m", "source fixture", cwd=xla_root)

    binary_relative = Path("runtime/jaxlib/_jax.so")
    wheel_relative = Path("artifacts/jaxlib.whl")
    binary_path = root / binary_relative
    wheel_path = root / wheel_relative
    binary_path.parent.mkdir(parents=True)
    wheel_path.parent.mkdir(parents=True)
    binary_path.write_bytes(b"ELF-selftest\n")
    wheel_path.write_bytes(b"wheel-selftest\n")

    stdout_relative = CAPTURE_RELATIVE.parent / "stdout.txt"
    stdout_path = root / stdout_relative
    stdout_path.write_text("ok\n")
    native_path = root / NATIVE_RELATIVE
    native = {
        "$schema": "https://jax-source-analysis.local/schema/native-binaries.schema.json",
        "schema_version": "1.0",
        "inventory_id": "native-binaries",
        "packages": [
            {
                "name": "jaxlib",
                "version": "1.0",
                "source_component": "jax",
                "build_revision": source_revision,
                "runtime_roles": ["python-extension", "cpu-backend"],
                "identity": {
                    "kind": "repository-artifact",
                    "artifact_path": wheel_relative.as_posix(),
                    "artifact_sha256": _sha256(wheel_path),
                    "artifact_size_bytes": wheel_path.stat().st_size,
                },
                "selection": "Mapped selftest binaries.",
                "binaries": [
                    {
                        "package_path": "_jax.so",
                        "artifact_path": binary_relative.as_posix(),
                        "sha256": _sha256(binary_path),
                        "size_bytes": binary_path.stat().st_size,
                    }
                ],
            }
        ],
    }
    _write_json(native_path, native)

    source_index_relative = TOPIC_RELATIVE.parent / "source-index.json"
    source_index = {
        "$schema": "https://jax-source-analysis.local/schema/source-index.schema.json",
        "schema_version": "1.0",
        "topic_id": "03-selftest",
        "components": [
            {
                "id": "jax",
                "name": "JAX",
                "root": "upstream/jax",
                "revision": source_revision,
            }
        ],
        "entries": [
            {
                "id": "jax.api.jit",
                "component": "jax",
                "path": "upstream/jax/api.py",
                "symbol": "jit",
                "kind": "function",
                "layer": "jax-python",
                "language": "python",
                "role": "Selftest source anchor.",
                "location": {"start_line": 1},
                "stability": "public",
                "consumes": ["callable"],
                "produces": ["callable"],
                "callers": [],
                "callees": [],
                "evidence_refs": ["cpu-observation"],
            }
        ],
    }
    _write_json(root / source_index_relative, source_index)
    readme_relative = TOPIC_RELATIVE.parent / "README.md"
    (root / readme_relative).write_text("# Selftest\n")

    producer = {
        "cwd": ".",
        "argv": ["python", "probe.py"],
        "stdout_path": stdout_relative.as_posix(),
        "required_environment": [],
    }
    capture = {
        "$schema": "https://jax-source-analysis.local/schema/capture.schema.json",
        "schema_version": "1.0",
        "capture_id": "cpu-run",
        "topic_id": "03-selftest",
        "evidence_level": "RUN-CPU",
        "qualifiers": [],
        "captured_at": "2026-09-08T00:00:00Z",
        "provenance": {
            "analysis_repository_revision": analysis_revision,
            "components": [
                {
                    "name": "analysis-repository",
                    "kind": "source",
                    "root": ".",
                    "revision": analysis_revision,
                    "scope_paths": ["probe.py", "unchanged.txt"],
                    "dirty": True,
                    "patches": [
                        {
                            "path": patch_relative.as_posix(),
                            "sha256": _sha256(patch_path),
                        }
                    ],
                },
                {
                    "name": "jax",
                    "kind": "source",
                    "root": "upstream/jax",
                    "revision": source_revision,
                    "dirty": False,
                },
                {
                    "name": "jaxlib-runtime",
                    "kind": "binary",
                    "package": "jaxlib",
                    "version": "1.0",
                    "source_component": "jax",
                    "build_revision": source_revision,
                    "runtime_roles": ["python-extension", "cpu-backend"],
                    "root": "runtime/jaxlib",
                    "artifact_path": binary_relative.as_posix(),
                    "artifact_sha256": _sha256(binary_path),
                    "artifact_size_bytes": binary_path.stat().st_size,
                },
            ],
        },
        "environment": {
            "host_os": "Linux",
            "arch": "x86_64",
            "backend": "cpu",
            "device_count": 1,
        },
        "producer": producer,
        "inputs": [
            {
                "path": "probe.py",
                "sha256": _sha256(root / "probe.py"),
                "role": "Probe source.",
            },
            {
                "path": "unchanged.txt",
                "sha256": _sha256(root / "unchanged.txt"),
                "role": "Unchanged scoped input.",
            },
        ],
        "artifacts": [
            {
                **_artifact(stdout_relative, "stdout", "stdout", "text/plain"),
                "sha256": _sha256(stdout_path),
                "size_bytes": stdout_path.stat().st_size,
            },
            {
                **_artifact(
                    NATIVE_RELATIVE,
                    "native-binaries",
                    "other",
                    "application/vnd.jax-source-analysis.native-binaries+json",
                ),
                "sha256": _sha256(native_path),
                "size_bytes": native_path.stat().st_size,
            },
        ],
        "runtime_inventory": "native-binaries",
        "outcome": {
            "status": "pass",
            "assertions": [
                {
                    "id": "selftest-pass",
                    "status": "pass",
                    "summary": "Selftest passed.",
                    "probe_ref": "probe.py",
                    "artifact_refs": ["stdout"],
                }
            ],
        },
        "sanitized": True,
        "limitations": ["Synthetic validator fixture only."],
    }
    _write_json(root / CAPTURE_RELATIVE, capture)

    topic = {
        "$schema": "https://jax-source-analysis.local/schema/topic.schema.json",
        "schema_version": "1.0",
        "topic_id": "03-selftest",
        "title": "Evidence validator selftest",
        "question": "Does the semantic validator reject malformed evidence?",
        "status": "verified",
        "scope": {"includes": ["Validator contracts"], "excludes": []},
        "provenance": {
            "components": [
                {
                    "name": "jax",
                    "root": "upstream/jax",
                    "revision": source_revision,
                    "dirty": False,
                }
            ]
        },
        "readme": readme_relative.as_posix(),
        "source_index": source_index_relative.as_posix(),
        "evidence": [
            {
                "id": "cpu-observation",
                "claim": "The synthetic CPU observation passed.",
                "evidence_level": "RUN-CPU",
                "qualifiers": [],
                "source_refs": ["jax.api.jit"],
                "artifact_refs": ["stdout", "native-binaries"],
                "capture_manifest": CAPTURE_RELATIVE.as_posix(),
                "reproduce": producer,
                "limitations": ["Synthetic validator fixture only."],
            }
        ],
    }
    _write_json(root / TOPIC_RELATIVE, topic)


def _validate(root: Path, *, require_live_source_state: bool = False) -> list[str]:
    validator = VALIDATOR.EvidenceValidator(
        root, require_live_source_state=require_live_source_state
    )
    try:
        for name in VALIDATOR.SCHEMAS:
            validator.schema(name)
        validator.validate_topic(root / TOPIC_RELATIVE)
        return list(validator.errors)
    finally:
        validator.close()


def _mutate_json(root: Path, relative: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / relative
    value = _read_json(path)
    mutate(value)
    _write_json(path, value)


def _refresh_native_artifact(root: Path) -> None:
    capture = _read_json(root / CAPTURE_RELATIVE)
    native_path = root / NATIVE_RELATIVE
    native_artifact = next(
        item for item in capture["artifacts"] if item["id"] == "native-binaries"
    )
    native_artifact["sha256"] = _sha256(native_path)
    native_artifact["size_bytes"] = native_path.stat().st_size
    _write_json(root / CAPTURE_RELATIVE, capture)


def _set_producer_argument(root: Path, argument: str) -> None:
    def mutate_capture(value: dict[str, Any]) -> None:
        value["producer"]["argv"].append(argument)

    def mutate_topic(value: dict[str, Any]) -> None:
        value["evidence"][0]["reproduce"]["argv"].append(argument)

    _mutate_json(root, CAPTURE_RELATIVE, mutate_capture)
    _mutate_json(root, TOPIC_RELATIVE, mutate_topic)


def _write_valid_wheel(path: Path, build: Any) -> None:
    version = str(build.EXPECTED_JAXLIB_VERSION)
    dist_info = f"jaxlib-{version}.dist-info"
    members = {
        "jaxlib/__init__.py": b"from .version import __version__\n",
        "jaxlib/version.py": f"__version__ = {version!r}\n".encode(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: jaxlib\n"
            f"Version: {version}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: evidence-selftest\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {build.EXPECTED_WHEEL_TAG}\n\n"
        ).encode(),
    }
    elf_header = bytearray(64)
    elf_header[:6] = b"\x7fELF\x02\x01"
    elf_header[18:20] = (62).to_bytes(2, "little")
    for name in build.REQUIRED_JAXLIB_NATIVE_MEMBERS:
        members[name] = bytes(elf_header)
    rows: list[list[str]] = []
    for name, content in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
        rows.append(
            [name, f"sha256={digest.rstrip(b'=').decode()}", str(len(content))]
        )
    record_name = f"{dist_info}/RECORD"
    rows.append([record_name, "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    members[record_name] = output.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _build_tool_record(build: Any, path: Path, version: str) -> dict[str, Any]:
    return {
        "path": build._repo_path(path) if path.is_relative_to(build.ROOT) else str(path),
        "resolved_path": str(path.resolve()),
        "version": version,
        "returncode": 0,
        "sha256": _sha256(path),
    }


def _configure_build_validator(build: Any, root: Path) -> None:
    source_revision = _run("git", "rev-parse", "HEAD", cwd=root / "upstream/jax")
    xla_revision = _run("git", "rev-parse", "HEAD", cwd=root / "upstream/xla")
    python_path = root / ".venv/bin/python"
    toolchain = root / "toolchain"
    clang_path = toolchain / "clang"
    clangxx_path = toolchain / "clang++"
    bazel_path = root / "upstream/jax/bazel-8.7.0-linux-x86_64"

    build.ROOT = root
    build.JAX_ROOT = root / "upstream/jax"
    build.SOURCE_ROOTS = {
        "jax": build.JAX_ROOT,
        "xla": root / "upstream/xla",
    }
    build.SCHEMA_PATH = root / "manifests/schema/build-jaxlib.schema.json"
    build.DEFAULT_MANIFEST_ROOT = root / "manifests/build-fingerprints"
    build.DEFAULT_ARTIFACT_ROOT = root / "artifacts/builds"
    build.GLOBAL_LOCK_PATH = build.DEFAULT_ARTIFACT_ROOT / ".jaxlib-build.lock"
    build.EXPECTED_PYTHON_PATH = python_path
    build.GENERATED_BAZELRC = build.JAX_ROOT / ".jax_configure.bazelrc"
    build.EXPECTED_SOURCE_COMMITS = {
        "jax": source_revision,
        "xla": xla_revision,
    }
    build.EXPECTED_TOOL_PATHS = {
        "bazel": bazel_path,
        "clang": clang_path,
        "clangxx": clangxx_path,
        "git": Path("/usr/bin/git"),
    }
    build.EXPECTED_BAZEL_SHA256 = _sha256(bazel_path)


def _validate_build_fixture(root: Path) -> list[str]:
    validator = VALIDATOR.EvidenceValidator(root)
    try:
        for name in VALIDATOR.SCHEMAS:
            validator.schema(name)
        _configure_build_validator(validator.build_validator(), root)
        validator.validate_topic(root / TOPIC_RELATIVE)
        return list(validator.errors)
    finally:
        validator.close()


def _install_valid_build_identity(root: Path) -> tuple[Path, Path]:
    for relative in (
        ".python-version",
        "pyproject.toml",
        "tools/check-jaxlib-build-env.py",
        "upstream-sources.lock",
        "uv.lock",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "3.12.3\n" if relative == ".python-version" else f"fixture {relative}\n"
        )
    python_path = root / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to("/usr/bin/python3.12")
    toolchain = root / "toolchain"
    toolchain.mkdir()
    clang_path = toolchain / "clang"
    clangxx_path = toolchain / "clang++"
    clang_path.write_bytes(b"fixture clang\n")
    clangxx_path.write_bytes(b"fixture clang++\n")
    bazel_path = root / "upstream/jax/bazel-8.7.0-linux-x86_64"
    bazel_path.write_bytes(b"fixture bazel\n")

    validator = VALIDATOR.EvidenceValidator(root)
    try:
        build = validator.build_validator()
        _configure_build_validator(build, root)
    finally:
        validator.close()

    analysis_inputs = [
        relative
        for relative in build.INPUT_PATHS
        if not relative.startswith("upstream/jax/")
    ]
    _run("git", "add", *analysis_inputs, cwd=root)
    _run(
        "git",
        "commit",
        "--quiet",
        "-m",
        "commit build inputs",
        cwd=root,
    )

    source_revision = _run("git", "rev-parse", "HEAD", cwd=root / "upstream/jax")
    build_id = "selftest_build-1"

    sources = {
        name: build._source_observation(name) for name in ("jax", "xla")
    }
    tools = {
        "python": {
            "executable": str(python_path),
            "resolved_executable": str(build.EXPECTED_BASE_PYTHON_PATH),
            "base_executable": str(build.EXPECTED_BASE_PYTHON_PATH),
            "base_resolved_path": str(build.EXPECTED_BASE_PYTHON_PATH),
            "base_size_bytes": build.EXPECTED_BASE_PYTHON_SIZE_BYTES,
            "base_sha256": build.EXPECTED_BASE_PYTHON_SHA256,
            "version": "3.12.3",
        },
        "bazel": _build_tool_record(build, bazel_path, build.EXPECTED_BAZEL_VERSION),
        "clang": _build_tool_record(
            build, clang_path, "Ubuntu clang version 18.1.3"
        ),
        "clangxx": _build_tool_record(
            build, clangxx_path, "Ubuntu clang version 18.1.3"
        ),
        "git": {
            "path": "/usr/bin/git",
            "resolved_path": str(Path("/usr/bin/git").resolve()),
            "version": build.EXPECTED_GIT_VERSION,
            "returncode": 0,
            "sha256": build.EXPECTED_GIT_SHA256,
        },
    }
    preflight_report = {"python": tools["python"], "sources": sources, **tools}
    config = build._build_config(preflight_report, 4, [], [], False, {})
    inputs, fingerprint = build._build_inputs(
        preflight_report, config, build_id, []
    )
    manifest_path = root / f"manifests/build-fingerprints/{build_id}.json"
    artifact_dir = root / f"artifacts/builds/{build_id}"
    attempt_dir = artifact_dir / "attempts/0001"
    wheel_dir = attempt_dir / "wheels"
    wheel_dir.mkdir(parents=True)
    (attempt_dir / "home").mkdir()
    (attempt_dir / "tmp").mkdir()
    log_path = attempt_dir / "build.log"
    bazelrc_path = attempt_dir / "generated.jax_configure.bazelrc"
    wheel_path = wheel_dir / (
        f"jaxlib-{build.EXPECTED_JAXLIB_VERSION}-{build.EXPECTED_WHEEL_TAG}.whl"
    )
    log_path.write_text("synthetic successful build\n")
    bazelrc_path.write_text("build --selftest\n")
    _write_valid_wheel(wheel_path, build)
    manifest_path.parent.mkdir(parents=True)
    manifest = build._new_manifest(
        build_id=build_id,
        config=config,
        inputs=inputs,
        input_fingerprint=fingerprint,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        build_lock_path=artifact_dir / ".build-id.lock",
        preflight={"ok": True, "blockers": [], "waived_blockers": []},
    )
    paths = {
        "attempt_dir": build._repo_path(attempt_dir),
        "log": build._repo_path(log_path),
        "wheel_dir": build._repo_path(wheel_dir),
        "generated_bazelrc_archive": build._repo_path(bazelrc_path),
        "home_dir": build._repo_path(attempt_dir / "home"),
        "tmp_dir": build._repo_path(attempt_dir / "tmp"),
    }
    argv = build._build_command(attempt_dir, config)
    process_identity = {
        "pid": 1,
        "boot_id": "00000000-0000-0000-0000-000000000000",
        "start_time_ticks": 0,
        "cmdline_sha256": "0" * 64,
    }
    wheel = build._artifact_record(wheel_path, wheel_dir)
    generated = build._artifact_record(bazelrc_path, attempt_dir)
    attempt = {
        "number": 1,
        "status": "build-succeeded",
        "started_at": "2026-09-08T00:00:00Z",
        "ended_at": "2026-09-08T00:00:01Z",
        "paths": paths,
        "command": {"cwd": "upstream/jax", "argv": argv, "shell": shlex.join(argv)},
        "environment": build._attempt_environment_record(attempt_dir, config),
        "wrapper_process": process_identity,
        "child_process": process_identity,
        "exit_code": 0,
        "error": None,
        "log_artifact": build._artifact_record(log_path, attempt_dir),
        "wheels": [wheel],
        "generated_bazelrc": generated,
    }
    manifest.update(
        {
            "created_at": "2026-09-08T00:00:00Z",
            "updated_at": "2026-09-08T00:00:01Z",
            "status": "build-succeeded",
            "attempts": [attempt],
            "result": {
                "attempt_number": 1,
                "exit_code": 0,
                "error": None,
                "wheels": [wheel],
                "generated_bazelrc": generated,
            },
        }
    )
    _write_json(manifest_path, manifest)
    build.load_and_validate_manifest(manifest_path)

    native = _read_json(root / NATIVE_RELATIVE)
    native["packages"][0]["identity"] = {
        "kind": "build-manifest",
        "build_manifest": manifest_path.relative_to(root).as_posix(),
        "build_manifest_sha256": _sha256(manifest_path),
        "build_manifest_size_bytes": manifest_path.stat().st_size,
        "build_id": build_id,
        "artifact_sha256": wheel["sha256"],
        "artifact_size_bytes": wheel["size_bytes"],
    }
    _write_json(root / NATIVE_RELATIVE, native)
    _refresh_native_artifact(root)
    return manifest_path, wheel_path


def _case_path_traversal(root: Path) -> None:
    _mutate_json(root, TOPIC_RELATIVE, lambda value: value.__setitem__("readme", "../escape.md"))


def _case_symlink_escape(root: Path) -> None:
    outside = root.parent / "outside.txt"
    outside.write_text("outside\n")
    link_relative = CAPTURE_RELATIVE.parent / "escaped.txt"
    link = root / link_relative
    link.symlink_to(outside)

    def mutate(value: dict[str, Any]) -> None:
        artifact = value["artifacts"][0]
        artifact["path"] = link_relative.as_posix()
        artifact["sha256"] = _sha256(outside)
        artifact["size_bytes"] = outside.stat().st_size

    _mutate_json(root, CAPTURE_RELATIVE, mutate)


def _case_external_uri(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        artifact = value["artifacts"][0]
        artifact.pop("path")
        artifact["uri"] = "file:///etc/passwd"

    _mutate_json(root, CAPTURE_RELATIVE, mutate)


def _case_failed_capture(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["outcome"].__setitem__("status", "fail"),
    )


def _case_producer_mismatch(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE,
        lambda value: value["evidence"][0]["reproduce"].__setitem__(
            "argv", ["python", "different.py"]
        ),
    )


def _case_normalization_cycle(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        first, second = value["artifacts"]
        first.update({"normalized_from": second["id"], "normalization": "first"})
        second.update({"normalized_from": first["id"], "normalization": "second"})

    _mutate_json(root, CAPTURE_RELATIVE, mutate)


def _case_duplicate_artifact_locator(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        duplicate = copy.deepcopy(value["artifacts"][0])
        duplicate["id"] = "duplicate-stdout"
        duplicate["kind"] = "log"
        value["artifacts"].append(duplicate)

    _mutate_json(root, CAPTURE_RELATIVE, mutate)


def _case_source_outside_component(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE.parent / "source-index.json",
        lambda value: value["entries"][0].__setitem__("path", "probe.py"),
    )


def _case_incomplete_patch(root: Path) -> None:
    (root / "unchanged.txt").write_text("unrecorded change\n")


def _case_live_source_head_mismatch(root: Path) -> None:
    (root / "head-move.txt").write_text("move live HEAD\n")
    _run("git", "add", "head-move.txt", cwd=root)
    _run("git", "commit", "--quiet", "-m", "move live HEAD", cwd=root)


def _case_missing_skew_qualifier(root: Path) -> None:
    mismatched_revision = "f" * 40

    def mutate_capture(value: dict[str, Any]) -> None:
        value["provenance"]["components"][2]["build_revision"] = mismatched_revision

    def mutate_inventory(value: dict[str, Any]) -> None:
        value["packages"][0]["build_revision"] = mismatched_revision

    _mutate_json(root, CAPTURE_RELATIVE, mutate_capture)
    _mutate_json(root, NATIVE_RELATIVE, mutate_inventory)
    capture = _read_json(root / CAPTURE_RELATIVE)
    native_path = root / NATIVE_RELATIVE
    native_artifact = next(
        item for item in capture["artifacts"] if item["id"] == "native-binaries"
    )
    native_artifact["sha256"] = _sha256(native_path)
    native_artifact["size_bytes"] = native_path.stat().st_size
    _write_json(root / CAPTURE_RELATIVE, capture)


def _case_cross_kind_field(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["provenance"]["components"][0].__setitem__(
            "artifact_path", "runtime/jaxlib/_jax.so"
        ),
    )


def _case_nonreciprocal_refs(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE.parent / "source-index.json",
        lambda value: value["entries"][0].__setitem__("evidence_refs", []),
    )


def _case_callgraph_self_reference(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        entry_id = value["entries"][0]["id"]
        value["entries"][0]["callers"] = [entry_id]
        value["entries"][0]["callees"] = [entry_id]

    _mutate_json(root, TOPIC_RELATIVE.parent / "source-index.json", mutate)


def _case_clean_source_dirty(root: Path) -> None:
    with (root / "upstream/jax/api.py").open("a", encoding="utf-8") as stream:
        stream.write("# unrecorded tracked change\n")


def _case_clean_source_ignored_bytecode(root: Path) -> None:
    bytecode = root / "upstream/jax/__pycache__/evil.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"ignored bytecode\n")


def _case_ignored_untracked_scope(root: Path) -> None:
    with (root / ".git/info/exclude").open("a", encoding="utf-8") as stream:
        stream.write("ignored-input.bin\n")
    (root / "ignored-input.bin").write_bytes(b"ignored but scoped\n")
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["provenance"]["components"][0]["scope_paths"].append(
            "ignored-input.bin"
        ),
    )


def _case_symlink_source_input(root: Path) -> None:
    def mutate(value: dict[str, Any]) -> None:
        component = value["provenance"]["components"][0]
        component["scope_paths"].append("alias-probe.py")
        value["inputs"].append(
            {
                "path": "alias-probe.py",
                "sha256": _sha256(root / "alias-probe.py"),
                "role": "Unsupported symlink input.",
            }
        )

    _mutate_json(root, CAPTURE_RELATIVE, mutate)


def _case_symlink_source_index(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE.parent / "source-index.json",
        lambda value: value["entries"][0].__setitem__(
            "path", "upstream/jax/alias.py"
        ),
    )


def _case_missing_symbol(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE.parent / "source-index.json",
        lambda value: value["entries"][0].__setitem__(
            "symbol", "definitely_not_in_the_revision_window"
        ),
    )


def _case_secret_producer(root: Path) -> None:
    _set_producer_argument(root, "--token=do-not-record-this")


def _case_absolute_producer_path(root: Path) -> None:
    _set_producer_argument(root, "/root/private/probe.py")


def _case_authority_producer(root: Path) -> None:
    _set_producer_argument(root, "ssh://user@internal.example/probe.py")


def _case_host_path_retrieval(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["artifacts"][0].__setitem__(
            "retrieval", "Copy from /usr/local/internal/capture.txt."
        ),
    )


def _set_retrieval(root: Path, value: str) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda capture: capture["artifacts"][0].__setitem__("retrieval", value),
    )


def _case_root_path_retrieval(root: Path) -> None:
    _set_retrieval(root, "Copy from /root/private/capture.txt.")


def _case_unc_retrieval(root: Path) -> None:
    _set_retrieval(root, r"Copy from \\internal-host\private\capture.txt.")


def _case_userinfo_retrieval(root: Path) -> None:
    _set_retrieval(root, "Ask analyst@internal-host for the capture.")


def _case_control_character_producer(root: Path) -> None:
    _set_producer_argument(root, "probe.py\n--unexpected")


def _case_environment_value(root: Path) -> None:
    def mutate_capture(value: dict[str, Any]) -> None:
        value["producer"]["required_environment"] = ["TOKEN=secret"]

    def mutate_topic(value: dict[str, Any]) -> None:
        value["evidence"][0]["reproduce"]["required_environment"] = [
            "TOKEN=secret"
        ]

    _mutate_json(root, CAPTURE_RELATIVE, mutate_capture)
    _mutate_json(root, TOPIC_RELATIVE, mutate_topic)


def _case_secret_flag_variant(root: Path) -> None:
    _set_producer_argument(root, "--auth-token")


def _case_noncanonical_scope(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["provenance"]["components"][0]["scope_paths"].__setitem__(
            0, "./probe.py"
        ),
    )


def _case_noncanonical_input(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["inputs"][0].__setitem__("path", "./probe.py"),
    )


def _case_symlink_component_root(root: Path) -> None:
    (root / "upstream/jax-alias").symlink_to("jax", target_is_directory=True)
    _mutate_json(
        root,
        TOPIC_RELATIVE,
        lambda value: value["provenance"]["components"][0].__setitem__(
            "root", "upstream/jax-alias"
        ),
    )


def _case_noncanonical_component_root(root: Path) -> None:
    _mutate_json(
        root,
        TOPIC_RELATIVE,
        lambda value: value["provenance"]["components"][0].__setitem__(
            "root", "./upstream/jax"
        ),
    )


def _case_zero_devices(root: Path) -> None:
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["environment"].__setitem__("device_count", 0),
    )


def _case_missing_cpu_runtime_role(root: Path) -> None:
    _mutate_json(
        root,
        NATIVE_RELATIVE,
        lambda value: value["packages"][0].__setitem__(
            "runtime_roles", ["python-extension"]
        ),
    )
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["provenance"]["components"][2].__setitem__(
            "runtime_roles", ["python-extension"]
        ),
    )
    _refresh_native_artifact(root)


def _case_missing_simulator_runtime_role(root: Path) -> None:
    def mutate_capture(value: dict[str, Any]) -> None:
        value["evidence_level"] = "SIM-TPU"
        value["environment"]["backend"] = "simulated-tpu"
        value["environment"]["device_model"] = "synthetic-tpu"

    _mutate_json(root, CAPTURE_RELATIVE, mutate_capture)
    _mutate_json(
        root,
        TOPIC_RELATIVE,
        lambda value: value["evidence"][0].__setitem__(
            "evidence_level", "SIM-TPU"
        ),
    )


def _case_runtime_role_disagreement(root: Path) -> None:
    _mutate_json(
        root,
        NATIVE_RELATIVE,
        lambda value: value["packages"][0].__setitem__(
            "runtime_roles", ["python-extension"]
        ),
    )
    _refresh_native_artifact(root)


def _case_fake_build_manifest(root: Path) -> None:
    manifest_relative = Path("manifests/build-fingerprints/fake_build.json")
    manifest_path = root / manifest_relative
    manifest_path.parent.mkdir(parents=True)
    fake_manifest = {
        "$schema": "../schema/build-jaxlib.schema.json",
        "schema_version": "1.0",
        "build_id": "fake_build",
        "kind": "jaxlib-cpu-source-build",
        "dry_run": False,
        "status": "build-succeeded",
    }
    _write_json(manifest_path, fake_manifest)

    native = _read_json(root / NATIVE_RELATIVE)
    package = native["packages"][0]
    previous = package["identity"]
    package["identity"] = {
        "kind": "build-manifest",
        "build_manifest": manifest_relative.as_posix(),
        "build_manifest_sha256": _sha256(manifest_path),
        "build_manifest_size_bytes": manifest_path.stat().st_size,
        "build_id": "fake_build",
        "artifact_sha256": previous["artifact_sha256"],
        "artifact_size_bytes": previous["artifact_size_bytes"],
    }
    _write_json(root / NATIVE_RELATIVE, native)
    _refresh_native_artifact(root)


def _case_native_package_path_mismatch(root: Path) -> None:
    _mutate_json(
        root,
        NATIVE_RELATIVE,
        lambda value: value["packages"][0]["binaries"][0].__setitem__(
            "package_path", "wrong.so"
        ),
    )
    _refresh_native_artifact(root)


def _make_origin_capture(root: Path) -> tuple[Path, dict[str, Any]]:
    origin_relative = TOPIC_RELATIVE.parent / "captures/origin-run/manifest.json"
    origin_directory = (root / origin_relative).parent
    origin_directory.mkdir(parents=True)
    origin = _read_json(root / CAPTURE_RELATIVE)
    origin["capture_id"] = "origin-run"
    for name in ("stdout.txt", "native-binaries.json"):
        shutil.copy2(root / CAPTURE_RELATIVE.parent / name, origin_directory / name)
    origin_stdout = origin_relative.parent / "stdout.txt"
    origin_native = origin_relative.parent / "native-binaries.json"
    origin["producer"]["stdout_path"] = origin_stdout.as_posix()
    for artifact in origin["artifacts"]:
        if artifact["id"] == "stdout":
            artifact["path"] = origin_stdout.as_posix()
        elif artifact["id"] == "native-binaries":
            artifact["path"] = origin_native.as_posix()
    _write_json(root / origin_relative, origin)
    return origin_relative, origin


def _make_replay(root: Path, origin_relative: Path, refs: list[str]) -> None:
    def mutate_capture(value: dict[str, Any]) -> None:
        value["evidence_level"] = "REPLAY-OFFLINE"
        value["environment"]["backend"] = "offline"
        value["replay"] = {
            "origin_capture": origin_relative.as_posix(),
            "origin_sha256": _sha256(root / origin_relative),
            "origin_artifact_refs": refs,
            "tool_component": "analysis-repository",
        }

    _mutate_json(root, CAPTURE_RELATIVE, mutate_capture)
    _mutate_json(
        root,
        TOPIC_RELATIVE,
        lambda value: value["evidence"][0].__setitem__(
            "evidence_level", "REPLAY-OFFLINE"
        ),
    )


def _case_replay_missing_origin_artifact(root: Path) -> None:
    origin_relative, _ = _make_origin_capture(root)
    _make_replay(root, origin_relative, ["missing-artifact"])


def _case_replay_invalid_origin(root: Path) -> None:
    origin_relative, origin = _make_origin_capture(root)
    origin["artifacts"][0]["sha256"] = "0" * 64
    _write_json(root / origin_relative, origin)
    _make_replay(root, origin_relative, ["stdout"])


def _case_replay_cycle(root: Path) -> None:
    origin_relative, origin = _make_origin_capture(root)
    _make_replay(root, origin_relative, ["stdout"])
    origin["evidence_level"] = "REPLAY-OFFLINE"
    origin["environment"]["backend"] = "offline"
    origin["replay"] = {
        "origin_capture": CAPTURE_RELATIVE.as_posix(),
        "origin_sha256": _sha256(root / CAPTURE_RELATIVE),
        "origin_artifact_refs": ["stdout"],
        "tool_component": "analysis-repository",
    }
    _write_json(root / origin_relative, origin)
    capture = _read_json(root / CAPTURE_RELATIVE)
    capture["replay"]["origin_sha256"] = _sha256(root / origin_relative)
    _write_json(root / CAPTURE_RELATIVE, capture)


def _case_replay_cross_topic(root: Path) -> None:
    origin_relative, origin = _make_origin_capture(root)
    origin["topic_id"] = "different-topic"
    _write_json(root / origin_relative, origin)
    _make_replay(root, origin_relative, ["stdout"])


CASES: tuple[tuple[str, Callable[[Path], None], str], ...] = (
    ("path traversal", _case_path_traversal, "does not match"),
    ("symlink escape", _case_symlink_escape, "resolves outside the root"),
    ("external URI", _case_external_uri, "does not match"),
    ("failed verified capture", _case_failed_capture, "requires a passing capture"),
    ("producer mismatch", _case_producer_mismatch, "must exactly match capture producer"),
    ("normalization cycle", _case_normalization_cycle, "normalization graph contains a cycle"),
    ("duplicate artifact locator", _case_duplicate_artifact_locator, "duplicate artifact paths"),
    ("source outside component", _case_source_outside_component, "outside component root"),
    ("incomplete patch", _case_incomplete_patch, "differ from revision plus patches"),
    (
        "live source HEAD mismatch",
        _case_live_source_head_mismatch,
        "live source HEAD differs from recorded revision",
    ),
    ("missing VERSION-SKEW", _case_missing_skew_qualifier, "VERSION-SKEW qualifier=False"),
    ("cross-kind field", _case_cross_kind_field, "not valid under any of the given schemas"),
    ("nonreciprocal refs", _case_nonreciprocal_refs, "refs are not reciprocal"),
    (
        "callgraph self reference",
        _case_callgraph_self_reference,
        "cannot reference the entry itself",
    ),
    ("dirty clean source", _case_clean_source_dirty, "source is declared clean"),
    (
        "clean source ignored bytecode",
        _case_clean_source_ignored_bytecode,
        "clean source contains ignored Python bytecode",
    ),
    (
        "ignored untracked scoped source",
        _case_ignored_untracked_scope,
        "untracked or ignored scoped paths",
    ),
    ("symlink source input", _case_symlink_source_input, "unsupported Git mode 120000"),
    ("symlink source index", _case_symlink_source_index, "unsupported Git mode 120000"),
    ("missing source symbol", _case_missing_symbol, "absent from the revision location window"),
    ("secret producer argument", _case_secret_producer, "credential or secret assignment"),
    ("absolute producer path", _case_absolute_producer_path, "absolute path or authority/userinfo"),
    ("producer authority", _case_authority_producer, "absolute path or authority/userinfo"),
    ("retrieval host path", _case_host_path_retrieval, "absolute path or authority/userinfo"),
    ("retrieval root path", _case_root_path_retrieval, "absolute path or authority/userinfo"),
    ("retrieval UNC path", _case_unc_retrieval, "absolute path or authority/userinfo"),
    ("retrieval userinfo", _case_userinfo_retrieval, "absolute path or authority/userinfo"),
    ("producer control character", _case_control_character_producer, "control characters"),
    ("environment value", _case_environment_value, "does not match"),
    ("secret flag variant", _case_secret_flag_variant, "credential or secret assignment"),
    ("noncanonical scope path", _case_noncanonical_scope, "canonical spelling"),
    ("noncanonical input path", _case_noncanonical_input, "canonical spelling"),
    ("symlink component root", _case_symlink_component_root, "canonical repository path"),
    ("noncanonical component root", _case_noncanonical_component_root, "canonical spelling"),
    ("zero execution devices", _case_zero_devices, "less than the minimum of 1"),
    ("missing CPU runtime role", _case_missing_cpu_runtime_role, "RUN-CPU requires one runtime role"),
    (
        "missing TPU simulator runtime role",
        _case_missing_simulator_runtime_role,
        "SIM-TPU requires one runtime role",
    ),
    ("runtime role disagreement", _case_runtime_role_disagreement, "runtime_roles disagree"),
    ("fake build manifest", _case_fake_build_manifest, "canonical build manifest validation failed"),
    ("native package path mismatch", _case_native_package_path_mismatch, "does not match any binary component root"),
    (
        "replay missing selected origin artifact",
        _case_replay_missing_origin_artifact,
        "origin_artifact_refs do not resolve",
    ),
    ("replay invalid origin capture", _case_replay_invalid_origin, "SHA-256 mismatch"),
    ("replay capture cycle", _case_replay_cycle, "replay capture cycle"),
    ("replay cross-topic origin", _case_replay_cross_topic, "different topic"),
)


def _check_ambient_git_isolation(root: Path) -> None:
    names = ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "PATH")
    previous = {name: os.environ.get(name) for name in names}
    fake_directory = root / "fake-bin"
    fake_directory.mkdir()
    marker = root / "fake-git-ran"
    fake_git = fake_directory / "git"
    fake_git.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    fake_git.chmod(0o755)
    fsmonitor_marker = root / "fsmonitor-ran"
    fsmonitor = root / "fake-fsmonitor"
    fsmonitor.write_text(f"#!/bin/sh\ntouch {fsmonitor_marker}\nexit 0\n")
    fsmonitor.chmod(0o755)
    _run("git", "config", "core.fsmonitor", str(fsmonitor), cwd=root)
    try:
        os.environ["GIT_DIR"] = str(root / "does-not-exist.git")
        os.environ["GIT_INDEX_FILE"] = str(root / "attacker-index")
        os.environ["GIT_WORK_TREE"] = str(root.parent)
        os.environ["PATH"] = str(fake_directory)
        errors = _validate(root)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if errors:
        raise AssertionError(
            "ambient GIT_* variables changed validation:\n  " + "\n  ".join(errors)
        )
    if marker.exists():
        raise AssertionError("ambient PATH git executable was invoked")
    if fsmonitor_marker.exists():
        raise AssertionError("target repository core.fsmonitor program was invoked")


def _check_uv_path_isolation(root: Path) -> None:
    capture_baseline = _load_baseline_capture_module()
    fake_directory = root / "fake-uv-bin"
    fake_directory.mkdir()
    marker = root / "fake-uv-ran"
    fake_uv = fake_directory / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        f"touch {shlex.quote(str(marker))}\n"
        "echo 'uv 0.12.9 (x86_64-unknown-linux-gnu)'\n"
    )
    fake_uv.chmod(0o755)
    previous_path = os.environ.get("PATH")
    os.environ["PATH"] = str(fake_directory)
    try:
        try:
            capture_baseline._uv_runtime_identity()
        except capture_baseline.BaselineError as error:
            if "bytes do not match" not in str(error):
                raise AssertionError(
                    f"fake uv failed for an unexpected reason: {error}"
                ) from error
        else:
            raise AssertionError("ambient PATH uv executable was accepted")
    finally:
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path
    if marker.exists():
        raise AssertionError(
            "ambient PATH uv executable was invoked before byte validation"
        )


def _check_dirty_patch_survives_clean_rollback(parent: Path, template: Path) -> None:
    root = parent / "dirty-patch-clean-rollback"
    shutil.copytree(template, root, symlinks=True)
    _run("git", "checkout", "--", "probe.py", cwd=root)
    _mutate_json(
        root,
        CAPTURE_RELATIVE,
        lambda value: value["provenance"]["components"][0]["scope_paths"].append(
            "scratch"
        ),
    )
    with (root / ".git/info/exclude").open("a", encoding="utf-8") as stream:
        stream.write("scratch/\n")
    (root / "scratch").mkdir()
    (root / "scratch/generated.bin").write_bytes(b"post-capture ignored artifact\n")
    errors = _validate(root)
    if errors:
        raise AssertionError(
            "durable patch evidence failed after a clean worktree rollback:\n  "
            + "\n  ".join(errors)
        )


def _check_source_index_survives_checkout_upgrade(parent: Path, template: Path) -> None:
    root = parent / "source-index-history"
    shutil.copytree(template, root, symlinks=True)
    source_root = root / "upstream/jax"
    _run("git", "rm", "--quiet", "api.py", cwd=source_root)
    _run("git", "commit", "--quiet", "-m", "remove indexed source", cwd=source_root)
    errors = _validate(root)
    if errors:
        raise AssertionError(
            "source index stopped resolving its historical revision after checkout upgrade:\n  "
            + "\n  ".join(errors)
        )


def _check_complete_docs_topic_discovery(root: Path) -> None:
    topic_path = root / "docs/alternate/04-discovery/topic.json"
    topic_path.parent.mkdir(parents=True)
    topic_path.write_text("{}\n")
    discovered = VALIDATOR.discover_topics(root, [])
    if topic_path.resolve() not in discovered:
        raise AssertionError("default evidence discovery omitted docs/**/topic.json")


def _check_baseline_mutation(
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    baseline = _read_json(REPOSITORY_ROOT / "manifests/baseline.json")
    mutate(baseline)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".evidence-selftest-baseline-",
        suffix=".json",
        dir=REPOSITORY_ROOT / "manifests",
    )
    os.close(descriptor)
    path = Path(raw_path)
    try:
        _write_json(path, baseline)
        validator = VALIDATOR.EvidenceValidator(REPOSITORY_ROOT)
        try:
            validator.validate_baseline(path)
            errors = list(validator.errors)
        finally:
            validator.close()
    finally:
        path.unlink(missing_ok=True)
    if not any(expected in error for error in errors):
        raise AssertionError(
            f"forged baseline case {name!r} did not report {expected!r}:\n  "
            + "\n  ".join(errors)
        )


def _check_baseline_negatives() -> None:
    _check_baseline_mutation(
        "alignment",
        lambda value: value["status"].__setitem__("reason", "Forged alignment."),
        "alignment fields do not match",
    )

    def mutate_source_status(value: dict[str, Any]) -> None:
        source = value["repository"]["sources"]["jax"]
        source["dirty"] = True
        source["status"] = [" M forged.py"]

    _check_baseline_mutation(
        "source status", mutate_source_status, "dirty flag disagrees"
    )
    _check_baseline_mutation(
        "locked distribution",
        lambda value: value["runtime"]["jaxlib"]["distribution"].__setitem__(
            "artifact_sha256", "0" * 64
        ),
        "does not resolve to one locked wheel",
    )
    _check_baseline_mutation(
        "native binary",
        lambda value: value["runtime"]["jaxlib"]["native_binaries"][0].__setitem__(
            "sha256", "0" * 64
        ),
        "SHA-256 mismatch",
    )


def _check_build_identity_cases(parent: Path, template: Path) -> None:
    valid_root = parent / "build-identity-valid"
    shutil.copytree(template, valid_root, symlinks=True)
    _install_valid_build_identity(valid_root)
    errors = _validate_build_fixture(valid_root)
    if errors:
        raise AssertionError(
            "complete synthetic build identity was rejected:\n  " + "\n  ".join(errors)
        )

    semantic_root = parent / "build-identity-semantic-tamper"
    shutil.copytree(template, semantic_root, symlinks=True)
    manifest_path, _ = _install_valid_build_identity(semantic_root)
    manifest = _read_json(manifest_path)
    manifest["input_fingerprint_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    native = _read_json(semantic_root / NATIVE_RELATIVE)
    identity = native["packages"][0]["identity"]
    identity["build_manifest_sha256"] = _sha256(manifest_path)
    identity["build_manifest_size_bytes"] = manifest_path.stat().st_size
    _write_json(semantic_root / NATIVE_RELATIVE, native)
    _refresh_native_artifact(semantic_root)
    errors = _validate_build_fixture(semantic_root)
    if not any("input_fingerprint_sha256" in error for error in errors):
        raise AssertionError(
            "schema-valid build semantic tamper was not rejected:\n  "
            + "\n  ".join(errors)
        )

    artifact_root = parent / "build-identity-artifact-tamper"
    shutil.copytree(template, artifact_root, symlinks=True)
    _, wheel_path = _install_valid_build_identity(artifact_root)
    wheel_path.write_bytes(b"tampered wheel bytes\n")
    errors = _validate_build_fixture(artifact_root)
    if not any("disagrees with recorded bytes" in error for error in errors):
        raise AssertionError(
            "build wheel byte tamper was not rejected:\n  " + "\n  ".join(errors)
        )

    validator_root = parent / "build-validator-code-tamper"
    shutil.copytree(template, validator_root, symlinks=True)
    _install_valid_build_identity(validator_root)
    marker = validator_root / "untrusted-build-validator-ran"
    with (validator_root / "tools/build-jaxlib.py").open("a", encoding="utf-8") as stream:
        stream.write(f"\nPath({str(marker)!r}).write_text('executed')\n")
    errors = _validate(validator_root)
    if not any("build validator bytes differ" in error for error in errors):
        raise AssertionError(
            "modified repository build validator was not rejected:\n  "
            + "\n  ".join(errors)
        )
    if marker.exists():
        raise AssertionError("modified repository build validator code was executed")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="evidence-selftest-template-") as temporary:
        template = Path(temporary) / "repository"
        template.mkdir()
        _create_valid_fixture(template)
        valid_errors = _validate(template)
        if valid_errors:
            print("FAILED: valid fixture was rejected", file=sys.stderr)
            for error in valid_errors:
                print(f"  {error}", file=sys.stderr)
            return 1

        try:
            _check_ambient_git_isolation(template)
        except AssertionError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        print("OK: ignored ambient GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE/PATH")

        try:
            _check_uv_path_isolation(template)
        except AssertionError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        print("OK: rejected an ambient PATH uv binary before execution")

        try:
            _check_dirty_patch_survives_clean_rollback(Path(temporary), template)
        except AssertionError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        print("OK: accepted durable patch evidence after clean worktree rollback")

        try:
            _check_source_index_survives_checkout_upgrade(Path(temporary), template)
            _check_complete_docs_topic_discovery(template)
        except AssertionError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        print("OK: preserved revision source indexes and discovered every docs topic")

        try:
            _check_build_identity_cases(Path(temporary), template)
        except Exception as error:
            print(f"FAILED: build identity cases: {error}", file=sys.stderr)
            return 1
        print("OK: accepted complete build identity and rejected semantic/artifact tampering")

        for index, (name, mutate, expected) in enumerate(CASES):
            case_root = Path(temporary) / f"case-{index:02d}"
            shutil.copytree(template, case_root, symlinks=True)
            mutate(case_root)
            errors = _validate(
                case_root,
                require_live_source_state=name
                in {
                    "clean source ignored bytecode",
                    "incomplete patch",
                    "ignored untracked scoped source",
                    "live source HEAD mismatch",
                },
            )
            if not errors:
                print(f"FAILED: {name} was accepted", file=sys.stderr)
                return 1
            if not any(expected in error for error in errors):
                print(
                    f"FAILED: {name} did not report {expected!r}", file=sys.stderr
                )
                for error in errors:
                    print(f"  {error}", file=sys.stderr)
                return 1
            print(f"OK: rejected {name}")

        try:
            _check_baseline_negatives()
        except AssertionError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        print("OK: rejected 4 forged baseline states")

    print(
        f"OK: valid fixture, Git/uv environment isolation, durable patch rollback, "
        f"{len(CASES)} negative evidence cases, and 4 baseline cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
