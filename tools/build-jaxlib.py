#!/usr/bin/env python3
"""Build pinned CPU jaxlib with durable attempts and validated provenance."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import copy
import csv
import datetime as dt
from email.parser import BytesParser
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator
import zipfile

from jsonschema import Draft202012Validator, FormatChecker
ROOT = Path(__file__).resolve().parents[1]
JAX_ROOT = ROOT / "upstream/jax"
SOURCE_ROOTS = {"jax": JAX_ROOT, "xla": ROOT / "upstream/xla"}
SCHEMA_PATH = ROOT / "manifests/schema/build-jaxlib.schema.json"
DEFAULT_MANIFEST_ROOT = ROOT / "manifests/build-fingerprints"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/builds"
GLOBAL_LOCK_PATH = DEFAULT_ARTIFACT_ROOT / ".jaxlib-build.lock"
EXPECTED_PYTHON_PATH = ROOT / ".venv/bin/python"
EXPECTED_BASE_PYTHON_PATH = Path("/usr/bin/python3.12")
EXPECTED_BASE_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
EXPECTED_BASE_PYTHON_SIZE_BYTES = 8020928
GENERATED_BAZELRC = JAX_ROOT / ".jax_configure.bazelrc"
BUILD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
GIT_PATH = Path("/usr/bin/git")
EXPECTED_GIT_VERSION = "git version 2.43.0"
EXPECTED_GIT_SHA256 = "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
EXPECTED_SOURCE_COMMITS = {
    "jax": "5832e866449a41c3eea6333416528039119a0fde",
    "xla": "496bd4bd49db9ecbffd85da630b49c860b724604",
}
EXPECTED_TOOL_PATHS = {
    "bazel": JAX_ROOT / "bazel-8.7.0-linux-x86_64",
    "clang": Path("/usr/bin/clang"),
    "clangxx": Path("/usr/bin/clang++"),
    "git": GIT_PATH,
}
EXPECTED_BAZEL_VERSION = "bazel 8.7.0"
EXPECTED_BAZEL_SHA256 = "d7606e679b78067c811096fb3d6cf135225b528835ca396e3a4dddf957859544"
EXPECTED_CLANG_VERSION = "18.1.3"
EXPECTED_JAXLIB_VERSION = "0.11.2.dev0+selfbuilt"
EXPECTED_WHEEL_TAG = "cp312-cp312-manylinux_2_27_x86_64"
REQUIRED_JAXLIB_NATIVE_MEMBERS = {
    "jaxlib/_jax.so",
    "jaxlib/cpu_feature_guard.so",
    "jaxlib/libjax_common.so",
}

FIXED_STARTUP_OPTIONS = ("--nosystem_rc", "--nohome_rc")
# This JAX revision has no tracked MODULE.bazel.lock and ignores generated locks.
# Disable both lockfile reads and writes so an ignored local lock cannot silently
# change module resolution.
FIXED_BUILD_OPTIONS = ("--lockfile_mode=off",)
SAFE_STARTUP_EXACT = {"--batch", "--nobatch"}
SAFE_STARTUP_PATTERNS = (
    re.compile(r"--max_idle_secs=[0-9]+"),
    re.compile(r"--host_jvm_args=-X(?:ms|mx)[0-9]+[kKmMgG]?"),
)
SAFE_BUILD_EXACT = {
    "--announce_rc", "--sandbox_debug", "--show_timestamps", "--subcommands"
}
SAFE_BUILD_PATTERNS = (
    re.compile(r"--local_resources=[A-Za-z0-9_.=,+-]+"),
    re.compile(r"--ram_utilization_factor=[0-9]+(?:\.[0-9]+)?"),
    re.compile(r"--local_(?:cpu|ram)_resources=[0-9]+(?:\.[0-9]+)?"),
    re.compile(r"--experimental_ui_max_stdouterr_bytes=[0-9]+"),
)
FIXED_ENVIRONMENT = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
INSPECTION_ENVIRONMENT = {
    **FIXED_ENVIRONMENT,
    "HOME": "/nonexistent",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}
GIT_FIXED_OPTIONS = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.attributesFile=/dev/null",
    "-c", "core.excludesFile=/dev/null",
)
INHERIT_ENVIRONMENT_NAMES = (
    "ALL_PROXY", "CURL_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_DIR", "SSL_CERT_FILE", "all_proxy",
    "https_proxy", "http_proxy", "no_proxy",
)
INPUT_PATHS = (
    ".python-version",
    "manifests/schema/build-jaxlib.schema.json",
    "pyproject.toml",
    "tools/build-jaxlib.py",
    "tools/check-jaxlib-build-env.py",
    "upstream-sources.lock",
    "upstream/jax/.bazelrc",
    "upstream/jax/.bazelrc.user",
    "upstream/jax/.bazelversion",
    "upstream/jax/MODULE.bazel.lock",
    "upstream/jax/build/build.py",
    "uv.lock",
)


class BuildError(RuntimeError):
    """A build cannot be safely started, read, or resumed."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _default_build_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("cpu-baseline-%Y%m%dT%H%M%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise BuildError(f"path escapes the repository: {path}") from error


def _repo_lexical_path(path: Path) -> str:
    try:
        return path.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError as error:
        raise BuildError(f"path is not repository-owned: {path}") from error


def _recorded_tool_path(path: Path) -> str:
    try:
        return _lexical_absolute(path).relative_to(_lexical_absolute(ROOT)).as_posix()
    except ValueError:
        return str(path)


def _normalize_recorded_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    return _recorded_tool_path(path)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, boundary: Path, label: str) -> Path:
    candidate = _lexical_absolute(path)
    base = _lexical_absolute(boundary)
    try:
        relative = candidate.relative_to(base)
    except ValueError as error:
        raise BuildError(f"{label} escapes {base}: {path}") from error
    repository = _lexical_absolute(ROOT)
    try:
        base.relative_to(repository)
        anchor = repository
    except ValueError:
        anchor = Path(base.anchor)
    current = anchor
    for part in candidate.relative_to(anchor).parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise BuildError(f"{label} traverses a symlink: {current}")
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise BuildError(f"{label} resolves outside {base}: {path}") from error
    return candidate


def _safe_source_root(path: Path, label: str) -> Path:
    safe = _reject_symlink_components(path, ROOT, label)
    if not safe.is_dir():
        raise BuildError(f"{label} must be an existing directory: {path}")
    return safe


def _validate_source_roots() -> None:
    for component, source in SOURCE_ROOTS.items():
        expected = ROOT / "upstream" / component
        if _lexical_absolute(source) != _lexical_absolute(expected):
            raise BuildError(f"{component} source root is not the fixed path {expected}")
        _safe_source_root(source, f"{component} source root")


def _safe_root(raw: Path, allowed: Path, label: str) -> Path:
    text = os.fspath(raw)
    pure = PurePosixPath(text)
    if "\\" in text or ".." in pure.parts or re.match(r"^[A-Za-z]:", text):
        raise BuildError(f"{label} contains an unsafe path component: {text}")
    candidate = raw if raw.is_absolute() else ROOT / raw
    return _reject_symlink_components(candidate, allowed, label)


def _safe_output_path(path: Path, boundary: Path, label: str) -> Path:
    return _reject_symlink_components(path, boundary, label)


def _safe_recorded_path(relative: str, boundary: Path, label: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative
            or re.match(r"^[A-Za-z]:", relative)):
        raise BuildError(f"{label} is not a safe repository-relative path: {relative}")
    return _safe_output_path(ROOT / relative, boundary, label)


def _safe_patch_path(relative: str) -> Path:
    path = _safe_recorded_path(relative, ROOT, "source patch")
    if not os.path.lexists(path):
        raise BuildError(f"source patch does not exist: {relative}")
    if not path.is_file():
        raise BuildError(f"source patch must be a regular non-symlink file: {relative}")
    return path


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


@contextmanager
def _exclusive_lock(path: Path, description: str) -> Iterator[int]:
    safe = _safe_output_path(path, DEFAULT_ARTIFACT_ROOT, f"{description} lock")
    safe.parent.mkdir(parents=True, exist_ok=True)
    safe = _safe_output_path(safe, DEFAULT_ARTIFACT_ROOT, f"{description} lock")
    parent_fd = os.open(safe.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            safe.name,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    os.close(parent_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BuildError(f"{description} lock is not a private regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BuildError(f"{description} is locked by another build: {_repo_path(path)}") from error
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as error:
        raise BuildError(f"cannot read Linux boot identity: {error}") from error


def _process_identity(pid: int) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text()
        command = (root / "cmdline").read_bytes()
        start_time_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None
    if not command:
        return None
    return {"pid": pid, "boot_id": _boot_id(), "start_time_ticks": start_time_ticks,
            "cmdline_sha256": _sha256_bytes(command)}


def _identity_is_live(record: Any) -> bool:
    return (isinstance(record, dict) and isinstance(record.get("pid"), int)
            and _process_identity(record["pid"]) == record)


def _git_head(path: Path) -> str | None:
    process = subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=INSPECTION_ENVIRONMENT,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _file_record(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"exists": False, "kind": "missing", "sha256": None, "size_bytes": None}
    if path.is_symlink():
        return {"exists": True, "kind": "symlink", "sha256": None, "size_bytes": None}
    if path.is_file():
        return {"exists": True, "kind": "file", "sha256": _sha256(path),
                "size_bytes": path.stat().st_size}
    return {"exists": True, "kind": "directory" if path.is_dir() else "other",
            "sha256": None, "size_bytes": None}


def _bytes_file_record(value: bytes) -> dict[str, Any]:
    return {
        "exists": True,
        "kind": "file",
        "sha256": _sha256_bytes(value),
        "size_bytes": len(value),
    }


def _missing_file_record() -> dict[str, Any]:
    return {"exists": False, "kind": "missing", "sha256": None, "size_bytes": None}


def _capture_input_files() -> dict[str, dict[str, Any]]:
    return {relative: _file_record(ROOT / relative) for relative in INPUT_PATHS}


def _run_preflight() -> dict[str, Any]:
    _validate_source_roots()
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools/check-jaxlib-build-env.py"), "--json"],
        cwd=ROOT, check=False, capture_output=True, text=True,
        env=INSPECTION_ENVIRONMENT,
    )
    if process.returncode != 0:
        raise BuildError("preflight inspection crashed: "
                         + (process.stderr.strip() or f"exit code {process.returncode}"))
    try:
        report = json.loads(process.stdout)
        inspected = Path(report["python"]["executable"]).absolute()
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BuildError(f"preflight emitted an invalid report: {error}") from error
    actual = Path(sys.executable).absolute()
    if inspected != actual:
        raise BuildError(f"preflight checked {inspected}, but the wrapper runs with {actual}")
    if actual != EXPECTED_PYTHON_PATH.absolute():
        report["blockers"].append({"code": "python-path",
            "message": f"the wrapper must run with {_repo_lexical_path(EXPECTED_PYTHON_PATH)}"})
        report["ok"] = False
    return report


def _toolchain_fingerprint(preflight: dict[str, Any]) -> dict[str, Any]:
    def selected(name: str, keys: tuple[str, ...]) -> dict[str, Any]:
        source = preflight.get(name) or {}
        return {
            key: (_normalize_recorded_path(source.get(key))
                  if key in {"path", "resolved_path"} else source.get(key))
            for key in keys
        }
    python = preflight.get("python") or {}
    return {
        "python": {
            key: (_normalize_recorded_path(python.get(key))
                  if key in {"executable", "resolved_executable"}
                  else python.get(key))
            for key in (
                "executable", "resolved_executable", "base_executable",
                "base_resolved_path", "base_size_bytes", "base_sha256", "version",
            )
        },
        "bazel": selected("bazel", ("path", "resolved_path", "version", "returncode", "sha256")),
        "clang": selected("clang", ("path", "resolved_path", "version", "returncode", "sha256")),
        "clangxx": selected("clangxx", ("path", "resolved_path", "version", "returncode", "sha256")),
        "git": selected("git", ("path", "resolved_path", "version", "returncode", "sha256")),
    }


def _git_bytes(source: Path, arguments: list[str]) -> bytes:
    _safe_source_root(source, "Git source root")
    process = subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(source), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=INSPECTION_ENVIRONMENT,
    )
    if process.returncode:
        detail = process.stderr.decode(errors="replace").strip()
        raise BuildError(f"git {' '.join(arguments)} failed in {_repo_path(source)}: {detail}")
    return process.stdout


def _git_process(
    source: Path, arguments: list[str], *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    _safe_source_root(source, "Git source root")
    return subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(source), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment or INSPECTION_ENVIRONMENT,
    )


def _tree_file_record(repository: Path, revision: str, relative: str) -> dict[str, Any]:
    listing = _git_process(
        repository, ["ls-tree", "-z", revision, "--", relative]
    )
    if listing.returncode:
        raise BuildError(
            f"cannot inspect {relative} at {revision}: "
            + listing.stderr.decode(errors="replace").strip()
        )
    if not listing.stdout:
        return _missing_file_record()
    entries = [entry for entry in listing.stdout.split(b"\0") if entry]
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise BuildError(f"ambiguous Git tree entry for {relative} at {revision}")
    metadata, recorded_path = entries[0].split(b"\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or recorded_path.decode(errors="strict") != relative:
        raise BuildError(f"malformed Git tree entry for {relative} at {revision}")
    mode, kind, object_id = fields
    if kind != b"blob":
        return {
            "exists": True,
            "kind": "directory" if kind == b"tree" else "other",
            "sha256": None,
            "size_bytes": None,
        }
    if mode == b"120000":
        return {
            "exists": True, "kind": "symlink", "sha256": None, "size_bytes": None
        }
    blob = _git_process(repository, ["cat-file", "blob", object_id.decode()])
    if blob.returncode:
        raise BuildError(
            f"cannot read {relative} at {revision}: "
            + blob.stderr.decode(errors="replace").strip()
        )
    return _bytes_file_record(blob.stdout)


@contextmanager
def _patched_source_index(
    source: Path, base_revision: str, patch_path: Path | None
) -> Iterator[dict[str, str]]:
    descriptor, index_name = tempfile.mkstemp(prefix="jaxlib-reconstruct-index-", dir="/tmp")
    os.close(descriptor)
    Path(index_name).unlink()
    environment = dict(INSPECTION_ENVIRONMENT)
    environment["GIT_INDEX_FILE"] = index_name
    try:
        read_tree = _git_process(
            source, ["read-tree", base_revision], environment=environment
        )
        if read_tree.returncode:
            raise BuildError(
                f"cannot reconstruct source base {base_revision}: "
                + read_tree.stderr.decode(errors="replace").strip()
            )
        if patch_path is not None:
            apply = _git_process(
                source,
                ["apply", "--cached", "--binary", str(patch_path)],
                environment=environment,
            )
            if apply.returncode:
                raise BuildError(
                    f"source patch cannot reconstruct {base_revision}: "
                    + apply.stderr.decode(errors="replace").strip()
                )
        yield environment
    finally:
        try:
            Path(index_name).unlink()
        except FileNotFoundError:
            pass


def _index_file_record(
    source: Path, environment: dict[str, str], relative: str
) -> dict[str, Any]:
    listing = _git_process(
        source, ["ls-files", "-s", "-z", "--", relative], environment=environment
    )
    if listing.returncode:
        raise BuildError(
            f"cannot inspect reconstructed source path {relative}: "
            + listing.stderr.decode(errors="replace").strip()
        )
    entries = [entry for entry in listing.stdout.split(b"\0") if entry]
    if not entries:
        return _missing_file_record()
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise BuildError(f"ambiguous reconstructed source path: {relative}")
    metadata, recorded_path = entries[0].split(b"\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or recorded_path.decode(errors="strict") != relative:
        raise BuildError(f"malformed reconstructed source path: {relative}")
    mode, object_id, stage = fields
    if stage != b"0":
        raise BuildError(f"reconstructed source path is unmerged: {relative}")
    if mode == b"120000":
        return {
            "exists": True, "kind": "symlink", "sha256": None, "size_bytes": None
        }
    blob = _git_process(
        source, ["cat-file", "blob", object_id.decode()], environment=environment
    )
    if blob.returncode:
        raise BuildError(
            f"cannot read reconstructed source path {relative}: "
            + blob.stderr.decode(errors="replace").strip()
        )
    return _bytes_file_record(blob.stdout)


def _source_diff(source: Path) -> bytes:
    return _git_bytes(source, ["diff", "--binary", "--full-index", "--no-color",
        "--no-ext-diff", "--no-textconv", "HEAD", "--", "."])


def _source_observation(component: str) -> dict[str, Any]:
    source = _safe_source_root(SOURCE_ROOTS[component], f"{component} source root")
    head = _git_bytes(source, ["rev-parse", "HEAD"]).decode().strip()
    top_level = _git_bytes(source, ["rev-parse", "--show-toplevel"]).decode().strip()
    if Path(top_level).resolve() != source.resolve():
        raise BuildError(f"{component} Git top-level is not its fixed source root")
    status = _git_bytes(source, ["status", "--porcelain=v1", "--untracked-files=all",
                                 "--ignore-submodules=none"]).decode(errors="replace").splitlines()
    flags = _git_bytes(source, ["ls-files", "-v", "-z"])
    index_flags = [item.decode(errors="replace") for item in flags.split(b"\0")
                   if item and (item[:1] == b"S" or item[:1].islower())]
    ignored_raw = _git_bytes(
        source, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
    )
    ignored = [item.decode(errors="replace")
               for item in ignored_raw.split(b"\0") if item]
    return {
        "path": f"upstream/{component}",
        "exists": True,
        "commit": head,
        "top_level": f"upstream/{component}",
        "expected_commit": EXPECTED_SOURCE_COMMITS[component],
        "status_returncode": 0,
        "dirty": bool(status),
        "status": status,
        "index_flags": index_flags,
        "index_flags_returncode": 0,
        "ignored": ignored,
        "ignored_returncode": 0,
    }


def _ignored_source_path_allowed(component: str, relative: str) -> bool:
    if relative in {"bazel-bin", "bazel-jax", "bazel-out", "bazel-testlogs", "bazel-xla"}:
        return True
    if component != "jax":
        return False
    if relative in {
        ".jax_configure.bazelrc",
        "MODULE.bazel.lock",
        "bazel-8.7.0-linux-x86_64",
    }:
        return True
    pure = PurePosixPath(relative)
    if pure.parts and pure.parts[0] == "jax.egg-info":
        return True
    return False


def _generated_ignored_source_path(component: str, relative: str) -> bool:
    """Return whether build.py/Bazel may create this non-input ignored path."""
    if relative in {"bazel-bin", "bazel-jax", "bazel-out", "bazel-testlogs", "bazel-xla"}:
        return True
    return component == "jax" and relative == ".jax_configure.bazelrc"


def _immutable_source_observation(component: str, value: dict[str, Any]) -> dict[str, Any]:
    """Remove controlled output-presence transitions from an input observation."""
    result = copy.deepcopy(value)
    result["ignored"] = [
        relative for relative in result.get("ignored", [])
        if not _generated_ignored_source_path(component, relative)
    ]
    return result


def _reject_untracked_or_submodule_dirty(component: str, source: Path) -> None:
    untracked = _git_bytes(source, ["ls-files", "--others", "--exclude-standard", "-z"])
    if untracked:
        names = [item.decode(errors="replace") for item in untracked.split(b"\0") if item]
        raise BuildError(f"{component} source patch cannot include untracked files: {names}")
    flags = _git_bytes(source, ["ls-files", "-v", "-z"])
    hidden = [item.decode(errors="replace") for item in flags.split(b"\0")
              if item and (item[:1] == b"S" or item[:1].islower())]
    if hidden:
        raise BuildError(
            f"{component} source contains assume-unchanged/skip-worktree entries: {hidden}"
        )
    ignored = _git_bytes(
        source, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
    )
    unsafe_ignored = [
        item.decode(errors="replace") for item in ignored.split(b"\0")
        if item and not _ignored_source_path_allowed(component, item.decode(errors="replace"))
    ]
    if unsafe_ignored:
        raise BuildError(
            f"{component} source contains non-allowlisted ignored files: {unsafe_ignored}"
        )
    porcelain = _git_bytes(source, ["status", "--porcelain=v2", "-z",
        "--untracked-files=all", "--ignore-submodules=none"])
    for record in porcelain.split(b"\0"):
        if record.startswith((b"1 ", b"2 ")):
            fields = record.split(b" ", 3)
            if len(fields) >= 3 and fields[2].startswith(b"S"):
                raise BuildError(f"{component} source contains a dirty or changed nested submodule")
        elif record.startswith(b"u "):
            raise BuildError(f"{component} source contains unresolved index entries")


def _git_apply_check(source: Path, base_revision: str, patch_path: Path) -> None:
    descriptor, index_name = tempfile.mkstemp(prefix="jaxlib-patch-index-", dir="/tmp")
    os.close(descriptor)
    Path(index_name).unlink()
    environment = dict(INSPECTION_ENVIRONMENT)
    environment["GIT_INDEX_FILE"] = index_name
    try:
        read_tree = subprocess.run([str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(source), "read-tree", base_revision],
                                   check=False, capture_output=True, env=environment)
        if read_tree.returncode:
            raise BuildError("cannot create the temporary patch-check index: "
                             + read_tree.stderr.decode(errors="replace").strip())
        apply = subprocess.run([str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(source), "apply", "--check", "--cached",
                                "--binary", str(patch_path)], check=False,
                               capture_output=True, env=environment)
        if apply.returncode:
            raise BuildError(f"source patch does not apply to {base_revision}: "
                             + apply.stderr.decode(errors="replace").strip())
    finally:
        try:
            Path(index_name).unlink()
        except FileNotFoundError:
            pass


def _parse_source_patch_args(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        component, separator, relative = value.partition("=")
        if not separator or component not in SOURCE_ROOTS or not relative:
            raise BuildError("--source-patch must be jax=REPO_RELATIVE_PATCH or xla=REPO_RELATIVE_PATCH")
        if component in result:
            raise BuildError(f"--source-patch was supplied more than once for {component}")
        _safe_patch_path(relative)
        result[component] = relative
    return result


def _validate_source_patches(specifications: dict[str, str],
                             preflight: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in sorted(specifications):
        source_report = (preflight.get("sources") or {}).get(component) or {}
        source, base = SOURCE_ROOTS[component], source_report.get("expected_commit")
        if not source_report.get("exists") or not isinstance(base, str):
            raise BuildError(f"cannot validate a patch for missing {component} source")
        if source_report.get("commit") != base:
            raise BuildError(f"{component} source patch requires HEAD at its pinned base {base}")
        _reject_untracked_or_submodule_dirty(component, source)
        patch_path = _safe_patch_path(specifications[component])
        patch_bytes, diff_bytes = patch_path.read_bytes(), _source_diff(source)
        if not diff_bytes:
            raise BuildError(f"{component} source patch was supplied but the source diff is empty")
        if patch_bytes != diff_bytes:
            raise BuildError(f"{component} patch bytes do not exactly match the pinned git diff command")
        _git_apply_check(source, base, patch_path)
        records.append({"component": component, "base_revision": base,
            "path": specifications[component], "size_bytes": len(patch_bytes),
            "sha256": _sha256_bytes(patch_bytes)})
    return records


def _input_blockers(files: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for relative, record in files.items():
        if relative == "upstream/jax/.bazelrc.user":
            if record["kind"] != "missing":
                result.append({"code": "bazelrc-user-present",
                    "message": "upstream/jax/.bazelrc.user must have no directory entry"})
        elif relative == "upstream/jax/MODULE.bazel.lock":
            if record["kind"] not in {"missing", "file"}:
                result.append({"code": "module-lock-invalid",
                    "message": "upstream/jax/MODULE.bazel.lock must be absent or a regular file"})
        elif record["kind"] != "file":
            result.append({"code": f"input-file:{relative}",
                "message": f"required build input is not a regular file: {relative}"})
    return result


def _effective_preflight(report: dict[str, Any], patch_records: list[dict[str, Any]],
                         files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patches = {record["component"]: record for record in patch_records}
    blockers: list[dict[str, str]] = []
    waived: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for blocker in [*(report.get("blockers") or []), *_input_blockers(files)]:
        key = (blocker["code"], blocker["message"])
        if key in seen:
            continue
        seen.add(key)
        component = blocker["code"].removeprefix("source-dirty:")
        if blocker["code"].startswith("source-dirty:") and component in patches:
            waived.append({**blocker, "source_patch_sha256": patches[component]["sha256"]})
        else:
            blockers.append(blocker)
    return {"ok": not blockers, "blockers": blockers, "waived_blockers": waived}


def _validate_output_user_root(option: str) -> None:
    raw = option.split("=", maxsplit=1)[1]
    path = Path(raw)
    if (not path.is_absolute() or ".." in PurePosixPath(raw).parts or "\\" in raw
            or not re.fullmatch(r"/[-A-Za-z0-9_./]+", raw)):
        raise BuildError("--output_user_root must be an absolute safe path")
    resolved = path.resolve()
    allowed = (Path("/tmp").resolve(), DEFAULT_ARTIFACT_ROOT.resolve())
    matching = next((root for root in allowed
                     if resolved == root or root in resolved.parents), None)
    if matching is None:
        raise BuildError("--output_user_root must stay under /tmp or artifacts/builds")
    _reject_symlink_components(path, matching, "--output_user_root")


def _validate_extension_options(startup_options: list[str], build_options: list[str]) -> None:
    for option in startup_options:
        if option.startswith("--output_user_root="):
            _validate_output_user_root(option)
        elif option in SAFE_STARTUP_EXACT or any(
                pattern.fullmatch(option) for pattern in SAFE_STARTUP_PATTERNS):
            continue
        else:
            raise BuildError(f"unsupported Bazel startup option: {option}")
    for option in build_options:
        if option in SAFE_BUILD_EXACT or any(
                pattern.fullmatch(option) for pattern in SAFE_BUILD_PATTERNS):
            continue
        raise BuildError(f"unsupported Bazel build option: {option}")


def _network_environment_snapshot() -> dict[str, str]:
    return {name: os.environ[name] for name in INHERIT_ENVIRONMENT_NAMES if name in os.environ}


def _build_config(preflight: dict[str, Any], jobs: int, startup_options: list[str],
                  build_options: list[str], dry_run: bool,
                  network_environment: dict[str, str]) -> dict[str, Any]:
    _validate_extension_options(startup_options, build_options)
    return {
        "cwd": "upstream/jax",
        "python_executable": _repo_lexical_path(Path(preflight["python"]["executable"])),
        "jobs": jobs,
        "bazel_startup_options": [*FIXED_STARTUP_OPTIONS, *startup_options],
        "bazel_options": [f"--jobs={jobs}", *FIXED_BUILD_OPTIONS, *build_options],
        "dry_run": dry_run,
        "environment_policy": {
            "fixed": dict(FIXED_ENVIRONMENT),
            "per_attempt_directories": {"HOME": "home", "TMPDIR": "tmp"},
            "inherited_names": sorted(network_environment),
        },
    }


def _attempt_environment_record(attempt_dir: Path,
                                config: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(config["environment_policy"]["fixed"])
    fixed["HOME"], fixed["TMPDIR"] = (_repo_path(attempt_dir / "home"),
                                        _repo_path(attempt_dir / "tmp"))
    return {"fixed": fixed,
            "inherited_names": config["environment_policy"]["inherited_names"]}


def _build_environment(config: dict[str, Any], attempt_dir: Path,
                       network_environment: dict[str, str]) -> dict[str, str]:
    if sorted(network_environment) != config["environment_policy"]["inherited_names"]:
        raise BuildError("network proxy/certificate environment names changed before launch")
    home, temporary = attempt_dir / "home", attempt_dir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    environment = dict(config["environment_policy"]["fixed"])
    environment.update({"HOME": str(home), "TMPDIR": str(temporary)})
    environment.update(network_environment)
    return environment


def _build_command(attempt_dir: Path, config: dict[str, Any]) -> list[str]:
    wheel_dir = attempt_dir / "wheels"
    command = [
        os.path.relpath(ROOT / config["python_executable"], JAX_ROOT),
        "-S", "build/build.py", "build", "--wheels=jaxlib", "--python_version=3.12",
        "--local_xla_path=../xla",
        f"--output_path={os.path.relpath(wheel_dir, JAX_ROOT)}",
        "--bazel_path=./bazel-8.7.0-linux-x86_64", "--clang_path=/usr/bin/clang",
    ]
    command.extend(f"--bazel_startup_options={item}"
                   for item in config["bazel_startup_options"])
    command.extend(f"--bazel_options={item}" for item in config["bazel_options"])
    command.append("--verbose")
    if config["dry_run"]:
        command.append("--dry_run")
    return command


def _fingerprint_payload(inputs: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(inputs)
    value.pop("analysis_repository_commit", None)
    for component, source in value.get("sources", {}).items():
        value["sources"][component] = _immutable_source_observation(component, source)
    return value


def _build_inputs(preflight: dict[str, Any], config: dict[str, Any], build_id: str,
                  patch_records: list[dict[str, Any]],
                  files: dict[str, dict[str, Any]] | None = None
                  ) -> tuple[dict[str, Any], str]:
    sources: dict[str, Any] = {}
    for name in ("jax", "xla"):
        source = (preflight.get("sources") or {}).get(name) or {}
        sources[name] = {key: source.get(key) for key in (
            "path", "exists", "commit", "top_level", "expected_commit", "status_returncode",
            "dirty", "status", "index_flags", "index_flags_returncode",
            "ignored", "ignored_returncode")}
    value = {
        "analysis_repository_commit": _git_head(ROOT),
        "sources": sources,
        "source_patches": patch_records,
        "files": files if files is not None else _capture_input_files(),
        "toolchain": _toolchain_fingerprint(preflight),
        "build_config": config,
        "build_id": build_id,
    }
    return value, _sha256_bytes(_canonical_bytes(_fingerprint_payload(value)))


def _empty_result(error: str | None = None) -> dict[str, Any]:
    return {"attempt_number": None, "exit_code": None, "error": error,
            "wheels": [], "generated_bazelrc": None}


def _new_manifest(*, build_id: str, config: dict[str, Any], inputs: dict[str, Any],
                  input_fingerprint: str, manifest_path: Path, artifact_dir: Path,
                  build_lock_path: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "$schema": os.path.relpath(SCHEMA_PATH, manifest_path.parent),
        "schema_version": "1.0", "build_id": build_id,
        "kind": "jaxlib-cpu-source-build", "dry_run": config["dry_run"],
        "status": "planned", "created_at": now, "updated_at": now,
        "paths": {
            "manifest": _repo_path(manifest_path), "artifact_dir": _repo_path(artifact_dir),
            "build_lock": _repo_path(build_lock_path), "global_lock": _repo_path(GLOBAL_LOCK_PATH),
            "generated_bazelrc": _repo_path(GENERATED_BAZELRC),
        },
        "build_config": config, "input_fingerprint_sha256": input_fingerprint,
        "inputs": inputs, "preflight": preflight, "attempts": [],
        "result": _empty_result(),
        "runtime_validation": {"status": "not-run", "checked_at": None,
            "evidence": [],
            "notes": "A successful build still requires isolated install and runtime validation."},
    }


def _load_schema() -> dict[str, Any]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text())
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise BuildError(f"cannot load build manifest schema: {error}") from error
    return schema


def _artifact_record(path: Path, boundary: Path) -> dict[str, Any]:
    safe = _safe_output_path(path, boundary, "artifact")
    if safe.is_symlink() or not safe.is_file():
        raise BuildError(f"artifact must be a regular non-symlink file: {_repo_lexical_path(path)}")
    return {"path": _repo_path(safe), "size_bytes": safe.stat().st_size,
            "sha256": _sha256(safe)}


def _validate_wheel_archive(path: Path) -> None:
    expected_filename = (
        f"jaxlib-{EXPECTED_JAXLIB_VERSION}-{EXPECTED_WHEEL_TAG}.whl"
    )
    if path.name != expected_filename:
        raise BuildError(
            "wheel filename must be the pinned jaxlib project/version/tag: "
            + expected_filename
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                raise BuildError("wheel contains duplicate ZIP member names")
            for name in names:
                pure = PurePosixPath(name)
                if (not name or pure.is_absolute() or ".." in pure.parts
                        or "\\" in name):
                    raise BuildError(f"wheel contains an unsafe member path: {name!r}")
            dist_info = {
                PurePosixPath(name).parts[0]
                for name in names
                if PurePosixPath(name).parts
                and PurePosixPath(name).parts[0].endswith(".dist-info")
            }
            if len(dist_info) != 1:
                raise BuildError("wheel must contain exactly one .dist-info directory")
            directory = next(iter(dist_info))
            expected_dist_info = f"jaxlib-{EXPECTED_JAXLIB_VERSION}.dist-info"
            if directory != expected_dist_info:
                raise BuildError(
                    "wheel .dist-info directory disagrees with the pinned project/version"
                )
            metadata_name = f"{directory}/METADATA"
            wheel_name = f"{directory}/WHEEL"
            record_name = f"{directory}/RECORD"
            missing = sorted({metadata_name, wheel_name, record_name} - set(names))
            if missing:
                raise BuildError("wheel omits required metadata: " + ", ".join(missing))
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            if metadata.get("Name", "").strip().lower().replace("_", "-") != "jaxlib":
                raise BuildError("wheel METADATA Name is not jaxlib")
            if metadata.get("Version", "").strip() != EXPECTED_JAXLIB_VERSION:
                raise BuildError("wheel METADATA Version disagrees with the pinned version")
            wheel_headers = BytesParser().parsebytes(archive.read(wheel_name))
            if not wheel_headers.get("Wheel-Version"):
                raise BuildError("wheel WHEEL has no Wheel-Version")
            if wheel_headers.get("Root-Is-Purelib", "").lower() != "false":
                raise BuildError("wheel WHEEL must declare Root-Is-Purelib: false")
            wheel_tags = set(wheel_headers.get_all("Tag", []))
            if wheel_tags != {EXPECTED_WHEEL_TAG}:
                raise BuildError("wheel WHEEL Tag disagrees with the pinned platform tag")
            missing_native = sorted(REQUIRED_JAXLIB_NATIVE_MEMBERS - set(names))
            if missing_native:
                raise BuildError(
                    "wheel omits required jaxlib native payload: "
                    + ", ".join(missing_native)
                )
            for native_name in sorted(REQUIRED_JAXLIB_NATIVE_MEMBERS):
                native_header = archive.read(native_name)[:64]
                if (len(native_header) < 20 or native_header[:4] != b"\x7fELF"
                        or native_header[4:6] != b"\x02\x01"
                        or int.from_bytes(native_header[18:20], "little") != 62):
                    raise BuildError(
                        f"wheel native member is not an x86-64 ELF object: {native_name}"
                    )
            rows: dict[str, tuple[str, str]] = {}
            record_text = archive.read(record_name).decode("utf-8")
            for row in csv.reader(io.StringIO(record_text, newline="")):
                if len(row) != 3 or not row[0] or row[0] in rows:
                    raise BuildError("wheel RECORD has a malformed or duplicate row")
                rows[row[0]] = (row[1], row[2])
            if set(rows) != set(names):
                raise BuildError("wheel RECORD paths do not exactly match ZIP members")
            for member in members:
                digest_field, size_field = rows[member.filename]
                if member.filename == record_name:
                    if digest_field or size_field:
                        raise BuildError("wheel RECORD must leave its own hash and size empty")
                    continue
                if not digest_field.startswith("sha256="):
                    raise BuildError(f"wheel RECORD lacks SHA-256 for {member.filename}")
                encoded = digest_field.removeprefix("sha256=")
                try:
                    expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                    expected_size = int(size_field)
                except (ValueError, TypeError) as error:
                    raise BuildError(
                        f"wheel RECORD has invalid hash/size for {member.filename}"
                    ) from error
                digest = hashlib.sha256()
                size = 0
                with archive.open(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                if digest.digest() != expected or size != expected_size:
                    raise BuildError(
                        f"wheel RECORD hash/size disagrees for {member.filename}"
                    )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise BuildError(f"wheel is not a valid readable archive: {error}") from error


def _verify_artifact(record: dict[str, Any], boundary: Path, label: str) -> str | None:
    try:
        path = _safe_recorded_path(record["path"], boundary, label)
        if path.is_symlink() or not path.is_file():
            return f"{label} is missing or not a regular non-symlink file: {record['path']}"
        if path.stat().st_size != record["size_bytes"]:
            return f"{label} size disagrees with recorded bytes: {record['path']}"
        if _sha256(path) != record["sha256"]:
            return f"{label} SHA-256 disagrees with recorded bytes: {record['path']}"
    except (BuildError, OSError, KeyError) as error:
        return str(error)
    return None


def _fixed_input_errors(value: dict[str, Any], verify_current: bool) -> list[str]:
    errors: list[str] = []
    inputs = value["inputs"]
    tools = inputs["toolchain"]
    expected_tools: dict[str, dict[str, Any]] = {
        "bazel": {
            "path": _repo_path(EXPECTED_TOOL_PATHS["bazel"]),
            "resolved_path": _recorded_tool_path(EXPECTED_TOOL_PATHS["bazel"]),
            "version": EXPECTED_BAZEL_VERSION,
            "returncode": 0,
            "sha256": EXPECTED_BAZEL_SHA256,
        },
        "git": {
            "path": str(GIT_PATH),
            "resolved_path": str(GIT_PATH.resolve()),
            "version": EXPECTED_GIT_VERSION,
            "returncode": 0,
            "sha256": EXPECTED_GIT_SHA256,
        },
    }
    for name in ("clang", "clangxx"):
        path = EXPECTED_TOOL_PATHS[name]
        expected_tools[name] = {
            "path": _recorded_tool_path(path),
            "resolved_path": _recorded_tool_path(path.resolve()),
            "returncode": 0,
            "sha256": _sha256(path) if path.is_file() else None,
        }
    for name, expected in expected_tools.items():
        record = tools[name]
        for key, wanted in expected.items():
            if record.get(key) != wanted:
                errors.append(f"toolchain {name}.{key} is not the fixed value")
        if name in {"clang", "clangxx"}:
            version = record.get("version")
            if (not isinstance(version, str)
                    or re.search(rf"\bclang version {re.escape(EXPECTED_CLANG_VERSION)}\b",
                                 version) is None):
                errors.append(f"toolchain {name}.version is not Clang {EXPECTED_CLANG_VERSION}")
    python = tools["python"]
    expected_python = {
        "executable": _repo_lexical_path(EXPECTED_PYTHON_PATH),
        "resolved_executable": str(EXPECTED_BASE_PYTHON_PATH),
        "base_executable": str(EXPECTED_BASE_PYTHON_PATH),
        "base_resolved_path": str(EXPECTED_BASE_PYTHON_PATH),
        "base_size_bytes": EXPECTED_BASE_PYTHON_SIZE_BYTES,
        "base_sha256": EXPECTED_BASE_PYTHON_SHA256,
        "version": "3.12.3",
    }
    if python != expected_python:
        errors.append("toolchain Python is not the fixed repository interpreter")

    patches = {item["component"]: item for item in inputs["source_patches"]}
    for component, source in inputs["sources"].items():
        expected_dirty = component in patches
        if source.get("path") != f"upstream/{component}":
            errors.append(f"{component} source path is not fixed")
        if source.get("top_level") != f"upstream/{component}":
            errors.append(f"{component} Git top-level is not the fixed source root")
        if (source.get("exists") is not True
                or source.get("commit") != EXPECTED_SOURCE_COMMITS[component]
                or source.get("expected_commit") != EXPECTED_SOURCE_COMMITS[component]
                or source.get("status_returncode") != 0):
            errors.append(f"{component} source revision/status is not fixed")
        if source.get("dirty") != bool(source.get("status")):
            errors.append(f"{component} source dirty flag disagrees with status")
        if source.get("dirty") != expected_dirty:
            errors.append(f"{component} source dirty state is not matched by one patch")
        if source.get("index_flags"):
            errors.append(f"{component} source contains hidden index flags")
        if source.get("index_flags_returncode") != 0:
            errors.append(f"{component} hidden-index check did not succeed")
        unsafe = [item for item in source.get("ignored", [])
                  if not _ignored_source_path_allowed(component, item)]
        if unsafe:
            errors.append(f"{component} source contains non-allowlisted ignored files")
        if source.get("ignored_returncode") != 0:
            errors.append(f"{component} ignored-file check did not succeed")
        if component in patches and patches[component]["base_revision"] != EXPECTED_SOURCE_COMMITS[component]:
            errors.append(f"{component} source patch base is not the fixed revision")
        if verify_current:
            try:
                observed = _source_observation(component)
                if (_immutable_source_observation(component, observed)
                        != _immutable_source_observation(component, source)):
                    errors.append(f"{component} recorded source state differs from the current tree")
            except (BuildError, OSError, UnicodeError) as error:
                errors.append(f"cannot replay {component} source state: {error}")

    file_blockers = _input_blockers(inputs["files"])
    if file_blockers:
        errors.append("recorded build inputs contain blockers: "
                      + ", ".join(item["code"] for item in file_blockers))
    if verify_current:
        try:
            if _capture_input_files() != inputs["files"]:
                errors.append("recorded input files differ from current bytes")
        except OSError as error:
            errors.append(f"cannot replay input file bytes: {error}")
    return errors


def _historical_input_errors(value: dict[str, Any]) -> list[str]:
    """Reconstruct every recorded input file from committed objects and patches."""
    errors: list[str] = []
    inputs = value["inputs"]
    analysis_revision = inputs.get("analysis_repository_commit")
    if not isinstance(analysis_revision, str):
        return ["analysis_repository_commit must identify the committed wrapper inputs"]
    try:
        top_level = _git_bytes(ROOT, ["rev-parse", "--show-toplevel"]).decode().strip()
        if Path(top_level).resolve() != ROOT.resolve():
            errors.append("analysis Git top-level is not the repository root")
        resolved_revision = _git_bytes(
            ROOT, ["rev-parse", "--verify", f"{analysis_revision}^{{commit}}"]
        ).decode().strip()
        if resolved_revision != analysis_revision:
            errors.append("analysis_repository_commit is not an exact commit object")
    except (BuildError, OSError, UnicodeError) as error:
        return [f"cannot resolve analysis_repository_commit: {error}"]

    expected: dict[str, dict[str, Any]] = {}
    try:
        for relative in INPUT_PATHS:
            if not relative.startswith("upstream/jax/"):
                expected[relative] = _tree_file_record(
                    ROOT, analysis_revision, relative
                )

        jax_relative = [
            relative.removeprefix("upstream/jax/")
            for relative in INPUT_PATHS
            if relative.startswith("upstream/jax/")
        ]
        patches = {
            item["component"]: item for item in inputs["source_patches"]
        }
        patch_path = (
            _safe_patch_path(patches["jax"]["path"]) if "jax" in patches else None
        )
        with _patched_source_index(
            SOURCE_ROOTS["jax"], EXPECTED_SOURCE_COMMITS["jax"], patch_path
        ) as environment:
            for relative in jax_relative:
                expected[f"upstream/jax/{relative}"] = _index_file_record(
                    SOURCE_ROOTS["jax"], environment, relative
                )
    except (BuildError, OSError, UnicodeError) as error:
        errors.append(f"cannot reconstruct recorded input files: {error}")
        return errors

    for relative in INPUT_PATHS:
        if expected.get(relative) != inputs["files"].get(relative):
            errors.append(
                f"recorded input file does not match committed reconstruction: {relative}"
            )
    return errors


def _semantic_manifest_errors(value: dict[str, Any], path: Path,
                              expected_build_id: str | None,
                              expected_artifact_dir: Path | None,
                              verify_artifacts: bool,
                              verify_current_inputs: bool) -> list[str]:
    errors: list[str] = []
    if expected_build_id is not None and value["build_id"] != expected_build_id:
        errors.append("build_id does not match the requested manifest")
    try:
        if (path.parent / value["$schema"]).resolve() != SCHEMA_PATH.resolve():
            errors.append("$schema does not resolve to the pinned build schema")
    except (OSError, ValueError):
        errors.append("$schema is not a safe local schema reference")
    if value["paths"]["manifest"] != _repo_path(path):
        errors.append("paths.manifest does not match the loaded file")
    if value["preflight"]["ok"] != (not value["preflight"]["blockers"]):
        errors.append("preflight.ok disagrees with preflight.blockers")
    if value["dry_run"] != value["build_config"]["dry_run"]:
        errors.append("dry_run disagrees with build_config.dry_run")
    if value["inputs"]["build_config"] != value["build_config"]:
        errors.append("inputs.build_config disagrees with build_config")
    if value["inputs"]["build_id"] != value["build_id"]:
        errors.append("inputs.build_id disagrees with build_id")
    if set(value["inputs"]["files"]) != set(INPUT_PATHS):
        errors.append("inputs.files does not contain the exact pinned input path set")
    if value["preflight"]["ok"]:
        errors.extend(_fixed_input_errors(value, verify_current_inputs))

    artifact_dir = ROOT / value["paths"]["artifact_dir"]
    artifact_dir_safe = True
    try:
        artifact_dir = _safe_output_path(artifact_dir, DEFAULT_ARTIFACT_ROOT,
                                         "recorded artifact directory")
    except BuildError as error:
        errors.append(str(error))
        artifact_dir_safe = False
        artifact_dir = DEFAULT_ARTIFACT_ROOT / "__invalid__"
    if artifact_dir_safe:
        if (expected_artifact_dir is not None
                and artifact_dir != _lexical_absolute(expected_artifact_dir)):
            errors.append("paths.artifact_dir does not match the requested build id")
        if artifact_dir.name != value["build_id"]:
            errors.append("paths.artifact_dir must end in the build id")
        try:
            expected_lock = _repo_path(artifact_dir / ".build-id.lock")
            if value["paths"]["build_lock"] != expected_lock:
                errors.append("paths.build_lock does not match paths.artifact_dir")
        except BuildError as error:
            errors.append(str(error))
    if path.name != f"{value['build_id']}.json":
        errors.append("manifest filename must equal build_id.json")

    config = value["build_config"]
    startup_options, build_options = (config["bazel_startup_options"],
                                      config["bazel_options"])
    if startup_options[:len(FIXED_STARTUP_OPTIONS)] != list(FIXED_STARTUP_OPTIONS):
        errors.append("build_config omits the fixed Bazel rc isolation options")
    expected_prefix = [f"--jobs={config['jobs']}", *FIXED_BUILD_OPTIONS]
    if build_options[:len(expected_prefix)] != expected_prefix:
        errors.append("build_config.bazel_options omits fixed jobs/lockfile options")
    try:
        _validate_extension_options(startup_options[len(FIXED_STARTUP_OPTIONS):],
                                    build_options[len(expected_prefix):])
    except BuildError as error:
        errors.append(str(error))
    inherited = config["environment_policy"]["inherited_names"]
    expected_policy = {
        "fixed": dict(FIXED_ENVIRONMENT),
        "per_attempt_directories": {"HOME": "home", "TMPDIR": "tmp"},
        "inherited_names": sorted(inherited),
    }
    if config["environment_policy"] != expected_policy:
        errors.append("build_config contains an unknown environment policy")
    if any(name not in INHERIT_ENVIRONMENT_NAMES for name in inherited):
        errors.append("build_config inherits a non-allowlisted environment name")
    python_input = value["inputs"]["toolchain"]["python"].get("executable")
    if (not isinstance(python_input, str)
            or python_input != config["python_executable"]):
        errors.append("build_config Python disagrees with the toolchain fingerprint")
    actual_fingerprint = _sha256_bytes(_canonical_bytes(
        _fingerprint_payload(value["inputs"])))
    if actual_fingerprint != value["input_fingerprint_sha256"]:
        errors.append("input_fingerprint_sha256 does not match build inputs")

    patches = {item["component"]: item for item in value["inputs"]["source_patches"]}
    if len(patches) != len(value["inputs"]["source_patches"]):
        errors.append("source patch components must be unique")
    for waived in value["preflight"]["waived_blockers"]:
        component = waived["code"].removeprefix("source-dirty:")
        if not waived["code"].startswith("source-dirty:") or component not in patches:
            errors.append("only a matching source-dirty blocker may be waived")
        elif waived["source_patch_sha256"] != patches[component]["sha256"]:
            errors.append("waived blocker does not name its matching source patch")
    if verify_artifacts:
        for item in value["inputs"]["source_patches"]:
            message = _verify_artifact(item, ROOT, f"{item['component']} source patch")
            if message:
                errors.append(message)
                continue
            try:
                _git_apply_check(
                    SOURCE_ROOTS[item["component"]],
                    item["base_revision"],
                    _safe_patch_path(item["path"]),
                )
            except (BuildError, OSError) as error:
                errors.append(
                    f"cannot reconstruct {item['component']} source patch: {error}"
                )
        if value["preflight"]["ok"] and patches and verify_current_inputs:
            try:
                replayed = _validate_source_patches(
                    {component: item["path"] for component, item in patches.items()},
                    {"sources": value["inputs"]["sources"]},
                )
                if replayed != value["inputs"]["source_patches"]:
                    errors.append("source patch records do not match replayed patches")
            except (BuildError, OSError) as error:
                errors.append(f"cannot replay source patches: {error}")
        if value["preflight"]["ok"]:
            errors.extend(_historical_input_errors(value))

    attempts = value["attempts"]
    if [item["number"] for item in attempts] != list(range(1, len(attempts) + 1)):
        errors.append("attempt numbers must be contiguous and start at one")
    terminal_attempts = {"dry-run-succeeded", "build-failed", "interrupted",
                         "build-succeeded"}
    try:
        created_at = _parse_datetime(value["created_at"])
        updated_at = _parse_datetime(value["updated_at"])
    except ValueError as error:
        errors.append(f"manifest timestamps cannot be ordered: {error}")
        created_at = updated_at = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    previous_end = created_at
    saw_success = False
    for attempt in attempts:
        number = attempt["number"]
        expected_dir = artifact_dir / "attempts" / f"{number:04d}"
        expected_paths = {
            "attempt_dir": _repo_path(expected_dir),
            "log": _repo_path(expected_dir / "build.log"),
            "wheel_dir": _repo_path(expected_dir / "wheels"),
            "generated_bazelrc_archive": _repo_path(
                expected_dir / "generated.jax_configure.bazelrc"),
            "home_dir": _repo_path(expected_dir / "home"),
            "tmp_dir": _repo_path(expected_dir / "tmp"),
        }
        if attempt["paths"] != expected_paths:
            errors.append(f"attempt {number} paths do not match its isolated directory")
        if saw_success:
            errors.append(f"attempt {number} follows an already successful attempt")
        if attempt["status"] in {"dry-run-succeeded", "build-succeeded"}:
            saw_success = True
        if (config["dry_run"] and attempt["status"] == "build-succeeded"):
            errors.append(f"attempt {number} reports a real build in a dry-run manifest")
        if (not config["dry_run"] and attempt["status"] == "dry-run-succeeded"):
            errors.append(f"attempt {number} reports a dry-run in a real-build manifest")
        try:
            started_at = _parse_datetime(attempt["started_at"])
            if started_at < created_at or started_at < previous_end:
                errors.append(f"attempt {number} starts before manifest/history time")
            if attempt["ended_at"] is not None:
                ended_at = _parse_datetime(attempt["ended_at"])
                if ended_at < started_at:
                    errors.append(f"attempt {number} ends before it starts")
                previous_end = ended_at
            else:
                previous_end = started_at
        except ValueError as error:
            errors.append(f"attempt {number} timestamps cannot be ordered: {error}")
        if (attempt["command"]["cwd"] != config["cwd"]
                or attempt["command"]["argv"] != _build_command(expected_dir, config)):
            errors.append(f"attempt {number} command differs from the pinned command")
        if attempt["command"]["shell"] != shlex.join(attempt["command"]["argv"]):
            errors.append(f"attempt {number} shell rendering disagrees with argv")
        if attempt["environment"] != _attempt_environment_record(expected_dir, config):
            errors.append(f"attempt {number} environment disagrees with fixed policy")
        generated = attempt["generated_bazelrc"]
        expected_bazelrc = _repo_path(expected_dir / "generated.jax_configure.bazelrc")
        if generated is not None and generated["path"] != expected_bazelrc:
            errors.append(f"attempt {number} generated Bazel rc is not its fixed archive")
        if generated is not None and generated["size_bytes"] == 0:
            errors.append(f"attempt {number} generated Bazel rc is empty")
        log_artifact = attempt["log_artifact"]
        if log_artifact is not None:
            if log_artifact["path"] != expected_paths["log"]:
                errors.append(f"attempt {number} log artifact is not its fixed build.log")
            if log_artifact["size_bytes"] == 0:
                errors.append(f"attempt {number} log artifact is empty")
        for wheel in attempt["wheels"]:
            try:
                wheel_path = _safe_recorded_path(
                    wheel["path"], expected_dir / "wheels", "wheel"
                )
                if wheel_path.parent != expected_dir / "wheels":
                    errors.append(f"attempt {number} wheel is not directly in wheel_dir")
                if not wheel_path.name.startswith("jaxlib-") or wheel_path.suffix != ".whl":
                    errors.append(f"attempt {number} wheel name is not jaxlib-*.whl")
                if verify_artifacts:
                    _validate_wheel_archive(wheel_path)
            except BuildError as error:
                errors.append(str(error))
        if attempt["status"] == "running" and attempt["log_artifact"] is not None:
            errors.append(f"running attempt {number} cannot have a closed log artifact")
        if attempt["status"] in terminal_attempts and attempt["log_artifact"] is None:
            errors.append(f"terminal attempt {number} must fingerprint its closed log")
        if verify_artifacts and attempt["status"] in terminal_attempts:
            records = [attempt["log_artifact"], *attempt["wheels"]]
            if generated is not None:
                records.append(generated)
            for record in records:
                if record is not None:
                    message = _verify_artifact(record, expected_dir,
                                               f"attempt {number} artifact")
                    if message:
                        errors.append(message)
    if attempts and updated_at < previous_end:
        errors.append("updated_at predates the latest attempt state")

    status = value["status"]
    terminal = {"dry-run-succeeded", "build-failed", "interrupted", "build-succeeded"}
    if status in {"planned", "preflight-failed"} and attempts:
        errors.append(f"{status} manifest must not contain attempts")
    if status == "running" and (not attempts or attempts[-1]["status"] != "running"):
        errors.append("running manifest must end in a running attempt")
    if status in terminal and (not attempts or attempts[-1]["status"] != status):
        errors.append("terminal manifest status must match its last attempt")
    if any(item["status"] == "running" for item in attempts[:-1]):
        errors.append("only the final attempt may be running")
    if attempts and not value["preflight"]["ok"]:
        errors.append("every attempted build requires preflight.ok=true")
    if status in terminal and attempts:
        try:
            if updated_at != _parse_datetime(attempts[-1]["ended_at"]):
                errors.append("terminal updated_at must equal final attempt ended_at")
        except (TypeError, ValueError):
            pass
    if updated_at < created_at:
        errors.append("updated_at predates created_at")

    result = value["result"]
    if status in terminal:
        last = attempts[-1]
        for key in ("exit_code", "error", "wheels", "generated_bazelrc"):
            if result[key] != last[key]:
                errors.append(f"result.{key} disagrees with final attempt")
        if result["attempt_number"] != last["number"]:
            errors.append("result.attempt_number disagrees with final attempt")
    elif status in {"planned", "running"} and result != _empty_result():
        errors.append(f"{status} manifest must have an empty result")
    elif status == "preflight-failed":
        if (result["attempt_number"] is not None or result["exit_code"] is not None
                or result["wheels"] or result["generated_bazelrc"] is not None
                or not isinstance(result["error"], str) or not result["error"]):
            errors.append("preflight-failed result must contain only a non-empty error")
    if status == "build-succeeded" and (value["dry_run"] or len(result["wheels"]) != 1):
        errors.append("build-succeeded requires one wheel from a non-dry-run attempt")
    if status in {"build-succeeded", "dry-run-succeeded"} and result["generated_bazelrc"] is None:
        errors.append("successful build command requires an archived generated Bazel rc")
    if status == "dry-run-succeeded" and (not value["dry_run"] or result["wheels"]):
        errors.append("dry-run-succeeded requires dry_run=true and no wheels")

    runtime = value["runtime_validation"]
    if (runtime["status"] != "not-run" or runtime["checked_at"] is not None
            or runtime["evidence"]):
        errors.append(
            "runtime_validation is fail-closed until P1 defines hashed capture evidence"
        )
    return errors


def _validate_manifest(value: dict[str, Any], path: Path,
                       expected_build_id: str | None = None,
                       expected_artifact_dir: Path | None = None,
                       *, verify_artifacts: bool = True,
                       verify_current_inputs: bool = True) -> None:
    validator = Draft202012Validator(_load_schema(), format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path))
    messages = ["schema "
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in schema_errors]
    if not messages:
        messages.extend(_semantic_manifest_errors(value, path, expected_build_id,
                                                  expected_artifact_dir,
                                                  verify_artifacts,
                                                  verify_current_inputs))
    if messages:
        raise BuildError("invalid build manifest: " + "; ".join(messages))


def _load_manifest(path: Path, expected_build_id: str | None = None,
                   expected_artifact_dir: Path | None = None, *,
                   verify_artifacts: bool = True,
                   verify_current_inputs: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot read build manifest {_repo_path(path)}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"build manifest is not a JSON object: {_repo_path(path)}")
    _validate_manifest(value, path, expected_build_id, expected_artifact_dir,
                       verify_artifacts=verify_artifacts,
                       verify_current_inputs=verify_current_inputs)
    return value


def load_and_validate_manifest(path: str | Path) -> dict[str, Any]:
    """Read-only public validator for a manifest and all recorded artifacts."""
    safe = _safe_output_path(Path(path), DEFAULT_MANIFEST_ROOT, "build manifest")
    return _load_manifest(
        safe,
        verify_artifacts=True,
        verify_current_inputs=False,
    )


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        stream.write((text.rstrip() + "\n").encode(errors="replace"))
        os.fsync(stream.fileno())


def _archive_generated_bazelrc(attempt_dir: Path) -> dict[str, Any] | None:
    if not os.path.lexists(GENERATED_BAZELRC):
        return None
    if GENERATED_BAZELRC.is_symlink() or not GENERATED_BAZELRC.is_file():
        raise BuildError("generated .jax_configure.bazelrc is not a regular non-symlink file")
    archive = attempt_dir / "generated.jax_configure.bazelrc"
    _atomic_write_bytes(archive, GENERATED_BAZELRC.read_bytes())
    return _artifact_record(archive, attempt_dir)


def _clear_generated_bazelrc() -> None:
    if not os.path.lexists(GENERATED_BAZELRC):
        return
    if GENERATED_BAZELRC.is_symlink() or not GENERATED_BAZELRC.is_file():
        raise BuildError("refusing to replace a non-regular .jax_configure.bazelrc entry")
    GENERATED_BAZELRC.unlink()


def _wheel_records(wheel_dir: Path) -> list[dict[str, Any]]:
    wheels = sorted(path for path in wheel_dir.iterdir()
                    if path.is_file() and not path.is_symlink() and path.suffix == ".whl")
    if len(wheels) != 1 or not wheels[0].name.startswith("jaxlib-"):
        names = ", ".join(path.name for path in wheels) or "none"
        raise BuildError(f"expected exactly one new jaxlib wheel, found: {names}")
    _validate_wheel_archive(wheels[0])
    return [_artifact_record(wheels[0], wheel_dir)]


def _current_input_state(manifest: dict[str, Any], patch_specs: dict[str, str]
                         ) -> tuple[dict[str, Any], str, dict[str, Any]]:
    report = _run_preflight()
    files = _capture_input_files()
    patch_records = _validate_source_patches(patch_specs, report)
    effective = _effective_preflight(report, patch_records, files)
    inputs, fingerprint = _build_inputs(report, manifest["build_config"],
        manifest["build_id"], patch_records, files)
    return inputs, fingerprint, effective


def _assert_current_inputs(manifest: dict[str, Any], patch_specs: dict[str, str],
                           phase: str) -> None:
    _inputs, fingerprint, effective = _current_input_state(manifest, patch_specs)
    if fingerprint != manifest["input_fingerprint_sha256"]:
        raise BuildError(f"build inputs drifted during {phase}")
    if effective != manifest["preflight"] or not effective["ok"]:
        raise BuildError(f"preflight state drifted or became blocked during {phase}")


def _finalize_stale_attempt(attempt: dict[str, Any], artifact_dir: Path) -> None:
    number = attempt["number"]
    attempt_dir = artifact_dir / "attempts" / f"{number:04d}"
    log_path = attempt_dir / "build.log"
    if not log_path.exists():
        _append_log(log_path,
                    f"[{_utc_now()}] stale running attempt recovered without an existing log")
    bazelrc = _archive_generated_bazelrc(attempt_dir)
    attempt.update({"status": "interrupted", "ended_at": _utc_now(),
        "error": "stale running identity recovered before resume", "wheels": [],
        "generated_bazelrc": bazelrc,
        "log_artifact": _artifact_record(log_path, attempt_dir)})


def _check_resume(manifest: dict[str, Any], input_fingerprint: str,
                  config: dict[str, Any], preflight: dict[str, Any],
                  artifact_dir: Path) -> None:
    if manifest["status"] in {"dry-run-succeeded", "build-succeeded"}:
        raise BuildError("a successful attempt cannot be resumed; use a new build id")
    if manifest["input_fingerprint_sha256"] != input_fingerprint:
        raise BuildError("build inputs changed; use a new build id instead of resuming")
    if manifest["build_config"] != config or manifest["preflight"] != preflight:
        raise BuildError("build configuration or preflight state changed; use a new build id")
    if manifest["status"] == "running":
        attempt = manifest["attempts"][-1]
        live = [name for name in ("wrapper_process", "child_process")
                if _identity_is_live(attempt[name])]
        if live:
            raise BuildError("the recorded build still has live identity: " + ", ".join(live))
        _finalize_stale_attempt(attempt, artifact_dir)
        manifest["status"] = "interrupted"
        manifest["updated_at"] = attempt["ended_at"]
        manifest["result"] = {"attempt_number": attempt["number"],
            "exit_code": attempt["exit_code"], "error": attempt["error"],
            "wheels": attempt["wheels"],
            "generated_bazelrc": attempt["generated_bazelrc"]}


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except OSError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_build(manifest: dict[str, Any], manifest_path: Path, artifact_dir: Path,
               inherited_lock_fds: tuple[int, ...], patch_specs: dict[str, str],
               network_environment: dict[str, str]) -> int:
    number = len(manifest["attempts"]) + 1
    attempt_dir = artifact_dir / "attempts" / f"{number:04d}"
    if attempt_dir.exists():
        raise BuildError(f"untracked attempt directory already exists: {_repo_path(attempt_dir)}")
    wheel_dir, log_path = attempt_dir / "wheels", attempt_dir / "build.log"
    command = _build_command(attempt_dir, manifest["build_config"])
    wrapper_identity = _process_identity(os.getpid())
    if wrapper_identity is None:
        raise BuildError("cannot record wrapper process identity")
    attempt: dict[str, Any] = {
        "number": number, "status": "running", "started_at": _utc_now(),
        "ended_at": None,
        "paths": {"attempt_dir": _repo_path(attempt_dir), "log": _repo_path(log_path),
            "wheel_dir": _repo_path(wheel_dir),
            "generated_bazelrc_archive": _repo_path(
                attempt_dir / "generated.jax_configure.bazelrc"),
            "home_dir": _repo_path(attempt_dir / "home"),
            "tmp_dir": _repo_path(attempt_dir / "tmp")},
        "command": {"cwd": "upstream/jax", "argv": command,
                    "shell": shlex.join(command)},
        "environment": _attempt_environment_record(attempt_dir, manifest["build_config"]),
        "wrapper_process": wrapper_identity, "child_process": None,
        "exit_code": None, "error": None, "log_artifact": None,
        "wheels": [], "generated_bazelrc": None,
    }
    manifest["attempts"].append(attempt)
    manifest["status"], manifest["updated_at"], manifest["result"] = (
        "running", _utc_now(), _empty_result())
    _validate_manifest(manifest, manifest_path, manifest["build_id"], artifact_dir)
    _atomic_write_json(manifest_path, manifest)

    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = None
    status, error = "build-failed", None
    wheels: list[dict[str, Any]] = []
    bazelrc: dict[str, Any] | None = None
    received_signal: int | None = None

    def interrupt_for_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_for_signal)
    try:
        wheel_dir.mkdir(parents=True)
        environment = _build_environment(manifest["build_config"], attempt_dir,
                                         network_environment)
        _assert_current_inputs(manifest, patch_specs, "pre-launch verification")
        _clear_generated_bazelrc()
        with log_path.open("ab", buffering=0) as log:
            log.write((f"\n[{attempt['started_at']}] attempt={number} "
                       f"command={attempt['command']['shell']}\n").encode())
            os.fsync(log.fileno())
            process = subprocess.Popen(command, cwd=JAX_ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=inherited_lock_fds, env=environment)
            attempt["child_process"] = _process_identity(process.pid)
            if attempt["child_process"] is None:
                raise BuildError("cannot record child process identity")
            manifest["updated_at"] = _utc_now()
            _validate_manifest(manifest, manifest_path, manifest["build_id"], artifact_dir)
            _atomic_write_json(manifest_path, manifest)
            print(f"build-id: {manifest['build_id']}", flush=True)
            print(f"attempt: {number}", flush=True)
            print(f"log: {_repo_path(log_path)}", flush=True)
            exit_code = process.wait()
        _assert_current_inputs(manifest, patch_specs, "post-build verification")
        if exit_code != 0:
            error = f"build command exited with code {exit_code}"
        elif manifest["dry_run"]:
            if any(wheel_dir.iterdir()):
                raise BuildError("dry-run unexpectedly produced wheel-directory contents")
            status = "dry-run-succeeded"
        else:
            wheels, status = _wheel_records(wheel_dir), "build-succeeded"
        bazelrc = _archive_generated_bazelrc(attempt_dir)
        if status in {"dry-run-succeeded", "build-succeeded"} and bazelrc is None:
            raise BuildError("build.py did not produce .jax_configure.bazelrc")
    except KeyboardInterrupt:
        status = "interrupted"
        error = (f"build interrupted by signal {received_signal}" if received_signal
                 else "build interrupted by user")
        if process is not None:
            _terminate_process(process)
            exit_code = process.returncode
    except BaseException as exc:
        status = "build-failed"
        error = f"build wrapper failed: {type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            _terminate_process(process)
            exit_code = process.returncode
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    if bazelrc is None:
        try:
            bazelrc = _archive_generated_bazelrc(attempt_dir)
        except BaseException as exc:
            status = "build-failed"
            error = f"{error + '; ' if error else ''}cannot archive generated Bazel rc: {exc}"
    if error:
        try:
            _append_log(log_path, f"[{_utc_now()}] wrapper: {error}")
        except OSError as exc:
            print(f"error: cannot append terminal log: {exc}", file=sys.stderr)
    elif not log_path.exists():
        _append_log(log_path, f"[{_utc_now()}] wrapper: command completed without output")

    ended_at = _utc_now()
    try:
        log_artifact = _artifact_record(log_path, attempt_dir)
    except BaseException as exc:
        print(f"error: cannot fingerprint terminal log: {exc}", file=sys.stderr)
        return 1
    attempt.update({"status": status, "ended_at": ended_at, "exit_code": exit_code,
        "error": error, "log_artifact": log_artifact, "wheels": wheels,
        "generated_bazelrc": bazelrc})
    manifest["status"], manifest["updated_at"] = status, ended_at
    manifest["result"] = {"attempt_number": number, "exit_code": exit_code,
        "error": error, "wheels": wheels, "generated_bazelrc": bazelrc}
    try:
        _validate_manifest(manifest, manifest_path, manifest["build_id"], artifact_dir)
        _atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        print(f"error: cannot persist terminal build state: {exc}", file=sys.stderr)
        return 1
    print(f"status: {status}")
    print(f"manifest: {_repo_path(manifest_path)}")
    return 0 if status in {"dry-run-succeeded", "build-succeeded"} else 1


def _show_status(path: Path, build_id: str, artifact_dir: Path) -> int:
    manifest = _load_manifest(path, build_id, artifact_dir)
    observed = "not-running"
    if manifest["status"] == "running":
        attempt = manifest["attempts"][-1]
        live = [name for name in ("wrapper_process", "child_process")
                if _identity_is_live(attempt[name])]
        observed = "+".join(live) if live else "stale"
    print(f"build-id: {manifest['build_id']}")
    print(f"status: {manifest['status']}")
    print(f"process-state: {observed}")
    print(f"updated-at: {manifest['updated_at']}")
    print(f"attempts: {len(manifest['attempts'])}")
    if manifest["attempts"]:
        print(f"last-log: {manifest['attempts'][-1]['paths']['log']}")
    print(f"runtime-validation: {manifest['runtime_validation']['status']}")
    print(f"manifest: {_repo_path(path)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", help="stable id used for status and resume")
    parser.add_argument("--jobs", type=int, default=4, help="Bazel job limit (default: 4)")
    parser.add_argument("--bazel-startup-option", action="append", default=[],
        help="allowlisted Bazel cache/resource startup option; repeat as needed")
    parser.add_argument("--bazel-option", action="append", default=[],
        help="allowlisted Bazel resource/diagnostic build option; repeat as needed")
    parser.add_argument("--source-patch", action="append", default=[],
        metavar="COMPONENT=REPO_RELATIVE_PATCH",
        help="permit an exact tracked jax/xla diff captured by the named patch")
    parser.add_argument("--dry-run", action="store_true", help="run JAX build.py --dry_run")
    parser.add_argument("--resume", action="store_true", help="resume a failed or stale build id")
    parser.add_argument("--status", action="store_true", help="show validated recorded status")
    parser.add_argument("--verify-manifest", type=Path,
        help="read-only schema, semantic, and artifact-byte validation")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT,
        help=argparse.SUPPRESS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT,
        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.resume or args.status) and not args.build_id:
        parser.error("--resume and --status require --build-id")
    if args.status and (args.resume or args.dry_run or args.bazel_startup_option
                        or args.bazel_option or args.source_patch or args.verify_manifest):
        parser.error("--status cannot be combined with build options")
    if args.verify_manifest and (args.build_id or args.resume or args.status or args.dry_run
            or args.bazel_startup_option or args.bazel_option or args.source_patch):
        parser.error("--verify-manifest cannot be combined with build options")
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.verify_manifest:
            manifest = load_and_validate_manifest(args.verify_manifest)
            print(f"valid: {_repo_path(args.verify_manifest)}")
            print(f"build-id: {manifest['build_id']}")
            print(f"status: {manifest['status']}")
            return 0
        build_id = args.build_id or _default_build_id()
        if not BUILD_ID_PATTERN.fullmatch(build_id):
            raise BuildError("build id must match [a-z0-9][a-z0-9._-]*")
        manifest_root = _safe_root(args.manifest_root, DEFAULT_MANIFEST_ROOT,
                                   "manifest root")
        artifact_root = _safe_root(args.artifact_root, DEFAULT_ARTIFACT_ROOT,
                                   "artifact root")
        manifest_path = _safe_output_path(manifest_root / f"{build_id}.json",
                                           DEFAULT_MANIFEST_ROOT, "manifest path")
        artifact_dir = _safe_output_path(artifact_root / build_id,
                                          DEFAULT_ARTIFACT_ROOT, "artifact directory")
        build_lock_path = artifact_dir / ".build-id.lock"
        if args.status:
            return _show_status(manifest_path, build_id, artifact_dir)

        patch_specs = _parse_source_patch_args(args.source_patch)
        network_environment = _network_environment_snapshot()
        with _exclusive_lock(GLOBAL_LOCK_PATH, "global jaxlib build") as global_lock_fd:
            with _exclusive_lock(build_lock_path, f"build id {build_id}") as build_lock_fd:
                raw_preflight = _run_preflight()
                files = _capture_input_files()
                patch_records = _validate_source_patches(patch_specs, raw_preflight)
                preflight = _effective_preflight(raw_preflight, patch_records, files)
                config = _build_config(raw_preflight, args.jobs,
                    args.bazel_startup_option, args.bazel_option, args.dry_run,
                    network_environment)
                inputs, fingerprint = _build_inputs(raw_preflight, config, build_id,
                                                     patch_records, files)
                if manifest_path.exists():
                    if not args.resume:
                        raise BuildError(f"build id already exists: {build_id}; use --resume or a new id")
                    manifest = _load_manifest(manifest_path, build_id, artifact_dir)
                    _check_resume(manifest, fingerprint, config, preflight, artifact_dir)
                    _validate_manifest(manifest, manifest_path, build_id, artifact_dir)
                    _atomic_write_json(manifest_path, manifest)
                else:
                    if args.resume:
                        raise BuildError(f"cannot resume missing build id: {build_id}")
                    manifest = _new_manifest(build_id=build_id, config=config,
                        inputs=inputs, input_fingerprint=fingerprint,
                        manifest_path=manifest_path, artifact_dir=artifact_dir,
                        build_lock_path=build_lock_path, preflight=preflight)
                _validate_manifest(manifest, manifest_path, build_id, artifact_dir)
                if preflight["blockers"]:
                    message = "; ".join(item["message"] for item in preflight["blockers"])
                    manifest["status"] = "preflight-failed"
                    manifest["updated_at"] = _utc_now()
                    manifest["preflight"] = preflight
                    manifest["result"] = _empty_result(message)
                    _validate_manifest(manifest, manifest_path, build_id, artifact_dir)
                    _atomic_write_json(manifest_path, manifest)
                    raise BuildError("preflight failed: " + message)
                return _run_build(manifest, manifest_path, artifact_dir,
                    (global_lock_fd, build_lock_fd), patch_specs, network_environment)
    except BaseException as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
