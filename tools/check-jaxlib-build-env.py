#!/usr/bin/env python3
"""Inspect prerequisites for the repository's pinned CPU jaxlib build.

The strict check is deliberately narrower than a generic JAX build check. It
verifies the exact Linux x86_64 Bazel and Clang toolchain documented by this
repository, plus clean JAX/XLA source trees at their pinned revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BAZEL_VERSION = "8.7.0"
EXPECTED_BAZEL_SHA256 = (
    "d7606e679b78067c811096fb3d6cf135225b528835ca396e3a4dddf957859544"
)
BAZEL_PATH = ROOT / "upstream/jax/bazel-8.7.0-linux-x86_64"

EXPECTED_CLANG_VERSION = "18.1.3"
CLANG_PATH = Path("/usr/bin/clang")
CLANGXX_PATH = Path("/usr/bin/clang++")
GIT_PATH = Path("/usr/bin/git")
BASE_PYTHON_PATH = Path("/usr/bin/python3.12")
EXPECTED_BASE_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
EXPECTED_BASE_PYTHON_SIZE_BYTES = 8020928
EXPECTED_GIT_VERSION = "git version 2.43.0"
EXPECTED_GIT_SHA256 = "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
CLEAN_ENVIRONMENT = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
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

EXPECTED_SOURCE_COMMITS = {
    "jax": "5832e866449a41c3eea6333416528039119a0fde",
    "xla": "496bd4bd49db9ecbffd85da630b49c860b724604",
}
SOURCE_PATHS = {
    "jax": ROOT / "upstream/jax",
    "xla": ROOT / "upstream/xla",
}
SOURCE_MARKERS = {
    "jax": "build/build.py",
    "xla": "xla",
}
BAZELRC_USER_PATH = ROOT / "upstream/jax/.bazelrc.user"


def _repo_path(path: Path) -> str:
    """Return repository-relative paths for repository-owned files."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _safe_source_root(path: Path) -> str | None:
    lexical = Path(os.path.abspath(os.fspath(path)))
    base = ROOT.resolve()
    try:
        relative = lexical.relative_to(base)
    except ValueError:
        return f"source path is outside the repository: {path}"
    current = base
    for part in relative.parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            return f"source path traverses a symlink: {current}"
    try:
        lexical.resolve().relative_to(base)
    except ValueError:
        return f"source path resolves outside the repository: {path}"
    if not lexical.is_dir():
        return f"source path is not an existing directory: {path}"
    return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(path: Path, *args: str) -> dict[str, Any]:
    exists = path.is_file()
    executable = exists and os.access(path, os.X_OK)
    report: dict[str, Any] = {
        "path": _repo_path(path),
        "resolved_path": str(path.resolve()) if exists else None,
        "exists": exists,
        "executable": executable,
        "version": None,
        "returncode": None,
        "sha256": _sha256(path),
    }
    if not executable:
        return report
    try:
        process = subprocess.run(
            [str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            env=CLEAN_ENVIRONMENT,
        )
    except OSError as error:
        report["error"] = str(error)
        return report
    lines = (process.stdout or process.stderr).strip().splitlines()
    report["version"] = lines[0] if lines else None
    report["returncode"] = process.returncode
    return report


def _git_source(name: str, path: Path) -> dict[str, Any]:
    marker = path / SOURCE_MARKERS[name]
    report: dict[str, Any] = {
        "path": _repo_path(path),
        "exists": marker.exists(),
        "expected_commit": EXPECTED_SOURCE_COMMITS[name],
        "commit": None,
        "top_level": None,
        "status_returncode": None,
        "dirty": None,
        "status": [],
        "path_error": _safe_source_root(path),
        "index_flags": [],
        "index_flags_returncode": None,
        "ignored": [],
        "ignored_returncode": None,
    }
    if not report["exists"] or report["path_error"] is not None:
        return report

    try:
        head = subprocess.run(
            [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            env=CLEAN_ENVIRONMENT,
        )
    except OSError as error:
        report["error"] = str(error)
        return report
    if head.returncode == 0:
        report["commit"] = head.stdout.strip()
    top_level = subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        env=CLEAN_ENVIRONMENT,
    )
    if top_level.returncode == 0:
        canonical_top_level = Path(top_level.stdout.strip()).resolve()
        if canonical_top_level == path.resolve():
            report["top_level"] = _repo_path(path)

    status = subprocess.run(
        [
            str(GIT_PATH),
            *GIT_FIXED_OPTIONS,
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=CLEAN_ENVIRONMENT,
    )
    report["status_returncode"] = status.returncode
    if status.returncode == 0:
        report["status"] = status.stdout.splitlines()
        report["dirty"] = bool(report["status"])
    flags = subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(path), "ls-files", "-v", "-z"],
        check=False,
        capture_output=True,
        env=CLEAN_ENVIRONMENT,
    )
    report["index_flags_returncode"] = flags.returncode
    if flags.returncode == 0:
        report["index_flags"] = [
            item.decode(errors="replace")
            for item in flags.stdout.split(b"\0")
            if item and (item[:1] == b"S" or item[:1].islower())
        ]
    ignored = subprocess.run(
        [str(GIT_PATH), *GIT_FIXED_OPTIONS, "-C", str(path), "ls-files", "--others",
         "--ignored", "--exclude-standard", "-z"],
        check=False,
        capture_output=True,
        env=CLEAN_ENVIRONMENT,
    )
    report["ignored_returncode"] = ignored.returncode
    if ignored.returncode == 0:
        report["ignored"] = [
            item.decode(errors="replace") for item in ignored.stdout.split(b"\0") if item
        ]
    return report


def _memory_bytes() -> tuple[int | None, int | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", maxsplit=1)
        fields = value.strip().split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values.get("MemTotal"), values.get("SwapTotal")


def _extract_clang_version(version_line: str | None) -> str | None:
    if version_line is None:
        return None
    match = re.search(r"\bclang version (\d+\.\d+\.\d+)\b", version_line)
    return match.group(1) if match else None


def _blocker(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _ignored_source_path_allowed(name: str, relative: str) -> bool:
    path = Path(relative)
    if relative in {"bazel-bin", "bazel-jax", "bazel-out", "bazel-testlogs", "bazel-xla"}:
        return True
    if name != "jax":
        return False
    if relative in {
        ".jax_configure.bazelrc",
        "MODULE.bazel.lock",
        "bazel-8.7.0-linux-x86_64",
    }:
        return True
    if path.parts and path.parts[0] == "jax.egg-info":
        return True
    return False


def blockers(report: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    host = report["host"]
    if host["system"] != "Linux" or host["machine"] != "x86_64":
        result.append(_blocker("host-platform", "the pinned build toolchain requires Linux x86_64"))

    python = report["python"]
    if not python["matches_repository_pin"]:
        result.append(_blocker(
            "python-version",
            "the preflight interpreter does not match the repository Python pin "
            f"{python['repository_pin']}",
        ))
    if python["executable"] != str(ROOT / ".venv/bin/python"):
        result.append(_blocker(
            "python-path", "preflight must run through .venv/bin/python"
        ))
    if (python["resolved_executable"] != str(BASE_PYTHON_PATH)
            or python["base_executable"] != str(BASE_PYTHON_PATH)
            or python["base_resolved_path"] != str(BASE_PYTHON_PATH)):
        result.append(_blocker(
            "python-base-path",
            f"the repository interpreter must resolve to {BASE_PYTHON_PATH}",
        ))
    if (python["base_size_bytes"] != EXPECTED_BASE_PYTHON_SIZE_BYTES
            or python["base_sha256"] != EXPECTED_BASE_PYTHON_SHA256):
        result.append(_blocker(
            "python-base-bytes", "the base Python executable bytes do not match the pin"
        ))

    if report["required_bazel_version"] != EXPECTED_BAZEL_VERSION:
        result.append(_blocker(
            "bazel-version-file",
            "upstream/jax/.bazelversion does not contain the pinned Bazel version "
            f"{EXPECTED_BAZEL_VERSION}",
        ))
    bazel = report["bazel"]
    if not bazel["exists"] or not bazel["executable"]:
        result.append(_blocker(
            "bazel-missing",
            f"the pinned Bazel binary is missing or not executable: {bazel['path']}",
        ))
    else:
        if bazel["returncode"] != 0:
            result.append(_blocker(
                "bazel-returncode",
                f"the pinned Bazel --version command returned {bazel['returncode']}",
            ))
        if bazel["version"] != f"bazel {EXPECTED_BAZEL_VERSION}":
            result.append(_blocker(
                "bazel-version",
                "the pinned Bazel binary reported an unexpected version: "
                f"{bazel['version']!r}",
            ))
        if bazel["sha256"] != EXPECTED_BAZEL_SHA256:
            result.append(_blocker(
                "bazel-sha256",
                "the pinned Bazel SHA-256 does not match: "
                f"expected {EXPECTED_BAZEL_SHA256}, got {bazel['sha256']}",
            ))

    for key, expected_path in (("clang", CLANG_PATH), ("clangxx", CLANGXX_PATH)):
        compiler = report[key]
        if not compiler["exists"] or not compiler["executable"]:
            result.append(_blocker(
                f"{key}-missing",
                f"the pinned compiler is missing or not executable: {expected_path}",
            ))
            continue
        if compiler["returncode"] != 0:
            result.append(_blocker(
                f"{key}-returncode",
                f"{expected_path} --version returned {compiler['returncode']}",
            ))
        if compiler["parsed_version"] != EXPECTED_CLANG_VERSION:
            result.append(_blocker(
                f"{key}-version",
                f"{expected_path} must be Clang {EXPECTED_CLANG_VERSION}; "
                f"got {compiler['version']!r}",
            ))

    git = report["git"]
    if not git["exists"] or not git["executable"]:
        result.append(_blocker("git-missing", f"the pinned Git is missing: {GIT_PATH}"))
    else:
        if git["path"] != str(GIT_PATH) or git["resolved_path"] != str(GIT_PATH.resolve()):
            result.append(_blocker("git-path", f"Git must resolve from {GIT_PATH}"))
        if git["returncode"] != 0 or git["version"] != EXPECTED_GIT_VERSION:
            result.append(_blocker(
                "git-version", f"Git must report {EXPECTED_GIT_VERSION!r}"
            ))
        if git["sha256"] != EXPECTED_GIT_SHA256:
            result.append(_blocker(
                "git-sha256",
                f"Git SHA-256 must be {EXPECTED_GIT_SHA256}, got {git['sha256']}",
            ))

    if report["bazelrc_user"]["exists"]:
        result.append(_blocker(
            "bazelrc-user-present",
            "upstream/jax/.bazelrc.user must have no directory entry "
            "(regular files, directories, and symlinks are all rejected)",
        ))

    for name, source in report["sources"].items():
        if source["path_error"] is not None:
            result.append(_blocker(f"source-path:{name}", source["path_error"]))
        if not source["exists"]:
            result.append(_blocker(
                f"source-missing:{name}", f"the pinned {name} source tree is missing"
            ))
            continue
        if source["commit"] != source["expected_commit"]:
            result.append(_blocker(
                f"source-revision:{name}",
                f"{name} is at {source['commit']}, expected {source['expected_commit']}",
            ))
        if (source["top_level"] is None
                or (ROOT / source["top_level"]).resolve() != SOURCE_PATHS[name].resolve()):
            result.append(_blocker(
                f"source-toplevel:{name}",
                f"{name} Git top-level is not its fixed source root",
            ))
        if source["status_returncode"] != 0:
            result.append(_blocker(
                f"source-status:{name}", f"git status failed for {name}"
            ))
        elif source["dirty"]:
            result.append(_blocker(
                f"source-dirty:{name}",
                f"the pinned {name} source tree is dirty: {source['status']}",
            ))
        if source["index_flags"]:
            result.append(_blocker(
                f"source-index-flags:{name}",
                f"{name} contains assume-unchanged/skip-worktree entries: "
                f"{source['index_flags']}",
            ))
        if source["index_flags_returncode"] != 0:
            result.append(_blocker(
                f"source-index-flags-check:{name}",
                f"cannot inspect assume-unchanged/skip-worktree flags for {name}",
            ))
        unsafe_ignored = [
            relative for relative in source["ignored"]
            if not _ignored_source_path_allowed(name, relative)
        ]
        if unsafe_ignored:
            result.append(_blocker(
                f"source-ignored:{name}",
                f"{name} contains non-allowlisted ignored paths: {unsafe_ignored}",
            ))
        if source["ignored_returncode"] != 0:
            result.append(_blocker(
                f"source-ignored-check:{name}",
                f"cannot inspect ignored paths for {name}",
            ))
    return result


def inspect() -> dict[str, Any]:
    bazel_version_file = ROOT / "upstream/jax/.bazelversion"
    required_bazel_version = (
        bazel_version_file.read_text(encoding="utf-8").strip()
        if bazel_version_file.exists()
        else None
    )
    memory_bytes, swap_bytes = _memory_bytes()
    disk = shutil.disk_usage(ROOT)

    bazel = _command(BAZEL_PATH, "--version")
    bazel["expected_version"] = EXPECTED_BAZEL_VERSION
    bazel["expected_sha256"] = EXPECTED_BAZEL_SHA256
    bazel["sha256"] = _sha256(BAZEL_PATH)

    clang = _command(CLANG_PATH, "--version")
    clang["expected_version"] = EXPECTED_CLANG_VERSION
    clang["parsed_version"] = _extract_clang_version(clang["version"])
    clangxx = _command(CLANGXX_PATH, "--version")
    clangxx["expected_version"] = EXPECTED_CLANG_VERSION
    clangxx["parsed_version"] = _extract_clang_version(clangxx["version"])

    git = _command(GIT_PATH, "--version")
    repository_pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()

    base_executable = Path(getattr(sys, "_base_executable", ""))
    report: dict[str, Any] = {
        "workspace": str(ROOT),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "memory_bytes": memory_bytes,
            "swap_bytes": swap_bytes,
            "workspace_disk_free_bytes": disk.free,
        },
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "base_executable": str(base_executable),
            "base_resolved_path": str(base_executable.resolve()),
            "base_size_bytes": (
                base_executable.stat().st_size if base_executable.is_file() else None
            ),
            "base_sha256": _sha256(base_executable),
            "version": platform.python_version(),
            "repository_pin": repository_pin,
            "matches_repository_pin": platform.python_version() == repository_pin,
        },
        "required_bazel_version": required_bazel_version,
        "bazel": bazel,
        "clang": clang,
        "clangxx": clangxx,
        "git": git,
        "bazelrc_user": {
            "path": _repo_path(BAZELRC_USER_PATH),
            "exists": os.path.lexists(BAZELRC_USER_PATH),
            "is_symlink": BAZELRC_USER_PATH.is_symlink(),
        },
        "sources": {
            name: _git_source(name, path) for name, path in SOURCE_PATHS.items()
        },
    }
    report["blockers"] = blockers(report)
    report["ok"] = not report["blockers"]
    return report


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.1f} GiB"


def print_human(report: dict[str, Any]) -> None:
    print(f"workspace: {report['workspace']}")
    python = report["python"]
    print(
        "python: "
        f"{python['version']} ({python['executable']}) "
        f"repository-pin={python['repository_pin']} "
        f"matches={python['matches_repository_pin']}"
    )
    print(f"required Bazel: {report['required_bazel_version']}")
    bazel = report["bazel"]
    print(
        f"bazel: {bazel['version'] or 'MISSING'} ({bazel['path']}) "
        f"returncode={bazel['returncode']} sha256={bazel['sha256']}"
    )
    for key, label in (("clang", "clang"), ("clangxx", "clang++"), ("git", "git")):
        item = report[key]
        print(
            f"{label}: {item['version'] or 'MISSING'} ({item['path']}) "
            f"returncode={item['returncode']}"
        )
    host = report["host"]
    print(
        "host: "
        f"{host['machine']}, cpus={host['logical_cpus']}, "
        f"memory={_gib(host['memory_bytes'])}, "
        f"swap={_gib(host['swap_bytes'])}, "
        f"workspace-free={_gib(host['workspace_disk_free_bytes'])}"
    )
    for name, source in report["sources"].items():
        state = "OK" if source["exists"] else "MISSING"
        print(
            f"source {name}: {state} commit={source['commit']} "
            f"expected={source['expected_commit']} dirty={source['dirty']} "
            f"path={source['path']}"
        )
    print(f"strict-ready: {report['ok']}")
    for item in report["blockers"]:
        print(f"BLOCKER [{item['code']}]: {item['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless every pinned build prerequisite matches exactly",
    )
    args = parser.parse_args()
    report = inspect()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    if args.strict and report["blockers"]:
        if args.json:
            for item in report["blockers"]:
                print(f"BLOCKER [{item['code']}]: {item['message']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
