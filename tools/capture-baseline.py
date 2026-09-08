#!/usr/bin/env python3
"""Capture or verify the local JAX source/runtime baseline without network access."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib
from typing import Any

import jax
import jax.version as jax_version
import jaxlib
import jaxlib.version as jaxlib_version


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "manifests" / "baseline.json"
SOURCE_PATHS = {
    "jax": "upstream/jax",
    "llvm": "upstream/llvm-project",
    "shardy": "upstream/shardy",
    "stablehlo": "upstream/stablehlo",
    "xla": "upstream/xla",
}


class BaselineError(RuntimeError):
    """Raised when the local baseline cannot be captured or verified."""


def _command(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise BaselineError(f"command failed: {' '.join(args)}: {detail.strip()}") from error
    return result.stdout.strip()


def _git_commit(path: Path) -> str:
    return _command("git", "rev-parse", "HEAD", cwd=path)


def _project_configuration() -> dict[str, Any]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    return project


def _uv_runtime_version() -> str:
    output = _command("uv", "--version")
    match = re.fullmatch(r"uv\s+(\S+)(?:\s+.*)?", output)
    if match is None:
        raise BaselineError(f"unexpected `uv --version` output: {output!r}")
    return match.group(1)


def _jax_install_type() -> str:
    source_root = (REPOSITORY_ROOT / SOURCE_PATHS["jax"]).resolve()
    imported_file = Path(jax.__file__).resolve()
    return "editable-source" if imported_file.is_relative_to(source_root) else "environment"


def _jaxlib_install_type() -> str:
    source_root = (REPOSITORY_ROOT / SOURCE_PATHS["jax"] / "jaxlib").resolve()
    imported_file = Path(jaxlib.__file__).resolve()
    return "source-tree" if imported_file.is_relative_to(source_root) else "installed-wheel"


def _alignment_status(jax_commit: str, jaxlib_git_hash: str | None) -> dict[str, Any]:
    matches = jaxlib_git_hash == jax_commit
    status = "ALIGNED" if matches else "VERSION-SKEW"
    tags = ["RUN-CPU"] if matches else ["RUN-CPU", "VERSION-SKEW"]
    reason = (
        "The imported JAX source commit matches the installed jaxlib build git hash."
        if matches
        else "The imported editable JAX source commit does not match the installed "
        "jaxlib build git hash; C++/XLA runtime observations are version-skewed."
    )
    return {
        "evidence_tags": tags,
        "jax_source_matches_jaxlib_build": matches,
        "reason": reason,
        "runtime_source_alignment": status,
    }


def capture_baseline() -> dict[str, Any]:
    configuration = _project_configuration()
    jax_commit = _git_commit(REPOSITORY_ROOT / SOURCE_PATHS["jax"])
    raw_jaxlib_hash = getattr(jaxlib_version, "_git_hash", None)
    jaxlib_git_hash = raw_jaxlib_hash or None
    devices = jax.devices()

    sources = {
        name: {
            "git_commit": _git_commit(REPOSITORY_ROOT / relative_path),
            "path": relative_path,
        }
        for name, relative_path in SOURCE_PATHS.items()
    }
    root_commit = _git_commit(REPOSITORY_ROOT)

    return {
        "external_inputs": {
            "libtpu": None,
            "tokamax": None,
            "tpu_target": None,
        },
        "repository": {
            "root": {
                "git_commit": root_commit,
                "path": ".",
                "verification": "ancestor",
            },
            "sources": sources,
        },
        "runtime": {
            "device": {
                "default_platform": jax.default_backend(),
                "platforms": sorted({device.platform for device in devices}),
            },
            "jax": {
                "git_hash": jax_commit,
                "install_type": _jax_install_type(),
                "reported_git_hash": getattr(jax_version, "_git_hash", None),
                "version": jax.__version__,
            },
            "jaxlib": {
                "git_hash": jaxlib_git_hash,
                "install_type": _jaxlib_install_type(),
                "version": jaxlib.__version__,
            },
        },
        "schema_version": 1,
        "status": _alignment_status(jax_commit, jaxlib_git_hash),
        "toolchain": {
            "python": {
                "project_constraint": configuration["project"]["requires-python"],
                "repository_pin": (REPOSITORY_ROOT / ".python-version").read_text().strip(),
                "runtime": platform.python_version(),
            },
            "uv": {
                "project_constraint": configuration["tool"]["uv"]["required-version"],
                "runtime": _uv_runtime_version(),
            },
        },
    }


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def verify_baseline(manifest_path: Path, current: dict[str, Any]) -> None:
    try:
        expected = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot read baseline manifest {manifest_path}: {error}") from error

    comparable_current = copy.deepcopy(current)
    try:
        expected_root = expected["repository"]["root"]
        current_root = comparable_current["repository"]["root"]
        if expected_root["verification"] != "ancestor":
            raise BaselineError("repository.root.verification must be 'ancestor'")
        if not _is_ancestor(expected_root["git_commit"], current_root["git_commit"]):
            raise BaselineError(
                "recorded root commit is not an ancestor of the current root commit: "
                f"{expected_root['git_commit']} -> {current_root['git_commit']}"
            )
        current_root["git_commit"] = expected_root["git_commit"]
    except (KeyError, TypeError) as error:
        raise BaselineError("baseline manifest has an invalid repository.root entry") from error

    if expected != comparable_current:
        diff = difflib.unified_diff(
            _canonical_json(expected).splitlines(),
            _canonical_json(comparable_current).splitlines(),
            fromfile=str(manifest_path),
            tofile="current local baseline",
            lineterm="",
        )
        raise BaselineError("baseline mismatch:\n" + "\n".join(diff))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        nargs="?",
        const=DEFAULT_MANIFEST,
        type=Path,
        metavar="PATH",
        help="verify PATH (default: manifests/baseline.json) against the local baseline",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        baseline = capture_baseline()
        if arguments.verify is None:
            sys.stdout.write(_canonical_json(baseline))
        else:
            verify_baseline(arguments.verify, baseline)
            status = baseline["status"]["runtime_source_alignment"]
            print(f"baseline verified: {arguments.verify} [{status}]")
    except BaselineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
