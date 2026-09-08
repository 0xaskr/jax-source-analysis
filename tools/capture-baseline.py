#!/usr/bin/env python3
"""Capture or verify the local JAX source/runtime baseline without network access."""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from typing import Any
from urllib.parse import unquote, urlsplit

import jax
import jax.version as jax_version
import jaxlib
import jaxlib.version as jaxlib_version
from jsonschema import Draft202012Validator, FormatChecker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "manifests/baseline.json"
BASELINE_SCHEMA = REPOSITORY_ROOT / "manifests/schema/baseline.schema.json"
BUILD_SCHEMA = REPOSITORY_ROOT / "manifests/schema/build-jaxlib.schema.json"
BUILD_VALIDATOR = REPOSITORY_ROOT / "tools/build-jaxlib.py"
PINNED_GIT = "/usr/bin/git"
EXPECTED_UV_VERSION = "0.12.9"
EXPECTED_UV_SHA256 = "671793498fe0a545432e2524b6691ffb9eea4540d9fda43ca2f978df2dbf8426"
EXPECTED_UV_SIZE_BYTES = 49313824
EXPECTED_UV_ARTIFACT_URI = "artifact:uv-0.12.9-x86_64-unknown-linux-gnu"
UV_RETRIEVAL = (
    "Install the pinned uv 0.12.9 x86_64 Linux standalone artifact and verify "
    "its recorded SHA-256."
)
SOURCE_PATHS = {
    "jax": "upstream/jax",
    "llvm": "upstream/llvm-project",
    "shardy": "upstream/shardy",
    "stablehlo": "upstream/stablehlo",
    "xla": "upstream/xla",
}
INPUT_PATHS = (
    ".gitmodules",
    ".python-version",
    "manifests/schema/baseline.schema.json",
    "pyproject.toml",
    "upstream-sources.lock",
    "upstream/jax/.bazelversion",
    "uv.lock",
)


class BaselineError(RuntimeError):
    """Raised when the local baseline cannot be captured or verified."""


def _clean_git_environment() -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")
        if name in os.environ
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return environment


def _command(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    command = (
        (
            PINNED_GIT,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args[1:],
        )
        if args and args[0] == "git"
        else args
    )
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_clean_git_environment() if args and args[0] == "git" else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise BaselineError(f"command failed: {' '.join(args)}: {detail.strip()}") from error
    return result.stdout.strip()


def _git_state(path: Path) -> dict[str, Any]:
    status = _command(
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
        cwd=path,
    ).splitlines()
    return {
        "git_commit": _command("git", "rev-parse", "HEAD", cwd=path),
        "dirty": bool(status),
        "status": status,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_build_validator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "jax_source_analysis_build_validator", BUILD_VALIDATOR
    )
    if spec is None or spec.loader is None:
        raise BaselineError(f"cannot load canonical build validator: {BUILD_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repository_path(path: Path) -> str | None:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return None


def _project_configuration() -> dict[str, Any]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _uv_runtime_identity() -> dict[str, Any]:
    executable = shutil.which("uv")
    if executable is None:
        raise BaselineError("the pinned uv executable is unavailable")
    path = Path(executable).resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BaselineError(f"cannot open resolved uv executable: {error}") from error
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise BaselineError("the resolved uv executable is not a regular file")
        digest_state = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        size = file_status.st_size
        if digest != EXPECTED_UV_SHA256 or size != EXPECTED_UV_SIZE_BYTES:
            raise BaselineError(
                "resolved uv executable bytes do not match the pinned standalone artifact"
            )
        try:
            process = subprocess.run(
                (f"/proc/self/fd/{descriptor}", "--version"),
                check=True,
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                pass_fds=(descriptor,),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise BaselineError(f"pinned uv --version failed: {error}") from error
    finally:
        os.close(descriptor)
    output = process.stdout.strip()
    match = re.fullmatch(r"uv\s+(\S+)\s+\(([^)]+)\)", output)
    if match is None:
        raise BaselineError(f"unexpected `uv --version` output: {output!r}")
    version, target = match.groups()
    artifact_uri = f"artifact:uv-{version}-{target.lower()}"
    if version != EXPECTED_UV_VERSION or artifact_uri != EXPECTED_UV_ARTIFACT_URI:
        raise BaselineError(f"unexpected pinned uv identity: {output!r}")
    return {
        "runtime": version,
        "artifact_uri": artifact_uri,
        "artifact_sha256": digest,
        "artifact_size_bytes": size,
        "retrieval": UV_RETRIEVAL,
    }


def _jax_install() -> tuple[str, str | None, str]:
    source_root = (REPOSITORY_ROOT / SOURCE_PATHS["jax"]).resolve()
    imported_file = Path(jax.__file__).resolve()
    install_type = (
        "editable-source" if imported_file.is_relative_to(source_root) else "environment"
    )
    return install_type, _repository_path(imported_file), _sha256(imported_file)


def _jaxlib_install_type(package_root: Path) -> str:
    source_root = (REPOSITORY_ROOT / SOURCE_PATHS["jax"] / "jaxlib").resolve()
    return "source-tree" if package_root.is_relative_to(source_root) else "installed-wheel"


def _installed_wheel_tags(distribution_name: str) -> set[str]:
    try:
        wheel_metadata = metadata.distribution(distribution_name).read_text("WHEEL") or ""
    except metadata.PackageNotFoundError as error:
        raise BaselineError(f"installed distribution is missing: {distribution_name}") from error
    return {
        line.partition(":")[2].strip()
        for line in wheel_metadata.splitlines()
        if line.startswith("Tag:") and line.partition(":")[2].strip()
    }


def _locked_wheel_identity(name: str, version: str) -> dict[str, Any]:
    lock_path = REPOSITORY_ROOT / "uv.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    package = next(
        (
            item
            for item in lock.get("package", [])
            if item.get("name") == name and item.get("version") == version
        ),
        None,
    )
    if package is None:
        raise BaselineError(f"uv.lock has no {name}=={version} package")

    tags = _installed_wheel_tags(name)
    candidates = []
    for wheel in package.get("wheels", []):
        filename = unquote(Path(urlsplit(wheel["url"]).path).name)
        if any(filename.endswith(f"-{tag}.whl") for tag in tags):
            candidates.append((filename, wheel))
    if len(candidates) != 1:
        raise BaselineError(
            f"expected one locked wheel for installed {name} tags {sorted(tags)}, "
            f"found {[filename for filename, _ in candidates]}"
        )
    filename, wheel = candidates[0]
    raw_hash = wheel.get("hash", "")
    if not raw_hash.startswith("sha256:"):
        raise BaselineError(f"locked {name} wheel lacks a SHA-256: {raw_hash!r}")
    return {
        "kind": "lock-artifact",
        "artifact_uri": "artifact:" + filename.lower(),
        "artifact_sha256": raw_hash.removeprefix("sha256:"),
        "artifact_size_bytes": wheel["size"],
        "lock_path": "uv.lock",
        "retrieval": "Resolve this exact wheel entry from uv.lock with uv sync --locked.",
    }


def _build_manifest_identity(
    manifest_path: Path, build_revision: str | None
) -> dict[str, Any]:
    resolved = manifest_path.resolve()
    relative = _repository_path(resolved)
    if relative is None or not resolved.is_file():
        raise BaselineError("jaxlib build manifest must be a repository-local file")
    try:
        manifest = _load_build_validator().load_and_validate_manifest(resolved)
    except Exception as error:
        raise BaselineError(
            f"canonical jaxlib build manifest validation failed: {error}"
        ) from error
    if (
        manifest.get("status") != "build-succeeded"
        or manifest.get("dry_run")
        or manifest.get("preflight", {}).get("ok") is not True
    ):
        raise BaselineError("jaxlib build manifest is not a successful non-dry-run build")
    inputs = manifest.get("inputs", {})
    if inputs.get("build_id") != manifest.get("build_id"):
        raise BaselineError("jaxlib build manifest build_id fields disagree")
    if inputs.get("sources", {}).get("jax", {}).get("commit") != build_revision:
        raise BaselineError("jaxlib runtime revision disagrees with build manifest JAX commit")
    attempts = manifest.get("attempts", [])
    result = manifest.get("result", {})
    if not attempts:
        raise BaselineError("successful jaxlib build manifest has no attempt")
    terminal = attempts[-1]
    if terminal.get("status") != "build-succeeded" or terminal.get("exit_code") != 0:
        raise BaselineError("jaxlib build manifest final attempt did not succeed")
    if result.get("attempt_number") != terminal.get("number"):
        raise BaselineError("jaxlib build result points to a different attempt")
    for field in ("exit_code", "error", "wheels", "generated_bazelrc"):
        if result.get(field) != terminal.get(field):
            raise BaselineError(f"jaxlib build result.{field} differs from final attempt")
    wheels = result.get("wheels", [])
    if len(wheels) != 1:
        raise BaselineError("successful jaxlib build must contain exactly one wheel")
    wheel = wheels[0]
    wheel_path = (REPOSITORY_ROOT / wheel["path"]).resolve()
    if _repository_path(wheel_path) is None or not wheel_path.is_file():
        raise BaselineError(f"built jaxlib wheel is missing: {wheel['path']}")
    if _sha256(wheel_path) != wheel["sha256"] or wheel_path.stat().st_size != wheel["size_bytes"]:
        raise BaselineError("built jaxlib wheel hash or size disagrees with build manifest")
    return {
        "kind": "build-manifest",
        "build_manifest": relative,
        "build_manifest_sha256": _sha256(resolved),
        "build_manifest_size_bytes": resolved.stat().st_size,
        "build_id": manifest["build_id"],
        "artifact_sha256": wheel["sha256"],
        "artifact_size_bytes": wheel["size_bytes"],
    }


def _direct_url_identity(name: str) -> dict[str, Any] | None:
    raw = metadata.distribution(name).read_text("direct_url.json")
    if not raw:
        return None
    try:
        direct = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BaselineError(f"invalid {name} direct_url.json: {error}") from error
    url = direct.get("url") if isinstance(direct, dict) else None
    if not isinstance(url, str):
        raise BaselineError(f"{name} direct_url.json has no URL")
    parsed = urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise BaselineError(
            f"{name} has a direct URL install; provide --jaxlib-build-manifest "
            "instead of attributing it to the lock file"
        )
    artifact_path = Path(unquote(parsed.path)).resolve()
    relative = _repository_path(artifact_path)
    if relative is None or not artifact_path.is_file():
        raise BaselineError(
            f"{name} was installed from a local source outside the repository; "
            "provide --jaxlib-build-manifest"
        )
    return {
        "kind": "repository-artifact",
        "artifact_path": relative,
        "artifact_sha256": _sha256(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
    }


def _distribution_identity(
    version: str,
    build_revision: str | None,
    build_manifest: Path | None,
) -> dict[str, Any]:
    if build_manifest is not None:
        return _build_manifest_identity(build_manifest, build_revision)
    direct_identity = _direct_url_identity("jaxlib")
    if direct_identity is not None:
        return direct_identity
    return _locked_wheel_identity("jaxlib", version)


def _mapped_native_binaries(package_root: Path) -> list[dict[str, Any]]:
    process_maps = Path("/proc/self/maps")
    if not process_maps.is_file():
        raise BaselineError("native runtime inventory requires Linux /proc/self/maps")
    binaries: set[Path] = set()
    for line in process_maps.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        candidate = Path(fields[5])
        if not candidate.is_absolute() or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved.is_relative_to(package_root) and ".so" in resolved.name:
            binaries.add(resolved)
    if not binaries:
        raise BaselineError("no mapped jaxlib native binaries were found")

    result = []
    for binary in sorted(binaries):
        item = {
            "package_path": binary.relative_to(package_root).as_posix(),
            "sha256": _sha256(binary),
            "size_bytes": binary.stat().st_size,
        }
        artifact_path = _repository_path(binary)
        if artifact_path is not None:
            item["artifact_path"] = artifact_path
        result.append(item)
    return result


def _alignment_status(
    jax_commit: str,
    jax_install_type: str,
    jaxlib_build_revision: str | None,
) -> dict[str, Any]:
    matches = (
        jax_install_type == "editable-source" and jaxlib_build_revision == jax_commit
    )
    reason = (
        "The imported editable JAX checkout and jaxlib build revision are aligned."
        if matches
        else "The imported JAX checkout or jaxlib build revision is not aligned; "
        "native runtime observations are version-skewed."
    )
    return {
        "evidence_level": "RUN-CPU",
        "qualifiers": [] if matches else ["VERSION-SKEW"],
        "jax_source_matches_jaxlib_build": matches,
        "reason": reason,
        "runtime_source_alignment": "ALIGNED" if matches else "VERSION-SKEW",
    }


def capture_baseline(jaxlib_build_manifest: Path | None = None) -> dict[str, Any]:
    configuration = _project_configuration()
    sources = {
        name: {**_git_state(REPOSITORY_ROOT / relative_path), "path": relative_path}
        for name, relative_path in SOURCE_PATHS.items()
    }
    jax_commit = sources["jax"]["git_commit"]
    raw_jaxlib_hash = getattr(jaxlib_version, "_git_hash", None)
    jaxlib_build_revision = raw_jaxlib_hash or None
    devices = jax.devices()
    jax_install_type, jax_imported_file, jax_imported_sha256 = _jax_install()
    jaxlib_root = Path(jaxlib.__file__).resolve().parent

    baseline = {
        "$schema": "schema/baseline.schema.json",
        "schema_version": "1.0",
        "external_inputs": {"libtpu": None, "tokamax": None, "tpu_target": None},
        "repository": {
            "inputs": {
                relative_path: {
                    "sha256": _sha256(REPOSITORY_ROOT / relative_path),
                    "size_bytes": (REPOSITORY_ROOT / relative_path).stat().st_size,
                }
                for relative_path in INPUT_PATHS
            },
            "root": {
                "git_commit": _command("git", "rev-parse", "HEAD"),
                "path": ".",
                "verification": "ancestor",
            },
            "sources": sources,
        },
        "runtime": {
            "device": {
                "default_platform": jax.default_backend(),
                "platforms": sorted({device.platform for device in devices}),
                "device_count": len(devices),
                "devices": [str(device) for device in devices],
                "host_os": platform.system(),
                "arch": platform.machine(),
            },
            "jax": {
                "version": jax.__version__,
                "install_type": jax_install_type,
                "imported_file": jax_imported_file,
                "imported_file_sha256": jax_imported_sha256,
                "source_checkout_git_hash": jax_commit,
                "reported_git_hash": getattr(jax_version, "_git_hash", None),
            },
            "jaxlib": {
                "version": jaxlib.__version__,
                "install_type": _jaxlib_install_type(jaxlib_root),
                "package_root": _repository_path(jaxlib_root),
                "build_revision": jaxlib_build_revision,
                "distribution": _distribution_identity(
                    jaxlib.__version__,
                    jaxlib_build_revision,
                    jaxlib_build_manifest,
                ),
                "native_binaries": _mapped_native_binaries(jaxlib_root),
            },
        },
        "status": _alignment_status(
            jax_commit, jax_install_type, jaxlib_build_revision
        ),
        "toolchain": {
            "python": {
                "project_constraint": configuration["project"]["requires-python"],
                "repository_pin": (REPOSITORY_ROOT / ".python-version").read_text().strip(),
                "runtime": platform.python_version(),
            },
            "uv": {
                "project_constraint": configuration["tool"]["uv"]["required-version"],
                "repository_pin": configuration["tool"]["uv"]["required-version"].removeprefix("=="),
                **_uv_runtime_identity(),
            },
        },
    }
    _validate_schema(baseline, BASELINE_SCHEMA)
    return baseline


def _validate_schema(
    value: dict[str, Any], schema_path: Path, description: str = "baseline"
) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot load {description} schema: {error}") from error
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise BaselineError(f"{description} schema validation failed: {details}")


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        (
            PINNED_GIT,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env=_clean_git_environment(),
    )
    return result.returncode == 0


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def verify_baseline(manifest_path: Path, current: dict[str, Any]) -> None:
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot read baseline manifest {manifest_path}: {error}") from error
    if not isinstance(expected, dict):
        raise BaselineError("baseline manifest must contain a JSON object")
    _validate_schema(expected, BASELINE_SCHEMA)
    schema_ref = expected.get("$schema")
    if not isinstance(schema_ref, str) or (manifest_path.parent / schema_ref).resolve() != BASELINE_SCHEMA:
        raise BaselineError("baseline $schema does not resolve to baseline.schema.json")

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
    parser.add_argument(
        "--jaxlib-build-manifest",
        type=Path,
        help=(
            "successful repository-local build manifest for a source-built "
            "jaxlib; required when direct install provenance is not a "
            "repository wheel"
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        baseline = capture_baseline(arguments.jaxlib_build_manifest)
        if arguments.verify is None:
            sys.stdout.write(_canonical_json(baseline))
        else:
            verify_baseline(arguments.verify.resolve(), baseline)
            status = baseline["status"]["runtime_source_alignment"]
            print(f"baseline verified: {arguments.verify} [{status}]")
    except (BaselineError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
