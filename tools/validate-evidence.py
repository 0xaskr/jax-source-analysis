#!/usr/bin/env python3
"""Validate source-analysis evidence bundles beyond JSON Schema structure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import types
from typing import Any
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator, FormatChecker


SCHEMAS = {
    "baseline": "manifests/schema/baseline.schema.json",
    "topic": "manifests/schema/topic.schema.json",
    "source-index": "manifests/schema/source-index.schema.json",
    "capture": "manifests/schema/capture.schema.json",
    "native-binaries": "manifests/schema/native-binaries.schema.json",
    "build-jaxlib": "manifests/schema/build-jaxlib.schema.json",
}
NATIVE_BINARY_FORMAT = "application/vnd.jax-source-analysis.native-binaries+json"
EXECUTION_LEVELS = {"RUN-CPU", "SIM-TPU", "COMPILE-TPU", "RUN-TPU"}
OPAQUE_URI_RE = re.compile(r"^artifact:[a-z0-9][a-z0-9._-]*$")
ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s='\"(:,;])(?:/(?!/)|[A-Za-z]:[\\/]|\\\\)"
)
USERINFO_RE = re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+")
SECRET_RE = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|authorization|credential|password|passwd|"
    r"client[_-]?secret|private[_-]?key|secret|(?:access[_-]|auth[_-])?token)"
    r"\s*[:=]|(?:^|\s)--?(?:api[_-]?key|authorization|credential|password|"
    r"passwd|client[_-]?secret|private[_-]?key|secret|(?:access[_-]|auth[_-])?token)"
    r"(?:[=_-]|\s|$))"
)
ALIGNMENT_REASON = {
    True: "The imported editable JAX checkout and jaxlib build revision are aligned.",
    False: "The imported JAX checkout or jaxlib build revision is not aligned; native runtime observations are version-skewed.",
}
REQUIRED_RUNTIME_ROLES = {
    "RUN-CPU": {"cpu-backend"},
    "SIM-TPU": {"tpu-simulator"},
    "RUN-TPU": {"tpu-runtime"},
    "COMPILE-TPU": {"tpu-compiler", "tpu-runtime"},
}
PINNED_GIT = "/usr/bin/git"
EXPECTED_UV_IDENTITY = {
    "project_constraint": "==0.12.9",
    "repository_pin": "0.12.9",
    "runtime": "0.12.9",
    "artifact_uri": "artifact:uv-0.12.9-x86_64-unknown-linux-gnu",
    "artifact_sha256": "671793498fe0a545432e2524b6691ffb9eea4540d9fda43ca2f978df2dbf8426",
    "artifact_size_bytes": 49313824,
    "retrieval": (
        "Install the pinned uv 0.12.9 x86_64 Linux standalone artifact and "
        "verify its recorded SHA-256."
    ),
}


@dataclass(frozen=True)
class ReconstructedSource:
    root: Path
    revision: str
    object_directory: Path
    alternate_object_directory: Path
    scopes: tuple[PurePosixPath, ...]
    base_files: dict[str, tuple[str, str]]
    files: dict[str, tuple[str, str]]
    changed_paths: frozenset[str]


class EvidenceValidator:
    def __init__(
        self, repository_root: Path, *, require_live_source_state: bool = False
    ) -> None:
        self.root = repository_root.resolve()
        self.require_live_source_state = require_live_source_state
        self.errors: list[str] = []
        self._schemas: dict[str, dict[str, Any]] = {}
        self._hashes: dict[Path, str] = {}
        self._checked_revisions: set[tuple[str, str]] = set()
        self._reconstructions: dict[tuple[Any, ...], ReconstructedSource | None] = {}
        self._temporary_directories: list[tempfile.TemporaryDirectory[str]] = []
        self._validated_captures: dict[Path, dict[str, Any] | None] = {}
        self._capture_stack: list[Path] = []
        self._build_validator: Any | None = None
        self.capture_count = 0

    def close(self) -> None:
        for directory in reversed(self._temporary_directories):
            directory.cleanup()
        self._temporary_directories.clear()

    def error(self, context: str | Path, message: str) -> None:
        self.errors.append(f"{context}: {message}")

    def load_json(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.error(path, f"cannot read JSON: {error}")
            return None
        if not isinstance(value, dict):
            self.error(path, "top-level JSON value must be an object")
            return None
        return value

    def schema(self, name: str) -> dict[str, Any]:
        if name not in self._schemas:
            path = self.root / SCHEMAS[name]
            value = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(value)
            self._schemas[name] = value
        return self._schemas[name]

    def build_validator(self) -> Any:
        if self._build_validator is None:
            path = self.root / "tools/build-jaxlib.py"
            if not path.is_file():
                raise ValueError(f"missing canonical build validator: {path}")
            source = path.read_bytes()
            canonical_path = Path(__file__).resolve().with_name("build-jaxlib.py")
            canonical_source = canonical_path.read_bytes()
            if source != canonical_source:
                raise ValueError(
                    "repository build validator bytes differ from the canonical "
                    f"validator at {canonical_path}"
                )
            module_name = "jax_source_analysis_build_validator_" + hashlib.sha256(
                str(self.root).encode()
            ).hexdigest()[:16]
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = ""
            sys.modules[module_name] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            self._build_validator = module
        return self._build_validator

    def validate_schema(
        self, instance: dict[str, Any], instance_path: Path, schema_name: str
    ) -> bool:
        schema = self.schema(schema_name)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            self.error(instance_path, f"schema {location}: {error.message}")

        schema_ref = instance.get("$schema")
        expected_path = (self.root / SCHEMAS[schema_name]).resolve()
        expected_id = schema.get("$id")
        if not isinstance(schema_ref, str):
            self.error(instance_path, "$schema must be a string")
        elif urlsplit(schema_ref).scheme:
            if schema_ref != expected_id:
                self.error(
                    instance_path,
                    f"$schema is {schema_ref!r}, expected schema id {expected_id!r}",
                )
        elif (instance_path.parent / schema_ref).resolve() != expected_path:
            self.error(
                instance_path,
                f"$schema does not resolve to {expected_path}",
            )
        return not errors

    def repo_path(self, raw_path: Any, context: str | Path) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path:
            self.error(context, "repository path must be a non-empty string")
            return None
        pure = PurePosixPath(raw_path)
        if (
            "\\" in raw_path
            or pure.is_absolute()
            or re.match(r"^[A-Za-z]:", raw_path)
            or ".." in pure.parts
        ):
            self.error(context, f"unsafe or non-POSIX repository path: {raw_path!r}")
            return None
        if pure.as_posix() != raw_path:
            self.error(context, f"repository path must use canonical spelling: {raw_path!r}")
            return None
        resolved = self.root.joinpath(*pure.parts).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            self.error(context, f"repository path resolves outside the root: {raw_path!r}")
            return None
        return resolved

    def require_path(
        self, raw_path: Any, context: str | Path, *, kind: str = "file"
    ) -> Path | None:
        path = self.repo_path(raw_path, context)
        if path is None:
            return None
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            self.error(context, f"missing {kind}: {raw_path}")
            return None
        return path

    def require_component_root(
        self, raw_path: Any, context: str | Path
    ) -> Path | None:
        path = self.require_path(raw_path, context, kind="directory")
        if path is None or not isinstance(raw_path, str):
            return None
        canonical = path.relative_to(self.root).as_posix()
        if canonical == "":
            canonical = "."
        if raw_path != canonical:
            self.error(
                context,
                f"component root must identify its canonical repository path: "
                f"{raw_path!r} resolves to {canonical!r}",
            )
            return None
        return path

    def sha256(self, path: Path) -> str:
        path = path.resolve()
        if path not in self._hashes:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._hashes[path] = digest.hexdigest()
        return self._hashes[path]

    def check_file(
        self,
        path: Path | None,
        expected_hash: Any,
        expected_size: Any,
        context: str | Path,
    ) -> None:
        if path is None:
            return
        if isinstance(expected_hash, str):
            actual = self.sha256(path)
            if actual != expected_hash:
                self.error(context, f"SHA-256 mismatch: expected {expected_hash}, got {actual}")
        if isinstance(expected_size, int) and path.stat().st_size != expected_size:
            self.error(context, "size_bytes does not match the file")

    def check_unique(
        self,
        items: Any,
        key: str,
        context: str | Path,
        description: str,
    ) -> set[str]:
        values = [
            item.get(key)
            for item in items
            if isinstance(item, dict) and isinstance(item.get(key), str)
        ] if isinstance(items, list) else []
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            self.error(context, f"duplicate {description}: {', '.join(duplicates)}")
        return set(values)

    def git(
        self,
        repository: Path,
        arguments: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        clean_environment = {
            name: os.environ[name]
            for name in ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")
            if name in os.environ
        }
        clean_environment.update(
            {
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
        )
        if env is not None:
            allowed = {
                "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            }
            unexpected = sorted(set(env) - allowed)
            if unexpected:
                raise ValueError(
                    "unsupported Git environment override(s): " + ", ".join(unexpected)
                )
            clean_environment.update(env)
        return subprocess.run(
            [
                PINNED_GIT,
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(repository),
                *arguments,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=clean_environment,
        )

    def git_root(self, root: Path, context: str | Path) -> Path | None:
        result = self.git(root, ["rev-parse", "--show-toplevel"])
        if result.returncode:
            self.error(context, result.stderr.decode(errors="replace").strip())
            return None
        top = Path(result.stdout.decode().strip()).resolve()
        if top != root.resolve():
            self.error(context, f"component root is not a Git toplevel: {root}")
            return None
        return top

    def validate_revision(
        self, component: dict[str, Any], context: str | Path
    ) -> Path | None:
        root_raw = component.get("root")
        revision = component.get("revision")
        if not isinstance(root_raw, str) or not isinstance(revision, str):
            return None
        root = self.require_component_root(root_raw, context)
        if root is None or self.git_root(root, context) is None:
            return None
        key = (str(root), revision)
        if key not in self._checked_revisions:
            result = self.git(root, ["cat-file", "-e", f"{revision}^{{commit}}"])
            if result.returncode:
                detail = result.stderr.decode(errors="replace").strip()
                self.error(context, f"Git revision {revision} is unavailable: {detail}")
                return None
            self._checked_revisions.add(key)
        return root

    @staticmethod
    def _component_relative(root_raw: str, raw_path: str) -> PurePosixPath | None:
        path = PurePosixPath(raw_path)
        if root_raw == ".":
            return path
        try:
            return path.relative_to(PurePosixPath(root_raw))
        except ValueError:
            return None

    @staticmethod
    def _in_scopes(path: PurePosixPath, scopes: tuple[PurePosixPath, ...]) -> bool:
        return any(path == scope or scope in path.parents for scope in scopes)

    def _component_scopes(
        self, component: dict[str, Any], context: str | Path
    ) -> tuple[PurePosixPath, ...] | None:
        root_raw = component.get("root")
        if not isinstance(root_raw, str):
            return None
        scopes = []
        for raw_scope in component.get("scope_paths", []):
            if not isinstance(raw_scope, str):
                continue
            if self.repo_path(raw_scope, context) is None:
                continue
            relative = self._component_relative(root_raw, raw_scope)
            if relative is None or relative == PurePosixPath("."):
                self.error(context, f"scope path must be a file or directory beneath component root: {raw_scope}")
                continue
            scopes.append(relative)
        return tuple(scopes) if scopes else None

    def _index_files(
        self, root: Path, env: dict[str, str], context: str | Path
    ) -> dict[str, tuple[str, str]] | None:
        result = self.git(root, ["ls-files", "--stage", "-z"], env=env)
        if result.returncode:
            self.error(context, result.stderr.decode(errors="replace").strip())
            return None
        files: dict[str, tuple[str, str]] = {}
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", maxsplit=1)
            mode, oid, stage_number = metadata.decode().split()
            path = raw_path.decode(errors="surrogateescape")
            if stage_number != "0":
                self.error(context, f"reconstructed index has an unmerged entry: {path}")
                continue
            files[path] = (mode, oid)
        return files

    def reconstruct_source(
        self, component: dict[str, Any], context: str | Path
    ) -> ReconstructedSource | None:
        root = self.validate_revision(component, context)
        revision = component.get("revision")
        scopes = self._component_scopes(component, context)
        patches = component.get("patches", [])
        if root is None or not isinstance(revision, str) or scopes is None:
            return None

        patch_records = []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            patch_path = self.require_path(patch.get("path"), context)
            if patch_path is None:
                continue
            self.check_file(patch_path, patch.get("sha256"), patch_path.stat().st_size, context)
            patch_records.append((patch_path, patch.get("sha256")))
        if len(patch_records) != len(patches):
            return None
        key = (
            str(root), revision, scopes,
            tuple((str(path), digest) for path, digest in patch_records),
        )
        if key in self._reconstructions:
            return self._reconstructions[key]

        temporary_directory = tempfile.TemporaryDirectory(prefix="evidence-reconstruction-")
        self._temporary_directories.append(temporary_directory)
        temporary_root = Path(temporary_directory.name)
        index_name = str(temporary_root / "index")
        object_directory = temporary_root / "objects"
        object_directory.mkdir()
        object_path = self.git(root, ["rev-parse", "--git-path", "objects"])
        if object_path.returncode:
            self.error(context, object_path.stderr.decode(errors="replace").strip())
            return None
        alternate_object_directory = Path(object_path.stdout.decode().strip())
        if not alternate_object_directory.is_absolute():
            alternate_object_directory = root / alternate_object_directory
        alternate_object_directory = alternate_object_directory.resolve()
        env = {
            "GIT_INDEX_FILE": index_name,
            "GIT_OBJECT_DIRECTORY": str(object_directory),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(alternate_object_directory),
        }
        reconstruction: ReconstructedSource | None = None
        try:
            read_tree = self.git(root, ["read-tree", revision], env=env)
            if read_tree.returncode:
                self.error(context, read_tree.stderr.decode(errors="replace").strip())
                return None
            base_all = self._index_files(root, env, context)
            if base_all is None:
                return None
            for patch_path, _ in patch_records:
                applied = self.git(
                    root,
                    [
                        "apply",
                        "--cached",
                        "--unidiff-zero",
                        "--whitespace=nowarn",
                        str(patch_path),
                    ],
                    env=env,
                )
                if applied.returncode:
                    detail = applied.stderr.decode(errors="replace").strip()
                    self.error(context, f"patch cannot be applied to {revision}: {detail}")
                    return None
            final_all = self._index_files(root, env, context)
            if final_all is None:
                return None
            changed = self.git(
                root,
                ["diff", "--cached", "--name-only", "--no-renames", "-z", revision],
                env=env,
            )
            if changed.returncode:
                self.error(context, changed.stderr.decode(errors="replace").strip())
                return None
            changed_paths = {
                item.decode(errors="surrogateescape")
                for item in changed.stdout.split(b"\0")
                if item
            }
            outside = sorted(
                path
                for path in changed_paths
                if not self._in_scopes(PurePosixPath(path), scopes)
            )
            if outside:
                self.error(context, "patch changes paths outside scope_paths: " + ", ".join(outside))
            if not changed_paths:
                self.error(context, "dirty source patches reconstruct no changes")
            base_files = {
                path: record
                for path, record in base_all.items()
                if self._in_scopes(PurePosixPath(path), scopes)
            }
            files = {
                path: record
                for path, record in final_all.items()
                if self._in_scopes(PurePosixPath(path), scopes)
            }
            reconstruction = ReconstructedSource(
                root=root,
                revision=revision,
                object_directory=object_directory,
                alternate_object_directory=alternate_object_directory,
                scopes=scopes,
                base_files=base_files,
                files=files,
                changed_paths=frozenset(changed_paths),
            )
            if self.require_live_source_state:
                self._validate_current_scoped_state(reconstruction, context)
            return reconstruction
        finally:
            self._reconstructions[key] = reconstruction

    def _blob_bytes(
        self,
        root: Path,
        oid: str,
        context: str | Path,
        reconstruction: ReconstructedSource | None = None,
    ) -> bytes | None:
        env = None
        if reconstruction is not None:
            env = {
                "GIT_OBJECT_DIRECTORY": str(reconstruction.object_directory),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(
                    reconstruction.alternate_object_directory
                ),
            }
        result = self.git(root, ["cat-file", "blob", oid], env=env)
        if result.returncode:
            self.error(context, result.stderr.decode(errors="replace").strip())
            return None
        return result.stdout

    def _validate_current_scoped_state(
        self, reconstruction: ReconstructedSource, context: str | Path
    ) -> None:
        head = self.git(reconstruction.root, ["rev-parse", "HEAD"])
        if head.returncode:
            detail = head.stderr.decode(errors="replace").strip()
            self.error(context, f"cannot inspect live source HEAD: {detail}")
            return
        live_revision = head.stdout.decode().strip()
        if live_revision != reconstruction.revision:
            self.error(
                context,
                f"live source HEAD differs from recorded revision: "
                f"{live_revision} != {reconstruction.revision}",
            )
            return
        scope_arguments = ["--", *(scope.as_posix() for scope in reconstruction.scopes)]
        status = self.git(
            reconstruction.root,
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
                "-z",
                *scope_arguments,
            ],
        )
        ignored = self.git(
            reconstruction.root,
            [
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                *scope_arguments,
            ],
        )
        if status.returncode or ignored.returncode:
            details = "; ".join(
                result.stderr.decode(errors="replace").strip()
                for result in (status, ignored)
                if result.returncode
            )
            self.error(context, f"cannot inspect current scoped source state: {details}")
            return
        # A durable patch is a reconstruction recipe. Once a capture is complete,
        # restoring the checkout to its cited clean revision must not invalidate it.
        clean_rollback = not status.stdout and not ignored.stdout
        expected_files = (
            reconstruction.base_files if clean_rollback else reconstruction.files
        )
        expected_description = (
            "cited clean revision" if clean_rollback else "revision plus patches"
        )
        all_expected_paths = set(reconstruction.base_files) | set(reconstruction.files)
        for relative in sorted(all_expected_paths):
            expected = expected_files.get(relative)
            path = reconstruction.root / relative
            if expected is None:
                if path.exists() or path.is_symlink():
                    self.error(context, f"current scoped path should be deleted by patches: {relative}")
                continue
            mode, oid = expected
            expected_bytes = self._blob_bytes(
                reconstruction.root, oid, context, reconstruction
            )
            if expected_bytes is None:
                continue
            if mode == "120000":
                if not path.is_symlink():
                    self.error(context, f"current scoped path is not the reconstructed symlink: {relative}")
                    continue
                actual_bytes = os.readlink(path).encode(errors="surrogateescape")
            elif mode == "160000":
                self.error(context, f"gitlink scope is unsupported; scope files within its component: {relative}")
                continue
            else:
                if not path.is_file() or path.is_symlink():
                    self.error(context, f"current scoped file is missing: {relative}")
                    continue
                actual_bytes = path.read_bytes()
                executable = bool(path.stat().st_mode & stat.S_IXUSR)
                if executable != (mode == "100755"):
                    self.error(context, f"current executable mode differs from reconstructed state: {relative}")
            if actual_bytes != expected_bytes:
                self.error(
                    context,
                    f"current scoped bytes differ from {expected_description}: {relative}",
                )

        untracked_results = [
            self.git(
                reconstruction.root,
                ["ls-files", "--others", "--exclude-standard", "-z", *scope_arguments],
            ),
            self.git(
                reconstruction.root,
                [
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "-z",
                    *scope_arguments,
                ],
            ),
        ]
        if any(result.returncode for result in untracked_results):
            details = "; ".join(
                result.stderr.decode(errors="replace").strip()
                for result in untracked_results
                if result.returncode
            )
            self.error(context, details)
            return
        untracked_records = {
            raw
            for result in untracked_results
            for raw in result.stdout.split(b"\0")
            if raw
        }
        extra = sorted(
            raw.decode(errors="surrogateescape")
            for raw in untracked_records
            if self._in_scopes(
                PurePosixPath(raw.decode(errors="surrogateescape")),
                reconstruction.scopes,
            )
            and raw.decode(errors="surrogateescape") not in expected_files
        )
        if extra:
            self.error(
                context,
                "untracked or ignored scoped paths are absent from patches: "
                + ", ".join(extra),
            )

    def validate_source_component(
        self, component: dict[str, Any], context: str | Path
    ) -> ReconstructedSource | None:
        root = self.validate_revision(component, context)
        if root is None:
            return None
        if component.get("dirty"):
            return self.reconstruct_source(component, context)
        revision = component.get("revision")
        head = self.git(root, ["rev-parse", "HEAD"])
        if self.require_live_source_state:
            if head.returncode:
                detail = head.stderr.decode(errors="replace").strip()
                self.error(context, f"cannot inspect live source HEAD: {detail}")
                return None
            live_revision = head.stdout.decode().strip()
            if live_revision != revision:
                self.error(
                    context,
                    f"live source HEAD differs from recorded revision: "
                    f"{live_revision} != {revision}",
                )
                return None
        if (
            isinstance(revision, str)
            and not head.returncode
            and head.stdout.decode().strip() == revision
        ):
            status_result = self.git(
                root,
                [
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                    "-z",
                ],
            )
            if status_result.returncode:
                self.error(
                    context,
                    "cannot verify clean source state: "
                    + status_result.stderr.decode(errors="replace").strip(),
                )
            elif status_result.stdout:
                changed = []
                for record in status_result.stdout.split(b"\0"):
                    if record:
                        changed.append(record.decode(errors="surrogateescape"))
                self.error(
                    context,
                    "source is declared clean but its current worktree differs: "
                    + ", ".join(changed),
                )
            if self.require_live_source_state:
                ignored_result = self.git(
                    root,
                    [
                        "ls-files",
                        "--others",
                        "--ignored",
                        "--exclude-standard",
                        "-z",
                    ],
                )
                if ignored_result.returncode:
                    self.error(
                        context,
                        "cannot inspect ignored source files: "
                        + ignored_result.stderr.decode(errors="replace").strip(),
                    )
                else:
                    ignored_bytecode = sorted(
                        raw.decode(errors="surrogateescape")
                        for raw in ignored_result.stdout.split(b"\0")
                        if raw.decode(errors="surrogateescape").endswith(
                            (".pyc", ".pyo")
                        )
                    )
                    if ignored_bytecode:
                        self.error(
                            context,
                            "clean source contains ignored Python bytecode: "
                            + ", ".join(ignored_bytecode),
                        )
        return None

    def revision_bytes(
        self,
        component: dict[str, Any],
        raw_path: str,
        context: str | Path,
    ) -> bytes | None:
        root_raw = component.get("root")
        revision = component.get("revision")
        if not isinstance(root_raw, str) or not isinstance(revision, str):
            return None
        relative = self._component_relative(root_raw, raw_path)
        if relative is None or relative == PurePosixPath("."):
            self.error(context, f"{raw_path} is outside component root {root_raw}")
            return None
        root = self.validate_revision(component, context)
        if root is None:
            return None
        result = self.git(
            root,
            ["ls-tree", "-z", "--full-tree", revision, "--", relative.as_posix()],
        )
        records = [record for record in result.stdout.split(b"\0") if record]
        if result.returncode or len(records) != 1:
            detail = result.stderr.decode(errors="replace").strip()
            self.error(context, f"{raw_path} is unavailable at {root_raw}@{revision}: {detail}")
            return None
        metadata, returned_path = records[0].split(b"\t", maxsplit=1)
        mode, object_type, oid = metadata.decode().split()
        if returned_path.decode(errors="surrogateescape") != relative.as_posix():
            self.error(context, f"{raw_path} does not resolve to one exact revision path")
            return None
        if mode not in {"100644", "100755"} or object_type != "blob":
            self.error(context, f"source input has unsupported Git mode {mode}: {raw_path}")
            return None
        return self._blob_bytes(root, oid, context)

    def reconstructed_bytes(
        self,
        component: dict[str, Any],
        reconstruction: ReconstructedSource,
        raw_path: str,
        context: str | Path,
    ) -> tuple[str, bytes] | None:
        root_raw = component.get("root")
        if not isinstance(root_raw, str):
            return None
        relative = self._component_relative(root_raw, raw_path)
        if relative is None:
            self.error(context, f"{raw_path} is outside component root {root_raw}")
            return None
        record = reconstruction.files.get(relative.as_posix())
        if record is None:
            self.error(context, f"{raw_path} is absent from revision plus patches")
            return None
        mode, oid = record
        if mode not in {"100644", "100755"}:
            self.error(context, f"source input has unsupported Git mode {mode}: {raw_path}")
            return None
        content = self._blob_bytes(
            reconstruction.root, oid, context, reconstruction
        )
        return (relative.as_posix(), content) if content is not None else None

    def matching_source_component(
        self,
        raw_path: str,
        components: list[dict[str, Any]],
        context: str | Path,
    ) -> dict[str, Any] | None:
        path = PurePosixPath(raw_path)
        matches: list[tuple[int, dict[str, Any]]] = []
        for component in components:
            if component.get("kind") != "source" or not isinstance(component.get("root"), str):
                continue
            root_raw = component["root"]
            relative = self._component_relative(root_raw, raw_path)
            if relative is None:
                continue
            scopes = self._component_scopes(component, context) if component.get("dirty") else None
            if scopes is not None and not self._in_scopes(relative, scopes):
                continue
            matches.append((len(PurePosixPath(root_raw).parts), component))
        if not matches:
            self.error(context, f"input has no source provenance component: {raw_path}")
            return None
        depth = max(item[0] for item in matches)
        best = [component for item_depth, component in matches if item_depth == depth]
        if len(best) != 1:
            self.error(
                context,
                f"input has ambiguous source provenance components: {[item.get('name') for item in best]}",
            )
            return None
        return best[0]

    def validate_opaque_uri(self, value: Any, context: str | Path) -> None:
        if not isinstance(value, str) or not OPAQUE_URI_RE.fullmatch(value):
            self.error(context, "artifact URI must be a sanitized opaque artifact:<id> locator")
            return
        parsed = urlsplit(value)
        if parsed.scheme != "artifact" or parsed.netloc or parsed.query or parsed.fragment:
            self.error(context, "artifact URI must not contain authority, userinfo, query, or fragment")

    def validate_sanitized_text(
        self, value: Any, context: str | Path, *, description: str
    ) -> None:
        if not isinstance(value, str) or not value.strip():
            self.error(context, f"{description} must be a non-empty string")
            return
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            self.error(context, f"{description} contains control characters")
        if (
            ABSOLUTE_PATH_RE.search(value)
            or value.startswith("//")
            or "=//" in value
            or "://" in value
            or USERINFO_RE.search(value)
        ):
            self.error(context, f"{description} contains an absolute path or authority/userinfo")
        if SECRET_RE.search(value):
            self.error(context, f"{description} contains a credential or secret assignment")

    def validate_retrieval(self, value: Any, context: str | Path) -> None:
        self.validate_sanitized_text(value, context, description="retrieval")

    def validate_invocation(
        self, value: Any, context: str | Path
    ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        self.require_path(value.get("cwd"), context, kind="directory")
        argv = value.get("argv")
        if isinstance(argv, list):
            for index, argument in enumerate(argv):
                self.validate_sanitized_text(
                    argument, f"{context} argv[{index}]", description="producer argument"
                )
        stdout_path = value.get("stdout_path")
        if isinstance(stdout_path, str):
            self.require_path(stdout_path, context)
        tool_version = value.get("tool_version")
        if tool_version is not None:
            self.validate_sanitized_text(
                tool_version, context, description="producer tool_version"
            )
        return value

    def validate_binary_component(
        self,
        component: dict[str, Any],
        source_by_name: dict[str, dict[str, Any]],
        context: str | Path,
    ) -> bool:
        source_name = component.get("source_component")
        source = source_by_name.get(source_name)
        if source is None:
            self.error(context, f"binary source_component does not resolve: {source_name}")
            skew = False
        else:
            skew = component.get("build_revision") != source.get("revision")
        artifact_path = component.get("artifact_path")
        if isinstance(artifact_path, str):
            path = self.require_path(artifact_path, context)
            self.check_file(
                path,
                component.get("artifact_sha256"),
                component.get("artifact_size_bytes"),
                context,
            )
            root = self.require_component_root(component.get("root"), context)
            if path is not None and root is not None:
                try:
                    path.relative_to(root)
                except ValueError:
                    self.error(context, "binary artifact_path resolves outside component root")
        else:
            self.validate_opaque_uri(component.get("artifact_uri"), context)
            self.validate_retrieval(component.get("retrieval"), context)
        return skew

    def validate_normalization_graph(
        self, artifacts: list[dict[str, Any]], context: str | Path
    ) -> None:
        parents = {
            artifact["id"]: artifact.get("normalized_from")
            for artifact in artifacts
            if isinstance(artifact.get("id"), str)
        }
        for artifact_id, parent in parents.items():
            if parent is not None and parent not in parents:
                self.error(context, f"artifact {artifact_id} normalized_from does not resolve: {parent}")
        for start in parents:
            seen: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in seen:
                    self.error(context, f"normalization graph contains a cycle through {current}")
                    break
                seen.add(current)
                parent = parents.get(current)
                current = parent if isinstance(parent, str) else None

    def _validate_identity(
        self, package: dict[str, Any], context: str | Path
    ) -> None:
        identity = package.get("identity", {})
        kind = identity.get("kind")
        expected_hash = identity.get("artifact_sha256")
        expected_size = identity.get("artifact_size_bytes")
        if kind == "lock-artifact":
            self.validate_opaque_uri(identity.get("artifact_uri"), context)
            self.validate_retrieval(identity.get("retrieval"), context)
            lock_path = self.require_path(identity.get("lock_path"), context)
            if lock_path is None:
                return
            try:
                with lock_path.open("rb") as file:
                    lock = tomllib.load(file)
            except (OSError, tomllib.TOMLDecodeError) as error:
                self.error(context, f"cannot read lock artifact identity: {error}")
                return
            matches = []
            for item in lock.get("package", []):
                if item.get("name") != package.get("name") or item.get("version") != package.get("version"):
                    continue
                matches.extend(
                    wheel
                    for wheel in item.get("wheels", [])
                    if wheel.get("hash") == f"sha256:{expected_hash}"
                    and wheel.get("size") == expected_size
                )
            if len(matches) != 1:
                self.error(context, "native package identity does not resolve to one locked wheel")
            elif isinstance(identity.get("artifact_uri"), str):
                filename = unquote(Path(urlsplit(matches[0].get("url", "")).path).name)
                expected_uri = "artifact:" + filename.lower()
                if identity["artifact_uri"] != expected_uri:
                    self.error(
                        context,
                        f"locked artifact URI disagrees with wheel filename: expected {expected_uri}",
                    )
        elif kind == "build-manifest":
            manifest_path = self.require_path(identity.get("build_manifest"), context)
            if manifest_path is None:
                return
            self.check_file(
                manifest_path,
                identity.get("build_manifest_sha256"),
                identity.get("build_manifest_size_bytes"),
                context,
            )
            manifest = self.load_json(manifest_path)
            if manifest is None:
                return
            try:
                validated_manifest = self.build_validator().load_and_validate_manifest(
                    manifest_path
                )
            except Exception as error:
                self.error(context, f"canonical build manifest validation failed: {error}")
                return
            if validated_manifest != manifest:
                self.error(context, "canonical build validator returned different manifest data")
                return
            build_id = identity.get("build_id")
            if manifest.get("build_id") != build_id:
                self.error(context, "native package build_id disagrees with build manifest")
            if manifest.get("status") != "build-succeeded" or manifest.get("dry_run"):
                self.error(context, "native package requires a non-dry-run build-succeeded manifest")
            if manifest.get("preflight", {}).get("ok") is not True:
                self.error(context, "native package build manifest did not pass preflight")
            inputs = manifest.get("inputs", {})
            if inputs.get("build_id") != build_id:
                self.error(context, "build manifest inputs.build_id disagrees with native identity")
            jax_commit = inputs.get("sources", {}).get("jax", {}).get("commit")
            if jax_commit != package.get("build_revision"):
                self.error(
                    context,
                    "native package build_revision disagrees with build manifest JAX commit",
                )
            attempts = manifest.get("attempts", [])
            result = manifest.get("result", {})
            if not attempts:
                self.error(context, "successful build manifest has no terminal attempt")
            else:
                terminal = attempts[-1]
                if terminal.get("status") != "build-succeeded" or terminal.get("exit_code") != 0:
                    self.error(context, "final build attempt is not a successful terminal attempt")
                if result.get("attempt_number") != terminal.get("number"):
                    self.error(context, "build result attempt_number disagrees with final attempt")
                for field in ("exit_code", "error", "wheels", "generated_bazelrc"):
                    if result.get(field) != terminal.get(field):
                        self.error(context, f"build result.{field} disagrees with final attempt")
            if result.get("exit_code") != 0 or result.get("error") is not None:
                self.error(context, "successful build result must have exit_code 0 and no error")
            wheels = result.get("wheels", [])
            matches = [
                wheel for wheel in wheels
                if wheel.get("sha256") == expected_hash and wheel.get("size_bytes") == expected_size
            ]
            if len(matches) != 1:
                self.error(context, "native package identity is absent from build manifest wheels")
            else:
                wheel_path = self.require_path(matches[0].get("path"), context)
                self.check_file(wheel_path, expected_hash, expected_size, context)
                artifact_dir = self.require_path(
                    manifest.get("paths", {}).get("artifact_dir"),
                    context,
                    kind="directory",
                )
                if wheel_path is not None and artifact_dir is not None:
                    try:
                        wheel_path.relative_to(artifact_dir)
                    except ValueError:
                        self.error(context, "build wheel is outside manifest artifact_dir")
        elif kind == "repository-artifact":
            artifact_path = self.require_path(identity.get("artifact_path"), context)
            self.check_file(artifact_path, expected_hash, expected_size, context)
        elif kind == "opaque-artifact":
            self.validate_opaque_uri(identity.get("artifact_uri"), context)
            self.validate_retrieval(identity.get("retrieval"), context)

    def validate_native_binary_manifest(
        self,
        path: Path,
        artifact_id: str,
        components: list[dict[str, Any]],
        context: str | Path,
    ) -> set[str]:
        value = self.load_json(path)
        if value is None or not self.validate_schema(value, path, "native-binaries"):
            return set()
        if value.get("inventory_id") != artifact_id:
            self.error(context, "runtime inventory_id disagrees with its capture artifact id")
        binary_components = [item for item in components if item.get("kind") == "binary"]
        packages = [item for item in value.get("packages", []) if isinstance(item, dict)]
        self.check_unique(packages, "name", context, "native package names")
        covered_components: set[str] = set()
        runtime_roles: set[str] = set()
        for package in packages:
            package_context = f"{context} package {package.get('name', '<missing>')}"
            self._validate_identity(package, package_context)
            matches = [
                item for item in binary_components
                if item.get("package") == package.get("name")
                and item.get("version") == package.get("version")
                and item.get("source_component") == package.get("source_component")
                and item.get("build_revision") == package.get("build_revision")
            ]
            if not matches:
                self.error(package_context, "native package has no matching binary component")
                continue
            package_roles = set(package.get("runtime_roles", []))
            runtime_roles.update(package_roles)
            for component in matches:
                component_roles = set(component.get("runtime_roles", []))
                if component_roles != package_roles:
                    self.error(
                        package_context,
                        f"runtime_roles disagree with binary component {component.get('name')}",
                    )
            covered_components.update(item["name"] for item in matches)
            binaries = [item for item in package.get("binaries", []) if isinstance(item, dict)]
            self.check_unique(binaries, "package_path", package_context, "package binary paths")
            artifact_paths = self.check_unique(
                [item for item in binaries if "artifact_path" in item],
                "artifact_path",
                package_context,
                "native artifact paths",
            )
            roots = [
                self.require_path(item.get("root"), package_context, kind="directory")
                for item in matches if isinstance(item.get("root"), str)
            ]
            roots = [root for root in roots if root is not None]
            for binary in binaries:
                binary_context = f"{package_context} binary {binary.get('package_path', '<missing>')}"
                raw_package_path = binary.get("package_path")
                if isinstance(raw_package_path, str):
                    canonical_package_path = PurePosixPath(raw_package_path).as_posix()
                    if canonical_package_path != raw_package_path:
                        self.error(binary_context, "package_path must use canonical spelling")
                raw_artifact = binary.get("artifact_path")
                if not isinstance(raw_artifact, str):
                    continue
                binary_path = self.require_path(raw_artifact, binary_context)
                self.check_file(
                    binary_path,
                    binary.get("sha256"),
                    binary.get("size_bytes"),
                    binary_context,
                )
                if binary_path is None:
                    continue
                relations = []
                for root in roots:
                    try:
                        relations.append(binary_path.relative_to(root).as_posix())
                    except ValueError:
                        pass
                if binary.get("package_path") not in relations:
                    self.error(binary_context, "package_path does not match any binary component root")
            for component in matches:
                component_path = component.get("artifact_path")
                if isinstance(component_path, str) and component_path not in artifact_paths:
                    self.error(package_context, f"binary component is absent from inventory: {component['name']}")
        missing = sorted(
            item["name"] for item in binary_components if item["name"] not in covered_components
        )
        if missing:
            self.error(context, "binary components absent from runtime inventory packages: " + ", ".join(missing))
        return runtime_roles

    def load_validated_capture(
        self, capture_path: Path, context: str | Path
    ) -> dict[str, Any] | None:
        capture_path = capture_path.resolve()
        if capture_path in self._validated_captures:
            return self._validated_captures[capture_path]
        if capture_path in self._capture_stack:
            cycle = self._capture_stack[self._capture_stack.index(capture_path):]
            cycle.append(capture_path)
            self.error(
                context,
                "replay capture cycle: " + " -> ".join(str(path) for path in cycle),
            )
            return None
        capture = self.load_json(capture_path)
        if capture is None:
            self._validated_captures[capture_path] = None
            return None
        self._capture_stack.append(capture_path)
        try:
            self.validate_capture(capture_path, capture)
        finally:
            self._capture_stack.pop()
        self._validated_captures[capture_path] = capture
        return capture

    def validate_replay(
        self,
        capture_path: Path,
        capture: dict[str, Any],
        component_names: set[str],
    ) -> None:
        replay = capture.get("replay")
        if not isinstance(replay, dict):
            return
        context = f"{capture_path} replay"
        origin = self.require_path(replay.get("origin_capture"), context)
        self.check_file(origin, replay.get("origin_sha256"), None, context)
        if origin is not None:
            if origin == capture_path.resolve():
                self.error(context, "replay origin_capture cannot be the replay capture itself")
            else:
                origin_value = self.load_validated_capture(origin, context)
                if origin_value is not None:
                    if origin_value.get("topic_id") != capture.get("topic_id"):
                        self.error(context, "origin capture belongs to a different topic")
                    origin_artifacts = origin_value.get("__artifacts_by_id", {})
                    missing = sorted(
                        set(replay.get("origin_artifact_refs", []))
                        - set(origin_artifacts)
                    )
                    if missing:
                        self.error(
                            context,
                            "origin_artifact_refs do not resolve: " + ", ".join(missing),
                        )
        if replay.get("tool_component") not in component_names:
            self.error(context, "replay tool_component does not resolve")

    def validate_capture(self, capture_path: Path, capture: dict[str, Any]) -> None:
        self.capture_count += 1
        if not self.validate_schema(capture, capture_path, "capture"):
            return
        if capture_path.name != "manifest.json":
            self.error(capture_path, "capture manifest must be named manifest.json")
        if capture_path.parent.name != capture.get("capture_id"):
            self.error(capture_path, "capture_id must match the containing directory")

        components = [
            item for item in capture.get("provenance", {}).get("components", [])
            if isinstance(item, dict)
        ]
        component_names = self.check_unique(components, "name", capture_path, "component names")
        source_components = [item for item in components if item.get("kind") == "source"]
        source_by_name = {
            item["name"]: item for item in source_components if isinstance(item.get("name"), str)
        }
        reconstructions: dict[str, ReconstructedSource] = {}
        for component in source_components:
            reconstructed = self.validate_source_component(component, capture_path)
            if reconstructed is not None:
                reconstructions[component["name"]] = reconstructed

        analysis_revision = capture.get("provenance", {}).get("analysis_repository_revision")
        analysis_components = [item for item in source_components if item.get("root") == "."]
        if len(analysis_components) != 1:
            self.error(capture_path, "capture must contain one source component rooted at '.'")
        elif analysis_components[0].get("revision") != analysis_revision:
            self.error(capture_path, "analysis_repository_revision disagrees with the '.' source component")

        expected_skew = False
        for component in components:
            if component.get("kind") == "binary":
                expected_skew |= self.validate_binary_component(
                    component, source_by_name, capture_path
                )
        has_skew = "VERSION-SKEW" in capture.get("qualifiers", [])
        if has_skew != expected_skew:
            self.error(
                capture_path,
                f"VERSION-SKEW qualifier={has_skew} disagrees with binary/source build revisions ({expected_skew})",
            )

        producer = self.validate_invocation(capture.get("producer"), capture_path)

        input_paths = self.check_unique(capture.get("inputs", []), "path", capture_path, "input paths")
        scoped_inputs: dict[str, set[str]] = {name: set() for name in reconstructions}
        for index, item in enumerate(capture.get("inputs", [])):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            context = f"{capture_path} input {index}"
            raw_path = item["path"]
            if self.repo_path(raw_path, context) is None:
                continue
            component = self.matching_source_component(raw_path, source_components, context)
            if component is None:
                continue
            content: bytes | None = None
            name = component.get("name")
            if name in reconstructions:
                reconstructed = self.reconstructed_bytes(
                    component, reconstructions[name], raw_path, context
                )
                if reconstructed is not None:
                    relative, content = reconstructed
                    scoped_inputs[name].add(relative)
            else:
                content = self.revision_bytes(component, raw_path, context)
            if content is not None:
                actual = hashlib.sha256(content).hexdigest()
                if actual != item.get("sha256"):
                    self.error(context, f"SHA-256 mismatch: expected {item.get('sha256')}, got {actual}")
        for name, reconstruction in reconstructions.items():
            expected = set(reconstruction.files)
            if scoped_inputs[name] != expected:
                missing = sorted(expected - scoped_inputs[name])
                extra = sorted(scoped_inputs[name] - expected)
                self.error(
                    capture_path,
                    f"dirty component {name} input inventory is not exact; missing={missing}, extra={extra}",
                )

        artifacts = [item for item in capture.get("artifacts", []) if isinstance(item, dict)]
        artifact_ids = self.check_unique(artifacts, "id", capture_path, "artifact ids")
        self.check_unique(
            [item for item in artifacts if isinstance(item.get("path"), str)],
            "path",
            capture_path,
            "artifact paths",
        )
        self.check_unique(
            [item for item in artifacts if isinstance(item.get("uri"), str)],
            "uri",
            capture_path,
            "artifact URIs",
        )
        artifacts_by_id = {
            item["id"]: item for item in artifacts if isinstance(item.get("id"), str)
        }
        for artifact in artifacts:
            context = f"{capture_path} artifact {artifact.get('id', '<missing>')}"
            self.validate_retrieval(artifact.get("retrieval"), context)
            if isinstance(artifact.get("path"), str):
                path = self.require_path(artifact["path"], context)
                self.check_file(path, artifact.get("sha256"), artifact.get("size_bytes"), context)
                if path is not None:
                    try:
                        path.relative_to(capture_path.parent.resolve())
                    except ValueError:
                        self.error(context, "local generated artifact must be inside its capture directory")
            else:
                self.validate_opaque_uri(artifact.get("uri"), context)
        assertions = [
            item
            for item in capture.get("outcome", {}).get("assertions", [])
            if isinstance(item, dict)
        ]
        self.check_unique(assertions, "id", capture_path, "assertion ids")
        outcome_status = capture.get("outcome", {}).get("status")
        for assertion in assertions:
            context = f"{capture_path} assertion {assertion.get('id', '<missing>')}"
            probe_ref = assertion.get("probe_ref")
            if probe_ref not in input_paths:
                self.error(context, "probe_ref does not resolve to a capture input")
            if producer is not None and probe_ref not in producer.get("argv", []):
                self.error(context, "probe_ref is not executed by producer.argv")
            missing = sorted(set(assertion.get("artifact_refs", [])) - artifact_ids)
            if missing:
                self.error(
                    context,
                    "artifact_refs do not resolve: " + ", ".join(missing),
                )
            if outcome_status == "pass" and assertion.get("status") != "pass":
                self.error(context, "passing capture requires every assertion to pass")
        if outcome_status != "pass" and assertions and all(
            assertion.get("status") == "pass" for assertion in assertions
        ):
            self.error(capture_path, "non-passing capture requires a failed assertion")
        self.validate_normalization_graph(artifacts, capture_path)
        if producer is not None and isinstance(producer.get("stdout_path"), str):
            stdout_matches = [
                artifact
                for artifact in artifacts
                if artifact.get("path") == producer["stdout_path"]
                and artifact.get("kind") == "stdout"
            ]
            if len(stdout_matches) != 1:
                self.error(
                    capture_path,
                    "producer.stdout_path must resolve to exactly one stdout artifact",
                )

        inventory_id = capture.get("runtime_inventory")
        runtime_roles: set[str] = set()
        if capture.get("evidence_level") in EXECUTION_LEVELS:
            inventory = artifacts_by_id.get(inventory_id)
            if inventory is None:
                self.error(capture_path, "runtime_inventory does not resolve to a capture artifact")
            elif inventory.get("format") != NATIVE_BINARY_FORMAT or not isinstance(inventory.get("path"), str):
                self.error(capture_path, "runtime_inventory must be a local native-binaries manifest artifact")
            else:
                inventory_path = self.require_path(inventory["path"], capture_path)
                if inventory_path is not None:
                    runtime_roles = self.validate_native_binary_manifest(
                        inventory_path, inventory_id, components, capture_path
                    )
            required_roles = REQUIRED_RUNTIME_ROLES.get(capture.get("evidence_level"))
            if required_roles and not runtime_roles.intersection(required_roles):
                self.error(
                    capture_path,
                    f"{capture.get('evidence_level')} requires one runtime role from "
                    + ", ".join(sorted(required_roles)),
                )
        self.validate_replay(capture_path, capture, component_names)
        capture["__artifact_ids"] = artifact_ids
        capture["__artifacts_by_id"] = artifacts_by_id
        capture["__source_components"] = source_components

    def validate_source_index(
        self,
        source_path: Path,
        source_index: dict[str, Any],
        evidence_ids: set[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self.validate_schema(source_index, source_path, "source-index"):
            return {}, {}
        components = [item for item in source_index.get("components", []) if isinstance(item, dict)]
        component_ids = self.check_unique(components, "id", source_path, "component ids")
        component_by_id = {
            item["id"]: item for item in components if isinstance(item.get("id"), str)
        }
        for component in components:
            self.validate_revision(component, source_path)

        entries = [item for item in source_index.get("entries", []) if isinstance(item, dict)]
        entry_ids = self.check_unique(entries, "id", source_path, "source entry ids")
        entry_by_id = {item["id"]: item for item in entries if isinstance(item.get("id"), str)}
        for entry in entries:
            context = f"{source_path} entry {entry.get('id', '<missing>')}"
            component_id = entry.get("component")
            component = component_by_id.get(component_id)
            if component is None:
                self.error(context, f"component does not resolve: {component_id}")
                continue
            for relation in ("callers", "callees"):
                if entry.get("id") in entry.get(relation, []):
                    self.error(context, f"{relation} cannot reference the entry itself")
                missing = sorted(set(entry.get(relation, [])) - entry_ids)
                if missing:
                    self.error(context, f"{relation} do not resolve: {', '.join(missing)}")
            missing_evidence = sorted(set(entry.get("evidence_refs", [])) - evidence_ids)
            if missing_evidence:
                self.error(context, f"evidence_refs do not resolve: {', '.join(missing_evidence)}")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            if self.repo_path(raw_path, context) is None:
                continue
            content = self.revision_bytes(component, raw_path, context)
            if content is None:
                continue
            lines = content.splitlines()
            location = entry.get("location", {})
            start_line = location.get("start_line")
            end_line = location.get("end_line", start_line)
            if isinstance(start_line, int) and isinstance(end_line, int):
                if end_line < start_line:
                    self.error(context, "location.end_line precedes start_line")
                elif end_line > len(lines):
                    self.error(context, f"location exceeds file length {len(lines)}")
                else:
                    symbol = entry.get("symbol")
                    window = b"\n".join(lines[start_line - 1:end_line])
                    if isinstance(symbol, str) and symbol.encode() not in window:
                        self.error(
                            context,
                            f"symbol {symbol!r} is absent from the revision location window",
                        )
        for entry in entries:
            entry_id = entry.get("id")
            for callee in entry.get("callees", []):
                if entry_id not in entry_by_id.get(callee, {}).get("callers", []):
                    self.error(source_path, f"callgraph is not reciprocal: {entry_id} -> {callee}")
            for caller in entry.get("callers", []):
                if entry_id not in entry_by_id.get(caller, {}).get("callees", []):
                    self.error(source_path, f"callgraph is not reciprocal: {caller} -> {entry_id}")
        return component_by_id, entry_by_id

    def validate_topic(self, topic_path: Path) -> None:
        topic = self.load_json(topic_path)
        if topic is None or not self.validate_schema(topic, topic_path, "topic"):
            return
        topic_path = topic_path.resolve()
        topic_id = topic.get("topic_id")
        if topic_path.name != "topic.json" or topic_path.parent.name != topic_id:
            self.error(topic_path, "topic must be stored as <topic_id>/topic.json")

        readme = self.require_path(topic.get("readme"), topic_path)
        if readme is not None and readme != topic_path.parent / "README.md":
            self.error(topic_path, "readme must be README.md in the topic directory")
        source_path = self.require_path(topic.get("source_index"), topic_path)
        if source_path is not None and source_path != topic_path.parent / "source-index.json":
            self.error(topic_path, "source_index must be source-index.json in the topic directory")

        evidence = [item for item in topic.get("evidence", []) if isinstance(item, dict)]
        evidence_ids = self.check_unique(evidence, "id", topic_path, "evidence ids")
        evidence_by_id = {
            item["id"]: item for item in evidence if isinstance(item.get("id"), str)
        }
        source_index: dict[str, Any] | None = None
        source_components: dict[str, dict[str, Any]] = {}
        entries: dict[str, dict[str, Any]] = {}
        if source_path is not None:
            source_index = self.load_json(source_path)
            if source_index is not None:
                if source_index.get("topic_id") != topic_id:
                    self.error(source_path, "topic_id disagrees with topic.json")
                source_components, entries = self.validate_source_index(
                    source_path, source_index, evidence_ids
                )

        for entry_id, entry in entries.items():
            for evidence_id in entry.get("evidence_refs", []):
                if entry_id not in evidence_by_id.get(evidence_id, {}).get("source_refs", []):
                    self.error(topic_path, f"entry/evidence refs are not reciprocal: {entry_id} -> {evidence_id}")
        for evidence_id, item in evidence_by_id.items():
            for entry_id in item.get("source_refs", []):
                if evidence_id not in entries.get(entry_id, {}).get("evidence_refs", []):
                    self.error(topic_path, f"evidence/entry refs are not reciprocal: {evidence_id} -> {entry_id}")

        topic_components = [
            item for item in topic.get("provenance", {}).get("components", [])
            if isinstance(item, dict)
        ]
        self.check_unique(topic_components, "name", topic_path, "component names")
        for component in topic_components:
            self.validate_source_component(component, topic_path)

        revisions_by_root: dict[Path, set[str]] = {}
        for component in [*topic_components, *source_components.values()]:
            root = component.get("root")
            revision = component.get("revision")
            if isinstance(root, str) and isinstance(revision, str):
                canonical_root = self.repo_path(root, topic_path)
                if canonical_root is not None:
                    revisions_by_root.setdefault(canonical_root, set()).add(revision)

        captures: dict[Path, dict[str, Any]] = {}
        for item in evidence:
            context = f"{topic_path} evidence {item.get('id', '<missing>')}"
            missing_sources = sorted(set(item.get("source_refs", [])) - set(entries))
            if missing_sources:
                self.error(context, f"source_refs do not resolve: {', '.join(missing_sources)}")
            self.validate_invocation(item.get("reproduce"), context)

            raw_capture = item.get("capture_manifest")
            if not isinstance(raw_capture, str):
                continue
            capture_path = self.require_path(raw_capture, context)
            if capture_path is None:
                continue
            expected_capture_parent = topic_path.parent / "captures" / capture_path.parent.name
            if capture_path.parent != expected_capture_parent or capture_path.name != "manifest.json":
                self.error(context, "capture must be <topic>/captures/<capture_id>/manifest.json")
            if capture_path not in captures:
                capture = self.load_validated_capture(capture_path, context)
                if capture is None:
                    continue
                captures[capture_path] = capture
                for component in capture.get("__source_components", []):
                    root = component.get("root")
                    revision = component.get("revision")
                    if isinstance(root, str) and isinstance(revision, str) and root != ".":
                        canonical_root = self.repo_path(root, context)
                        if canonical_root is not None:
                            revisions_by_root.setdefault(canonical_root, set()).add(revision)
            capture = captures.get(capture_path)
            if capture is None:
                continue
            if capture.get("topic_id") != topic_id:
                self.error(context, "capture topic_id disagrees with topic.json")
            if capture.get("evidence_level") != item.get("evidence_level"):
                self.error(context, "capture evidence_level disagrees with topic evidence")
            if capture.get("qualifiers", []) != item.get("qualifiers", []):
                self.error(context, "capture qualifiers disagree with topic evidence")
            if capture.get("producer") != item.get("reproduce"):
                self.error(context, "topic reproduce must exactly match capture producer")
            missing_artifacts = sorted(
                set(item.get("artifact_refs", [])) - capture.get("__artifact_ids", set())
            )
            if missing_artifacts:
                self.error(context, "artifact_refs are absent from the capture: " + ", ".join(missing_artifacts))
            if item.get("evidence_level") in EXECUTION_LEVELS and capture.get("runtime_inventory") not in item.get("artifact_refs", []):
                self.error(context, "execution evidence must reference its runtime_inventory artifact")
            if topic.get("status") == "verified" and capture.get("outcome", {}).get("status") != "pass":
                self.error(context, "verified evidence requires a passing capture outcome")

            if item.get("evidence_level") != "SOURCE-ONLY":
                capture_sources = set()
                for component in capture.get("__source_components", []):
                    root = component.get("root")
                    revision = component.get("revision")
                    if isinstance(root, str) and isinstance(revision, str):
                        canonical_root = self.repo_path(root, context)
                        if canonical_root is not None:
                            capture_sources.add((canonical_root, revision))
                for entry_id in item.get("source_refs", []):
                    entry = entries.get(entry_id, {})
                    component = source_components.get(entry.get("component"), {})
                    root = component.get("root")
                    revision = component.get("revision")
                    canonical_root = (
                        self.repo_path(root, context) if isinstance(root, str) else None
                    )
                    key = (canonical_root, revision)
                    if key not in capture_sources:
                        self.error(context, f"capture provenance omits source component for {entry_id}")

        for root, revisions in revisions_by_root.items():
            if len(revisions) > 1:
                display_root = root.relative_to(self.root).as_posix()
                self.error(topic_path, f"source revision mismatch for {display_root}: {', '.join(sorted(revisions))}")

    def _git_status(self, root: Path, context: str | Path) -> list[str] | None:
        result = self.git(
            root,
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
        )
        if result.returncode:
            self.error(context, result.stderr.decode(errors="replace").strip())
            return None
        return result.stdout.decode(errors="surrogateescape").splitlines()

    def validate_baseline(self, baseline_path: Path | None = None) -> None:
        path = baseline_path or self.root / "manifests/baseline.json"
        if not path.is_file():
            self.error(path, "missing baseline manifest")
            return
        baseline = self.load_json(path)
        if baseline is None or not self.validate_schema(baseline, path, "baseline"):
            return
        for raw_path, record in baseline.get("repository", {}).get("inputs", {}).items():
            input_path = self.require_path(raw_path, path)
            self.check_file(input_path, record.get("sha256"), record.get("size_bytes"), path)

        repository = baseline.get("repository", {})
        root_record = repository.get("root", {})
        root_component = {
            "root": root_record.get("path"),
            "revision": root_record.get("git_commit"),
        }
        root_repository = self.validate_revision(root_component, path)
        if root_repository is not None:
            head = self.git(root_repository, ["rev-parse", "HEAD"])
            ancestor = self.git(
                root_repository,
                [
                    "merge-base",
                    "--is-ancestor",
                    str(root_record.get("git_commit")),
                    head.stdout.decode().strip(),
                ],
            ) if not head.returncode else head
            if ancestor.returncode:
                self.error(path, "baseline repository root commit is not an ancestor of HEAD")

        source_records = repository.get("sources", {})
        for name, source in source_records.items():
            context = f"{path} source {name}"
            component = {
                "root": source.get("path"),
                "revision": source.get("git_commit"),
            }
            source_root = self.validate_revision(component, context)
            if source_root is None:
                continue
            head = self.git(source_root, ["rev-parse", "HEAD"])
            if head.returncode or head.stdout.decode().strip() != source.get("git_commit"):
                self.error(context, "source HEAD disagrees with baseline git_commit")
            current_status = self._git_status(source_root, context)
            if current_status is not None:
                if source.get("dirty") != bool(current_status):
                    self.error(context, "dirty flag disagrees with current Git status")
                if source.get("status") != current_status:
                    self.error(context, "recorded status disagrees with current Git status")

        runtime = baseline.get("runtime", {})
        jax_runtime = runtime.get("jax", {})
        imported_file = jax_runtime.get("imported_file")
        imported: Path | None = None
        if isinstance(imported_file, str):
            imported = self.require_path(imported_file, path)
            self.check_file(imported, jax_runtime.get("imported_file_sha256"), None, path)
        jax_source = source_records.get("jax", {})
        source_commit = jax_runtime.get("source_checkout_git_hash")
        if source_commit != jax_source.get("git_commit"):
            self.error(path, "runtime JAX source_checkout_git_hash disagrees with source baseline")
        if jax_runtime.get("install_type") == "editable-source":
            source_root = self.require_path(jax_source.get("path"), path, kind="directory")
            if not isinstance(imported_file, str) or imported is None or source_root is None:
                self.error(path, "editable JAX runtime requires a repository-local imported_file")
            else:
                try:
                    imported.relative_to(source_root)
                except ValueError:
                    self.error(path, "editable JAX imported_file is outside the JAX source root")
                revision_content = self.revision_bytes(
                    {"root": jax_source.get("path"), "revision": source_commit},
                    imported_file,
                    path,
                )
                if revision_content is not None and hashlib.sha256(revision_content).hexdigest() != jax_runtime.get("imported_file_sha256"):
                    self.error(path, "editable JAX imported_file differs from its source revision")
        reported_hash = jax_runtime.get("reported_git_hash")
        if isinstance(reported_hash, str) and reported_hash != source_commit:
            self.error(path, "reported JAX git hash disagrees with source checkout")

        jaxlib_runtime = runtime.get("jaxlib", {})
        package_root_raw = jaxlib_runtime.get("package_root")
        package_root = (
            self.require_path(package_root_raw, path, kind="directory")
            if isinstance(package_root_raw, str)
            else None
        )
        native_binaries = [
            item for item in jaxlib_runtime.get("native_binaries", [])
            if isinstance(item, dict)
        ]
        self.check_unique(native_binaries, "package_path", path, "baseline native package paths")
        self.check_unique(
            [item for item in native_binaries if "artifact_path" in item],
            "artifact_path",
            path,
            "baseline native artifact paths",
        )
        for binary in native_binaries:
            package_path_raw = binary.get("package_path")
            package_path = PurePosixPath(package_path_raw) if isinstance(package_path_raw, str) else None
            if (
                package_path is None
                or package_path.is_absolute()
                or ".." in package_path.parts
                or package_path.as_posix() != package_path_raw
            ):
                self.error(path, f"invalid native package_path: {package_path_raw!r}")
            if isinstance(binary, dict) and isinstance(binary.get("artifact_path"), str):
                binary_path = self.require_path(binary["artifact_path"], path)
                self.check_file(binary_path, binary.get("sha256"), binary.get("size_bytes"), path)
                if binary_path is not None and package_root is not None and package_path is not None:
                    try:
                        relative = binary_path.relative_to(package_root).as_posix()
                    except ValueError:
                        self.error(path, "baseline native artifact is outside package_root")
                    else:
                        if relative != package_path_raw:
                            self.error(path, "baseline native package_path disagrees with artifact_path")

        build_revision = jaxlib_runtime.get("build_revision")
        distribution = jaxlib_runtime.get("distribution")
        if isinstance(distribution, dict):
            self._validate_identity(
                {
                    "name": "jaxlib",
                    "version": jaxlib_runtime.get("version"),
                    "build_revision": build_revision,
                    "identity": distribution,
                },
                f"{path} jaxlib distribution",
            )
        aligned = jax_runtime.get("install_type") == "editable-source" and source_commit == build_revision
        status = baseline.get("status", {})
        expected_qualifiers = [] if aligned else ["VERSION-SKEW"]
        expected_alignment = "ALIGNED" if aligned else "VERSION-SKEW"
        if (
            status.get("jax_source_matches_jaxlib_build") != aligned
            or status.get("qualifiers") != expected_qualifiers
            or status.get("runtime_source_alignment") != expected_alignment
            or status.get("reason") != ALIGNMENT_REASON[aligned]
        ):
            self.error(path, "baseline alignment fields do not match derived runtime/source state")

        uv_identity = baseline.get("toolchain", {}).get("uv")
        if uv_identity != EXPECTED_UV_IDENTITY:
            self.error(path, "baseline uv identity does not match the pinned artifact")
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            self.error(path, "pinned uv executable is unavailable")
        else:
            uv_path = Path(uv_executable).resolve()
            if not uv_path.is_file():
                self.error(path, "resolved uv executable is not a regular file")
            else:
                self.check_file(
                    uv_path,
                    EXPECTED_UV_IDENTITY["artifact_sha256"],
                    EXPECTED_UV_IDENTITY["artifact_size_bytes"],
                    path,
                )

    def validate_schema_contract(self) -> None:
        for name in SCHEMAS:
            self.schema(name)
        topic = self.schema("topic")
        capture = self.schema("capture")
        topic_levels = topic.get("$defs", {}).get("evidenceLevel", {}).get("enum")
        capture_levels = capture.get("$defs", {}).get("evidenceLevel", {}).get("enum")
        topic_qualifiers = topic.get("$defs", {}).get("qualifier", {}).get("enum")
        capture_qualifiers = capture.get("$defs", {}).get("qualifier", {}).get("enum")
        if topic_levels != capture_levels:
            self.error("manifests/schema", "topic and capture evidence levels differ")
        if topic_qualifiers != capture_qualifiers:
            self.error("manifests/schema", "topic and capture qualifiers differ")
        self.validate_baseline()


def discover_topics(root: Path, arguments: list[Path]) -> list[Path]:
    if arguments:
        topics = []
        for argument in arguments:
            path = argument if argument.is_absolute() else root / argument
            path = (path / "topic.json" if path.is_dir() else path).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as error:
                raise ValueError(f"topic path is outside repository root: {path}") from error
            topics.append(path)
        return sorted({path.resolve() for path in topics})
    docs = root / "docs"
    return sorted(path.resolve() for path in docs.rglob("topic.json")) if docs.is_dir() else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "topics", nargs="*", type=Path,
        help="topic.json files or topic directories (default: scan repository topics)",
    )
    parser.add_argument(
        "--repository-root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root used to resolve evidence paths",
    )
    parser.add_argument(
        "--require-live-source-state",
        action="store_true",
        help=(
            "also require dirty source worktrees at cited HEADs to match their "
            "patch reconstruction; use while recording, not for durable replay"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate without modifying evidence (the default behavior)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    topics = discover_topics(root, args.topics)
    if not topics:
        print("ERROR: no topic.json files found", file=sys.stderr)
        return 1
    validator = EvidenceValidator(
        root, require_live_source_state=args.require_live_source_state
    )
    try:
        validator.validate_schema_contract()
        for topic in topics:
            validator.validate_topic(topic)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        validator.error("validator", f"unexpected validation failure: {type(error).__name__}: {error}")
    finally:
        validator.close()
    if validator.errors:
        for error in validator.errors:
            print(f"ERROR {error}", file=sys.stderr)
        print(
            f"FAILED: {len(validator.errors)} error(s) in {len(topics)} topic bundle(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: {len(topics)} topic bundle(s), {validator.capture_count} capture(s), "
        f"{len(validator._hashes)} file hash(es)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
