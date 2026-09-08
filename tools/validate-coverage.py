#!/usr/bin/env python3
"""Validate and summarize the JAX-to-TPU coverage inventory."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE = ROOT / "manifests/coverage.json"
SCHEMA = ROOT / "manifests/schema/coverage.schema.json"
EVIDENCE_VALIDATOR = ROOT / "tools/validate-evidence.py"
DEPTH = {f"L{level}": level for level in range(6)}
EXECUTION_LEVELS = {"RUN-CPU", "SIM-TPU", "COMPILE-TPU", "RUN-TPU"}
IR_ARTIFACT_KINDS = {"jaxpr", "stablehlo", "shardy", "hlo", "mosaic", "llo"}
IR_TEXT_FORMATS = {
    "jaxpr": "text/vnd.jax.jaxpr",
    "stablehlo": "text/x-mlir; dialect=stablehlo",
    "shardy": "text/x-mlir; dialect=shardy",
    "hlo": "text/vnd.xla.hlo",
    "mosaic": "text/x-mlir; dialect=mosaic-tpu",
    "llo": "text/vnd.tpu.llo",
}
MAX_REVIEWABLE_IR_BYTES = 1024 * 1024
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class CoverageError(RuntimeError):
  """Raised when the coverage inventory cannot be trusted."""


@dataclass(frozen=True)
class SourceReference:
  entry: dict[str, Any]
  component: dict[str, Any]


@dataclass
class ResolvedEvidence:
  claims: dict[str, tuple[str, dict[str, Any]]]
  sources: dict[str, SourceReference]
  captures: dict[str, dict[str, Any]]
  documentation: list[Path]
  probes: list[Path]


def _load_object(path: Path, context: str) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise CoverageError(f"{context}: cannot read JSON from {path}: {error}") from error
  if not isinstance(value, dict):
    raise CoverageError(f"{context}: {path} must contain a JSON object")
  return value


def _schema_errors(schema: dict[str, Any], value: dict[str, Any]) -> list[str]:
  validator = Draft202012Validator(schema, format_checker=FormatChecker())
  errors = sorted(
      validator.iter_errors(value),
      key=lambda error: tuple(str(part) for part in error.absolute_path),
  )
  messages = []
  for error in errors:
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    messages.append(f"schema {location}: {error.message}")
  return messages


def _validate_evidence_contract() -> None:
  validator = EVIDENCE_VALIDATOR.resolve()
  try:
    validator.relative_to(ROOT.resolve())
  except ValueError as error:
    raise CoverageError(
        f"evidence validator resolves outside the repository: {EVIDENCE_VALIDATOR}"
    ) from error
  if not validator.is_file():
    raise CoverageError(f"evidence validator is missing: {EVIDENCE_VALIDATOR}")
  try:
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--repository-root",
            str(ROOT.resolve()),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
  except OSError as error:
    raise CoverageError(f"cannot run evidence validator: {error}") from error
  if result.returncode:
    detail = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    raise CoverageError(
        "evidence validation failed before coverage validation"
        + (f":\n{detail}" if detail else "")
    )


def _validate_schema_reference(
    coverage_path: Path, coverage: dict[str, Any], schema: dict[str, Any]
) -> None:
  raw = coverage.get("$schema")
  if not isinstance(raw, str):
    raise CoverageError("$schema must be a string")
  if raw == schema.get("$id"):
    return
  if "://" in raw or raw.startswith("file:"):
    raise CoverageError(
        f"$schema must identify the fixed local schema {SCHEMA}, got {raw!r}"
    )
  if (coverage_path.parent / raw).resolve() != SCHEMA.resolve():
    raise CoverageError(
        f"$schema resolves to {(coverage_path.parent / raw).resolve()}, "
        f"expected {SCHEMA.resolve()}"
    )


def _repository_file(raw: str, context: str) -> Path:
  pure = PurePosixPath(raw)
  if (
      not raw
      or "\\" in raw
      or pure.is_absolute()
      or re.match(r"^[A-Za-z]:", raw)
      or ".." in pure.parts
  ):
    raise CoverageError(
        f"{context} is not a safe repository-relative path: {raw!r}"
    )
  resolved = ROOT.joinpath(*pure.parts).resolve()
  try:
    resolved.relative_to(ROOT)
  except ValueError as error:
    raise CoverageError(
        f"{context} resolves outside the repository: {raw!r}"
    ) from error
  if not resolved.is_file():
    raise CoverageError(f"{context} references a missing regular file: {raw!r}")
  return resolved


def _repository_relative(path: Path) -> str:
  return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _looks_like_textual_ir(kind: str, text: str) -> bool:
  """Apply a deliberately small signature check, not a full IR parse."""
  if kind == "jaxpr":
    return bool(
        re.search(r"(?s)^\s*\{\s*lambda\b.*\blet\b.*\bin\s*\(", text)
    )
  if kind == "stablehlo":
    return bool(
        re.search(r"\bmodule\b", text)
        and re.search(r"\bfunc\.func\b", text)
        and ("stablehlo." in text or re.search(r"\breturn\b", text))
    )
  if kind == "shardy":
    return bool(re.search(r"\bmodule\b", text) and "sdy." in text)
  if kind == "hlo":
    return bool(
        re.search(r"(?m)^\s*HloModule\b", text)
        and re.search(r"(?m)^\s*ENTRY\b", text)
    )
  if kind == "mosaic":
    return bool(
        re.search(r"\bmodule\b", text)
        and ("tpu." in text or "mosaic." in text)
    )
  if kind == "llo":
    return bool(
        re.search(r"\b(?:module|LloModule)\b", text)
        and re.search(r"\b(?:llo|tpu)\.", text, re.IGNORECASE)
    )
  return False


def _validate_local_ir_artifact(
    artifact: dict[str, Any], context: str
) -> None:
  """Validate the reviewable bytes used as an L3 IR witness."""
  artifact_id = artifact.get("id", "<missing>")
  artifact_context = f"{context}: IR artifact {artifact_id!r}"
  kind = artifact.get("kind")
  expected_format = IR_TEXT_FORMATS.get(kind)
  if artifact.get("generated") is not True:
    raise CoverageError(f"{artifact_context} must be marked generated")
  if artifact.get("format") != expected_format:
    raise CoverageError(
        f"{artifact_context} kind {kind!r} requires format {expected_format!r}, "
        f"got {artifact.get('format')!r}"
    )
  raw_path = artifact.get("path")
  if not isinstance(raw_path, str):
    raise CoverageError(
        f"{artifact_context} must be a local artifact so its IR bytes can be checked"
    )
  path = _repository_file(raw_path, artifact_context)
  actual_size = path.stat().st_size
  if actual_size == 0:
    raise CoverageError(f"{artifact_context} is empty")
  if actual_size > MAX_REVIEWABLE_IR_BYTES:
    raise CoverageError(
        f"{artifact_context} exceeds the 1 MiB reviewable IR limit; add a "
        "normalized or minimal textual artifact"
    )
  if artifact.get("size_bytes") != actual_size:
    raise CoverageError(
        f"{artifact_context} size_bytes does not match its local bytes"
    )
  try:
    payload = path.read_bytes()
  except OSError as error:
    raise CoverageError(f"{artifact_context} cannot be read: {error}") from error
  digest = hashlib.sha256(payload).hexdigest()
  if artifact.get("sha256") != digest:
    raise CoverageError(
        f"{artifact_context} SHA-256 does not match its local bytes"
    )
  try:
    text = payload.decode("utf-8")
  except UnicodeDecodeError as error:
    raise CoverageError(
        f"{artifact_context} format spoof: L3 review IR must be UTF-8 text"
    ) from error
  if "\x00" in text or not _looks_like_textual_ir(kind, text):
    raise CoverageError(
        f"{artifact_context} format spoof: bytes lack the minimal {kind} text signature"
    )


class ReferenceIndex:
  """Resolve coverage references through the repository's topic bundles."""

  def __init__(self) -> None:
    self._topic_paths: dict[str, Path] = {}
    self._topics: dict[str, dict[str, Any]] = {}
    self._claims: dict[str, dict[str, dict[str, Any]]] = {}
    self._sources: dict[str, dict[str, SourceReference]] = {}
    self._capture_declarations: dict[str, dict[str, set[str]]] = {}
    self._captures: dict[str, dict[str, Any]] = {}
    self._discover_topics()

  def _discover_topics(self) -> None:
    duplicates: dict[str, list[Path]] = {}
    for path in sorted((ROOT / "docs").rglob("topic.json")):
      repository_path = path.relative_to(ROOT).as_posix()
      path = _repository_file(repository_path, "topic registry")
      topic = _load_object(path, "topic registry")
      topic_id = topic.get("topic_id")
      if not isinstance(topic_id, str):
        raise CoverageError(f"topic registry: {path} has no string topic_id")
      if topic_id in self._topic_paths:
        duplicates.setdefault(topic_id, [self._topic_paths[topic_id]]).append(path)
      else:
        self._topic_paths[topic_id] = path.resolve()
        self._topics[topic_id] = topic
    if duplicates:
      details = "; ".join(
          f"{topic_id}: {', '.join(str(path) for path in paths)}"
          for topic_id, paths in sorted(duplicates.items())
      )
      raise CoverageError(f"duplicate topic ids make references ambiguous: {details}")

  def require_topic(self, topic_id: str, context: str) -> dict[str, Any]:
    topic = self._topics.get(topic_id)
    if topic is None:
      raise CoverageError(f"{context} references unknown topic {topic_id!r}")
    return topic

  def claims(self, topic_id: str, context: str) -> dict[str, dict[str, Any]]:
    self.require_topic(topic_id, context)
    if topic_id not in self._claims:
      evidence = self._topics[topic_id].get("evidence")
      if not isinstance(evidence, list):
        raise CoverageError(
            f"topic {topic_id!r}: evidence must be an array to resolve {context}"
        )
      claims: dict[str, dict[str, Any]] = {}
      declarations: dict[str, set[str]] = {}
      for index, claim in enumerate(evidence):
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
          raise CoverageError(
              f"topic {topic_id!r}: evidence[{index}] has no string id"
          )
        claim_id = claim["id"]
        if claim_id in claims:
          raise CoverageError(
              f"topic {topic_id!r} has duplicate claim id: {claim_id}"
          )
        claims[claim_id] = claim
        capture = claim.get("capture_manifest")
        if capture is not None:
          if not isinstance(capture, str):
            raise CoverageError(
                f"topic {topic_id!r}: evidence[{index}].capture_manifest "
                "must be a string"
            )
          path = _repository_file(capture, f"topic {topic_id}.capture_manifest")
          canonical = _repository_relative(path)
          declarations.setdefault(canonical, set()).add(claim_id)
      self._claims[topic_id] = claims
      self._capture_declarations[topic_id] = declarations
    return self._claims[topic_id]

  def claim(
      self, topic_id: str, claim_id: str, context: str
  ) -> dict[str, Any]:
    claim = self.claims(topic_id, context).get(claim_id)
    if claim is None:
      raise CoverageError(
          f"{context} references unknown claim {topic_id}#{claim_id}"
      )
    return claim

  def sources(
      self, topic_id: str, context: str
  ) -> dict[str, SourceReference]:
    topic = self.require_topic(topic_id, context)
    if topic_id not in self._sources:
      raw = topic.get("source_index")
      if not isinstance(raw, str):
        raise CoverageError(f"topic {topic_id!r} has no string source_index")
      path = _repository_file(raw, f"topic {topic_id}.source_index")
      source_index = _load_object(path, f"topic {topic_id}.source_index")
      if source_index.get("topic_id") != topic_id:
        raise CoverageError(
            f"{raw}: topic_id {source_index.get('topic_id')!r} does not match "
            f"{topic_id!r}"
        )
      entries = source_index.get("entries")
      if not isinstance(entries, list):
        raise CoverageError(f"{raw}: entries must be an array")
      components = source_index.get("components")
      if not isinstance(components, list):
        raise CoverageError(f"{raw}: components must be an array")
      component_by_id: dict[str, dict[str, Any]] = {}
      for index, component in enumerate(components):
        if not isinstance(component, dict) or not isinstance(
            component.get("id"), str
        ):
          raise CoverageError(f"{raw}: components[{index}] has no string id")
        component_id = component["id"]
        if component_id in component_by_id:
          raise CoverageError(f"{raw} has duplicate component id: {component_id}")
        component_by_id[component_id] = component
      sources: dict[str, SourceReference] = {}
      for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
          raise CoverageError(f"{raw}: entries[{index}] has no string id")
        source_id = entry["id"]
        if source_id in sources:
          raise CoverageError(f"{raw} has duplicate source id: {source_id}")
        component_id = entry.get("component")
        component = component_by_id.get(component_id)
        if component is None:
          raise CoverageError(
              f"{raw}: entry {source_id!r} has unknown component {component_id!r}"
          )
        sources[source_id] = SourceReference(entry=entry, component=component)
      self._sources[topic_id] = sources
    return self._sources[topic_id]

  def source(
      self, topic_id: str, source_id: str, context: str
  ) -> SourceReference:
    source = self.sources(topic_id, context).get(source_id)
    if source is None:
      raise CoverageError(
          f"{context} references unknown source {topic_id}#{source_id}"
      )
    return source

  def validate_capture(
      self, raw_path: str, declared_topics: set[str], context: str
  ) -> tuple[str, dict[str, Any]]:
    path = _repository_file(raw_path, context)
    canonical = _repository_relative(path)
    if canonical not in self._captures:
      self._captures[canonical] = _load_object(path, context)
    capture = self._captures[canonical]
    topic_id = capture.get("topic_id")
    capture_id = capture.get("capture_id")
    if not isinstance(topic_id, str) or not isinstance(capture_id, str):
      raise CoverageError(
          f"{context}: capture must have string topic_id and capture_id"
      )
    self.require_topic(topic_id, context)
    if topic_id not in declared_topics:
      raise CoverageError(
          f"{context}: capture topic {topic_id!r} is absent from topic_refs"
      )
    self.claims(topic_id, context)
    if canonical not in self._capture_declarations[topic_id]:
      raise CoverageError(
          f"{context}: {canonical!r} is not declared by a claim in topic "
          f"{topic_id!r}"
      )
    return canonical, capture


def _split_qualified(raw: str, context: str) -> tuple[str, str]:
  parts = raw.split("#")
  if len(parts) != 2 or not all(parts):
    raise CoverageError(f"{context} is not a topic-qualified reference: {raw!r}")
  return parts[0], parts[1]


def _validate_status(entry: dict[str, Any], context: str) -> None:
  status = entry["status"]
  present = set(entry)
  evidence = entry["evidence"]
  evidence_count = sum(len(evidence[field]) for field in evidence)
  if status == "covered":
    if evidence_count == 0:
      raise CoverageError(f"{context}: covered entries must cite evidence")
    forbidden = present & {"blockers", "rationale"}
  elif status == "blocked":
    forbidden = present & {"achieved_depth", "rationale"}
  elif status == "unobserved":
    forbidden = present & {"achieved_depth", "blockers", "rationale"}
  else:
    forbidden = present & {"achieved_depth", "blockers", "next_action"}
  if forbidden:
    raise CoverageError(
        f"{context}: status {status!r} forbids fields: {', '.join(sorted(forbidden))}"
    )


def _resolve_entry_evidence(
    entry: dict[str, Any], context: str, references: ReferenceIndex
) -> ResolvedEvidence:
  evidence = entry["evidence"]
  documentation = [
      _repository_file(raw, f"{context}.evidence.documentation")
      for raw in evidence["documentation"]
  ]
  probes = [
      _repository_file(raw, f"{context}.evidence.probe_refs")
      for raw in evidence["probe_refs"]
  ]

  declared_topics = set(evidence["topic_refs"])
  claims: dict[str, tuple[str, dict[str, Any]]] = {}
  used_topics: set[str] = set()
  for raw in evidence["claim_refs"]:
    topic_id, claim_id = _split_qualified(
        raw, f"{context}.evidence.claim_refs"
    )
    if topic_id not in declared_topics:
      raise CoverageError(
          f"{context}.evidence.claim_refs: {raw!r} uses a topic absent "
          "from topic_refs"
      )
    claims[raw] = (topic_id, references.claim(topic_id, claim_id, context))
    used_topics.add(topic_id)
  unused_topics = sorted(declared_topics - used_topics)
  if unused_topics:
    raise CoverageError(
        f"{context}.evidence.topic_refs are not consumed by claim_refs: "
        + ", ".join(unused_topics)
    )

  allowed_sources: set[str] = set()
  for topic_id, claim in claims.values():
    source_ids = claim.get("source_refs", [])
    if not isinstance(source_ids, list):
      raise CoverageError(
          f"{context}: claim {topic_id}#{claim.get('id')} has invalid source_refs"
      )
    allowed_sources.update(f"{topic_id}#{source_id}" for source_id in source_ids)
  sources: dict[str, SourceReference] = {}
  for raw in evidence["source_refs"]:
    if raw not in allowed_sources:
      raise CoverageError(
          f"{context}.evidence.source_refs is not linked by a referenced "
          f"claim: {raw!r}"
      )
    topic_id, source_id = _split_qualified(
        raw, f"{context}.evidence.source_refs"
    )
    sources[raw] = references.source(topic_id, source_id, context)

  allowed_captures: set[str] = set()
  capture_claims: dict[str, set[str]] = {}
  for qualified_claim, (topic_id, claim) in claims.items():
    raw_capture = claim.get("capture_manifest")
    if raw_capture is None:
      continue
    if not isinstance(raw_capture, str):
      raise CoverageError(
          f"{context}: claim {qualified_claim} has invalid capture_manifest"
      )
    capture_path = _repository_file(
        raw_capture, f"{context}: claim {qualified_claim}.capture_manifest"
    )
    canonical = _repository_relative(capture_path)
    allowed_captures.add(canonical)
    capture_claims.setdefault(canonical, set()).add(qualified_claim)

  captures: dict[str, dict[str, Any]] = {}
  for raw in evidence["capture_refs"]:
    canonical_path = _repository_relative(
        _repository_file(raw, f"{context}.evidence.capture_refs")
    )
    if canonical_path not in allowed_captures:
      raise CoverageError(
          f"{context}.evidence.capture_refs is not the capture_manifest of "
          f"a referenced claim: {raw!r}"
      )
    canonical, capture = references.validate_capture(
        raw, declared_topics, f"{context}.evidence.capture_refs"
    )
    captures[canonical] = capture

  if entry["status"] == "covered":
    for qualified_claim, (topic_id, _) in claims.items():
      topic = references.require_topic(topic_id, context)
      if topic.get("status") != "verified":
        raise CoverageError(
            f"{context}: covered evidence consumes non-verified topic "
            f"for {qualified_claim}: {topic.get('status')!r}"
        )
    for canonical, capture in captures.items():
      if capture.get("outcome", {}).get("status") != "pass":
        raise CoverageError(
            f"{context}: covered evidence consumes a non-passing capture "
            f"{canonical!r}"
        )

  if probes:
    capture_inputs = {
        item.get("path")
        for capture in captures.values()
        for item in capture.get("inputs", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for probe in probes:
      relative = _repository_relative(probe)
      if relative not in capture_inputs:
        raise CoverageError(
            f"{context}.evidence.probe_refs is not an input of a referenced "
            f"capture: {relative!r}"
        )

  resolved = ResolvedEvidence(
      claims=claims,
      sources=sources,
      captures=captures,
      documentation=documentation,
      probes=probes,
  )
  _validate_depth_gate(entry, context, resolved, capture_claims)
  return resolved


def _validate_depth_gate(
    entry: dict[str, Any],
    context: str,
    resolved: ResolvedEvidence,
    capture_claims: dict[str, set[str]],
) -> None:
  if entry["status"] != "covered":
    return
  depth = DEPTH[entry["achieved_depth"]]
  if not resolved.documentation:
    raise CoverageError(f"{context}: L0+ coverage requires documentation")
  if depth >= 4:
    raise CoverageError(
        f"{context}: achieved_depth {entry['achieved_depth']} is fail-closed; "
        "the coverage schema does not yet encode the L4 build/change/test/"
        "install/rollback contract or the L5 target-bound RUN-TPU contract"
    )
  if depth >= 2:
    if not resolved.claims or not resolved.sources:
      raise CoverageError(
          f"{context}: L2+ coverage requires a verified claim and fixed source"
      )
    for raw, source in resolved.sources.items():
      revision = source.component.get("revision")
      if not isinstance(revision, str) or not GIT_COMMIT_RE.fullmatch(revision):
        raise CoverageError(
            f"{context}: L2+ source has no fixed 40-hex revision: {raw}"
        )
      callers = source.entry.get("callers", [])
      callees = source.entry.get("callees", [])
      if not callers and not callees:
        raise CoverageError(
            f"{context}: L2+ source has no actual caller/callee relation: {raw}"
        )
  if depth >= 3:
    if not resolved.captures:
      raise CoverageError(
          f"{context}: L3+ coverage requires a passing execution capture"
      )
    if not resolved.probes:
      raise CoverageError(f"{context}: L3+ coverage requires a probe_ref")
    execution_captures = [
        (path, capture)
        for path, capture in resolved.captures.items()
        if capture.get("evidence_level") in EXECUTION_LEVELS
        and capture.get("outcome", {}).get("status") == "pass"
        and isinstance(capture.get("outcome", {}).get("assertions"), list)
        and bool(capture["outcome"]["assertions"])
    ]
    if not execution_captures:
      raise CoverageError(
          f"{context}: L3+ coverage requires a passing execution capture "
          "with non-empty assertions"
      )
    uncovered_probes = {
        _repository_relative(probe) for probe in resolved.probes
    }
    for capture_path, capture in execution_captures:
      allowed_artifact_ids: set[str] = set()
      for qualified_claim in capture_claims.get(capture_path, set()):
        claim = resolved.claims[qualified_claim][1]
        artifact_refs = claim.get("artifact_refs", [])
        if isinstance(artifact_refs, list):
          allowed_artifact_ids.update(
              item for item in artifact_refs if isinstance(item, str)
          )
      artifacts_by_id = {
          artifact["id"]: artifact
          for artifact in capture.get("artifacts", [])
          if isinstance(artifact, dict) and isinstance(artifact.get("id"), str)
      }
      capture_inputs = {
          item.get("path")
          for item in capture.get("inputs", [])
          if isinstance(item, dict) and isinstance(item.get("path"), str)
      }
      for assertion in capture.get("outcome", {}).get("assertions", []):
        if not isinstance(assertion, dict) or assertion.get("status") != "pass":
          continue
        probe_ref = assertion.get("probe_ref")
        if probe_ref not in uncovered_probes or probe_ref not in capture_inputs:
          continue
        assertion_artifacts = assertion.get("artifact_refs", [])
        if not isinstance(assertion_artifacts, list):
          continue
        candidate_ids = [
            artifact_id
            for artifact_id in assertion_artifacts
            if artifact_id in allowed_artifact_ids
            and artifact_id in artifacts_by_id
            and artifacts_by_id[artifact_id].get("kind") in IR_ARTIFACT_KINDS
        ]
        for artifact_id in candidate_ids:
          _validate_local_ir_artifact(
              artifacts_by_id[artifact_id],
              f"{context} capture {capture_path!r} assertion "
              f"{assertion.get('id', '<missing>')!r}",
          )
          uncovered_probes.remove(probe_ref)
          break
    if uncovered_probes:
      kinds = ", ".join(sorted(IR_ARTIFACT_KINDS))
      raise CoverageError(
          f"{context}: L3+ coverage requires every probe_ref to be joined by "
          "one passing structured assertion to a claim-linked, local real IR "
          f"artifact of kind ({kinds}); uncovered: "
          + ", ".join(sorted(uncovered_probes))
      )


def _validate_dag(entries: list[dict[str, Any]]) -> None:
  known = {entry["id"] for entry in entries}
  for entry in entries:
    missing = sorted(set(entry["depends_on"]) - known)
    if missing:
      raise CoverageError(
          f"entry {entry['id']!r} has unknown dependencies: {', '.join(missing)}"
      )

  state: dict[str, int] = {}
  stack: list[str] = []
  by_id = {entry["id"]: entry for entry in entries}

  def visit(entry_id: str) -> None:
    if state.get(entry_id) == 2:
      return
    if state.get(entry_id) == 1:
      start = stack.index(entry_id)
      cycle = stack[start:] + [entry_id]
      raise CoverageError("dependency cycle: " + " -> ".join(cycle))
    state[entry_id] = 1
    stack.append(entry_id)
    for dependency in by_id[entry_id]["depends_on"]:
      visit(dependency)
    stack.pop()
    state[entry_id] = 2

  for entry_id in sorted(known):
    visit(entry_id)


def _validate_completion(
    entries: list[dict[str, Any]],
    resolved: dict[str, ResolvedEvidence],
    references: ReferenceIndex,
) -> None:
  by_id = {entry["id"]: entry for entry in entries}
  census = by_id.get("workload-feature-census")
  if (
      census is None
      or census["status"] != "covered"
      or DEPTH[census.get("achieved_depth", "L0")]
      < DEPTH[census["target_depth"]]
  ):
    raise CoverageError(
        "completion requires workload-feature-census to be covered at its "
        "target_depth"
    )

  complete_ids: set[str] = set()
  for entry in entries:
    entry_id = entry["id"]
    status = entry["status"]
    if status in {"blocked", "unobserved"}:
      raise CoverageError(
          f"completion forbids {status} entry: {entry_id}"
      )
    if status == "covered":
      if DEPTH[entry["achieved_depth"]] < DEPTH[entry["target_depth"]]:
        raise CoverageError(
            f"completion requires {entry_id} achieved_depth "
            f"{entry['achieved_depth']} >= target_depth {entry['target_depth']}"
        )
      complete_ids.add(entry_id)
      continue

    claims = resolved[entry_id].claims
    if not claims:
      raise CoverageError(
          f"completion requires not-applicable entry {entry_id} to cite a "
          "verified claim"
      )
    for qualified_claim, (topic_id, _) in claims.items():
      topic_status = references.require_topic(topic_id, entry_id).get("status")
      if topic_status != "verified":
        raise CoverageError(
            f"completion requires not-applicable claim {qualified_claim} "
            f"to come from a verified topic, got {topic_status!r}"
        )
    complete_ids.add(entry_id)

  for entry in entries:
    incomplete = sorted(set(entry["depends_on"]) - complete_ids)
    if incomplete:
      raise CoverageError(
          f"completion requires dependencies of {entry['id']} to be complete: "
          + ", ".join(incomplete)
      )


def validate(
    coverage_path: Path, *, require_complete: bool = False
) -> dict[str, Any]:
  _validate_evidence_contract()
  schema = _load_object(SCHEMA, "coverage schema")
  Draft202012Validator.check_schema(schema)
  coverage = _load_object(coverage_path, "coverage")
  errors = _schema_errors(schema, coverage)
  if errors:
    raise CoverageError("schema validation failed:\n" + "\n".join(errors))
  _validate_schema_reference(coverage_path, coverage, schema)

  entries = coverage["layers"] + coverage["features"]
  ids = [entry["id"] for entry in entries]
  duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
  if duplicates:
    raise CoverageError(
        "layer/feature ids must be globally unique; duplicates: "
        + ", ".join(duplicates)
    )
  _validate_dag(entries)

  references = ReferenceIndex()
  resolved: dict[str, ResolvedEvidence] = {}
  for collection in ("layers", "features"):
    for index, entry in enumerate(coverage[collection]):
      context = f"{collection}[{index}] ({entry['id']})"
      _validate_status(entry, context)
      resolved[entry["id"]] = _resolve_entry_evidence(
          entry, context, references
      )
  if require_complete:
    _validate_completion(entries, resolved, references)
  return coverage


def _display_path(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def print_summary(coverage: dict[str, Any]) -> None:
  print(f"updated: {coverage['updated_at']}")
  for collection in ("layers", "features"):
    entries = coverage[collection]
    states = Counter(entry["status"] for entry in entries)
    counts = ", ".join(
        f"{state}={states.get(state, 0)}"
        for state in ("covered", "unobserved", "blocked", "not-applicable")
    )
    print(f"{collection}: {len(entries)} ({counts})")
    for entry in entries:
      achieved = entry.get("achieved_depth", "-")
      print(
          f"  - {entry['id']} [{entry['status']}] "
          f"{achieved}/{entry['target_depth']}"
      )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--coverage-file",
      type=Path,
      default=DEFAULT_COVERAGE,
      help="coverage JSON to validate",
  )
  parser.add_argument(
      "--check", action="store_true", help="validate without printing the summary"
  )
  parser.add_argument(
      "--require-complete",
      action="store_true",
      help="also enforce the final workload-census and target-depth contract",
  )
  args = parser.parse_args()
  coverage_path = args.coverage_file.resolve()
  try:
    coverage = validate(
        coverage_path, require_complete=args.require_complete
    )
  except (CoverageError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  if args.check:
    print(f"coverage valid: {_display_path(coverage_path)}")
  else:
    print_summary(coverage)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
