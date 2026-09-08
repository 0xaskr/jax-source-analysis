#!/usr/bin/env python3
"""Run isolated positive and negative tests for coverage validation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "tools/validate-coverage.py"
COVERAGE_RELATIVE = Path("manifests/coverage.json")
TOPIC_RELATIVE = Path("docs/topics/03-selftest/topic.json")
SOURCE_INDEX_RELATIVE = TOPIC_RELATIVE.parent / "source-index.json"
CAPTURE_A_RELATIVE = TOPIC_RELATIVE.parent / "captures/run-a/manifest.json"
CAPTURE_B_RELATIVE = TOPIC_RELATIVE.parent / "captures/run-b/manifest.json"
CAPTURE_A_IR_RELATIVE = CAPTURE_A_RELATIVE.parent / "program.jaxpr"
CAPTURE_B_IR_RELATIVE = CAPTURE_B_RELATIVE.parent / "program.jaxpr"
PROBE_RELATIVE = Path("docs/topics/03-selftest/probes/probe.py")


def _load_validator() -> Any:
  spec = importlib.util.spec_from_file_location(
      "jax_source_analysis_validate_coverage", VALIDATOR_PATH
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {VALIDATOR_PATH}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


VALIDATOR = _load_validator()


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  assert isinstance(value, dict)
  return value


def _evidence(
    *,
    documentation: list[str],
    topic_refs: list[str] | None = None,
    claim_refs: list[str] | None = None,
    source_refs: list[str] | None = None,
    capture_refs: list[str] | None = None,
    probe_refs: list[str] | None = None,
) -> dict[str, Any]:
  return {
      "documentation": documentation,
      "topic_refs": topic_refs or [],
      "claim_refs": claim_refs or [],
      "source_refs": source_refs or [],
      "capture_refs": capture_refs or [],
      "probe_refs": probe_refs or [],
  }


def _create_capture(
    capture_id: str, ir_relative: Path, ir_payload: bytes
) -> dict[str, Any]:
  return {
      "capture_id": capture_id,
      "topic_id": "03-selftest",
      "evidence_level": "RUN-CPU",
      "inputs": [{"path": PROBE_RELATIVE.as_posix()}],
      "artifacts": [
          {
              "id": "ir",
              "path": ir_relative.as_posix(),
              "sha256": hashlib.sha256(ir_payload).hexdigest(),
              "size_bytes": len(ir_payload),
              "kind": "jaxpr",
              "stage": "trace",
              "format": "text/vnd.jax.jaxpr",
              "generated": True,
          }
      ],
      "outcome": {
          "status": "pass",
          "assertions": [
              {
                  "id": "ir-shape",
                  "status": "pass",
                  "summary": "The generated Jaxpr contains the expected multiply.",
                  "probe_ref": PROBE_RELATIVE.as_posix(),
                  "artifact_refs": ["ir"],
              }
          ],
      },
  }


def _create_fixture(root: Path) -> None:
  schema = root / "manifests/schema/coverage.schema.json"
  schema.parent.mkdir(parents=True)
  shutil.copy2(
      REPOSITORY_ROOT / "manifests/schema/coverage.schema.json", schema
  )
  fake_evidence = root / "tools/validate-evidence.py"
  fake_evidence.parent.mkdir(parents=True)
  fake_evidence.write_text("raise SystemExit(0)\n")

  guide = root / "docs/guide.md"
  guide.parent.mkdir(parents=True)
  guide.write_text("# Coverage fixture\n")
  readme = root / TOPIC_RELATIVE.parent / "README.md"
  readme.parent.mkdir(parents=True, exist_ok=True)
  readme.write_text("# Topic fixture\n")
  probe = root / PROBE_RELATIVE
  probe.parent.mkdir(parents=True)
  probe.write_text("print('probe')\n")

  source_index = {
      "topic_id": "03-selftest",
      "components": [
          {
              "id": "jax",
              "root": "upstream/jax",
              "revision": "a" * 40,
          }
      ],
      "entries": [
          {
              "id": "source.a",
              "component": "jax",
              "callers": [],
              "callees": ["source.helper"],
          },
          {
              "id": "source.helper",
              "component": "jax",
              "callers": ["source.a"],
              "callees": [],
          },
          {
              "id": "source.b",
              "component": "jax",
              "callers": ["source.helper"],
              "callees": [],
          },
      ],
  }
  _write_json(root / SOURCE_INDEX_RELATIVE, source_index)
  topic = {
      "topic_id": "03-selftest",
      "status": "verified",
      "source_index": SOURCE_INDEX_RELATIVE.as_posix(),
      "evidence": [
          {
              "id": "claim-a",
              "source_refs": ["source.a"],
              "artifact_refs": ["ir"],
              "capture_manifest": CAPTURE_A_RELATIVE.as_posix(),
          },
          {
              "id": "claim-b",
              "source_refs": ["source.b"],
              "artifact_refs": ["ir"],
              "capture_manifest": CAPTURE_B_RELATIVE.as_posix(),
          },
      ],
  }
  _write_json(root / TOPIC_RELATIVE, topic)
  ir_payload = b"{ lambda ; a:f32[]. let b:f32[] = mul a 2.0 in (b,) }\n"
  for ir_relative in (CAPTURE_A_IR_RELATIVE, CAPTURE_B_IR_RELATIVE):
    ir_path = root / ir_relative
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    ir_path.write_bytes(ir_payload)
  _write_json(
      root / CAPTURE_A_RELATIVE,
      _create_capture("run-a", CAPTURE_A_IR_RELATIVE, ir_payload),
  )
  _write_json(
      root / CAPTURE_B_RELATIVE,
      _create_capture("run-b", CAPTURE_B_IR_RELATIVE, ir_payload),
  )

  coverage = {
      "$schema": "schema/coverage.schema.json",
      "schema_version": "1.0",
      "updated_at": "2026-09-08T00:00:00Z",
      "scope": {
          "boundary": "selftest",
          "coverage_unit": "selftest entry",
          "covered_semantics": "covered at achieved depth",
          "completion_rule": "all targets reached after census",
      },
      "layers": [
          {
              "id": "layer-base",
              "title": "L0 is representable",
              "status": "covered",
              "target_depth": "L0",
              "achieved_depth": "L0",
              "evidence": _evidence(documentation=["docs/guide.md"]),
              "depends_on": [],
              "notes": [],
          }
      ],
      "features": [
          {
              "id": "workload-feature-census",
              "title": "Reviewed workload census",
              "status": "covered",
              "target_depth": "L1",
              "achieved_depth": "L1",
              "evidence": _evidence(documentation=["docs/guide.md"]),
              "depends_on": ["layer-base"],
              "notes": [],
          },
          {
              "id": "execution-feature",
              "title": "L3 execution evidence",
              "status": "covered",
              "target_depth": "L3",
              "achieved_depth": "L3",
              "evidence": _evidence(
                  documentation=["docs/guide.md"],
                  topic_refs=["03-selftest"],
                  claim_refs=["03-selftest#claim-a"],
                  source_refs=["03-selftest#source.a"],
                  capture_refs=[CAPTURE_A_RELATIVE.as_posix()],
                  probe_refs=[PROBE_RELATIVE.as_posix()],
              ),
              "depends_on": ["workload-feature-census"],
              "notes": [],
          },
      ],
  }
  _write_json(root / COVERAGE_RELATIVE, coverage)


def _configure(root: Path) -> None:
  VALIDATOR.ROOT = root.resolve()
  VALIDATOR.SCHEMA = VALIDATOR.ROOT / "manifests/schema/coverage.schema.json"
  VALIDATOR.EVIDENCE_VALIDATOR = VALIDATOR.ROOT / "tools/validate-evidence.py"


def _entry(coverage: dict[str, Any], entry_id: str) -> dict[str, Any]:
  return next(
      item
      for item in coverage["layers"] + coverage["features"]
      if item["id"] == entry_id
  )


def _mutate_json(
    root: Path, relative: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
  path = root / relative
  value = _read_json(path)
  mutate(value)
  _write_json(path, value)


def _replace_ir_bytes(root: Path, payload: bytes) -> None:
  (root / CAPTURE_A_IR_RELATIVE).write_bytes(payload)

  def mutate(value: dict[str, Any]) -> None:
    artifact = value["artifacts"][0]
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    artifact["size_bytes"] = len(payload)

  _mutate_json(root, CAPTURE_A_RELATIVE, mutate)


def _case_fake_l5(root: Path, coverage: dict[str, Any]) -> None:
  entry = _entry(coverage, "execution-feature")
  entry["target_depth"] = "L5"
  entry["achieved_depth"] = "L5"
  entry["evidence"] = _evidence(documentation=["docs/guide.md"])


def _case_draft_topic(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      TOPIC_RELATIVE,
      lambda value: value.__setitem__("status", "draft"),
  )


def _case_failed_capture(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["outcome"].__setitem__("status", "fail"),
  )


def _case_capture_splice(root: Path, coverage: dict[str, Any]) -> None:
  entry = _entry(coverage, "execution-feature")
  entry["evidence"]["capture_refs"] = [CAPTURE_B_RELATIVE.as_posix()]


def _case_source_splice(root: Path, coverage: dict[str, Any]) -> None:
  entry = _entry(coverage, "execution-feature")
  entry["evidence"]["source_refs"] = ["03-selftest#source.b"]


def _case_above_target(root: Path, coverage: dict[str, Any]) -> None:
  _entry(coverage, "execution-feature")["target_depth"] = "L2"


def _case_all_not_applicable(root: Path, coverage: dict[str, Any]) -> None:
  for entry in coverage["layers"] + coverage["features"]:
    entry["status"] = "not-applicable"
    entry["rationale"] = "review pending"
    entry.pop("achieved_depth", None)
    entry.pop("blockers", None)
    entry.pop("next_action", None)
    entry["evidence"] = _evidence(documentation=[])


def _case_path_traversal(root: Path, coverage: dict[str, Any]) -> None:
  _entry(coverage, "layer-base")["evidence"]["documentation"] = [
      "../outside.md"
  ]


def _case_symlink_escape(root: Path, coverage: dict[str, Any]) -> None:
  outside = root.parent / "outside.md"
  outside.write_text("outside\n")
  link = root / "docs/escaped.md"
  link.symlink_to(outside)
  _entry(coverage, "layer-base")["evidence"]["documentation"] = [
      "docs/escaped.md"
  ]


def _case_dag_cycle(root: Path, coverage: dict[str, Any]) -> None:
  _entry(coverage, "layer-base")["depends_on"] = ["execution-feature"]


def _case_missing_call_relation(root: Path, coverage: dict[str, Any]) -> None:
  def mutate(value: dict[str, Any]) -> None:
    source = next(item for item in value["entries"] if item["id"] == "source.a")
    source["callers"] = []
    source["callees"] = []

  _mutate_json(root, SOURCE_INDEX_RELATIVE, mutate)


def _case_missing_ir(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["artifacts"][0].__setitem__("kind", "stdout"),
  )


def _case_missing_probe(root: Path, coverage: dict[str, Any]) -> None:
  _entry(coverage, "execution-feature")["evidence"]["probe_refs"] = []


def _case_probe_not_capture_input(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["inputs"][0].__setitem__(
          "path", "docs/topics/03-selftest/probes/different.py"
      ),
  )


def _case_unstructured_assertion(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["outcome"].__setitem__(
          "assertions", ["free-form text cannot bind probe and IR"]
      ),
  )


def _case_assertion_probe_splice(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["outcome"]["assertions"][0].__setitem__(
          "probe_ref", "docs/topics/03-selftest/probes/different.py"
      ),
  )


def _case_assertion_ir_splice(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["outcome"]["assertions"][0].__setitem__(
          "artifact_refs", ["other-ir"]
      ),
  )


def _case_failed_assertion(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["outcome"]["assertions"][0].__setitem__(
          "status", "fail"
      ),
  )


def _case_empty_ir(root: Path, coverage: dict[str, Any]) -> None:
  _replace_ir_bytes(root, b"")


def _case_ir_hash_mismatch(root: Path, coverage: dict[str, Any]) -> None:
  (root / CAPTURE_A_IR_RELATIVE).write_text(
      "{ lambda ; a:f32[]. let b:f32[] = add a 1.0 in (b,) }\n"
  )


def _case_ir_format_mismatch(root: Path, coverage: dict[str, Any]) -> None:
  _mutate_json(
      root,
      CAPTURE_A_RELATIVE,
      lambda value: value["artifacts"][0].__setitem__(
          "format", "text/x-mlir; dialect=stablehlo"
      ),
  )


def _case_ir_format_spoof(root: Path, coverage: dict[str, Any]) -> None:
  _replace_ir_bytes(root, b"this is ordinary text, not Jaxpr\n")


def _case_evidence_failure(root: Path, coverage: dict[str, Any]) -> None:
  (root / "tools/validate-evidence.py").write_text(
      "print('synthetic evidence failure')\nraise SystemExit(1)\n"
  )


Case = tuple[
    str,
    Callable[[Path, dict[str, Any]], None],
    bool,
    bool,
    str | None,
]


CASES: tuple[Case, ...] = (
    ("positive default", lambda root, coverage: None, False, True, None),
    ("positive complete", lambda root, coverage: None, True, True, None),
    ("README cannot prove L5", _case_fake_l5, False, False, "fail-closed"),
    ("draft topic", _case_draft_topic, False, False, "non-verified"),
    ("failed capture", _case_failed_capture, False, False, "non-passing"),
    ("claim/capture splice", _case_capture_splice, False, False, "capture_manifest"),
    ("claim/source splice", _case_source_splice, False, False, "not linked"),
    ("achieved above target", _case_above_target, False, True, None),
    (
        "all not-applicable completion",
        _case_all_not_applicable,
        True,
        False,
        "workload-feature-census",
    ),
    ("path traversal", _case_path_traversal, False, False, "schema validation"),
    ("symlink escape", _case_symlink_escape, False, False, "outside the repository"),
    ("DAG cycle", _case_dag_cycle, False, False, "dependency cycle"),
    (
        "missing caller/callee",
        _case_missing_call_relation,
        False,
        False,
        "caller/callee",
    ),
    ("missing IR", _case_missing_ir, False, False, "real IR artifact"),
    ("missing probe", _case_missing_probe, False, False, "probe_ref"),
    (
        "probe absent from capture inputs",
        _case_probe_not_capture_input,
        False,
        False,
        "not an input",
    ),
    (
        "unstructured assertion",
        _case_unstructured_assertion,
        False,
        False,
        "passing structured assertion",
    ),
    (
        "assertion probe splice",
        _case_assertion_probe_splice,
        False,
        False,
        "passing structured assertion",
    ),
    (
        "assertion IR splice",
        _case_assertion_ir_splice,
        False,
        False,
        "passing structured assertion",
    ),
    (
        "failed assertion",
        _case_failed_assertion,
        False,
        False,
        "passing structured assertion",
    ),
    ("empty IR", _case_empty_ir, False, False, "is empty"),
    (
        "IR hash mismatch",
        _case_ir_hash_mismatch,
        False,
        False,
        "SHA-256",
    ),
    (
        "IR kind/format mismatch",
        _case_ir_format_mismatch,
        False,
        False,
        "requires format",
    ),
    (
        "IR format spoof",
        _case_ir_format_spoof,
        False,
        False,
        "format spoof",
    ),
    (
        "evidence validation failure",
        _case_evidence_failure,
        False,
        False,
        "evidence validation failed",
    ),
)


def main() -> int:
  with tempfile.TemporaryDirectory(prefix="coverage-selftest-") as temporary:
    template = Path(temporary) / "template"
    template.mkdir()
    _create_fixture(template)
    for index, (name, mutate, require_complete, should_pass, expected) in enumerate(
        CASES
    ):
      root = Path(temporary) / f"case-{index:02d}"
      shutil.copytree(template, root, symlinks=True)
      _configure(root)
      coverage_path = root / COVERAGE_RELATIVE
      coverage = _read_json(coverage_path)
      mutate(root, coverage)
      _write_json(coverage_path, coverage)
      try:
        VALIDATOR.validate(
            coverage_path, require_complete=require_complete
        )
      except VALIDATOR.CoverageError as error:
        if should_pass:
          print(f"FAILED: {name} was rejected: {error}", file=sys.stderr)
          return 1
        if expected is not None and expected not in str(error):
          print(
              f"FAILED: {name} did not report {expected!r}: {error}",
              file=sys.stderr,
          )
          return 1
        print(f"OK: rejected {name}")
      else:
        if not should_pass:
          print(f"FAILED: {name} was accepted", file=sys.stderr)
          return 1
        print(f"OK: accepted {name}")

  print(f"OK: {len(CASES)} isolated coverage validation cases")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
