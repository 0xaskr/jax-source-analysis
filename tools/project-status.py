#!/usr/bin/env python3
"""Validate and summarize the long-running JAX source-analysis project status."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "manifests/status.json"
SCHEMA = ROOT / "manifests/schema/project-status.schema.json"
COVERAGE_VALIDATOR = ROOT / "tools/validate-coverage.py"
AUTHORITATIVE_REFERENCES = {
    "plan_path": "PLAN.md",
    "baseline_path": "manifests/baseline.json",
    "coverage_path": "manifests/coverage.json",
}


class StatusError(RuntimeError):
  """Raised when the project status is malformed or references missing files."""


def _load_object(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise StatusError(f"cannot read {path}: {error}") from error
  if not isinstance(value, dict):
    raise StatusError(f"{path} must contain a JSON object")
  return value


def _repository_path(raw: str, context: str) -> Path:
  pure = PurePosixPath(raw)
  if (
      not raw
      or "\\" in raw
      or pure.is_absolute()
      or re.match(r"^[A-Za-z]:", raw)
      or ".." in pure.parts
  ):
    raise StatusError(f"{context} is not a safe repository-relative path: {raw!r}")
  resolved = ROOT.joinpath(*pure.parts).resolve()
  try:
    resolved.relative_to(ROOT)
  except ValueError as error:
    raise StatusError(
        f"{context} resolves outside the repository: {raw!r}"
    ) from error
  return resolved


def _repository_file(raw: str, context: str) -> Path:
  path = _repository_path(raw, context)
  if not path.is_file():
    raise StatusError(f"{context} references a missing regular file: {raw!r}")
  return path


def _status_file(path: Path) -> Path:
  resolved = path.resolve()
  try:
    resolved.relative_to(ROOT.resolve())
  except ValueError as error:
    raise StatusError(f"status file resolves outside the repository: {path}") from error
  if not resolved.is_file():
    raise StatusError(f"status file is not a regular file: {_display_path(resolved)}")
  return resolved


def _display_path(path: Path) -> str:
  try:
    return path.relative_to(ROOT).as_posix()
  except ValueError:
    return str(path)


def _timestamp(raw: str, context: str) -> datetime:
  try:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
  except ValueError as error:
    raise StatusError(f"{context} is not a valid date-time: {raw!r}") from error


def _validate_action_graph(
    actions: list[dict[str, Any]], completed_ids: set[str]
) -> None:
  action_by_id = {item["id"]: item for item in actions}
  known_dependencies = set(action_by_id) | completed_ids
  for action in actions:
    missing = sorted(set(action["depends_on"]) - known_dependencies)
    if missing:
      raise StatusError(
          f"next_actions.{action['id']} has unknown dependencies: "
          + ", ".join(missing)
      )

  visiting: set[str] = set()
  visited: set[str] = set()

  def visit(action_id: str, trail: tuple[str, ...]) -> None:
    if action_id in visiting:
      start = trail.index(action_id)
      cycle = trail[start:] + (action_id,)
      raise StatusError("next_actions dependency cycle: " + " -> ".join(cycle))
    if action_id in visited:
      return
    visiting.add(action_id)
    action = action_by_id[action_id]
    for dependency in action["depends_on"]:
      if dependency in action_by_id:
        visit(dependency, trail + (action_id,))
    visiting.remove(action_id)
    visited.add(action_id)

  for action_id in action_by_id:
    visit(action_id, ())

  for action in actions:
    for dependency in action["depends_on"]:
      dependency_action = action_by_id.get(dependency)
      if dependency_action is not None and dependency_action["order"] >= action["order"]:
        raise StatusError(
            f"next_actions.{action['id']} depends on {dependency}, whose order "
            f"{dependency_action['order']} is not earlier than {action['order']}"
        )


def _validate_coverage_contract(coverage_path: Path) -> dict[str, Any]:
  validator = COVERAGE_VALIDATOR.resolve()
  try:
    validator.relative_to(ROOT.resolve())
  except ValueError as error:
    raise StatusError(
        f"coverage validator resolves outside the repository: {COVERAGE_VALIDATOR}"
    ) from error
  if not validator.is_file():
    raise StatusError(f"coverage validator is missing: {COVERAGE_VALIDATOR}")
  try:
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--coverage-file",
            str(coverage_path),
            "--check",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
  except OSError as error:
    raise StatusError(f"cannot run coverage validator: {error}") from error
  if result.returncode:
    detail = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    raise StatusError(
        "coverage validation failed"
        + (f":\n{detail}" if detail else "")
    )
  return _load_object(coverage_path)


def validate(status_path: Path) -> dict[str, Any]:
  status_path = _status_file(status_path)
  schema = _load_object(SCHEMA)
  Draft202012Validator.check_schema(schema)
  status = _load_object(status_path)
  errors = sorted(
      Draft202012Validator(
          schema, format_checker=FormatChecker()
      ).iter_errors(status),
      key=lambda error: tuple(str(part) for part in error.absolute_path),
  )
  if errors:
    messages = []
    for error in errors:
      location = ".".join(str(part) for part in error.absolute_path) or "<root>"
      messages.append(f"{location}: {error.message}")
    raise StatusError("schema validation failed:\n" + "\n".join(messages))

  schema_ref = status["$schema"]
  resolved_schema = (status_path.parent / schema_ref).resolve()
  if resolved_schema != SCHEMA.resolve():
    raise StatusError(
        f"$schema resolves to {resolved_schema}, expected {SCHEMA.resolve()}"
    )

  for field, expected in AUTHORITATIVE_REFERENCES.items():
    if status[field] != expected:
      raise StatusError(
          f"{field} must name the authoritative repository file {expected!r}"
      )

  references: list[tuple[str, str]] = [
      (field, status[field]) for field in AUTHORITATIVE_REFERENCES
  ]
  for collection_name in ("milestones", "active_work"):
    for item in status[collection_name]:
      for raw_path in item["artifacts"]:
        references.append((f"{collection_name}.{item['id']}.artifacts", raw_path))
  for context, raw_path in references:
    _repository_file(raw_path, context)

  ids = [item["id"] for item in status["milestones"] + status["active_work"]]
  ids.extend(item["id"] for item in status["next_actions"])
  ids.extend(item["id"] for item in status["external_inputs"])
  duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
  if duplicates:
    raise StatusError("duplicate ids: " + ", ".join(duplicates))

  completed_ids = {
      item["id"] for item in status["milestones"] if item["state"] == "complete"
  }
  orders = [item["order"] for item in status["next_actions"]]
  expected_orders = list(range(1, len(orders) + 1))
  if sorted(orders) != expected_orders:
    raise StatusError(
        "next_actions.order values must be consecutive starting at 1; got "
        + ", ".join(str(order) for order in sorted(orders))
    )
  _validate_action_graph(status["next_actions"], completed_ids)

  updated_at = _timestamp(status["updated_at"], "updated_at")
  for milestone in status["milestones"]:
    completed_at = _timestamp(
        milestone["completed_at"], f"milestones.{milestone['id']}.completed_at"
    )
    if completed_at > updated_at:
      raise StatusError(
          f"milestones.{milestone['id']}.completed_at is later than updated_at"
      )

  coverage_path = _repository_file(status["coverage_path"], "coverage_path")
  coverage = _validate_coverage_contract(coverage_path)
  coverage_updated_at = _timestamp(
      coverage["updated_at"], "coverage.updated_at"
  )
  if coverage_updated_at > updated_at:
    raise StatusError("coverage.updated_at is later than project status updated_at")
  return status


def print_summary(status: dict[str, Any]) -> None:
  phase = status["current_phase"]
  print(f"updated: {status['updated_at']}")
  print(f"phase: {phase['id']} [{phase['state']}] {phase['title']}")
  for note in phase.get("notes", []):
    print(f"  note: {note}")

  milestones = sorted(
      status["milestones"],
      key=lambda item: _timestamp(item["completed_at"], "completed_at"),
      reverse=True,
  )
  if milestones:
    latest = milestones[0]
    print(
        f"latest-milestone: {latest['id']} [{latest['completed_at']}] "
        f"{latest['title']}"
    )
    for note in latest["notes"]:
      print(f"  note: {note}")
  else:
    print("latest-milestone: none")

  coverage = _load_object(
      _repository_file(status["coverage_path"], "coverage_path")
  )
  print("coverage:")
  for collection in ("layers", "features"):
    entries = coverage[collection]
    counts = Counter(item["status"] for item in entries)
    rendered = ", ".join(
        f"{state}={counts.get(state, 0)}"
        for state in ("covered", "unobserved", "blocked", "not-applicable")
    )
    print(f"  {collection}: {len(entries)} ({rendered})")

  print("active:")
  for item in status["active_work"]:
    print(f"  - {item['id']} [{item['state']}]: {item['title']}")
    for note in item["notes"]:
      print(f"    note: {note}")
    if item.get("next_command"):
      print(f"    next-command: {item['next_command']}")

  completed_ids = {item["id"] for item in status["milestones"]}
  ready = [
      item
      for item in sorted(status["next_actions"], key=lambda value: value["order"])
      if set(item["depends_on"]).issubset(completed_ids)
  ]
  if ready:
    first = ready[0]
    print(f"first-ready-action: {first['id']} ({first['order']}) {first['title']}")
  else:
    print("first-ready-action: none")
  print("next-actions:")
  for item in sorted(status["next_actions"], key=lambda value: value["order"]):
    dependencies = ",".join(item["depends_on"]) or "none"
    print(f"  {item['order']}. {item['id']} (depends-on: {dependencies})")
    blockers = sorted(set(item["depends_on"]) - completed_ids)
    if blockers:
      print(f"    blocked-by: {','.join(blockers)}")
  unavailable = [
      item for item in status["external_inputs"] if item["state"] != "available"
  ]
  print("external-inputs:")
  for item in unavailable:
    print(f"  - {item['id']} [{item['state']}]")
    print(f"    note: {item['notes']}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--status-file", type=Path, default=DEFAULT_STATUS, help="status JSON to read"
  )
  parser.add_argument(
      "--check", action="store_true", help="validate without printing the summary"
  )
  parser.add_argument("--json", action="store_true", help="print canonical status JSON")
  args = parser.parse_args()
  if args.check and args.json:
    parser.error("--check and --json are mutually exclusive")
  try:
    status_path = _status_file(args.status_file)
    status = validate(status_path)
  except StatusError as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  if args.check:
    print(f"project status valid: {_display_path(status_path)}")
  elif args.json:
    print(json.dumps(status, indent=2, sort_keys=True))
  else:
    print_summary(status)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
