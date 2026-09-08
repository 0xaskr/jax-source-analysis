#!/usr/bin/env python3
"""Exercise project-status recovery validation with isolated status fixtures."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/project-status.py"


def _relative(path: Path) -> str:
  return path.absolute().relative_to(ROOT.absolute()).as_posix()


def _base_status() -> dict[str, Any]:
  return {
      "$schema": "../schema/project-status.schema.json",
      "schema_version": "1.0",
      "updated_at": "2099-09-08T12:00:00+08:00",
      "objective": "Exercise durable project recovery state.",
      "plan_path": "PLAN.md",
      "baseline_path": "manifests/baseline.json",
      "coverage_path": "manifests/coverage.json",
      "current_phase": {
          "id": "P0/P1",
          "title": "Foundation and source build",
          "state": "running",
          "notes": ["Phase note used by the summary contract."],
      },
      "milestones": [
          {
              "id": "newest-milestone",
              "title": "Newest completed work",
              "state": "complete",
              "completed_at": "2026-09-08T11:00:00+08:00",
              "artifacts": ["PLAN.md"],
              "notes": ["Newest milestone note."],
          },
          {
              "id": "older-milestone",
              "title": "Older completed work",
              "state": "complete",
              "completed_at": "2026-09-08T10:00:00+08:00",
              "artifacts": ["README.md"],
              "notes": ["Older milestone note."],
          },
      ],
      "active_work": [
          {
              "id": "active-build",
              "title": "Active build",
              "state": "running",
              "artifacts": ["tools/project-status.py"],
              "notes": ["Active work note."],
              "next_command": ".venv/bin/python tools/project-status.py --check",
          }
      ],
      "next_actions": [
          {
              "order": 1,
              "id": "ready-action",
              "title": "First runnable action",
              "depends_on": ["newest-milestone"],
          },
          {
              "order": 2,
              "id": "dependent-action",
              "title": "Action waiting for its predecessor",
              "depends_on": ["ready-action"],
          },
      ],
      "external_inputs": [
          {
              "id": "external-source",
              "state": "missing",
              "required_for": ["Future phase"],
              "notes": "External blocker note.",
          }
      ],
  }


def _write(path: Path, value: dict[str, Any]) -> None:
  path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      [sys.executable, str(VALIDATOR), "--status-file", str(path), *arguments],
      cwd=ROOT,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
  )


def _expect_valid(path: Path, label: str, *arguments: str) -> str:
  result = _run(path, *arguments)
  if result.returncode:
    raise AssertionError(
        f"{label}: expected success, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
  print(f"OK: accepted {label}")
  return result.stdout


def _expect_invalid(
    directory: Path,
    base: dict[str, Any],
    label: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
  value = deepcopy(base)
  mutate(value)
  path = directory / f"{label}.json"
  _write(path, value)
  result = _run(path, "--check")
  output = result.stdout + result.stderr
  if result.returncode == 0:
    raise AssertionError(f"{label}: invalid fixture was accepted")
  if expected not in output:
    raise AssertionError(
        f"{label}: expected error containing {expected!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
  print(f"OK: rejected {label}")


def main() -> int:
  manifests = ROOT / "manifests"
  with tempfile.TemporaryDirectory(
      prefix=".project-status-selftest-", dir=manifests
  ) as internal_name, tempfile.TemporaryDirectory(
      prefix="project-status-selftest-external-"
  ) as external_name:
    internal = Path(internal_name)
    external = Path(external_name)
    base = _base_status()
    positive = internal / "positive.json"
    _write(positive, base)

    _expect_valid(positive, "positive check", "--check")
    summary = _expect_valid(positive, "positive summary")
    for expected in (
        "latest-milestone: newest-milestone",
        "coverage:",
        "layers:",
        "features:",
        "first-ready-action: ready-action",
        "blocked-by: ready-action",
        "Phase note used by the summary contract.",
        "External blocker note.",
    ):
      if expected not in summary:
        raise AssertionError(f"summary is missing {expected!r}\n{summary}")
    _expect_valid(positive, "positive JSON", "--json")

    outside_status = external / "outside.json"
    _write(outside_status, base)
    result = _run(outside_status, "--check")
    if result.returncode == 0 or "outside the repository" not in result.stderr:
      raise AssertionError(f"outside status: unexpected result\n{result.stderr}")
    print("OK: rejected outside status")

    status_symlink = internal / "outside-status-link.json"
    status_symlink.symlink_to(outside_status)
    result = _run(status_symlink, "--check")
    if result.returncode == 0 or "outside the repository" not in result.stderr:
      raise AssertionError(f"status symlink escape: unexpected result\n{result.stderr}")
    print("OK: rejected status symlink escape")

    escaped_artifact = external / "artifact.txt"
    escaped_artifact.write_text("outside\n", encoding="utf-8")
    artifact_symlink = internal / "artifact-link.txt"
    artifact_symlink.symlink_to(escaped_artifact)
    _expect_invalid(
        internal,
        base,
        "artifact-symlink-escape",
        lambda value: value["active_work"][0].update(
            artifacts=[_relative(artifact_symlink)]
        ),
        "resolves outside the repository",
    )

    _expect_invalid(
        internal,
        base,
        "directory-artifact",
        lambda value: value["active_work"][0].update(
            artifacts=[_relative(internal)]
        ),
        "missing regular file",
    )
    _expect_invalid(
        internal,
        base,
        "missing-coverage-path",
        lambda value: value.pop("coverage_path"),
        "schema validation failed",
    )
    _expect_invalid(
        internal,
        base,
        "duplicate-id",
        lambda value: value["active_work"][0].update(id="newest-milestone"),
        "duplicate ids",
    )
    _expect_invalid(
        internal,
        base,
        "non-authoritative-plan",
        lambda value: value.update(plan_path="README.md"),
        "plan_path must name the authoritative repository file",
    )
    _expect_invalid(
        internal,
        base,
        "non-authoritative-baseline",
        lambda value: value.update(baseline_path="README.md"),
        "baseline_path must name the authoritative repository file",
    )
    _expect_invalid(
        internal,
        base,
        "non-authoritative-coverage",
        lambda value: value.update(coverage_path="README.md"),
        "coverage_path must name the authoritative repository file",
    )
    _expect_invalid(
        internal,
        base,
        "unknown-dependency",
        lambda value: value["next_actions"][0].update(
            depends_on=["unknown-action"]
        ),
        "unknown dependencies",
    )
    _expect_invalid(
        internal,
        base,
        "dependency-cycle",
        lambda value: value["next_actions"][0].update(
            depends_on=["dependent-action"]
        ),
        "dependency cycle",
    )
    _expect_invalid(
        internal,
        base,
        "reverse-dependency",
        lambda value: (
            value["next_actions"][0].update(depends_on=["dependent-action"]),
            value["next_actions"][1].update(depends_on=[]),
        ),
        "is not earlier",
    )
    _expect_invalid(
        internal,
        base,
        "nonconsecutive-order",
        lambda value: value["next_actions"][1].update(order=3),
        "must be consecutive",
    )
    _expect_invalid(
        internal,
        base,
        "complete-active-work",
        lambda value: value["active_work"][0].update(
            state="complete", completed_at="2026-09-08T11:30:00+08:00"
        ),
        "schema validation failed",
    )
    _expect_invalid(
        internal,
        base,
        "active-completed-at",
        lambda value: value["active_work"][0].update(
            completed_at="2026-09-08T11:30:00+08:00"
        ),
        "schema validation failed",
    )

    def make_milestone_incomplete(value: dict[str, Any]) -> None:
      value["milestones"][0]["state"] = "running"
      value["milestones"][0].pop("completed_at")

    _expect_invalid(
        internal,
        base,
        "incomplete-milestone",
        make_milestone_incomplete,
        "schema validation failed",
    )
    _expect_invalid(
        internal,
        base,
        "incomplete-milestone-completed-at",
        lambda value: value["milestones"][0].update(state="running"),
        "schema validation failed",
    )
    _expect_invalid(
        internal,
        base,
        "timestamp-inversion",
        lambda value: value.update(updated_at="2026-09-08T10:30:00+08:00"),
        "later than updated_at",
    )

    def make_coverage_newer(value: dict[str, Any]) -> None:
      value["updated_at"] = "2001-01-01T00:00:00+00:00"
      for index, milestone in enumerate(value["milestones"]):
        milestone["completed_at"] = f"2000-01-0{index + 1}T00:00:00+00:00"

    _expect_invalid(
        internal,
        base,
        "coverage-timestamp-inversion",
        make_coverage_newer,
        "coverage.updated_at is later",
    )

    invalid_coverage = internal / "invalid-coverage.json"
    _write(invalid_coverage, {})
    _expect_invalid(
        internal,
        base,
        "invalid-coverage-contract",
        lambda value: value.update(coverage_path=_relative(invalid_coverage)),
        "coverage_path must name the authoritative repository file",
    )

  print("OK: project-status positive summary plus 20 negative contracts")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
