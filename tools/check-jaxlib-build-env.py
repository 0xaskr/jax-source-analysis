#!/usr/bin/env python3
"""Inspect local prerequisites for the pinned CPU jaxlib build.

The script is read-only and uses only the Python standard library. It does not
download a compiler, Bazel, or build dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _command_version(candidates: tuple[str, ...], args: tuple[str, ...]) -> dict[str, Any]:
  for candidate in candidates:
    path = shutil.which(candidate)
    if path is None:
      continue
    process = subprocess.run(
        [path, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = (process.stdout or process.stderr).strip().splitlines()
    return {
        "command": candidate,
        "path": path,
        "version": lines[0] if lines else None,
        "returncode": process.returncode,
    }
  return {
      "command": None,
      "path": None,
      "version": None,
      "returncode": None,
  }


def _git_head(path: Path) -> str | None:
  process = subprocess.run(
      ["git", "-C", str(path), "rev-parse", "HEAD"],
      check=False,
      capture_output=True,
      text=True,
  )
  return process.stdout.strip() if process.returncode == 0 else None


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


def inspect() -> dict[str, Any]:
  bazel_version_file = ROOT / "upstream/jax/.bazelversion"
  required_bazel_version = (
      bazel_version_file.read_text(encoding="utf-8").strip()
      if bazel_version_file.exists()
      else None
  )
  downloaded_bazel = tuple(
      str(path)
      for path in sorted((ROOT / "upstream/jax").glob("bazel-*-*"))
      if path.is_file() and os.access(path, os.X_OK)
  )
  memory_bytes, swap_bytes = _memory_bytes()
  disk = shutil.disk_usage(ROOT)
  compiler = _command_version(("clang++", "g++"), ("--version",))
  return {
      "workspace": str(ROOT),
      "host": {
          "platform": platform.platform(),
          "machine": platform.machine(),
          "logical_cpus": os.cpu_count(),
          "memory_bytes": memory_bytes,
          "swap_bytes": swap_bytes,
          "workspace_disk_free_bytes": disk.free,
      },
      "python": {
          "executable": sys.executable,
          "version": platform.python_version(),
          "is_required_3_12": sys.version_info[:2] == (3, 12),
      },
      "required_bazel_version": required_bazel_version,
      "bazel": _command_version(
          (*downloaded_bazel, "bazel", "bazelisk"), ("--version",)
      ),
      "compiler": compiler,
      "git": _command_version(("git",), ("--version",)),
      "sources": {
          "jax": {
              "path": str(ROOT / "upstream/jax"),
              "exists": (ROOT / "upstream/jax/build/build.py").exists(),
              "commit": _git_head(ROOT / "upstream/jax"),
          },
          "xla": {
              "path": str(ROOT / "upstream/xla"),
              "exists": (ROOT / "upstream/xla/xla").is_dir(),
              "commit": _git_head(ROOT / "upstream/xla"),
          },
      },
  }


def _gib(value: int | None) -> str:
  if value is None:
    return "unknown"
  return f"{value / (1024 ** 3):.1f} GiB"


def print_human(report: dict[str, Any]) -> None:
  print(f"workspace: {report['workspace']}")
  print(
      "python: "
      f"{report['python']['version']} ({report['python']['executable']}) "
      f"required-3.12={report['python']['is_required_3_12']}"
  )
  print(f"required Bazel: {report['required_bazel_version']}")
  for name in ("bazel", "compiler", "git"):
    item = report[name]
    if item["path"]:
      print(f"{name}: {item['version']} ({item['path']})")
    else:
      print(f"{name}: MISSING")
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
    print(f"source {name}: {state} commit={source['commit']} path={source['path']}")
  if not report["bazel"]["path"]:
    print("note: JAX build.py can download required Bazel when network is available")


def blockers(report: dict[str, Any]) -> list[str]:
  result: list[str] = []
  if not report["python"]["is_required_3_12"]:
    result.append("the preflight interpreter is not Python 3.12")
  if not report["compiler"]["path"]:
    result.append("no local clang++ or g++ compiler was found")
  if not report["git"]["path"]:
    result.append("git was not found")
  for name, source in report["sources"].items():
    if not source["exists"]:
      result.append(f"the pinned {name} source tree is missing")
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--json", action="store_true", help="print JSON")
  parser.add_argument(
      "--strict",
      action="store_true",
      help="fail when a local build prerequisite is missing",
  )
  args = parser.parse_args()
  report = inspect()
  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
  else:
    print_human(report)
  missing = blockers(report)
  if args.strict and missing:
    for item in missing:
      print(f"BLOCKER: {item}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
