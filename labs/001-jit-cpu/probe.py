#!/usr/bin/env python3
"""Inspect the JAX-to-CPU path for a minimal jitted function."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import jax
import jaxlib
import jaxlib.version as jaxlib_version

from lab_core import cache_experiment
from lab_core import environment_info
from lab_core import jaxpr_text
from lab_core import source_rows
from lab_core import stablehlo_text


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "manifests/baseline.json"
NATIVE_BINARY_SCHEMA_ID = (
    "https://jax-source-analysis.local/schema/native-binaries.schema.json"
)


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError("size must be at least 1")
  return parsed


def heading(title: str) -> None:
  print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_environment() -> None:
  heading("Environment")
  for label, value in environment_info().items():
    print(f"{label + ':':<16} {value}")


def show_sources() -> None:
  heading("Live source anchors")
  for row in source_rows():
    print(f"{row['Symbol']:<36} {row['本地源码']}")


def show_jaxpr(size: int, scale: float, bias: float) -> None:
  heading("Jaxpr")
  print(jaxpr_text(size, scale, bias))


def show_stablehlo(size: int, scale: float, bias: float) -> None:
  heading("StableHLO")
  print(stablehlo_text(size, scale, bias))


def show_run(size: int, scale: float, bias: float) -> None:
  heading("Tracing cache and CPU execution")
  observations = cache_experiment(size, scale, bias)
  trace_counts = tuple(observation.trace_count for observation in observations)
  shapes = tuple(observation.shape for observation in observations)
  expected_shapes = ((size,), (size,), (size + 1,))
  if trace_counts != (1, 1, 2) or shapes != expected_shapes:
    raise AssertionError(
        "jit cache invariant failed: "
        f"expected counts/shapes {(1, 1, 2)!r}/{expected_shapes!r}, "
        f"got {trace_counts!r}/{shapes!r}"
    )
  for observation in observations:
    print(
        f"call={observation.call} shape={observation.shape} "
        f"trace_count={observation.trace_count} result={observation.result}"
    )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_native_binary_manifest(output_path: Path, baseline_path: Path) -> None:
  """Record jaxlib native binaries mapped after running the selected stage."""
  process_maps = Path("/proc/self/maps")
  if not process_maps.is_file():
    raise RuntimeError("native binary capture requires Linux /proc/self/maps")

  jaxlib_root = Path(jaxlib.__file__).resolve().parent
  binaries: set[Path] = set()
  for line in process_maps.read_text().splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) != 6:
      continue
    candidate = Path(fields[5])
    if not candidate.is_absolute() or not candidate.is_file():
      continue
    try:
      candidate.relative_to(jaxlib_root)
    except ValueError:
      continue
    if ".so" in candidate.name:
      binaries.add(candidate.resolve())

  if not binaries:
    raise RuntimeError("no mapped jaxlib native binaries were found")

  entries = []
  for binary in sorted(binaries):
    entry = {
        "package_path": binary.relative_to(jaxlib_root).as_posix(),
        "sha256": _sha256(binary),
        "size_bytes": binary.stat().st_size,
    }
    try:
      entry["artifact_path"] = binary.relative_to(REPO_ROOT).as_posix()
    except ValueError:
      # A temporary validation environment can still provide package-relative
      # paths and hashes without leaking a host path into the capture.
      pass
    entries.append(entry)

  baseline_path = baseline_path.resolve()
  try:
    baseline_path.relative_to(REPO_ROOT)
  except ValueError as error:
    raise RuntimeError("baseline manifest must be repository-local") from error
  baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
  runtime = baseline["runtime"]["jaxlib"]
  build_revision = getattr(jaxlib_version, "_git_hash", None)
  if runtime["version"] != jaxlib.__version__:
    raise RuntimeError("baseline jaxlib version differs from the imported runtime")
  if runtime["build_revision"] != build_revision or not build_revision:
    raise RuntimeError("baseline jaxlib build revision differs from the imported runtime")

  identity = copy.deepcopy(runtime["distribution"])

  payload = {
      "$schema": NATIVE_BINARY_SCHEMA_ID,
      "schema_version": "1.0",
      "inventory_id": "native-binaries",
      "packages": [
          {
              "name": "jaxlib",
              "version": jaxlib.__version__,
              "source_component": "jax",
              "build_revision": build_revision,
              "runtime_roles": ["python-extension", "cpu-backend"],
              "identity": identity,
              "selection": (
                  "jaxlib shared objects mapped after the selected probe stage; "
                  "the lock identity selects a hashed wheel candidate but is not "
                  "cryptographic proof of installation membership"
              ),
              "binaries": entries,
          }
      ],
  }
  output_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--stage",
      choices=("all", "environment", "sources", "jaxpr", "stablehlo", "run"),
      default="all",
      help="Only display one pipeline stage (default: all).",
  )
  parser.add_argument(
      "--baseline-manifest",
      type=Path,
      default=BASELINE_PATH,
      help="Baseline whose exact jaxlib distribution identity is reused.",
  )
  parser.add_argument(
      "--size",
      type=positive_int,
      default=4,
      help="Input vector length (default: 4).",
  )
  parser.add_argument("--scale", type=float, default=2.0)
  parser.add_argument("--bias", type=float, default=1.0)
  parser.add_argument(
      "--log-compiles",
      action="store_true",
      help="Enable JAX compilation logs.",
  )
  parser.add_argument(
      "--native-binaries-manifest",
      type=Path,
      help="Write hashes for mapped jaxlib shared objects after the selected stage.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.log_compiles:
    jax.config.update("jax_log_compiles", True)

  stages = {
      "environment": show_environment,
      "sources": show_sources,
      "jaxpr": lambda: show_jaxpr(args.size, args.scale, args.bias),
      "stablehlo": lambda: show_stablehlo(args.size, args.scale, args.bias),
      "run": lambda: show_run(args.size, args.scale, args.bias),
  }
  if args.stage == "all":
    for action in stages.values():
      action()
  else:
    stages[args.stage]()
  if args.native_binaries_manifest is not None:
    write_native_binary_manifest(args.native_binaries_manifest, args.baseline_manifest)


if __name__ == "__main__":
  main()
