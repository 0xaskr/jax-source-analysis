#!/usr/bin/env python3
"""Inspect the JAX-to-CPU path for a minimal jitted function."""

from __future__ import annotations

import argparse

import jax

from lab_core import cache_experiment
from lab_core import environment_info
from lab_core import jaxpr_text
from lab_core import source_rows
from lab_core import stablehlo_text


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
  for observation in cache_experiment(size, scale, bias):
    print(
        f"call={observation.call} shape={observation.shape} "
        f"trace_count={observation.trace_count} result={observation.result}"
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


if __name__ == "__main__":
  main()
