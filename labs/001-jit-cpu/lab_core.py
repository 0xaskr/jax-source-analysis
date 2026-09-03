"""Shared analysis logic for the CLI probe and interactive notebook."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax._src import api as api_internal
from jax._src import compiler
from jax._src import pjit
from jax._src import xla_bridge
from jax._src.interpreters import mlir


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SYMBOLS = (
    ("用户 API", "jax.jit", api_internal.jit),
    ("包装与参数处理", "pjit.make_jit", pjit.make_jit),
    ("Tracing", "pjit._trace_for_jit", pjit._trace_for_jit),
    ("MLIR lowering", "mlir.lower_jaxpr_to_module", mlir.lower_jaxpr_to_module),
    ("编译与缓存", "compiler.compile_or_get_cached", compiler.compile_or_get_cached),
    ("CPU client", "xla_bridge.make_cpu_client", xla_bridge.make_cpu_client),
)

jax.config.update("jax_platforms", "cpu")


@dataclass(frozen=True)
class CacheObservation:
  call: int
  shape: tuple[int, ...]
  trace_count: int
  result: list[float]

  def as_row(self) -> dict[str, object]:
    return {
        "调用": self.call,
        "输入 shape": str(self.shape),
        "累计 tracing 次数": self.trace_count,
        "结果": self.result,
    }


def make_affine(
    scale: float = 2.0, bias: float = 1.0
) -> Callable[[jax.Array], jax.Array]:
  def affine(x: jax.Array) -> jax.Array:
    return x * scale + bias

  return affine


def cpu_input(size: int) -> jax.Array:
  cpu = jax.local_devices(backend="cpu")[0]
  with jax.default_device(cpu):
    return jnp.arange(size, dtype=jnp.float32)


def local_source_location(symbol: Callable[..., Any]) -> str:
  symbol = inspect.unwrap(symbol)
  filename = inspect.getsourcefile(symbol)
  if filename is None:
    return "<source unavailable>"
  source_path = Path(filename).resolve()
  try:
    display_path = source_path.relative_to(REPO_ROOT)
  except ValueError:
    display_path = source_path
  _, line = inspect.getsourcelines(symbol)
  return f"{display_path}:{line}"


def environment_info() -> dict[str, str]:
  return {
      "JAX": jax.__version__,
      "JAX source": str(Path(jax.__file__).resolve()),
      "CPU device": str(jax.local_devices(backend="cpu")[0]),
  }


def source_rows() -> list[dict[str, str]]:
  return [
      {"阶段": stage, "Symbol": name, "本地源码": local_source_location(symbol)}
      for stage, name, symbol in SOURCE_SYMBOLS
  ]


def source_excerpt(name: str, max_lines: int = 36) -> tuple[str, str]:
  for _, symbol_name, symbol in SOURCE_SYMBOLS:
    if symbol_name != name:
      continue
    unwrapped = inspect.unwrap(symbol)
    lines, _ = inspect.getsourcelines(unwrapped)
    excerpt = "".join(lines[:max_lines]).rstrip()
    if len(lines) > max_lines:
      excerpt += "\n  # … excerpt truncated …"
    return local_source_location(unwrapped), excerpt
  raise ValueError(f"Unknown source symbol: {name}")


def jaxpr_text(size: int, scale: float = 2.0, bias: float = 1.0) -> str:
  function = make_affine(scale, bias)
  return str(jax.make_jaxpr(function)(cpu_input(size)))


def stablehlo_text(
    size: int,
    scale: float = 2.0,
    bias: float = 1.0,
    *,
    debug_info: bool = False,
) -> str:
  function = make_affine(scale, bias)
  lowered = jax.jit(function).lower(cpu_input(size))
  return lowered.as_text(dialect="stablehlo", debug_info=debug_info)


def cache_experiment(
    size: int, scale: float = 2.0, bias: float = 1.0
) -> list[CacheObservation]:
  trace_count = 0
  function = make_affine(scale, bias)

  def counted_affine(x: jax.Array) -> jax.Array:
    nonlocal trace_count
    trace_count += 1
    return function(x)

  compiled = jax.jit(counted_affine)
  inputs = (cpu_input(size), cpu_input(size), cpu_input(size + 1))
  observations = []
  for call_number, value in enumerate(inputs, start=1):
    result = compiled(value)
    result.block_until_ready()
    observations.append(
        CacheObservation(
            call=call_number,
            shape=value.shape,
            trace_count=trace_count,
            result=result.tolist(),
        )
    )
  return observations
