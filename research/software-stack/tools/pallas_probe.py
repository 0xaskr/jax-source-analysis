#!/usr/bin/env python3
"""Compare identical matmul inputs across ordinary JAX and Pallas interpreters."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil

from matmul_probe import ROOT, fingerprint, write_json


def make_tiled_matmul(m, k, n, *, mode, recorder=None):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if m % 2 or n % 2:
        raise ValueError("This comparison uses exact 2x2 output tiles.")

    def tiled_kernel(a_ref, w_ref, out_ref):
        out_ref[...] = jnp.matmul(a_ref[...], w_ref[...],
                                 precision=jax.lax.Precision.HIGHEST)

    interpret = mode
    if mode == "tpu":
        interpret = pltpu.InterpretParams(random_seed=20260914,
                                          grid_point_recorder=recorder)
    return pl.pallas_call(
        tiled_kernel, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        grid=(m // 2, n // 2),
        in_specs=[pl.BlockSpec((2, k), lambda i, j: (i, 0)),
                  pl.BlockSpec((k, 2), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((2, 2), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret, debug=True,
    )


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    jax.config.update("jax_enable_compilation_cache", False)
    origin = ROOT / "artifacts/jax-stack/cpu-matmul-003/inputs.npz"
    shutil.copy2(origin, output / "inputs.npz")
    with np.load(origin, allow_pickle=False) as inputs:
        a, w, x = [inputs[name].copy() for name in ("a", "w", "x")]
    reference = a.astype(np.float64) @ w.astype(np.float64)
    batch_reference = x.astype(np.float64) @ w.astype(np.float64)
    a_device, w_device, x_device = [jax.device_put(value, jax.devices("cpu")[0])
                                    for value in (a, w, x)]
    grid_points = []

    def record_grid_point(token, coordinates, core_id):
        grid_points.append({"coordinates": [int(v) for v in coordinates],
                            "core": int(core_id)})
        return token

    generic = make_tiled_matmul(4, 8, 6, mode=True)
    tpu_interpret = make_tiled_matmul(4, 8, 6, mode="tpu", recorder=record_grid_point)
    cases = [
        ("regular", lambda lhs, rhs: jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST),
         (a_device, w_device), reference, "RUN-CPU", "ordinary JAX"),
        ("pallas-generic", generic, (a_device, w_device), reference, "RUN-CPU",
         "generic Pallas HLO interpreter; not TPU-specific simulation"),
        ("pallas-tpu-interpret", tpu_interpret, (a_device, w_device), reference,
         "SIM-TPU", "Mosaic TPU semantic interpreter executing on CPU"),
        ("pallas-generic-vmap", jax.vmap(generic, in_axes=(0, None)),
         (x_device, w_device), batch_reference, "RUN-CPU", "vmap over generic Pallas interpreter"),
    ]
    summaries = []
    for name, function, args, expected, level, mode in cases:
        directory = output / name
        directory.mkdir()
        print("CASE", name)
        (directory / "jaxpr.txt").write_text(str(jax.make_jaxpr(function)(*args)) + "\n")
        lowered = jax.jit(function).lower(*args)
        stablehlo = lowered.as_text("stablehlo")
        (directory / "stablehlo.mlir").write_text(stablehlo)
        compiled = lowered.compile()
        (directory / "optimized-hlo.txt").write_text(compiled.as_text())
        actual = compiled(*args)
        actual.block_until_ready()
        jax.effects_barrier()
        actual = np.asarray(actual)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.save(directory / "output.npy", actual, allow_pickle=False)
        summary = {
            "case": name, "evidence_level": level, "execution_backend": "cpu", "mode": mode,
            "input_shapes": [list(arg.shape) for arg in args], "output_shape": list(actual.shape),
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
            "numerics_passed": True,
            "stablehlo_operation_mentions": dict(sorted(Counter(re.findall(r"\bstablehlo\.([a-z_]+)", stablehlo)).items())),
            "custom_call_targets": sorted(set(re.findall(r"stablehlo\.custom_call\s+@([A-Za-z0-9_]+)", stablehlo))),
            "ir_counts_boundary": "Lexical observations, not a complete semantic IR analysis.",
        }
        write_json(directory / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps(summary))
    check_points = {tuple(point["coordinates"]) for point in grid_points}
    assert len(grid_points) == 6 and check_points == {(i, j) for i in range(2) for j in range(3)}
    write_json(output / "grid-points.json", {
        "observations": grid_points, "expected_grid": [2, 3], "num_cores": 1,
        "scope": "Interpreter visitation of six grid points; not hardware scheduling or concurrency.",
    })
    pltpu.reset_tpu_interpret_mode_state()

    try:
        jax.jit(make_tiled_matmul(4, 8, 6, mode=False)).lower(a_device, w_device)
    except ValueError as error:
        assert "Only interpret mode is supported on CPU backend" in str(error)
        failure = {"outcome": "expected-failure", "stage": "CPU lowering without interpret",
                   "exception_type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("Non-interpreted Pallas unexpectedly lowered on CPU")
    write_json(output / "non-interpret-cpu.json", failure)

    spec = importlib.util.spec_from_file_location("baseline_capture", ROOT / "tools/capture-baseline.py")
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    environment = baseline_module.capture_baseline()
    assert all(not entry["dirty"] for entry in environment["repository"]["sources"].values())
    environment["$schema"] = os.path.relpath(ROOT / "manifests/schema/baseline.schema.json", output)
    write_json(output / "environment.json", environment)
    return summaries, environment, {"path": str(origin.relative_to(ROOT)), **fingerprint(origin)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        parser.error("output must be within artifacts/jax-stack/")
    if os.environ.get("XLA_FLAGS"):
        parser.error("unset ambient XLA_FLAGS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    with (output / "run.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cases, environment, input_origin = collect(output)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
    artifacts = [{"path": str(path.relative_to(ROOT)), **fingerprint(path)}
                 for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {
        "capture_id": output.name, "outcome": "pass", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "qualifiers": environment["status"]["qualifiers"], "evidence_levels": ["RUN-CPU", "SIM-TPU"],
        "producer": {"argv": [".venv/bin/python", "-B", "research/software-stack/tools/pallas_probe.py",
            "--output", str(output.relative_to(ROOT))], "cwd": ".",
            "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "support_sources": [{"path": str(path.relative_to(ROOT)), **fingerprint(path)} for path in (
            ROOT / "research/software-stack/tools/matmul_probe.py", ROOT / "tools/capture-baseline.py")],
        "input_origin": input_origin, "cases": cases, "artifacts": artifacts,
        "limitations": ["2x2 tiles illustrate semantics only and are not asserted legal or efficient TPU hardware tiles.",
            "The TPU interpreter case validates only this matmul and grid traversal, not DMA, race freedom, communication or cycle timing.",
            "No Mosaic TPU compiler MLIR, LLO or real TPU execution is captured.",
            "All native CPU observations retain VERSION-SKEW relative to the pinned source."],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"capture": str(output.relative_to(ROOT)), "cases": len(cases),
                      "artifacts": len(artifacts), "qualifiers": manifest["qualifiers"]}))


if __name__ == "__main__":
    main()
