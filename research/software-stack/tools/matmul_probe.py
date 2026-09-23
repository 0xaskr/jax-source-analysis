#!/usr/bin/env python3
"""Capture CPU matmul transformations and native pass dumps from pinned JAX."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
SEED = 20260914


def fingerprint(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_capture(output, build_manifest=None):
    import jax
    import jax.numpy as jnp
    from jax.experimental import serialize_executable
    import numpy as np

    jax.config.update("jax_enable_compilation_cache", False)
    jax.config.update("jax_log_compiles", True)
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(SEED)
    a = (rng.standard_normal((4, 8)) * 0.25).astype(np.float32)
    w = (rng.standard_normal((8, 6)) * 0.25).astype(np.float32)
    x = (rng.standard_normal((3, 4, 8)) * 0.25).astype(np.float32)
    a_changed = (rng.standard_normal((5, 8)) * 0.25).astype(np.float32)
    np.savez(output / "inputs.npz", a=a, w=w, x=x, a_changed=a_changed)

    def matmul(lhs, rhs):
        return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)

    batched_matmul = jax.vmap(matmul, in_axes=(0, None))

    def loss(lhs, rhs):
        y = matmul(lhs, rhs)
        return jnp.sum(jnp.square(y))

    def batched_loss(lhs, rhs):
        y = batched_matmul(lhs, rhs)
        return jnp.sum(jnp.square(y))

    a64, w64, x64 = [v.astype(np.float64) for v in (a, w, x)]
    y64, z64 = a64 @ w64, x64 @ w64
    cases = [
        ("matmul", matmul, (a, w), y64),
        ("vmap_matmul", batched_matmul, (x, w), z64),
        ("grad_matmul", jax.grad(loss, argnums=(0, 1)), (a, w),
         (2 * y64 @ w64.T, 2 * a64.T @ y64)),
        ("jit_grad_vmap_matmul", jax.grad(batched_loss, argnums=(0, 1)), (x, w),
         (2 * z64 @ w64.T, 2 * np.einsum("bmk,bmn->kn", x64, z64))),
    ]
    summaries = []
    for name, function, arrays, expected in cases:
        directory = output / name
        directory.mkdir()
        args = tuple(jax.device_put(value, cpu) for value in arrays)
        jaxpr = jax.make_jaxpr(function)(*args)
        lowered = jax.jit(function).lower(*args)
        (directory / "jaxpr.txt").write_text(str(jaxpr) + "\n")
        (directory / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
        # This conversion is an inspection API; actual backend passes are
        # captured independently through XLA_FLAGS.
        (directory / "exported-hlo.txt").write_text(lowered.as_text("hlo"))
        before_cost = lowered.cost_analysis()
        compiled = lowered.compile()
        (directory / "optimized-hlo.txt").write_text(compiled.as_text())
        actual = compiled(*args)
        jax.block_until_ready(actual)
        actual_leaves = jax.tree.leaves(actual)
        expected_leaves = jax.tree.leaves(expected)
        assert len(actual_leaves) == len(expected_leaves)
        serialized, in_tree, out_tree = serialize_executable.serialize(compiled)
        (directory / "executable.bin").write_bytes(serialized)
        # Only reload bytes produced in this process, never external pickle or
        # executable input. This validates the captured native payload boundary.
        reloaded = serialize_executable.deserialize_and_load(
            serialized, in_tree, out_tree, backend="cpu", execution_devices=[cpu])
        replayed = reloaded(*args)
        jax.block_until_ready(replayed)
        for original, replay in zip(actual_leaves, jax.tree.leaves(replayed), strict=True):
            np.testing.assert_array_equal(np.asarray(original), np.asarray(replay))
        comparisons = []
        for observed, reference in zip(actual_leaves, expected_leaves):
            observed = np.asarray(observed)
            np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=2e-5)
            comparisons.append({
                "shape": list(observed.shape), "dtype": str(observed.dtype),
                "max_absolute_error": float(np.max(np.abs(observed - reference))),
                "reference": "NumPy float64 forward or analytic gradient",
            })
        np.savez(directory / "outputs.npz", **{
            f"output_{i}": np.asarray(value) for i, value in enumerate(actual_leaves)
        })
        memory = compiled.memory_analysis()
        memory_fields = (
            "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
            "temp_size_in_bytes", "generated_code_size_in_bytes",
            "host_argument_size_in_bytes", "host_output_size_in_bytes",
            "host_alias_size_in_bytes", "host_temp_size_in_bytes",
        )
        summary = {
            "name": name, "input_shapes": [list(v.shape) for v in arrays],
            "comparison": comparisons, "numerics_passed": True,
            "serialized_executable_reload": "pass; outputs match original bit-for-bit",
            "cost_before_optimization": before_cost,
            "cost_after_optimization": compiled.cost_analysis(),
            "memory_analysis": None if memory is None else {
                field: getattr(memory, field) for field in memory_fields
                if hasattr(memory, field)
            },
            "memory_scope": "Compiler estimates, not measured live process/device peak memory.",
        }
        write_json(directory / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps({"case": name, "numerics": "pass", "errors": comparisons}))

    trace_count = 0

    def cache_matmul(lhs, rhs):
        nonlocal trace_count
        trace_count += 1
        return matmul(lhs, rhs)

    cached = jax.jit(cache_matmul)
    observations = []
    for i, lhs in enumerate((a, a.copy(), a_changed), 1):
        result = cached(jax.device_put(lhs, cpu), jax.device_put(w, cpu))
        result.block_until_ready()
        np.testing.assert_allclose(np.asarray(result), lhs.astype(np.float64) @ w64,
                                   rtol=2e-5, atol=2e-5)
        observations.append({"call": i, "lhs_shape": list(lhs.shape),
                             "trace_count": trace_count})
    assert [item["trace_count"] for item in observations] == [1, 1, 2]
    write_json(output / "tracing.json", {
        "observations": observations, "assertion": "trace counts equal [1, 1, 2]",
        "boundary": "Python tracing counts; compiler activity is recorded separately in run.log and xla-dump/.",
    })

    from capture_runtime import environment
    baseline = environment(output, build_manifest)
    # Keep the reused schema reference valid at this capture's location.
    baseline["$schema"] = os.path.relpath(
        ROOT / "manifests/schema/baseline.schema.json", output)
    write_json(output / "environment.json", baseline)
    return summaries, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jaxlib-build-manifest", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        parser.error("output must be under artifacts/jax-stack/")
    if any(c.isspace() for c in str(output)):
        parser.error("output path must not contain whitespace for XLA_FLAGS")
    if os.environ.get("XLA_FLAGS"):
        parser.error("unset ambient XLA_FLAGS so this capture has an explicit flag set")
    for variable in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
        if os.environ.get(variable) not in (None, "cpu"):
            parser.error(f"unset non-CPU {variable}")
    output.mkdir(parents=True, exist_ok=False)
    # HloPassPipeline treats the literal .* specially and skips unchanged
    # passes. .+ matches all non-empty pass names without that special case.
    flags = [f"--xla_dump_to={output / 'xla-dump'}",
             "--xla_dump_hlo_as_text", "--xla_dump_hlo_pass_re=.+",
             "--xla_dump_emitter_re=mlir-fusion|llvm"]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = " ".join(flags)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from capture_runtime import preflight
    build_manifest = preflight(output, args.jaxlib_build_manifest)
    started = datetime.now(timezone.utc).isoformat()
    producer = output / "producer.py"
    shutil.copy2(Path(__file__), producer)
    with (output / "run.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cases, baseline = run_capture(output, build_manifest)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            artifacts.append({"path": str(path.relative_to(ROOT)), **fingerprint(path)})
    relative_output = str(output.relative_to(ROOT))
    manifest = {
        "capture_id": output.name, "evidence_level": "RUN-CPU",
        "qualifiers": baseline["status"]["qualifiers"],
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, *sys.orig_argv[1:]],
            "cwd": ".", "source": str(producer.relative_to(ROOT)), **fingerprint(producer)},
        "input_parameters": {"seed": SEED, "dtype": "float32", "precision": "HIGHEST",
                             "rtol": 2e-5, "atol": 2e-5},
        "environment_path": f"{relative_output}/environment.json",
        "jaxlib_build_manifest": str(build_manifest.relative_to(ROOT)) if build_manifest else None,
        "xla_flags": [f"--xla_dump_to={relative_output}/xla-dump", *flags[1:]],
        "cases": [case["name"] for case in cases],
        "outcome": "pass", "artifacts": artifacts,
        "limitations": [
            "Native behavior belongs to the captured wheel; source attribution additionally requires the recorded build/native binding.",
            "No TPU compiler, LLO, TPU execution, real inference model, fusion injection or peak-memory split is validated.",
            "A dump file shows a recorded pass boundary; missing dump files do not prove that a pass did not run.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"capture": relative_output, "cases": len(cases),
                      "files": len(artifacts), "qualifiers": manifest["qualifiers"]}))


if __name__ == "__main__":
    main()
