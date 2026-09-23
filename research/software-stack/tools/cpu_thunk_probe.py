#!/usr/bin/env python3
"""Capture source-built CPU executables and warm thunk traces for four matmuls."""

import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

from matmul_probe import ROOT, fingerprint, write_json
from capture_runtime import preflight, environment


def check_environment_transition(before, after):
    a, b = copy.deepcopy(before), copy.deepcopy(after)
    old = {v["package_path"]: v for v in a["runtime"]["jaxlib"].pop("native_binaries")}
    new = {v["package_path"]: v for v in b["runtime"]["jaxlib"].pop("native_binaries")}
    if a != b or not old.keys() <= new.keys() or any(new[k] != v for k, v in old.items()):
        raise ValueError("environment changed beyond newly loaded source-bound native libraries")
    return sorted(new.keys() - old.keys())


def collect(output, build_manifest):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.serialize_executable import serialize

    jax.config.update("jax_enable_compilation_cache", False)
    before = environment(output, build_manifest)
    write_json(output / "environment-before.json", before)
    shutil.copy2(output / "build-binding.json", output / "build-binding-before.json")
    rng = np.random.default_rng(20260914)
    a, w, x = [(rng.standard_normal(shape) * 0.25).astype(np.float32)
               for shape in [(4, 8), (8, 6), (3, 4, 8)]]
    np.savez(output / "inputs.npz", a=a, w=w, x=x)
    a64, w64, x64 = [v.astype(np.float64) for v in (a, w, x)]
    y, z = a64 @ w64, x64 @ w64

    def matmul(lhs, rhs):
        return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)

    batched = jax.vmap(matmul, in_axes=(0, None))

    def loss(lhs, rhs):
        return jnp.sum(jnp.square(matmul(lhs, rhs)))

    def batched_loss(lhs, rhs):
        return jnp.sum(jnp.square(batched(lhs, rhs)))

    specs = [("matmul", matmul, (a, w), (y,)),
             ("vmap_matmul", batched, (x, w), (z,)),
             ("grad_matmul", jax.grad(loss, argnums=(0, 1)), (a, w), (2*y@w64.T, 2*a64.T@y)),
             ("jit_grad_vmap_matmul", jax.grad(batched_loss, argnums=(0, 1)), (x, w),
              (2*z@w64.T, 2*np.einsum("bmk,bmn->kn", x64, z)))]
    cases = []
    for name, function, arrays, references in specs:
        directory = output / name
        directory.mkdir()
        args = tuple(jax.device_put(a) for a in arrays)
        compiled = jax.jit(function).lower(*args).compile()
        (directory / "optimized-hlo.txt").write_text(compiled.as_text())
        package, _, _ = serialize(compiled)
        (directory / "executable.bin").write_bytes(package)
        jax.block_until_ready(compiled(*args))
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.enable_hlo_proto = True
        values = []
        jax.profiler.start_trace(directory / "trace", create_perfetto_trace=True, profiler_options=options)
        try:
            for iteration in range(3):
                with jax.profiler.TraceAnnotation("research_thunk_invocation", case=name, iteration=iteration):
                    result = compiled(*args)
                    jax.block_until_ready(result)
                values.append(result)
            jax.effects_barrier()
        finally:
            jax.profiler.stop_trace()
        errors, saved = [], {}
        for iteration, value in enumerate(values):
            leaves = jax.tree.leaves(value)
            assert len(leaves) == len(references)
            for i, (leaf, reference) in enumerate(zip(leaves, references, strict=True)):
                actual = np.asarray(leaf)
                np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
                saved[f"iteration_{iteration}_output_{i}"] = actual
                errors.append(float(np.max(np.abs(actual - reference))))
        np.savez(directory / "outputs.npz", **saved)
        trace_files = list((directory / "trace").rglob("perfetto_trace.json.gz"))
        assert len(trace_files) == 1
        with gzip.open(trace_files[0], "rt") as stream:
            trace = json.load(stream)
        selected = [e for e in trace["traceEvents"] if e.get("name") == "research_thunk_invocation" or
                    "hlo_op" in e.get("args", {}) or e.get("name", "").startswith("end: ")]
        write_json(directory / "selected-events.json", selected)
        cases.append({"name": name, "invocations": 3, "numerics_passed": True,
                      "max_absolute_error": max(errors), "selected_trace_events": len(selected)})
    after = environment(output, build_manifest)
    write_json(output / "environment.json", after)
    added = check_environment_transition(before, after)
    write_json(output / "summary.json", {"cases": cases,
        "newly_loaded_native_libraries": added,
        "scope": "Fresh CPU compile and warm execution; serialized bytes retained but never reloaded by this probe."})
    return after, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jaxlib-build-manifest", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("use a fresh artifacts/jax-stack output and unset ambient XLA_FLAGS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    build_manifest = preflight(output, args.jaxlib_build_manifest)
    started = datetime.now(timezone.utc).isoformat()
    observed, cases = collect(output, build_manifest)
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass",
        "evidence_level": "RUN-CPU", "qualifiers": observed["status"]["qualifiers"],
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv], "cwd": ".",
                     "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "jaxlib_build_manifest": str(build_manifest.relative_to(ROOT)),
        "cases": [c["name"] for c in cases],
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)}
                      for p in sorted(output.rglob("*")) if p.is_file()],
        "limitations": ["Host-thread CPU thunk trace; not a TPU device/core trace.",
                        "No executable package is loaded or replayed by this probe.",
                        "Trace instrumentation changes overhead; durations are not benchmark results."]})
    print(json.dumps(cases))


if __name__ == "__main__":
    main()
