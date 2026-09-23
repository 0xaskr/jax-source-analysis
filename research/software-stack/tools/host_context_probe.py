#!/usr/bin/env python3
"""Exercise explicitly correlated, overlapping host operations across threads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading

from matmul_probe import ROOT, fingerprint, write_json
from capture_runtime import preflight, environment
from cpu_thunk_probe import check_environment_transition

CONTEXT_IDS = [(1 << 60), (1 << 60) + 1, (1 << 63) + 7, (1 << 64) - 1]


def collect(output, build_manifest):
    import jax
    import jax.numpy as jnp
    import numpy as np

    initial = environment(output, build_manifest)
    write_json(output / "environment-before.json", initial)
    shutil.copy2(output / "build-binding.json", output / "build-binding-before.json")
    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(20260915)
    arrays = [(rng.standard_normal((16, 16)) * 0.1).astype(np.float32) for _ in range(4)]
    weight = (rng.standard_normal((16, 24)) * 0.1).astype(np.float32)
    np.savez(output / "inputs.npz", **{f"a_{i}": a for i, a in enumerate(arrays)}, weight=weight)
    device_arrays = [jax.device_put(a) for a in arrays]
    device_weight = jax.device_put(weight)
    compiled = jax.jit(lambda a, w: jnp.matmul(a, w, precision=jax.lax.Precision.HIGHEST)).lower(
        device_arrays[0], device_weight).compile()
    jax.block_until_ready(compiled(device_arrays[0], device_weight))
    (output / "optimized-hlo.txt").write_text(compiled.as_text())
    # All sends finish before workers compute; all computations finish before
    # completion markers are released in reverse order. This is a controlled
    # correlation experiment, not a throughput benchmark.
    start = threading.Barrier(5)
    calculated = threading.Barrier(4)
    release = [threading.Event() for _ in range(4)]
    release[3].set()
    values = [None] * 4
    completion_order = []

    def worker(index, context_id):
        outcome = "success"
        try:
            start.wait(timeout=20)
            value = compiled(device_arrays[index], device_weight)
            jax.block_until_ready(value)
            values[index] = np.asarray(value)
            calculated.wait(timeout=20)
            if not release[index].wait(timeout=20):
                raise TimeoutError("completion release did not arrive")
            if index == 3:
                outcome = "expected-error"
                raise ValueError("intentional error after device completion")
            return values[index]
        except BaseException:
            if outcome != "expected-error":
                outcome = "unexpected-error"
            raise
        finally:
            # Internal TraceMe context fields are pinned-version research
            # interfaces. Two independent annotations keep each local scope on
            # its originating thread; no live TraceAnnotation is transferred.
            with jax.profiler.TraceAnnotation("research_link_receive", _ct=0, _c=context_id,
                                             work=index, outcome=outcome):
                pass
            completion_order.append(index)
            if index:
                release[index-1].set()

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.python_tracer_level = 0
    options.enable_hlo_proto = True
    errors = []
    jax.profiler.start_trace(output / "trace", create_perfetto_trace=True, profiler_options=options)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for index, context_id in enumerate(CONTEXT_IDS):
                with jax.profiler.TraceAnnotation("research_link_send", _pt=0, _p=context_id, work=index):
                    futures.append(executor.submit(worker, index, context_id))
            start.wait(timeout=20)
            for index, future in enumerate(futures):
                try:
                    future.result(timeout=20)
                except ValueError as error:
                    assert index == 3 and str(error) == "intentional error after device completion"
                    errors.append({"work": index, "error": str(error)})
            jax.effects_barrier()
    finally:
        jax.profiler.stop_trace()
    assert errors == [{"work": 3, "error": "intentional error after device completion"}]
    assert completion_order == [3, 2, 1, 0]
    max_errors = []
    for actual, a in zip(values, arrays, strict=True):
        reference = a.astype(np.float64) @ weight.astype(np.float64)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        max_errors.append(float(np.max(np.abs(actual-reference))))
    np.savez(output / "outputs.npz", **{f"output_{i}": v for i, v in enumerate(values)})
    after = environment(output, build_manifest)
    write_json(output / "environment.json", after)
    added = check_environment_transition(initial, after)
    summary = {"context_ids": CONTEXT_IDS, "context_type": 0, "dispatch_order": [0, 1, 2, 3],
               "completion_order": completion_order, "expected_errors": errors,
               "max_absolute_errors": max_errors, "newly_loaded_native_libraries": added,
               "limits": ["Controlled host correlation with reverse completion gates, not a performance benchmark.",
                          "Uses internal TraceMe context metadata through this pinned TraceAnnotation wrapper.",
                          "Four CPU computations complete before markers, including the intentional error path; no TPU evidence."]}
    write_json(output / "summary.json", summary)
    return after, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaxlib-build-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("use a fresh artifacts/jax-stack output; unset ambient XLA_FLAGS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    shutil.copy2(Path(__file__).with_name("cpu_thunk_probe.py"), output / "environment-check-helper.py")
    build_manifest = preflight(output, args.jaxlib_build_manifest)
    started = datetime.now(timezone.utc).isoformat()
    observed, summary = collect(output, build_manifest)
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass",
        "evidence_level": "RUN-CPU", "qualifiers": observed["status"]["qualifiers"],
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv], "source": str((output / "producer.py").relative_to(ROOT)),
                     **fingerprint(output / "producer.py")},
        "jaxlib_build_manifest": str(build_manifest.relative_to(ROOT)),
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
