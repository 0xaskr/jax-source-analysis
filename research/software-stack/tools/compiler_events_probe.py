#!/usr/bin/env python3
"""Capture cold compilation separately from repeated CPU execution."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil

from matmul_probe import ROOT, fingerprint, write_json


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(20260914)
    a = (rng.standard_normal((64, 128)) * .02).astype(np.float32)
    w = (rng.standard_normal((128, 64)) * .02).astype(np.float32)
    np.savez(output / "inputs.npz", a=a, w=w)
    args = tuple(jax.device_put(v) for v in (a, w))
    for arg in args:
        arg.block_until_ready()
    trace_count = 0

    def workload(x, y):
        nonlocal trace_count
        trace_count += 1
        with jax.profiler.TraceAnnotation("research_python_inside_jit"):
            with jax.named_scope("research_matmul_op"):
                return jnp.tanh(jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST))

    function = jax.jit(workload)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 3
    options.python_tracer_level = 0
    options.enable_hlo_proto = True
    values = []
    jax.profiler.start_trace(output / "trace", create_perfetto_trace=True, profiler_options=options)
    try:
        with jax.profiler.TraceAnnotation("research_cold_lower"):
            lowered = function.lower(*args)
        with jax.profiler.TraceAnnotation("research_cold_compile"):
            executable = lowered.compile()
        for step in range(3):
            with jax.profiler.TraceAnnotation("research_warm_execute", step=step):
                # Invoke the JIT wrapper again to distinguish Python tracing from execution.
                value = function(*args)
                value.block_until_ready()
                values.append(value)
    finally:
        jax.profiler.stop_trace()

    (output / "stablehlo.mlir").write_text(lowered.as_text("stablehlo", debug_info=True))
    (output / "optimized-hlo.txt").write_text(executable.as_text())
    np.savez(output / "outputs.npz", **{f"output_{i}":np.asarray(v) for i,v in enumerate(values)})
    reference = np.tanh(a.astype(np.float64) @ w.astype(np.float64))
    for value in values:
        np.testing.assert_allclose(np.asarray(value), reference, rtol=2e-5, atol=2e-5)
    trace_files = list((output / "trace").rglob("perfetto_trace.json.gz"))
    assert len(trace_files) == 1
    with gzip.open(trace_files[0], "rt") as stream:
        raw = json.load(stream)
    events = [e for e in raw["traceEvents"] if e.get("ph") == "X"]
    counts = Counter(e["name"] for e in events)
    assert trace_count == 1 and counts["research_python_inside_jit"] == 1
    assert counts["research_warm_execute"] == 3

    def contains(parent, child):
        return (parent["pid"],parent["tid"]) == (child["pid"],child["tid"]) and (
            parent["ts"] <= child["ts"] and child["ts"]+child["dur"] <= parent["ts"]+parent["dur"])

    def scope(name):
        matches = [e for e in events if e["name"] == name]
        assert len(matches) == 1
        return matches[0]

    lower, compile = scope("research_cold_lower"), scope("research_cold_compile")
    assert contains(lower,scope("research_python_inside_jit"))
    compile_children = [e for e in events if e is not compile and contains(compile,e)]
    warm_scopes = [e for e in events if e["name"] == "research_warm_execute"]
    warm_children = [e for e in events if any(e is not s and contains(s,e) for s in warm_scopes)]
    write_json(output / "compile-thread-events.json", compile_children)
    write_json(output / "event-counts.json", dict(sorted(counts.items())))
    # Candidate names are observed data; native revision attribution is separately bounded.
    expected_passes = {"algsimp", "constant_folding", "layout-assignment"}
    observed_passes = expected_passes & {e["name"] for e in compile_children}
    assert observed_passes, "No expected native compiler pass events found"
    assert not (expected_passes & {e["name"] for e in warm_children})
    summary = {
        "outcome":"pass", "evidence_level":"RUN-CPU", "trace_count":trace_count,
        "python_inside_jit_events":counts["research_python_inside_jit"],
        "warm_execution_scopes":len(warm_scopes),
        "compile_thread_child_events":len(compile_children),
        "observed_known_pass_names":sorted(observed_passes),
        "compile_event_names":dict(sorted(Counter(e["name"] for e in compile_children).items())),
        "warm_event_names":dict(sorted(Counter(e["name"] for e in warm_children).items())),
        "max_absolute_errors":[float(np.max(np.abs(np.asarray(v)-reference))) for v in values],
        "bounds":["Child membership requires the same pid/tid and time containment; worker-thread compilation events are not included in this count.",
                  "A TraceAnnotation inside jit traces Python once; it does not create one runtime device span per call.",
                  "Existing CPU wheel pass events do not execute or validate a custom compiler-source patch."]}
    write_json(output / "summary.json",summary)
    spec = importlib.util.spec_from_file_location("baseline_capture",ROOT / "tools/capture-baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    environment = baseline.capture_baseline()
    assert all(not s["dirty"] for s in environment["repository"]["sources"].values())
    write_json(output / "environment.json",environment)
    return summary,environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack directory and unset ambient XLA_FLAGS")
    os.environ.update(JAX_PLATFORMS="cpu",XLA_FLAGS="--xla_cpu_enable_xprof_traceme=true",PYTHONDONTWRITEBYTECODE="1")
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    with (output / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            summary,env = collect(output)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    manifest = {"capture_id":output.name,"outcome":"pass","evidence_level":"RUN-CPU",
        "qualifiers":env["status"]["qualifiers"],"started_at":started,
        "finished_at":datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[".venv/bin/python","-B","research/software-stack/tools/compiler_events_probe.py","--output",str(output.relative_to(ROOT))],
                    "source":str((output / "producer.py").relative_to(ROOT)),**fingerprint(output / "producer.py")},
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]}
    write_json(output / "manifest.json",manifest)
    print(json.dumps({"capture":output.name,"passes":summary["observed_known_pass_names"],"qualifiers":manifest["qualifiers"]}))


if __name__ == "__main__":
    main()
