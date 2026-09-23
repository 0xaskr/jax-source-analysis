#!/usr/bin/env python3
"""Verify user-defined host profiling spans in real exported CPU traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

from matmul_probe import ROOT, fingerprint, write_json


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(20260914)
    a = (rng.standard_normal((256, 128)) * 0.02).astype(np.float32)
    w = (rng.standard_normal((128, 192)) * 0.02).astype(np.float32)
    reference = a.astype(np.float64) @ w.astype(np.float64)
    a_device, w_device = jax.device_put(a), jax.device_put(w)
    function = jax.jit(lambda x, y: jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST))
    function(a_device, w_device).block_until_ready()
    np.savez(output / "inputs.npz", a=a, w=w)

    def start_event(name, **metadata):
        stack = contextlib.ExitStack()
        event = stack.enter_context(jax.profiler.TraceAnnotation(name, **metadata))
        return stack, event

    def stop_event(handle):
        handle[0].close()

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.python_tracer_level = 0
    options.enable_hlo_proto = True
    trace_directory = output / "trace"
    computed = []
    with jax.profiler.TraceAnnotation("research_outside_session"):
        pass
    jax.profiler.start_trace(trace_directory, create_perfetto_trace=True, profiler_options=options)
    try:
        for step in range(3):
            with jax.profiler.StepTraceAnnotation("research_step", step_num=step):
                with jax.profiler.TraceAnnotation("research_dispatch", step=step):
                    value = function(a_device, w_device)
                with jax.profiler.TraceAnnotation("research_wait", step=step):
                    value.block_until_ready()
                computed.append(value)

        explicit = jax.profiler.TraceAnnotation("research_constructor_span", purpose="constructor-vs-enter")
        with jax.profiler.TraceAnnotation("research_before_enter"):
            time.sleep(0.004)
        explicit.__enter__()
        try:
            with jax.profiler.TraceAnnotation("research_after_enter"):
                time.sleep(0.002)
            explicit.set_metadata(phase="metadata-added-before-stop")
        finally:
            explicit.__exit__(None, None, None)

        handle = start_event("research_manual_span", phase="split-functions")
        try:
            with jax.profiler.TraceAnnotation("research_manual_work"):
                time.sleep(0.002)
            handle[1].set_metadata(completed=True)
        finally:
            stop_event(handle)

        try:
            with jax.profiler.TraceAnnotation("research_exception_span"):
                raise ValueError("intentional scope-unwind probe")
        except ValueError as error:
            assert str(error) == "intentional scope-unwind probe"
        jax.effects_barrier()
    finally:
        jax.profiler.stop_trace()

    maximum_errors = []
    for value in computed:
        observed = np.asarray(value)
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=2e-5)
        maximum_errors.append(float(np.max(np.abs(observed - reference))))
    trace_files = list(trace_directory.rglob("perfetto_trace.json.gz"))
    assert len(trace_files) == 1
    with gzip.open(trace_files[0], "rt") as stream:
        raw = json.load(stream)
    events = [e for e in raw["traceEvents"] if e.get("ph") == "X"
              and e.get("name", "").startswith("research_")]
    groups = defaultdict(list)
    for event in events:
        groups[event["name"]].append(event)
    expected_counts = {"research_step": 3, "research_dispatch": 3, "research_wait": 3,
        "research_constructor_span": 1, "research_before_enter": 1, "research_after_enter": 1,
        "research_manual_span": 1, "research_manual_work": 1, "research_exception_span": 1}
    assert dict(Counter(e["name"] for e in events)) == expected_counts

    def contains(parent, child):
        return (parent["pid"], parent["tid"]) == (child["pid"], child["tid"]) and (
            parent["ts"] <= child["ts"] and
            parent["ts"] + parent["dur"] >= child["ts"] + child["dur"])

    constructor = groups["research_constructor_span"][0]
    assert contains(constructor, groups["research_before_enter"][0])
    assert contains(constructor, groups["research_after_enter"][0])
    assert contains(groups["research_manual_span"][0], groups["research_manual_work"][0])
    step_records = []
    for parent in sorted(groups["research_step"], key=lambda e:e["ts"]):
        children = [e for name in ("research_dispatch", "research_wait")
                    for e in groups[name] if contains(parent, e)]
        assert len(children) == 2
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in children)
        union = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                union += end - start
                start, end = next_start, next_end
        union += end - start
        exclusive = parent["dur"] - union
        assert exclusive >= -1e-6
        step_records.append({"args": parent.get("args", {}), "inclusive_us": parent["dur"],
                             "selected_children_union_us": union, "remaining_scope_us": max(0.0, exclusive)})
    selected_events = [{key: e[key] for key in ("name", "pid", "tid", "ts", "dur", "args") if key in e}
                       for e in events]
    write_json(output / "selected-events.json", selected_events)
    stats = {name:{"count":len(group), "inclusive_total_us":sum(e["dur"] for e in group),
                   "min_us":min(e["dur"] for e in group), "max_us":max(e["dur"] for e in group)}
             for name,group in sorted(groups.items())}
    summary = {
        "outcome": "pass", "evidence_level": "RUN-CPU", "event_counts": expected_counts,
        "stats": stats, "step_scopes": step_records, "max_absolute_errors": maximum_errors,
        "assertions": {"constructor_starts_before_enter":True, "manual_span_contains_work":True,
            "exception_scope_closed":True, "outside_session_event_absent":True,
            "three_step_scopes_contain_dispatch_and_wait":True},
        "units_source": "upstream/xla/xla/tsl/profiler/convert/trace_events_to_json.cc:87; ps converted to us for ts/dur",
        "trace_formats": {"xplane_count":len(list(trace_directory.rglob('*.xplane.pb'))),
                          "perfetto_json_count":len(trace_files)},
        "scope": "Host event lifecycle and exported trace statistics. Durations are diagnostic observations, not a performance benchmark or TPU device timing.",
    }
    assert summary["trace_formats"]["xplane_count"] >= 1
    write_json(output / "summary.json", summary)
    spec = importlib.util.spec_from_file_location("baseline_capture", ROOT / "tools/capture-baseline.py")
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    environment = baseline_module.capture_baseline()
    assert all(not s["dirty"] for s in environment["repository"]["sources"].values())
    environment["$schema"] = os.path.relpath(ROOT / "manifests/schema/baseline.schema.json", output)
    write_json(output / "environment.json", environment)
    return summary, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        parser.error("output must be within artifacts/jax-stack/")
    if os.environ.get("XLA_FLAGS"):
        parser.error("unset ambient XLA_FLAGS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_cpu_enable_xprof_traceme=true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    with (output / "run.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                summary, environment = collect(output)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise
    manifest = {"capture_id": output.name, "outcome": "pass", "evidence_level": "RUN-CPU",
        "qualifiers":environment["status"]["qualifiers"], "started_at":started,
        "finished_at":datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[".venv/bin/python","-B","research/software-stack/tools/profile_events_probe.py",
            "--output",str(output.relative_to(ROOT))],"cwd":".",
            "source":str((output/'producer.py').relative_to(ROOT)), **fingerprint(output/'producer.py')},
        "xla_flags":["--xla_cpu_enable_xprof_traceme=true"],
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)}
                     for p in sorted(output.rglob('*')) if p.is_file()],
        "limitations":["CPU host trace only; no TPU timing, LLO attribution or cross-thread span lifecycle is validated.",
            "XSpace and Trace Event JSON are captured; an XProf UI custom aggregation feature is not implemented.",
            "Inclusive durations must not be added across nested or concurrent spans as if they were elapsed wall time."]}
    write_json(output/'manifest.json',manifest)
    print(json.dumps({"capture":str(output.relative_to(ROOT)),"events":sum(summary['event_counts'].values()),
                      "qualifiers":manifest['qualifiers']}))


if __name__ == "__main__":
    main()
