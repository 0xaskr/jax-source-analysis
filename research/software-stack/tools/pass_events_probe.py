#!/usr/bin/env python3
"""Capture cold/warm/filter controls and bind positive events to a built wheel."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import zipfile

from matmul_probe import ROOT, fingerprint, write_json
from pass_event_checks import analyze, require


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("pass_probe_baseline", ROOT / "tools/capture-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_gate(path, expected, observation=None):
    if path is None:
        require(expected == "absent", "present events require a successful patched build manifest")
        return None
    path = path.resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "build manifest must be repository-local")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if observation is not None:
        snapshot = observation.with_suffix(".input.json")
        snapshot.write_bytes(payload)
        write_json(observation, {"observed_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(path.relative_to(ROOT)), "snapshot_path": str(snapshot.relative_to(ROOT)),
            "snapshot_sha256": hashlib.sha256(payload).hexdigest(), "build_id": manifest["build_id"],
            "status_at_read": manifest["status"], "expected_events": expected})
    require(manifest["status"] == "build-succeeded" and not manifest["dry_run"], "build manifest is not a successful real build")
    require(manifest["result"]["exit_code"] == 0 and len(manifest["result"]["wheels"]) == 1, "build result lacks one successful wheel")
    patches = manifest["inputs"]["source_patches"]
    if expected == "present":
        require(len(patches) == 1 and patches[0]["component"] == "xla", "positive control requires exactly the XLA event patch")
        approved = [ROOT / "research/software-stack/tools/xla-hlo-pass-events.patch",
                    ROOT / "research/software-stack/tools/xla-hlo-pass-events-with-tests.patch",
                    ROOT / "research/software-stack/tools/xla-hlo-pass-events-exportable.patch"]
        require(patches[0]["sha256"] in {fingerprint(p)["sha256"] for p in approved if p.is_file()}, "build uses another source patch")
    else:
        require(not patches, "bound absent control must use an unpatched source build")
    return manifest


def bind_native(environment, path, expected):
    manifest = build_gate(path, expected)
    runtime = environment["runtime"]["jaxlib"]
    if manifest is None:
        return {"kind": "unbound-wheel-negative-control", "source_build_verified": False,
                "runtime_distribution": runtime["distribution"],
                "scope": "Observed absence in the current wheel; not the required matching-source build baseline."}
    # Reuse the canonical archive/manifest validator, which checks historical
    # inputs and artifacts without requiring the current checkout to be dirty.
    validator = load_baseline_module()._load_build_validator()
    manifest = validator.load_and_validate_manifest(path)
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    for component in ["jax", "xla"]:
        require(manifest["inputs"]["sources"][component]["commit"] == baseline["repository"]["sources"][component]["git_commit"], "build source pin differs")
    require(runtime["build_revision"] == manifest["inputs"]["sources"]["jax"]["commit"], "loaded jaxlib revision differs from build")
    require(environment["runtime"]["jax"]["install_type"] == "editable-source", "bound runtime must import the pinned editable JAX source")
    require(environment["runtime"]["jax"]["source_checkout_git_hash"] == runtime["build_revision"], "Python JAX and native build differ")
    wheel = manifest["result"]["wheels"][0]
    wheel_path = ROOT / wheel["path"]
    matched = []
    with zipfile.ZipFile(wheel_path) as archive:
        for binary in runtime["native_binaries"]:
            require("artifact_path" in binary, "loaded native binary lacks a repository-local path")
            loaded_path = (ROOT / binary["artifact_path"]).resolve()
            require(loaded_path.is_relative_to(ROOT), "loaded native binary is outside repository")
            require(fingerprint(loaded_path) == {k: binary[k] for k in ["sha256", "size_bytes"]}, "loaded binary file changed")
            member = "jaxlib/" + binary["package_path"]
            data = archive.read(member)
            require(hashlib.sha256(data).hexdigest() == binary["sha256"] and len(data) == binary["size_bytes"], "loaded native payload differs from built wheel")
            matched.append({"wheel_member": member, **binary})
    require({"jaxlib/_jax.so", "jaxlib/libjax_common.so"} <= {b["wheel_member"] for b in matched}, "core loaded binary identity missing")
    return {"kind": "source-build-native-payload-bound", "source_build_verified": True,
            "build_manifest_path": str(path.resolve().relative_to(ROOT)), "build_manifest": fingerprint(path),
            "build_id": manifest["build_id"], "input_fingerprint_sha256": manifest["input_fingerprint_sha256"],
            "source_patches": manifest["inputs"]["source_patches"], "wheel": wheel, "matched_loaded_payloads": matched,
            "scope": "Loaded jaxlib payloads match the recorded wheel. This does not add GPU/TPU evidence or prove a complete offline dependency closure."}


def read_events(output):
    paths = list((output / "trace").rglob("perfetto_trace.json.gz"))
    require(len(paths) == 1, "expected one exported trace")
    with gzip.open(paths[0], "rt") as stream:
        return [e for e in json.load(stream)["traceEvents"] if e.get("ph") == "X"]


def collect(output, expected, disabled_pass, build_manifest):
    build_gate(build_manifest, expected, output / "build-gate-observation.json")
    import jax
    import jax.numpy as jnp
    from jaxlib import _hlo  # Load this native reader before identity capture.
    import numpy as np

    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(20260915)
    a = (rng.standard_normal((64, 128)) * .02).astype(np.float32)
    w = (rng.standard_normal((128, 64)) * .02).astype(np.float32)
    np.savez(output / "inputs.npz", a=a, w=w)
    args = tuple(jax.device_put(v) for v in [a, w])
    jax.block_until_ready(args)
    baseline = load_baseline_module()
    before = baseline.capture_baseline(build_manifest)
    write_json(output / "environment-before.json", before)
    bind_native(before, build_manifest, expected)
    trace_count = 0
    def pass_event_workload(a, w):
        nonlocal trace_count
        trace_count += 1
        with jax.profiler.TraceAnnotation("pass_probe_python_trace"):
            return jnp.tanh(jnp.matmul(a, w, precision=jax.lax.Precision.HIGHEST) * 1.0 + 0.0)
    function = jax.jit(pass_event_workload)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 3
    options.python_tracer_level = 0
    options.enable_hlo_proto = True
    values = []
    jax.profiler.start_trace(output / "trace", create_perfetto_trace=True, profiler_options=options)
    try:
        with jax.profiler.TraceAnnotation("pass_probe_lower"):
            lowered = function.lower(*args)
        with jax.profiler.TraceAnnotation("pass_probe_compile"):
            compiled = lowered.compile()
        for step in range(3):
            with jax.profiler.TraceAnnotation("pass_probe_warm", step=step):
                value = function(*args)
                value.block_until_ready()
                values.append(value)
    finally:
        jax.profiler.stop_trace()
    require(trace_count == 1, "unexpected Python retracing")
    reference = np.tanh(a.astype(np.float64) @ w.astype(np.float64))
    outputs = [np.asarray(value) for value in values]
    for value in outputs:
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-5)
    np.savez(output / "outputs.npz", **{f"output_{i}": v for i, v in enumerate(outputs)})
    (output / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
    text = compiled.as_text()
    (output / "optimized-hlo.txt").write_text(text)
    module_name = re.match(r"HloModule ([^,\s]+)", text)[1]
    environment = baseline.capture_baseline(build_manifest)
    require(all(not s["dirty"] for s in environment["repository"]["sources"].values()), "original source tree became dirty")
    binding = bind_native(environment, build_manifest, expected)
    write_json(output / "environment.json", environment)
    write_json(output / "build-binding.json", binding)
    summary = analyze(read_events(output), module_name, expected, disabled_pass)
    summary.update(outcome="pass", evidence_level="RUN-CPU", python_trace_count=trace_count,
                   max_absolute_errors=[float(np.max(np.abs(v-reference))) for v in outputs],
                   runtime_binding_kind=binding["kind"],
                   positive_patch_runtime_accepted=(expected == "present" and binding["source_build_verified"]))
    write_json(output / "summary.json", summary)
    return summary, environment, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-events", choices=["absent", "present"], required=True)
    parser.add_argument("--disable-pass", choices=["algsimp"])
    parser.add_argument("--jaxlib-build-manifest", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack directory and unset ambient XLA_FLAGS")
    flags = ["--xla_cpu_enable_xprof_traceme=true"]
    if args.disable_pass:
        flags.append("--xla_disable_hlo_passes=" + args.disable_pass)
    os.environ.update(JAX_PLATFORMS="cpu", XLA_FLAGS=" ".join(flags), PYTHONDONTWRITEBYTECODE="1")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    shutil.copy2(Path(__file__).with_name("pass_event_checks.py"), output / "event-rules.py")
    started = datetime.now(timezone.utc).isoformat()
    failure = None
    with (output / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            summary, environment, binding = collect(output, args.expected_events, args.disable_pass, args.jaxlib_build_manifest)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            traceback.print_exc()
            write_json(output / "failure.json", failure)
    if failure:
        before_path = output / "environment-before.json"
        qualifiers = json.loads(before_path.read_text())["status"]["qualifiers"] if before_path.is_file() else []
    else:
        qualifiers = list(environment["status"]["qualifiers"])
    if not failure and binding.get("source_patches"):
        qualifiers.append("SOURCE-PATCHED")
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "fail" if failure else "pass",
        "evidence_level": "RUN-CPU" if not failure or (output / "trace").exists() else "SOURCE-ONLY", "qualifiers": qualifiers,
        "runtime_acceptance": failure is None,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "xla_flags": flags,
        "expected_events": args.expected_events, "disabled_pass": args.disable_pass,
        "producer": {"argv": [sys.executable, *sys.argv], "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    if failure:
        print(json.dumps({"capture": output.name, "outcome": "fail", "error": failure}))
        return 1
    print(json.dumps({"capture": output.name, "custom_events": summary["selected_custom_event_count"],
                      "binding": binding["kind"], "qualifiers": qualifiers}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
