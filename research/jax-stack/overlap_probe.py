#!/usr/bin/env python3
"""Trace explicit collective start/done and control hints through CPU compilation."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

from matmul_probe import ROOT, fingerprint, write_json


def hlo_counts(text):
    from jaxlib import _hlo
    module = _hlo.hlo_module_from_text(text)
    return {"opcodes": dict(sorted(Counter(i.opcode.name for c in module.computations() for i in c.instructions()).items())),
            "custom_call_targets": dict(sorted(Counter(re.findall(r'custom_call_target="([^"]+)"', text)).items()))}


def observe_previous_failures(output, make_function, args, x, w):
    """Record changed outcomes; numerical mismatches still fail the probe."""
    import jax
    import numpy as np

    observations = []
    x, w = x.astype(np.float64), w.astype(np.float64)
    total = x[:64] + x[64:]
    configurations = [("scheduled-varying-checked", "scheduled", True, True),
                      ("scheduled-varying-unchecked", "scheduled", False, True),
                      ("scheduled-invariant-default-layout", "scheduled", True, False),
                      ("explicit-layout-backend-error", "explicit-layout", True, False)]
    for name, mode, checked, varying in configurations:
        directory = output / name
        directory.mkdir()
        phase = "lowering"
        try:
            lowered = jax.jit(make_function(mode, checked, varying_math=varying)).lower(*args)
            (directory / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
            (directory / "exported-hlo.txt").write_text(lowered.as_text("hlo"))
            phase = "CPU-native-compilation"
            compiled = lowered.compile()
            (directory / "optimized-hlo.txt").write_text(compiled.as_text())
            phase = "CPU-execution"
            value = compiled(*args)
            value.block_until_ready()
            actual = np.asarray(value)
        except Exception as error:
            observed = {"case": name, "outcome": "error", "phase": phase,
                        "type": type(error).__name__, "message": str(error), "varying_math": varying}
            write_json(directory / "observed-error.json", observed)
        else:
            reference = np.concatenate([total + (piece @ w if varying else w @ w) for piece in [x[:64], x[64:]]])
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
            np.save(directory / "output.npy", actual, allow_pickle=False)
            observed = {"case": name, "outcome": "numerical-pass", "phase": "CPU-execution",
                        "varying_math": varying, "max_absolute_error": float(np.max(np.abs(actual-reference)))}
            write_json(directory / "observed-success.json", observed)
        observations.append(observed)
    write_json(output / "prior-failure-observations.json", observations)


def collect(output, build_manifest=None, observe_prior_failures=False):
    import jax
    import numpy as np
    from jax._src.lax import parallel
    from jax.experimental import overlap
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    jax.config.update("jax_enable_compilation_cache", False)
    devices = jax.devices("cpu")
    assert len(devices) == 2 and jax.process_count() == 1
    mesh = Mesh(np.array(devices), ("d",))
    rng = np.random.default_rng(20260915)
    x = (rng.standard_normal((128, 64)) * .125).astype(np.float32)
    w = (rng.standard_normal((64, 64)) * .125).astype(np.float32)
    np.savez(output / "inputs.npz", x=x, w=w)
    args = (jax.device_put(x, NamedSharding(mesh, P("d", None))),
            jax.device_put(w, NamedSharding(mesh, P(None, None))))
    pieces = np.split(x.astype(np.float64), 2, axis=0)
    total = pieces[0] + pieces[1]
    reference = np.concatenate([total + w.astype(np.float64) @ w.astype(np.float64)] * 2)

    def make_function(mode, check_vma, varying_math=False):
        def local(x, w):
            if mode in {"sync", "sync-control"}:
                total = jax.lax.psum(x, "d")
                math = w @ w
                if mode == "sync-control":
                    overlap.schedule([math, total])
            else:
                future = parallel.psum_start(x, "d")
                math = x @ w if varying_math else w @ w
                total = future.done()
                if mode == "scheduled":
                    overlap.schedule([future, math, total])
                elif mode == "explicit-layout":
                    # Same compiler-recognized target as overlap.control_dep,
                    # but explicit rank-2 major-to-minor layouts for this probe.
                    control_dep = jax.ffi.ffi_call("control_dep", (), has_side_effect=True,
                                                  input_layouts=((0, 1), (0, 1)))
                    control_dep(future, math)
                    control_dep(math, total)
            return total + math
        local.__name__ = "collective_" + mode + ("_checked" if check_vma else "_unchecked")
        return jax.shard_map(local, mesh=mesh, in_specs=(P("d", None), P(None, None)),
                             out_specs=P("d", None), check_vma=check_vma)

    if observe_prior_failures:
        observe_previous_failures(output, make_function, args, x, w)
    else:
        for name, check_vma, varying_math, exception, message in [
            ("scheduled-varying-checked", True, True, ValueError, "requires varying manual axes to match"),
            ("scheduled-varying-unchecked", False, True, AttributeError, "has no attribute done"),
            ("scheduled-invariant-default-layout", True, False, ValueError, "incorrect layout dense<>"),
        ]:
            failure = output / name
            failure.mkdir()
            try:
                jax.jit(make_function("scheduled", check_vma, varying_math=varying_math)).lower(*args)
            except exception as error:
                assert message in str(error)
                phase = "MLIR-verification" if name == "scheduled-invariant-default-layout" else "tracing"
                write_json(failure / "expected-error.json", {"phase": phase, "type": type(error).__name__,
                            "message": str(error), "check_vma": check_vma, "math": "x @ w" if varying_math else "w @ w"})
            else:
                raise AssertionError(f"Expected failure: {name}")

        failure = output / "explicit-layout-backend-error"
        failure.mkdir()
        lowered = jax.jit(make_function("explicit-layout", True)).lower(*args)
        (failure / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
        (failure / "exported-hlo.txt").write_text(lowered.as_text("hlo"))
        try:
            lowered.compile()
        except jax.errors.JaxRuntimeError as error:
            assert "instruction->IsDead()" in str(error) and "is live and cannot be removed" in str(error)
            write_json(failure / "expected-error.json", {"phase": "CPU-native-compilation", "type": type(error).__name__,
                        "message": str(error), "check_vma": True, "explicit_input_layouts": [[0, 1], [0, 1]]})
        else:
            raise AssertionError("Expected version-skew CPU failure with start control dependency")

    cases = []
    for name, mode, check_vma in [("sync", "sync", True), ("async", "async", True),
                                   ("sync-control", "sync-control", True)]:
        directory = output / name
        directory.mkdir()
        function = make_function(mode, check_vma)
        (directory / "jaxpr.txt").write_text(str(jax.make_jaxpr(function)(*args)) + "\n")
        lowered = jax.jit(function).lower(*args)
        (directory / "stablehlo.mlir").write_text(lowered.as_text("stablehlo", debug_info=True))
        (directory / "exported-hlo.txt").write_text(lowered.as_text("hlo"))
        compiled = lowered.compile()
        value = compiled(*args)
        value.block_until_ready()
        actual = np.asarray(value)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        np.save(directory / "output.npy", actual, allow_pickle=False)
        optimized = compiled.as_text()
        (directory / "optimized-hlo.txt").write_text(optimized)
        matches = [p for p in (output / "xla-dump").glob("*.cpu_after_optimizations.txt")
                   if p.read_text().splitlines()[0] == optimized.splitlines()[0]]
        assert len(matches) == 1
        prefix = matches[0].name.removesuffix(".cpu_after_optimizations.txt")
        selected = sorted(p for p in (output / "xla-dump").glob(prefix + ".*.txt")
                          if "async-collective" in p.name or "control-dep-rewriter" in p.name)
        assert selected
        boundaries = [{"path": str(p.relative_to(ROOT)), **hlo_counts(p.read_text())} for p in selected]
        summary = {"case": name, "check_vma": check_vma, "native_module_prefix": prefix,
                   "max_absolute_error": float(np.max(np.abs(actual-reference))),
                   "exported": hlo_counts((directory / "exported-hlo.txt").read_text()),
                   "optimized": hlo_counts(optimized), "async_pipeline_boundaries": boundaries}
        write_json(directory / "summary.json", summary)
        cases.append(summary)
    topology = {"backend": "cpu", "process_count": jax.process_count(), "device_count": len(devices),
                "devices": [str(d) for d in devices], "mesh_axes": {"d": 2}, "x_global_shape": [128, 64],
                "x_local_shape": [64, 64], "w_replicated_shape": [64, 64],
                "local_expression": "psum(x, 'd') + w @ w", "output_global_shape": [128, 64],
                "scope": "Two logical CPU devices on one host; no TPU or inter-host network."}
    limits = ["Numerical equivalence and compiler IR are verified; no communication/compute overlap time or speedup is measured.",
              "psum_start is a private JAX implementation API at this pin; overlap.schedule is experimental.",
              "The three core cases keep check_vma=True. Prior failure cases are either checked against historical outcomes or recorded independently when observe_prior_failures is enabled.",
              "The explicit-layout case uses the public FFI constructor with the private compiler control_dep target; its compilation outcome must be read from the capture.",
              "Native pass behavior belongs to the captured binary; source attribution additionally requires recorded build/native binding.",
              "CPU synchronous rewriting and CPU scheduling cannot establish TPU/libtpu overlap support or scheduler behavior."]
    write_json(output / "summary.json", {"outcome": "pass", "topology": topology, "cases": cases, "limits": limits})
    from capture_runtime import environment
    env = environment(output, build_manifest)
    write_json(output / "environment.json", env)
    return cases, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaxlib-build-manifest", type=Path)
    parser.add_argument("--observe-prior-failures", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS") or any(c.isspace() for c in str(output)):
        parser.error("Use a new artifacts/jax-stack directory and unset ambient XLA_FLAGS")
    flags = ["--xla_force_host_platform_device_count=2", f"--xla_dump_to={output/'xla-dump'}",
             "--xla_dump_hlo_as_text=true", "--xla_dump_hlo_pass_re=.+"]
    os.environ.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", XLA_FLAGS=" ".join(flags))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    from capture_runtime import preflight
    build_manifest = preflight(output, args.jaxlib_build_manifest)
    started = datetime.now(timezone.utc).isoformat()
    with (output / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            cases, env = collect(output, build_manifest, args.observe_prior_failures)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass", "evidence_level": "RUN-CPU",
        "qualifiers": env["status"]["qualifiers"], "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "xla_flags": flags, "cases": [c["case"] for c in cases],
        "jaxlib_build_manifest": str(build_manifest.relative_to(ROOT)) if build_manifest else None,
        "observe_prior_failures": args.observe_prior_failures,
        "producer": {"argv": [sys.executable, *sys.orig_argv[1:]],
                     "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture": output.name, "cases": len(cases), "qualifiers": env["status"]["qualifiers"]}))


if __name__ == "__main__":
    main()
