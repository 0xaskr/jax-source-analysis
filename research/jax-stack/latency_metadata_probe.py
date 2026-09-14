#!/usr/bin/env python3
"""Separate latency annotation propagation from the CPU cost-analysis API."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil

from matmul_probe import ROOT, fingerprint, write_json


CASES = [("plain", None), ("integer", 30000), ("string", "30000"), ("zero", 0),
         ("negative", -1), ("fractional", 30000.5), ("invalid", "slow"),
         ("int64-overflow", 2**63)]


def collect(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.xla_metadata import set_xla_metadata
    from jaxlib import _hlo

    jax.config.update("jax_enable_compilation_cache", False)
    rng = np.random.default_rng(20260915)
    a = (rng.standard_normal((4, 8)) * .25).astype(np.float32)
    w = (rng.standard_normal((8, 6)) * .25).astype(np.float32)
    np.savez(output / "inputs.npz", a=a, w=w)
    args = jax.device_put(a), jax.device_put(w)
    reference = a.astype(np.float64) @ w.astype(np.float64)
    cases = []
    for name, value in CASES:
        def matmul(a, w):
            with set_xla_metadata(**({} if value is None else {"latency_metadata": value})):
                return jnp.matmul(a, w, precision=jax.lax.Precision.HIGHEST)
        directory = output / name
        directory.mkdir()
        lowered = jax.jit(matmul).lower(*args)
        (directory / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
        exported = lowered.as_text("hlo")
        (directory / "exported-hlo.txt").write_text(exported)
        compiled = lowered.compile()
        optimized = compiled.as_text()
        (directory / "optimized-hlo.txt").write_text(optimized)
        actual = np.asarray(compiled(*args))
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        np.save(directory / "output.npy", actual, allow_pickle=False)
        attrs = {}
        for phase, text in [("exported", exported), ("optimized", optimized)]:
            module = _hlo.hlo_module_from_text(text)
            attrs[phase] = [{"name": i.name, "opcode": i.opcode.name,
                             "value": i.get_frontend_attribute("latency_metadata")}
                            for c in module.computations() for i in c.instructions()
                            if i.get_frontend_attribute("latency_metadata") is not None]
            if value is None:
                assert not attrs[phase]
            else:
                assert attrs[phase] and all(i["value"] == str(value) for i in attrs[phase])
        summary = {"case": name, "annotation_input": value, "python_type": type(value).__name__,
                   "lowered_cost": lowered.cost_analysis(), "compiled_cost": compiled.cost_analysis(),
                   "attributes": attrs, "max_absolute_error": float(np.max(np.abs(actual-reference)))}
        write_json(directory / "summary.json", summary)
        cases.append(summary)
    for case in cases:
        assert case["lowered_cost"] == cases[0]["lowered_cost"]
        assert case["compiled_cost"] == cases[0]["compiled_cost"]
    assert cases[0]["lowered_cost"]["flops"] == 384
    assert cases[0]["lowered_cost"]["bytes accessed"] == 416
    limits = ["Executed only the CPU metadata/export/compile/cost path. No GPU/TPU latency estimator or latency parser was executed.",
              "Successful propagation, including invalid/negative/overflow strings, is not validation of latency values or proof of scheduling use.",
              "latency_metadata is a recognized key in inspected GPU estimator source, but does not define FLOPs or bytes for this CPU cost API.",
              "No timing, speedup, interconnect model, real inference split or TPU runtime behavior is established.",
              "Native observations remain VERSION-SKEW until matching source-built jaxlib is loaded and rechecked."]
    write_json(output / "summary.json", {"outcome": "pass", "cases": cases, "limits": limits})
    spec = importlib.util.spec_from_file_location("baseline_capture", ROOT / "tools/capture-baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    env = baseline.capture_baseline()
    assert all(not s["dirty"] for s in env["repository"]["sources"].values())
    write_json(output / "environment.json", env)
    return cases, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("Use a new artifacts/jax-stack directory and unset XLA_FLAGS")
    os.environ.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    with (output / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            cases, env = collect(output)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass", "evidence_level": "RUN-CPU",
        "qualifiers": env["status"]["qualifiers"], "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [".venv/bin/python", "-B", "research/jax-stack/latency_metadata_probe.py", "--output", str(output.relative_to(ROOT))],
                     "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "cases": [c["case"] for c in cases],
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture": output.name, "cases": len(cases), "qualifiers": env["status"]["qualifiers"]}))


if __name__ == "__main__":
    main()
