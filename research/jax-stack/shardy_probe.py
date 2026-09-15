#!/usr/bin/env python3
"""Compare source-built Shardy and GSPMD on the same two-CPU matmul."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from capture_runtime import preflight, environment
from cpu_thunk_probe import check_environment_transition
from matmul_probe import ROOT, fingerprint, write_json


def collect(output, build_manifest, mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    jax.config.update("jax_enable_compilation_cache", False)
    jax.config.update("jax_use_shardy_partitioner", mode == "shardy")
    devices = jax.devices("cpu")
    assert len(devices) == 2 and jax.process_count() == 1
    initial = environment(output, build_manifest)
    write_json(output / "environment-before.json", initial)
    shutil.copy2(output / "build-binding.json", output / "build-binding-before.json")
    mesh = Mesh(np.array(devices), ("rows",))
    row = NamedSharding(mesh, P("rows", None))
    replicated = NamedSharding(mesh, P())
    rng = np.random.default_rng(20260915)
    a = (rng.standard_normal((8, 16)) * .125).astype(np.float32)
    w = (rng.standard_normal((16, 12)) * .125).astype(np.float32)
    bias = (rng.standard_normal((12,)) * .125).astype(np.float32)
    np.savez(output / "inputs.npz", a=a, w=w, bias=bias)
    args = tuple(jax.device_put(v, s) for v,s in zip((a,w,bias), (row,replicated,replicated), strict=True))

    def row_matmul(a, w, bias):
        return jnp.maximum(jnp.matmul(a, w, precision="highest") + bias, 0)

    lowered = jax.jit(row_matmul, in_shardings=(row,replicated,replicated), out_shardings=row).lower(*args)
    (output / "stablehlo.mlir").write_text(lowered.as_text("stablehlo"))
    (output / "exported-hlo.txt").write_text(lowered.as_text("hlo"))
    compiled = lowered.compile()
    (output / "optimized-hlo.txt").write_text(compiled.as_text())
    reference = np.maximum(a.astype(np.float64) @ w.astype(np.float64) + bias.astype(np.float64), 0)
    errors, shards, saved = [], [], {}
    for invocation in range(3):
        result = compiled(*args)
        result.block_until_ready()
        actual = np.asarray(result)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        saved[f"result_{invocation}"] = actual
        errors.append(float(np.max(np.abs(actual-reference))))
        if invocation == 0:
            for shard in result.addressable_shards:
                piece = np.asarray(shard.data)
                np.testing.assert_allclose(piece, reference[shard.index], rtol=2e-5, atol=2e-5)
                index = [[s.start, s.stop, s.step] for s in shard.index]
                shards.append({"device_id": shard.device.id, "index": index, "shape": list(piece.shape)})
                saved[f"shard_{shard.device.id}"] = piece
    jax.effects_barrier()
    np.savez(output / "outputs.npz", **saved)
    after = environment(output, build_manifest)
    write_json(output / "environment.json", after)
    added = check_environment_transition(initial, after)
    summary = {"mode": mode, "devices": [str(d) for d in devices], "device_count": len(devices),
               "process_count": jax.process_count(), "jax_use_shardy_partitioner": jax.config.jax_use_shardy_partitioner,
               "shape": [8,12], "input_shapes": [[8,16],[16,12],[12]], "output_shards": shards,
               "invocations": 3, "max_absolute_errors": errors, "newly_loaded_native_libraries": added,
               "limits": ["Two logical CPU devices in one process; not TPU or multi-host execution.",
                          "Numerical/IR sharding comparison; no communication overlap or memory/performance measurement.",
                          "Explicit row-sharded input/output with replicated weight/bias; not automatic partition search."]}
    write_json(output / "summary.json", summary)
    return after, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["shardy","gspmd"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaxlib-build-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack") or os.environ.get("XLA_FLAGS"):
        parser.error("choose a fresh artifacts/jax-stack output; unset XLA_FLAGS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count=2 --xla_dump_to={output}/xla-dump "
        "--xla_dump_hlo_as_text --xla_dump_hlo_pass_re=shardy-xla|sharding-propagation|spmd-partitioning")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    shutil.copy2(Path(__file__).with_name("cpu_thunk_probe.py"), output / "environment-check-helper.py")
    build_manifest = preflight(output, args.jaxlib_build_manifest)
    started = datetime.now(timezone.utc).isoformat()
    observed, summary = collect(output, build_manifest, args.mode)
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome":"pass", "evidence_level":"RUN-CPU",
        "qualifiers": observed["status"]["qualifiers"], "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[sys.executable,"-B",*sys.argv],"source":str((output/"producer.py").relative_to(ROOT)),**fingerprint(output/"producer.py")},
        "jaxlib_build_manifest":str(build_manifest.relative_to(ROOT)),
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob('*')) if p.is_file()]})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
