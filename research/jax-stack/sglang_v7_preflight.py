#!/usr/bin/env python3
"""Bounded TPU compatibility capture; random weights are not model acceptance."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, default=str) + '\n')

    devices = jax.devices()
    identity = {
        'python': platform.python_version(),
        'packages': {n: metadata.version(n) for n in ('jax', 'jaxlib', 'libtpu')},
        'devices': [{'id': d.id, 'kind': d.device_kind, 'platform': d.platform,
                     'process_index': d.process_index, 'coords': list(d.coords),
                     'core_on_chip': d.core_on_chip} for d in devices],
        'evidence_level': 'RUN-TPU', 'qualifiers': ['VERSION-SKEW'],
        'source_alignment': 'SGLang compatibility baseline; not pinned research JAX/XLA',
    }
    save('identity.json', identity)
    assert identity['packages'] == {'jax': '0.8.1', 'jaxlib': '0.8.1', 'libtpu': '0.0.30'}
    assert len(devices) == 8 and all(d.platform == 'tpu' for d in devices)
    assert len({tuple(d.coords) for d in devices}) == 4
    assert all('7' in d.device_kind for d in devices)
    natives = []
    for package in ('jaxlib', 'libtpu'):
        dist = metadata.distribution(package)
        for f in dist.files or []:
            if str(f).endswith('.so'):
                p = Path(dist.locate_file(f))
                natives.append({'package': package, 'path': str(f), 'size': p.stat().st_size,
                                'sha256': digest(p)})
    save('native-packages.json', natives)
    save('packages.json', sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))

    rng = np.random.default_rng(20260915)
    a = (rng.normal(size=(128, 128)) * .02).astype(np.float32)
    b = (rng.normal(size=(128, 128)) * .02).astype(np.float32)
    np.savez(output / 'inputs.npz', a=a, b=b)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    rows = []
    for d in devices:
        x, w = jax.device_put((a, b), d)
        y = jax.jit(lambda x, w: jnp.matmul(x, w, precision=jax.lax.Precision.HIGHEST))(x, w)
        actual = np.asarray(y.block_until_ready())
        np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=2e-5)
        rows.append({'device': d.id, 'max_abs_error': float(np.max(np.abs(actual-ref)))})
    save('regular-matmul.json', rows)
    print('EIGHT_DEVICE_MATMUL_OK', flush=True)

    def kernel(x, w, y):
        with jax.named_scope('research_v7_kernel'):
            with jax.named_scope('research_v7_dot'):
                z = jnp.matmul(x[...], w[...], precision=jax.lax.Precision.HIGHEST)
            with jax.named_scope('research_v7_store'):
                y[...] = z

    fn = jax.jit(pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32),
        grid=(1,), in_specs=[pl.BlockSpec((128, 128), lambda i: (0, 0))] * 2,
        out_specs=pl.BlockSpec((128, 128), lambda i: (0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',))))
    x, w = jax.device_put((a, b), devices[0])
    lowered = fn.lower(x, w)
    (output / 'pallas-stablehlo.mlir').write_text(lowered.as_text())
    executable = lowered.compile()
    (output / 'pallas-optimized-hlo.txt').write_text(executable.as_text())
    actual = np.asarray(executable(x, w).block_until_ready())
    np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=2e-5)
    np.save(output / 'pallas-output.npy', actual, allow_pickle=False)
    with jax.profiler.trace(str(output / 'profile'), create_perfetto_trace=False):
        for step in range(5):
            with jax.profiler.StepTraceAnnotation('research_v7_step', step_num=step):
                executable(x, w).block_until_ready()
    save('pallas.json', {'numerical_pass': True, 'max_abs_error': float(np.max(np.abs(actual-ref))),
                         'profile_captured': True, 'device_events_verified': False})
    print('PALLAS_EXECUTION_OK_PROFILE_REQUIRES_AUDIT', flush=True)

    from sgl_jax.srt.models.qwen3 import Qwen3MLP
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh
    mesh = create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
    with jax.set_mesh(mesh):
        model = Qwen3MLP(4096, 12288, mesh=mesh, dtype=jnp.bfloat16)
        x = jax.device_put(np.zeros((128, 4096), np.float32), NamedSharding(mesh, P('data', None)))
        x = x.astype(jnp.bfloat16)
        start = time.monotonic()
        y = nnx.jit(lambda m, x: m(x))(model, x)
        actual = np.asarray(y.block_until_ready())
        assert actual.shape == (128, 4096) and np.array_equal(actual, np.zeros_like(actual))
        save('sglang-mlp-smoke.json', {'shape': list(actual.shape), 'tp': 8,
              'seconds_including_compile': time.monotonic()-start,
              'weights': 'random initialization', 'inputs': 'zeros',
              'model_acceptance': False, 'scope': 'unmodified Qwen3MLP import/compile/TP8 execution only'})
    maps = Path('/proc/self/maps').read_text().splitlines()
    loaded = sorted({line.split()[-1] for line in maps if '/' in line and
                     ('libtpu' in line or 'jaxlib' in line)})
    save('loaded-native.json', [{'path': p, 'sha256': digest(Path(p))} for p in loaded if Path(p).is_file()])
    print('SGLANG_V7_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
