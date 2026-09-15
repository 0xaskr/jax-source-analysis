#!/usr/bin/env python3
"""Fresh-process default/enabled libtpu custom-region trace comparison."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260915)
    a, b = [(rng.normal(size=(128, 128)) * .02).astype(np.float32) for _ in range(2)]
    np.savez(output / 'inputs.npz', a=a, b=b)
    reference = a.astype(np.float64) @ b.astype(np.float64)
    values = jax.device_put((a, b), jax.devices()[0])
    rows = []
    for named in (False, True):
        case = output / ('named' if named else 'plain')
        case.mkdir()
        def scope(name):
            return jax.named_scope(name) if named else nullcontext()
        def kernel(x, w, y):
            with scope('research_v7_kernel'):
                with scope('research_v7_dot'):
                    z = jnp.matmul(x[...], w[...], precision=jax.lax.Precision.HIGHEST)
                with scope('research_v7_store'):
                    y[...] = z
        fn = jax.jit(pl.pallas_call(kernel, name='research_region_matmul',
            out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32), grid=(1,),
            in_specs=[pl.BlockSpec((128, 128), lambda i: (0, 0))] * 2,
            out_specs=pl.BlockSpec((128, 128), lambda i: (0, 0)),
            compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',))))
        lowered = fn.lower(*values)
        (case / 'stablehlo.mlir').write_text(lowered.as_text())
        executable = lowered.compile()
        (case / 'optimized-hlo.txt').write_text(executable.as_text())
        actual = np.asarray(executable(*values).block_until_ready())
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        np.save(case / 'output.npy', actual, allow_pickle=False)
        with jax.profiler.trace(str(case / 'profile')):
            for step in range(5):
                with jax.profiler.StepTraceAnnotation('research_region_step', step_num=step):
                    executable(*values).block_until_ready()
        rows.append({'case': case.name, 'max_abs_error': float(np.max(np.abs(actual-reference)))})
    (output / 'result.json').write_text(json.dumps({
        'packages': {n: metadata.version(n) for n in ('jax', 'jaxlib', 'libtpu')},
        'libtpu_init_args': os.environ.get('LIBTPU_INIT_ARGS', ''),
        'device': str(jax.devices()[0]), 'evidence_level': 'RUN-TPU',
        'qualifiers': ['VERSION-SKEW'], 'cases': rows, 'device_regions_verified': False,
        'loaded_libtpu': [{'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                          for p in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                           if '/libtpu/' in line and line.split()[-1].endswith('.so')})],
    }, indent=2) + '\n')
    print('TPU_REGION_CAPTURE_OK', output.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
