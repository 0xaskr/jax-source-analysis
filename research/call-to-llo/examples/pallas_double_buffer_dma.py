"""One grid=() Pallas call with manual ping-pong VMEM buffers and async DMA.

Run from the repository root:
  JAX_PLATFORMS=cpu .venv/bin/python -B \
      research/call-to-llo/examples/pallas_double_buffer_dma.py

The default run dumps non-interpret TPU lowering and separately checks the
kernel with the TPU interpreter on CPU. It does not compile or run on a TPU.
"""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path

import jax
from jax import lax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir, passmanager
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as tpu_interpret
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.mosaic.dialects import tpu


TILE_ROWS = 8
TILE_COLS = 128


def make_add(num_tiles, *, interpret=False):
    if num_tiles < 1:
        raise ValueError("num_tiles must be positive")

    def kernel(x_hbm, y_hbm, out_hbm, x_buf, y_buf, out_buf,
               x_sem, y_sem, out_sem):
        def load_x(tile):
            slot = lax.rem(tile, 2)
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(tile * TILE_ROWS, TILE_ROWS), :],
                x_buf.at[slot], x_sem.at[slot],
            )

        def load_y(tile):
            slot = lax.rem(tile, 2)
            return pltpu.make_async_copy(
                y_hbm.at[pl.ds(tile * TILE_ROWS, TILE_ROWS), :],
                y_buf.at[slot], y_sem.at[slot],
            )

        def store(tile):
            slot = lax.rem(tile, 2)
            return pltpu.make_async_copy(
                out_buf.at[slot],
                out_hbm.at[pl.ds(tile * TILE_ROWS, TILE_ROWS), :],
                out_sem.at[slot],
            )

        # Fill slot 0 before the loop. Subsequent input DMAs start one tile ahead.
        load_x(0).start()
        load_y(0).start()

        def step(i, unused):
            slot = lax.rem(i, 2)
            load_x(i).wait()
            load_y(i).wait()

            @pl.when(i + 1 < num_tiles)
            def prefetch():
                load_x(i + 1).start()
                load_y(i + 1).start()

            # The old store may still be reading this output slot.
            @pl.when(i >= 2)
            def release_output_slot():
                store(i - 2).wait()

            out_buf[slot, ...] = x_buf[slot, ...] + y_buf[slot, ...]
            store(i).start()
            return unused

        lax.fori_loop(0, num_tiles, step, ())

        # Earlier stores were consumed by the slot-reuse waits in the loop.
        for tile in range(max(0, num_tiles - 2), num_tiles):
            store(tile).wait()

    shape = jax.ShapeDtypeStruct((num_tiles * TILE_ROWS, TILE_COLS), jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=shape,
        grid=(),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),
                  pl.BlockSpec(memory_space=pltpu.HBM)),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((2, TILE_ROWS, TILE_COLS), jnp.float32),
            pltpu.VMEM((2, TILE_ROWS, TILE_COLS), jnp.float32),
            pltpu.VMEM((2, TILE_ROWS, TILE_COLS), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        interpret=interpret,
        name="manual_double_buffer_add",
    )


def walk_operations(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk_operations(child.operation)


def pallas_equations(value):
    if isinstance(value, (core.Jaxpr, core.ClosedJaxpr)):
        for equation in value.jaxpr.eqns:
            if equation.primitive.name == "pallas_call":
                yield equation
            yield from pallas_equations(equation.params)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from pallas_equations(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from pallas_equations(nested)


def capture(num_tiles, output):
    shape = jax.ShapeDtypeStruct((num_tiles * TILE_ROWS, TILE_COLS), jnp.float32)
    add = make_add(num_tiles)
    jaxpr = jax.make_jaxpr(add)(shape, shape)
    calls = list(pallas_equations(jaxpr))
    assert len(calls) == 1
    assert calls[0].params["grid_mapping"].grid == ()
    lowered = jax.jit(add).trace(shape, shape).lower(lowering_platforms=("tpu",))
    outer = lowered.compiler_ir("stablehlo")
    assert outer.operation.verify()
    outer_ops = list(walk_operations(outer.operation))
    custom_calls = [op for op in outer_ops if op.name == "stablehlo.custom_call"]
    assert len(custom_calls) == 1
    assert not any(op.name in ("stablehlo.add", "stablehlo.while") for op in outer_ops)
    call = custom_calls[0]
    assert ir.StringAttr(call.attributes["call_target_name"]).value == "tpu_custom_call"
    config = json.loads(ir.StringAttr(call.attributes["backend_config"]).value)
    body = config["custom_call_config"]["body"]
    payload = base64.b64decode(body, validate=True)

    ctx = mlir.make_ir_context()
    tpu.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
        mosaic = ir.Module.parse(payload)
        pm = passmanager.PassManager.parse("builtin.module(mosaic-serde{serialize=false})")
        pm.run(mosaic.operation)
        assert mosaic.operation.verify()
        counts = Counter(op.name for op in walk_operations(mosaic.operation))
        assert counts["scf.for"] == 1
        assert counts["tpu.enqueue_dma"] > 0
        assert counts["tpu.wait_dma2"] > 0
        assert counts["arith.addf"] == 1
        mosaic_text = mosaic.operation.get_asm(enable_debug_info=False)

    output.with_suffix(".stablehlo.mlir").write_text(
        outer.operation.get_asm(enable_debug_info=False) + "\n")
    output.with_suffix(".mosaic.mlir").write_text(mosaic_text + "\n")
    return {
        "num_tiles": num_tiles,
        "shape": list(shape.shape),
        "grid": [],
        "pallas_calls": len(calls),
        "outer_custom_calls": len(custom_calls),
        "payload_bytes": len(payload),
        "base64_characters": len(body),
        "vmem_buffer_shapes": [[2, TILE_ROWS, TILE_COLS]] * 3,
        "vmem_buffer_bytes": 3 * 2 * TILE_ROWS * TILE_COLS * 4,
        "dma_semaphores": 6,
        "mosaic_static_op_counts": dict(sorted(counts.items())),
    }


def check_interpreter():
    cases = []
    for mode in ("eager", "on_wait"):
        for num_tiles in (1, 2, 5, 6):
            shape = (num_tiles * TILE_ROWS, TILE_COLS)
            rng = np.random.default_rng(num_tiles)
            x = rng.integers(-128, 128, size=shape).astype(np.float32) / 8
            y = rng.integers(-128, 128, size=shape).astype(np.float32) / 4
            add = make_add(num_tiles, interpret=pltpu.InterpretParams(
                detect_races=True, dma_execution_mode=mode,
                uninitialized_memory="nan", buffer_bounds="logical",
            ))
            actual = np.asarray(jax.jit(add)(x, y).block_until_ready())
            np.testing.assert_array_equal(actual, x + y)
            assert tpu_interpret.races is not None
            assert not tpu_interpret.races.races_found
            case = {"tiles": num_tiles, "dma_mode": mode,
                    "exact_match": True, "races_found": False}
            cases.append(case)
            print("SIM-TPU", json.dumps(case), flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=int, default=5)
    parser.add_argument("--skip-interpret", action="store_true")
    args = parser.parse_args()
    output = Path(__file__).resolve().with_suffix("")
    report = {
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "host_backend": jax.default_backend(),
        "lowering_platform": "tpu",
        "tpu_compiled": False, "tpu_executed": False,
        "hardware_overlap_measured": False,
        "lowering": capture(args.tiles, output),
    }
    print("OFFLINE-LOWERING", json.dumps(report), flush=True)
    report["interpreter_cases"] = [] if args.skip_interpret else check_interpreter()
    output.with_suffix(".results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
