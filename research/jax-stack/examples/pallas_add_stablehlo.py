"""Dump a TPU Pallas add kernel without compiling or executing on a TPU.

From the repository root:
  JAX_PLATFORMS=cpu .venv/bin/python -B \
      research/jax-stack/examples/pallas_add_stablehlo.py
"""

import base64
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir, passmanager
from jax.experimental import pallas as pl
from jax.experimental.mosaic.dialects import tpu


def add_kernel(x_ref, y_ref, out_ref):
    out_ref[...] = x_ref[...] + y_ref[...]


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def main():
    shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    add = pl.pallas_call(add_kernel, out_shape=shape, name="pallas_add")
    lowered = jax.jit(add).trace(shape, shape).lower(lowering_platforms=("tpu",))
    stablehlo = lowered.compiler_ir("stablehlo")
    assert stablehlo.operation.verify()

    ops = list(walk(stablehlo.operation))
    calls = [op for op in ops if op.name == "stablehlo.custom_call"]
    assert len(calls) == 1
    assert not any(op.name == "stablehlo.add" for op in ops)
    call = calls[0]
    assert ir.StringAttr(call.attributes["call_target_name"]).value == "tpu_custom_call"
    config = json.loads(ir.StringAttr(call.attributes["backend_config"]).value)
    body = config["custom_call_config"]["body"]
    payload = base64.b64decode(body, validate=True)

    # Decode the actual custom_call payload, rather than separately lowering again.
    ctx = mlir.make_ir_context()
    tpu.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
        mosaic = ir.Module.parse(payload)
        pm = passmanager.PassManager.parse("builtin.module(mosaic-serde{serialize=false})")
        pm.run(mosaic.operation)
        assert mosaic.operation.verify()
        assert sum(op.name == "arith.addf" for op in walk(mosaic.operation)) == 1
        mosaic_text = mosaic.operation.get_asm(enable_debug_info=False)

    output = Path(__file__).resolve().parent
    (output / "pallas_add.stablehlo.mlir").write_text(
        stablehlo.operation.get_asm(enable_debug_info=False) + "\n"
    )
    (output / "pallas_add.mosaic.mlir").write_text(mosaic_text + "\n")
    print(json.dumps({
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "host_backend": jax.default_backend(),
        "lowering_platform": "tpu",
        "payload_bytes": len(payload),
        "base64_characters": len(body),
        "outer_custom_calls": len(calls),
        "decoded_addf_count": 1,
        "tpu_compiled": False,
        "tpu_executed": False,
        "output_directory": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
