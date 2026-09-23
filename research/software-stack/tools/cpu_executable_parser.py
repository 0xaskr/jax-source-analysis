#!/usr/bin/env python3
"""Bounded data-only inspection of this pinned CPU executable packaging format."""

import hashlib
import json
import math
import os
from pathlib import Path
import pickletools
import sys

from verify_research import ROOT, check, local_path, read_json, sha256

SCHEMA = ROOT / "artifacts/jax-stack/cpu-thunk-schema-002"


def inventory(capture, level, manifest=None):
    manifest = read_json(capture / "manifest.json") if manifest is None else manifest
    check(manifest["outcome"] == "pass" and manifest["evidence_level"] == level,
          "wrong capture outcome or evidence level")
    check(manifest["qualifiers"] == [], "source-aligned evidence required")
    paths = set()
    for item in manifest["artifacts"]:
        path = local_path(item["path"])
        check(path.is_relative_to(capture) and path not in paths, "invalid inventory path")
        check(path.is_file() and path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"],
              "artifact hash or size differs")
        paths.add(path)
    check(paths == {p for p in capture.rglob("*") if p.is_file() and p != capture / "manifest.json"},
          "incomplete artifact inventory")
    return {"path": str((capture / "manifest.json").relative_to(ROOT)),
            "sha256": sha256(capture / "manifest.json"), "artifact_count": len(paths)}


def schema_pool(schema=SCHEMA):
    inventory(schema, "SOURCE-ONLY")
    for item in read_json(schema / "source-inputs.json"):
        check(sha256(local_path(item["source_path"])) == item["sha256"] ==
              sha256(local_path(item["snapshot_path"])), "schema source changed")
    generation = read_json(schema / "generation.json")
    check(sha256(Path(generation["argv"][0])) == generation["protoc_sha256"], "protoc binary changed")
    check((schema / "cpu-executable.descriptor.pb").read_bytes() == (schema / "replayed.descriptor.pb").read_bytes(),
          "schema replay differs")
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    sys.path.insert(0, str(schema / "python-tool"))
    import google.protobuf
    from google.protobuf import descriptor_pb2, descriptor_pool
    from google.protobuf.internal import api_implementation
    check(Path(google.protobuf.__file__).resolve().is_relative_to(schema / "python-tool") and
          google.protobuf.__version__ == "7.34.0" and api_implementation.Type() == "python",
          "unexpected protobuf analysis runtime")
    files = descriptor_pb2.FileDescriptorSet.FromString((schema / "cpu-executable.descriptor.pb").read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file in files.file:
        pool.Add(file)
    check(len(files.file) == 28, "unexpected schema closure")
    return pool


def message(pool, name, payload):
    from google.protobuf import message_factory
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(name))
    obj = cls()
    check(obj.ParseFromString(payload) == len(payload), "incomplete protobuf parse")
    known = cls()
    known.CopyFrom(obj)
    known.DiscardUnknownFields()
    check(obj.SerializeToString(deterministic=True) == known.SerializeToString(deterministic=True),
          "unknown fields outside pinned schema")
    return obj


def extract_exec(package):
    check(0 < len(package) <= 16 * 1024 * 1024, "package size outside bounded audit")
    # genops decodes the opcode stream as data. No Unpickler, globals, reducers,
    # constructors, persistent_load callbacks or native loader are invoked.
    ops = list(pickletools.genops(package))
    check(ops[0][0].name == "PROTO" and ops[0][1] == 4 and
          ops[-1][0].name == "STOP" and ops[-1][2] == len(package) - 1,
          "unsupported or trailing pickle stream")
    candidates = [(i, op, value, pos) for i, (op, value, pos) in enumerate(ops)
                  if op.name in {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8"}]
    check(len(candidates) == 1, "ambiguous executable byte payload")
    i, op, value, pos = candidates[0]
    check(i >= 2 and ops[i-2][0].name == "SHORT_BINUNICODE" and ops[i-2][1] == "exec" and
          ops[i-1][0].name == "MEMOIZE" and
          [o.name for o, _, _ in ops[i+1:i+5]] == ["MEMOIZE", "TUPLE2", "MEMOIZE", "BINPERSID"],
          "bytes are not the supported exec persistent-id pattern")
    header = {"BINBYTES": 5, "SHORT_BINBYTES": 2, "BINBYTES8": 9}[op.name]
    check(package[pos+header:pos+header+len(value)] == value, "pickle byte location differs")
    return value, {"pickle_protocol": 4, "pickle_opcode": op.name,
                   "payload_offset": pos + header, "payload_size": len(value),
                   "payload_sha256": hashlib.sha256(value).hexdigest(),
                   "scope": "Opcode scanning only; no pickle execution."}


def delimited(payload):
    length = 0
    for i, value in enumerate(payload[:5]):
        length |= (value & 127) << (7 * i)
        if value < 128:
            check((i == 0 or length >= 1 << (7*i)) and 0 < length < len(payload) - i,
                  "invalid or truncated metadata length")
            return payload[i+1:i+1+length], payload[i+1+length:], i+1
    raise ValueError("invalid metadata length varint")


def decode_package(package, pool):
    payload, packaging = extract_exec(package)
    meta_bytes, pjrt_bytes, prefix_size = delimited(payload)
    meta = message(pool, "xla.ifrt.SerializedXlaExecutableMetadata", meta_bytes)
    wrapper = message(pool, "xla.ExecutableAndOptionsProto", pjrt_bytes)
    check(meta.runtime_name == "pjrt_ifrt" and meta.ifrt_version_number == 3 and
          wrapper.HasField("compile_options") and bool(wrapper.serialized_executable), "unexpected IFRT/PJRT wrapper")
    cpu = message(pool, "xla.cpu.CompilationResultProto", wrapper.serialized_executable)
    check(cpu.obj_files_kind == 2 and cpu.hlo_module.hlo_module.name == meta.computation_name,
          "unexpected CPU executable kind or computation")
    packaging.update(metadata_length_prefix_bytes=prefix_size, metadata_bytes=len(meta_bytes),
                     pjrt_bytes=len(pjrt_bytes), cpu_proto_bytes=len(wrapper.serialized_executable))
    return cpu, meta, packaging, {"ifrt-metadata.pb": meta_bytes, "pjrt-wrapper.pb": pjrt_bytes,
                                 "cpu-compilation.pb": wrapper.serialized_executable}


def project(cpu, package):
    from google.protobuf import json_format
    module = cpu.hlo_module.hlo_module
    entries = [c for c in module.computations if c.id == module.entry_computation_id]
    check(len(entries) == 1, "missing unique entry computation")
    by_name = {i.name: i for i in entries[0].instructions}
    by_id = {i.id: i for c in module.computations for i in c.instructions}
    allocations = {a.index: a.size for a in cpu.buffer_assignment.buffer_allocations}
    symbols = {s.name: s.function_type_id for s in cpu.compiled_symbols}
    check(len(symbols) == len(cpu.compiled_symbols), "duplicate compiled symbol")

    def slice_record(item):
        shape, value = item.shape, item.slice
        check(shape.element_type == 11, "bounded matmul audit requires f32 slices")
        check(value.size == math.prod(shape.dimensions) * 4 and value.offset >= 0 and
              value.buffer_allocation_index in allocations and
              value.offset + value.size <= allocations[value.buffer_allocation_index], "invalid shaped allocation slice")
        return {"shape": list(shape.dimensions), "layout": list(shape.layout.minor_to_major),
                "allocation": value.buffer_allocation_index, "offset": value.offset, "size": value.size}

    thunks = []
    for thunk in cpu.thunk_sequence.thunks:
        impl = thunk.WhichOneof("impl")
        check({"dot_thunk": "dot", "kernel_thunk": "kernel", "ynn_fusion_thunk": "ynn-fusion"}.get(impl) == thunk.kind,
              "unsupported or inconsistent thunk kind/oneof")
        check(thunk.info.module_name == module.name and thunk.info.module_id == module.id and
              thunk.info.op_name in by_name, "thunk module or HLO instruction binding differs")
        hlo = by_name[thunk.info.op_name]
        record = {"kind": thunk.kind, "impl": impl, "op_name": thunk.info.op_name,
                  "hlo_instruction_id": hlo.id, "hlo_opcode": hlo.opcode}
        body = getattr(thunk, impl)
        if impl == "dot_thunk":
            check(hlo.opcode == "dot" and body.dot_dimensions == hlo.dot_dimension_numbers, "dot dimensions differ from HLO")
            operands = [by_id[i] for i in hlo.operand_ids]
            check(len(operands) == 2 and body.lhs_buffer_shape.shape == operands[0].shape and
                  body.rhs_buffer_shape.shape == operands[1].shape and body.out_buffer_shape.shape == hlo.shape,
                  "dot shapes differ from HLO")
            record.update(dimensions=json_format.MessageToDict(body.dot_dimensions, preserving_proto_field_name=True),
                          lhs=slice_record(body.lhs_buffer_shape), rhs=slice_record(body.rhs_buffer_shape),
                          output=slice_record(body.out_buffer_shape))
        elif impl == "ynn_fusion_thunk":
            check(hlo.opcode == "fusion" and body.instruction_id == hlo.id and b"__ynn_fusion" in hlo.backend_config,
                  "YNN instruction identity differs")
            check([s.shape for s in body.arguments_shapes] == [by_id[i].shape for i in hlo.operand_ids] and
                  [s.shape for s in body.results_shapes] == [hlo.shape], "YNN argument/result shapes differ from HLO")
            record.update(arguments=[slice_record(s) for s in body.arguments_shapes],
                          results=[slice_record(s) for s in body.results_shapes])
        else:
            check(hlo.opcode == "fusion" and body.kernel_name == hlo.name and symbols.get(body.kernel_name) == 1,
                  "kernel symbol does not match HLO and compiled symbol list")
            check([s.shape for s in body.arguments_buffers] == [by_id[i].shape for i in hlo.operand_ids] and
                  [s.shape for s in body.results_buffers] == [hlo.shape], "kernel argument/result shapes differ from HLO")
            record.update(kernel_name=body.kernel_name,
                          workgroups=[body.num_workgroups.x, body.num_workgroups.y, body.num_workgroups.z],
                          arguments=[slice_record(s) for s in body.arguments_buffers],
                          results=[slice_record(s) for s in body.results_buffers])
        thunks.append(record)
    check(set(symbols) == {t["op_name"] for t in thunks if t["kind"] == "kernel"}, "kernel/symbol inventory differs")
    objects = []
    for obj in cpu.object_files:
        check(obj.contents.startswith(b"\x7fELF\x02\x01") and package.count(obj.contents) == 1,
              "object not uniquely embedded as ELF64")
        objects.append({"name": obj.name, "size_bytes": len(obj.contents),
                        "sha256": hashlib.sha256(obj.contents).hexdigest(), "package_offset": package.find(obj.contents)})
    return {"module_name": module.name, "module_id": module.id, "entry_computation": entries[0].name,
            "object_kind": "KERNELS", "compiled_symbols": symbols, "objects": objects, "thunks": thunks}
