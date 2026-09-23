#!/usr/bin/env python3
"""Read raw XSpace contexts without using event names as correlation keys."""

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from cpu_executable_parser import SCHEMA as TOOL_SCHEMA, schema_pool, message
from matmul_probe import ROOT, fingerprint, write_json
from verify_research import check, read_json, sha256

PROTO = ROOT / "upstream/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto"
CONTEXT_FIELDS = {"_pt", "_p", "_ct", "_c", "_pid"}
FILTERED_FIELDS = {"_pt", "_p", "_ct", "_c", "program_id"}


def prepare_schema(output):
    pool = schema_pool()
    from google.protobuf import descriptor_pb2
    source = output / "xplane.proto"
    shutil.copy2(PROTO, source)
    protoc = read_json(TOOL_SCHEMA / "generation.json")["argv"][0]
    argv = [protoc, f"-I{PROTO.parent}", "--include_imports",
            f"--descriptor_set_out={output}/xplane.descriptor.pb", "xplane.proto"]
    subprocess.run(argv, capture_output=True, check=True)
    replay = [protoc, f"-I{output}", "--include_imports",
              f"--descriptor_set_out={output}/xplane.replayed.pb", "xplane.proto"]
    subprocess.run(replay, capture_output=True, check=True)
    check((output / "xplane.descriptor.pb").read_bytes() == (output / "xplane.replayed.pb").read_bytes(), "XPlane descriptor replay differs")
    write_json(output / "schema.json", {"source": str(PROTO.relative_to(ROOT)), **fingerprint(PROTO),
        "argv": argv, "replay_argv": replay, "tool_schema": str(TOOL_SCHEMA.relative_to(ROOT)),
        "tool_schema_manifest_sha256": sha256(TOOL_SCHEMA / "manifest.json")})
    files = descriptor_pb2.FileDescriptorSet.FromString((output / "xplane.descriptor.pb").read_bytes())
    check(len(files.file) == 1, "unexpected XSpace schema closure")
    pool.Add(files.file[0])
    return pool


def load_schema(capture):
    pool = schema_pool()
    from google.protobuf import descriptor_pb2
    record = read_json(capture / "schema.json")
    check(record["source"] == str(PROTO.relative_to(ROOT)) and sha256(PROTO) == record["sha256"] == sha256(capture / "xplane.proto"), "XPlane schema source changed")
    check(record["tool_schema_manifest_sha256"] == sha256(TOOL_SCHEMA / "manifest.json"), "protobuf analysis tool changed")
    check((capture / "xplane.descriptor.pb").read_bytes() == (capture / "xplane.replayed.pb").read_bytes(), "descriptor replay changed")
    files = descriptor_pb2.FileDescriptorSet.FromString((capture / "xplane.descriptor.pb").read_bytes())
    check(len(files.file) == 1, "unexpected XSpace schema closure")
    pool.Add(files.file[0])
    return pool


def integer(stat, unsigned=False):
    check(stat["kind"] in {"int64_value", "uint64_value"} and isinstance(stat["value"], str) and
          re.fullmatch(r"-?[0-9]+", stat["value"]), "context requires exact integer stat")
    value = int(stat["value"])
    if stat["kind"] == "uint64_value":
        check(0 <= value < 1 << 64, "invalid uint64 stat")
    else:
        check(-(1 << 63) <= value < 1 << 63, "invalid int64 stat")
    return value % (1 << 64) if unsigned else value


def decode_stats(plane, stats, duplicates=None):
    result = {}
    for stat in stats:
        check(stat.metadata_id in plane.stat_metadata, "missing stat metadata")
        name = plane.stat_metadata[stat.metadata_id].name
        kind = stat.WhichOneof("value")
        value = getattr(stat, kind) if kind else None
        if kind == "ref_value":
            check(value in plane.stat_metadata, "missing referenced stat value")
            value = plane.stat_metadata[value].name
        elif kind in {"int64_value", "uint64_value"}:
            value = str(value)  # Keep uint64 and signed timeline IDs lossless in JSON.
        elif kind == "bytes_value":
            value = {"size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        item = {"kind": kind, "value": value}
        if name in result:
            check(name not in CONTEXT_FIELDS or result[name] == item, "conflicting duplicate context stat")
            if duplicates is not None:
                duplicates.append({"name": name, "previous": result[name], "replacement": item})
        result[name] = item
    return result


def flatten(space):
    check(not space.errors and not space.warnings, "XSpace reports collection problems")
    check(len(space.hostnames) <= 1, "this audit requires one host namespace per XSpace")
    rows = []
    for pi, plane in enumerate(space.planes):
        check(all(key == value.id for key, value in plane.event_metadata.items()) and
              all(key == value.id for key, value in plane.stat_metadata.items()), "metadata map id mismatch")
        plane_stats = decode_stats(plane, plane.stats)
        pid = integer(plane_stats["process_id"]) if "process_id" in plane_stats else None
        for li, line in enumerate(plane.lines):
            for ei, event in enumerate(line.events):
                check(event.metadata_id in plane.event_metadata, "missing event metadata")
                metadata = plane.event_metadata[event.metadata_id]
                common_duplicates, occurrence_duplicates = [], []
                common = decode_stats(plane, metadata.stats, common_duplicates)
                occurrence = decode_stats(plane, event.stats, occurrence_duplicates)
                merged = {**common, **occurrence}
                # GroupingEventStats reads occurrence stats. JSON export merges
                # metadata stats first, then occurrence stats; keep both views.
                check(not CONTEXT_FIELDS.intersection(common), "metadata-only contexts need a separate grouping contract")
                display_name = metadata.display_name or metadata.name
                if "step_name" in merged:
                    display_name = merged["step_name"]["value"]
                rows.append({"node": f"{pi}:{li}:{ei}", "plane": plane.name, "plane_index": pi,
                    "process_id": pid, "line_id": str(line.id), "display_id": str(line.display_id),
                    "viewer_tid": (line.display_id or line.id) & 0xffffffff,
                    "name": metadata.name, "export_name": display_name,
                    "timed": event.WhichOneof("data") == "offset_ps",
                    "timestamp_ps": line.timestamp_ns * 1000 + event.offset_ps,
                    "duration_ps": event.duration_ps, "stats": merged, "context_stats": occurrence,
                    "metadata_stat_overrides": common_duplicates, "occurrence_stat_overrides": occurrence_duplicates})
    return rows


def context_groups(rows, scope):
    groups = defaultdict(lambda: {"producers": [], "consumers": []})
    for row in rows:
        stats = row["context_stats"]
        for role, type_name, id_name in [("producers", "_pt", "_p"), ("consumers", "_ct", "_c")]:
            if type_name not in stats and id_name not in stats:
                continue
            check(type_name in stats and id_name in stats and row["timed"], "incomplete or untimed context")
            context_type = integer(stats[type_name])
            check(context_type in {0, 15}, "this fixture only supports Generic and ThreadpoolEvent contexts")
            context_id = integer(stats[id_name], unsigned=True)
            pid = row["process_id"]
            if role == "consumers" and "_pid" in stats:
                pid = integer(stats["_pid"], unsigned=True)
                pid = (pid + (1 << 31)) % (1 << 32) - (1 << 31)
            check(pid is None or -(1 << 31) <= pid < 1 << 31, "unsupported process id")
            groups[(scope, pid, context_type, context_id)][role].append(row["node"])
    result = []
    for (scope, pid, kind, ident), group in sorted(groups.items(), key=lambda item: repr(item[0])):
        result.append({"scope": scope, "process_id": pid, "context_type": kind,
                       "context_id": str(ident), **group})
    return result


def unique_pairs(rows, groups, selected=None):
    by_node = {r["node"]: r for r in rows}
    check(len(by_node) == len(rows), "duplicate raw event coordinate")
    pairs = []
    for group in groups:
        members = [by_node[n] for role in ("producers", "consumers") for n in group[role]]
        if selected is not None and not any(selected(r) for r in members):
            continue
        check(len(group["producers"]) == len(group["consumers"]) == 1, "context is missing or not one-to-one")
        producer, consumer = [by_node[group[k][0]] for k in ("producers", "consumers")]
        check(producer["node"] != consumer["node"] and producer["timestamp_ps"] <= consumer["timestamp_ps"], "invalid endpoint time or self link")
        pairs.append({"context": {k: group[k] for k in ("scope", "process_id", "context_type", "context_id")},
                      "producer": producer["node"], "consumer": consumer["node"],
                      "producer_name": producer["name"], "consumer_name": consumer["name"],
                      "cross_thread": (producer["process_id"], producer["line_id"]) != (consumer["process_id"], consumer["line_id"]),
                      "producer_duration_ps": producer["duration_ps"],
                      "start_to_consumer_ps": consumer["timestamp_ps"] - producer["timestamp_ps"]})
    return pairs


def json_correspondence(rows, events, pairs):
    table = defaultdict(list)
    for i, event in enumerate(events):
        if event.get("ph") == "X":
            key = (event["name"], event["pid"], event["tid"],
                   Decimal(str(event["ts"])) * 1000000, Decimal(str(event["dur"])) * 1000000)
            table[key].append(i)
    selected = {p[k] for p in pairs for k in ("producer", "consumer")}
    matches = []
    for row in rows:
        if row["node"] not in selected:
            continue
        check(row["plane"] == "/host:CPU", "non-host JSON device mapping requires separate handling")
        # AddTraceEvent clamps zero-duration raw events to one picosecond.
        key = (row["export_name"], 701, row["viewer_tid"], row["timestamp_ps"], max(1, row["duration_ps"]))
        candidates = table.get(key, [])
        check(len(candidates) == 1, "raw event lacks a unique JSON counterpart")
        event = events[candidates[0]]
        hidden = FILTERED_FIELDS.intersection(row["stats"])
        check(not hidden.intersection(event.get("args", {})), "internal context/program fields unexpectedly exported")
        for name in {"hlo_op", "hlo_module", "run_id", "work", "outcome", "_src"}.intersection(row["stats"]):
            check(str(row["stats"][name]["value"]) == event.get("args", {}).get(name), "public event metadata changed")
        matches.append({"node": row["node"], "json_index": candidates[0], "filtered_fields": sorted(hidden),
                        "raw_duration_ps": row["duration_ps"], "exported_duration_ps": max(1, row["duration_ps"])})
    check(len(matches) == len(selected), "incomplete raw/JSON matching")
    return matches


def flow_overlay(rows, pairs):
    by_node = {r["node"]: r for r in rows}
    flows, used = [], set()
    for pair in pairs:
        key = json.dumps(pair["context"], sort_keys=True).encode()
        ident = "0x" + hashlib.sha256(key).hexdigest()[:16]
        check(ident not in used, "derived viewer flow-id collision")
        used.add(ident)
        for role, phase in [("producer", "s"), ("consumer", "f")]:
            row = by_node[pair[role]]
            flows.append({"name": "research_context", "cat": "derived_raw_xspace_context", "ph": phase,
                          "id": ident, "pid": 701, "tid": row["viewer_tid"],
                          "ts": float(Decimal(row["timestamp_ps"]) / 1000000),
                          "args": {"context_id_decimal": pair["context"]["context_id"],
                                   "context_type": pair["context"]["context_type"]}})
    return flows
