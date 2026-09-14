#!/usr/bin/env python3
"""Recheck raw context links and reject name/time/float based miscorrelation."""

import argparse
import copy
import gzip
import json
from pathlib import Path
import subprocess

from audit_xspace_contexts import collect, inspect_trace, check_no_exported_flows
from cpu_executable_parser import inventory, message, SCHEMA as TOOL_SCHEMA
from verify_research import ROOT, check, read_json, local_path
from xspace_contexts import (load_schema, flatten, context_groups, unique_pairs,
                             json_correspondence, integer)


def selftest(capture):
    negatives, positives = [], []
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError("invalid input accepted: " + name)

    rejects("legacy-flow-already-present", lambda: check_no_exported_flows([{"ph": "s"}]))
    rejects("flow-v2-already-present", lambda: check_no_exported_flows([{"ph": "X", "bind_id": 0, "flow_out": True}]))

    for mode in ["hash", "missing", "level"]:
        fake = copy.deepcopy(read_json(capture / "manifest.json"))
        if mode == "hash": fake["artifacts"][0]["sha256"] = "0" * 64
        elif mode == "missing": fake["artifacts"].pop()
        else: fake["evidence_level"] = "RUN-TPU"
        rejects(mode, lambda: inventory(capture, "REPLAY-OFFLINE", fake))
    real = read_json(capture / "explicit_host/rows.json")
    p = copy.deepcopy(next(r for r in real if r["name"] == "research_link_send"))
    c = copy.deepcopy(next(r for r in real if r["name"] == "research_link_receive"))
    def stat(value, kind="int64_value"):
        return {"kind": kind, "value": str(value)}
    p.update(node="p", process_id=0, timestamp_ps=0, duration_ps=0,
             context_stats={"_pt": stat(0), "_p": stat(0)})
    c.update(node="c", process_id=0, timestamp_ps=10, duration_ps=0,
             context_stats={"_ct": stat(0), "_c": stat(0)})
    def pair(rows, scope="synthetic"):
        return unique_pairs(rows, context_groups(rows, scope))
    check(len(pair([p, c])) == 1, "zero-valued identity dropped")
    positives.append("zero-context-type-id-and-pid-retained")
    a, b = copy.deepcopy(p), copy.deepcopy(c)
    a["context_stats"]["_p"] = stat(-1)
    b["context_stats"]["_c"] = stat((1 << 64)-1, "uint64_value")
    check(pair([a, b])[0]["context"]["context_id"] == str((1 << 64)-1), "signed-to-uint64 normalization differs")
    positives.append("signed-minus-one-matches-uint64-max")
    a, b = copy.deepcopy(p), copy.deepcopy(c)
    a.update(node="p2", context_stats={"_pt": stat(15), "_p": stat(0)})
    b.update(node="c2", context_stats={"_ct": stat(15), "_c": stat(0)})
    check(len(pair([p, c, a, b])) == 2, "context type namespace collapsed")
    positives.append("context-type-is-part-of-key")
    bad = copy.deepcopy(c); bad["process_id"] = 1
    rejects("different-pid-joined", lambda: pair([p, bad]))
    bad["context_stats"]["_pid"] = stat(0)
    check(len(pair([p, bad])) == 1, "consumer pid override lost")
    positives.append("explicit-consumer-pid-override")
    groups = context_groups([p], "file-a") + context_groups([c], "file-b")
    rejects("different-captures-joined", lambda: unique_pairs([p, c], groups))
    rejects("missing-consumer", lambda: pair([p]))
    duplicate = copy.deepcopy(c); duplicate["node"] = "c2"
    rejects("ambiguous-consumer", lambda: pair([p, c, duplicate]))
    bad = copy.deepcopy(c); bad["timestamp_ps"] = -1
    rejects("completion-before-producer", lambda: pair([p, bad]))
    bad = copy.deepcopy(p); del bad["context_stats"]["_pt"]
    rejects("missing-context-type", lambda: pair([bad, c]))
    rejects("float-context-id", lambda: integer({"kind": "double_value", "value": float((1 << 60)+1)}, True))
    pool = load_schema(capture)
    summary = read_json(capture / "summary.json")
    host = next(r for r in summary["cases"] if r["label"] == "explicit_host")
    space = message(pool, "tensorflow.profiler.XSpace", local_path(host["xspace"]["path"]).read_bytes())
    bad_space = copy.deepcopy(space); bad_space.warnings.append("incomplete capture")
    rejects("capture-warning", lambda: flatten(bad_space))
    def first_event(value, with_stats=False):
        return next(e for plane in value.planes for line in plane.lines for e in line.events if not with_stats or e.stats)
    bad_space = copy.deepcopy(space); first_event(bad_space).metadata_id = 10**12
    rejects("missing-event-metadata", lambda: flatten(bad_space))
    bad_space = copy.deepcopy(space); first_event(bad_space, True).stats[0].metadata_id = 10**12
    rejects("missing-stat-metadata", lambda: flatten(bad_space))
    bad_space = copy.deepcopy(space)
    for plane in bad_space.planes:
        for line in plane.lines:
            for event in line.events:
                if plane.event_metadata[event.metadata_id].name == "research_link_send":
                    original = next(s for s in event.stats if plane.stat_metadata[s.metadata_id].name == "_p")
                    extra = event.stats.add(); extra.CopyFrom(original); extra.uint64_value = 0
                    break
            else: continue
            break
        else: continue
        break
    rejects("conflicting-context-stat", lambda: flatten(bad_space))
    row_data = read_json(capture / "vmap_matmul/rows.json")
    pair_data = read_json(capture / "vmap_matmul/pairs.json")
    case = next(r for r in summary["cases"] if r["label"] == "vmap_matmul")
    with gzip.open(local_path(case["json"]["path"]), "rt") as stream:
        events = json.load(stream)["traceEvents"]
    matches = json_correspondence(row_data, events, pair_data)
    zero = next(m for m in matches if m["raw_duration_ps"] == 0)
    check(zero["exported_duration_ps"] == 1, "zero duration clamp differs")
    positives.append("raw-zero-duration-exported-as-one-picosecond")
    bad_events = copy.deepcopy(events); bad_events[zero["json_index"]]["dur"] = 0
    rejects("json-duration-clamp-ignored", lambda: json_correspondence(row_data, bad_events, pair_data))
    bad_events = copy.deepcopy(events); bad_events.append(copy.deepcopy(events[matches[0]["json_index"]]))
    rejects("ambiguous-json-counterpart", lambda: json_correspondence(row_data, bad_events, pair_data))
    producer_match = next(m for m in matches if "_pt" in m["filtered_fields"])
    bad_events = copy.deepcopy(events); bad_events[producer_match["json_index"]].setdefault("args", {})["_pt"] = "0"
    rejects("internal-context-export-claim", lambda: json_correspondence(row_data, bad_events, pair_data))
    override_row = next(r for r in row_data if r["occurrence_stat_overrides"])
    check(override_row["stats"]["_src"] == override_row["occurrence_stat_overrides"][-1]["replacement"], "ordinary stat last-wins rule differs")
    positives.append("duplicate-source-stat-history-preserved")
    match = next(m for m in matches if m["node"] == override_row["node"])
    bad_events = copy.deepcopy(events)
    bad_events[match["json_index"]]["args"]["_src"] = override_row["occurrence_stat_overrides"][0]["previous"]["value"]
    rejects("first-source-stat-used", lambda: json_correspondence(row_data, bad_events, pair_data))
    return {"negative_tests": negatives, "positive_edge_cases": positives}


def verify(capture, run_selftest=False):
    checked = inventory(capture, "REPLAY-OFFLINE")
    saved = read_json(capture / "summary.json")
    thunks = local_path(saved["thunk_runtime"]["path"]).parent
    host = local_path(saved["host_runtime"]["path"]).parent
    current = collect(schema_capture=capture, thunks=thunks, host=host)
    check(saved == current, "saved correlation result differs from full source replay")
    pool = load_schema(capture)
    witnesses = {r["case"]: r for r in read_json(local_path(saved["thunk_witness"]["path"]))["cases"]}
    protoc = read_json(TOOL_SCHEMA / "generation.json")["argv"][0]
    for case in saved["cases"]:
        directory = thunks / case["label"] / "trace" if case["mode"] == "thunks" else host / "trace"
        _, artifacts, raw = inspect_trace(directory, pool, case["label"], case["mode"], witnesses.get(case["label"]))
        target = capture / case["label"]
        for name, value in artifacts.items():
            check(read_json(target / name) == value, "saved normalized rows, pairs or overlay differ")
        expected = message(pool, "tensorflow.profiler.XSpace", raw)
        check(message(pool, "tensorflow.profiler.XSpace", (target / "protoc-roundtrip.pb").read_bytes()) == expected,
              "saved C++ codec roundtrip differs")
        command = [protoc, f"--descriptor_set_in={capture}/xplane.descriptor.pb", "--encode=tensorflow.profiler.XSpace"]
        encoded = subprocess.run(command, input=(target / "xspace.txt").read_bytes(), capture_output=True, check=True).stdout
        check(message(pool, "tensorflow.profiler.XSpace", encoded) == expected, "saved XSpace text differs")
    tests = selftest(capture) if run_selftest else {}
    return {"audit_manifest": checked, **saved, **tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/xspace-context-audit-002/capture")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    result = verify(args.capture.resolve(), args.selftest)
    if args.result:
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pairs": sum(len(c["pairs"]) for c in result["cases"]),
                      "negative_tests": result.get("negative_tests", []), "positive_edge_cases": result.get("positive_edge_cases", [])}))


if __name__ == "__main__":
    main()
