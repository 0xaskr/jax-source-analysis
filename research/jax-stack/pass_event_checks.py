#!/usr/bin/env python3
"""Rules for raw compiler-pass traces; synthetic tests are not runtime evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

EVENT = "research_hlo_pass_run"
FIELDS = {"pass", "pipeline", "module", "program_id", "status", "changed"}
KNOWN_PASSES = {"algsimp", "constant_folding", "layout-assignment"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contains(parent, child, same_thread=False):
    return (parent["pid"] == child["pid"] and
            (not same_thread or parent["tid"] == child["tid"]) and
            parent["ts"] - 1e-3 <= child["ts"] and
            child["ts"] + child["dur"] <= parent["ts"] + parent["dur"] + 1e-3)


def overlaps(left, right):
    return (left["pid"] == right["pid"] and
            left["ts"] < right["ts"] + right["dur"] and
            right["ts"] < left["ts"] + left["dur"])


def union_duration(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total = 0.0
    left = right = None
    for start, end in intervals:
        if right is None or start > right:
            if right is not None:
                total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (0.0 if right is None else right - left)


def analyze(events, module_name, expected, disabled_pass=None):
    require(expected in {"absent", "present"}, "invalid custom-event expectation")
    def only(name):
        found = [e for e in events if e.get("name") == name]
        require(len(found) == 1, f"expected one {name} scope")
        return found[0]
    lower, cold = only("pass_probe_lower"), only("pass_probe_compile")
    python = only("pass_probe_python_trace")
    warm = sorted((e for e in events if e.get("name") == "pass_probe_warm"), key=lambda e: e["ts"])
    require(len(warm) == 3 and contains(lower, python, True), "tracing/warm scopes differ")
    for scope in [lower, cold, python, *warm]:
        require(all(math.isfinite(scope[k]) for k in ["ts", "dur"]) and scope["dur"] >= 0, "invalid host scope interval")
    require(lower["ts"] + lower["dur"] <= cold["ts"] + 1e-3, "lower/compile order differs")
    require((lower["pid"], lower["tid"]) == (cold["pid"], cold["tid"]), "lower/compile driver differs")
    previous_end = cold["ts"] + cold["dur"]
    for scope in warm:
        require((scope["pid"], scope["tid"]) == (cold["pid"], cold["tid"]) and scope["ts"] + 1e-3 >= previous_end, "cold/warm ordering or driver thread differs")
        previous_end = scope["ts"] + scope["dur"]
    generic = [e for e in events if e.get("name") in KNOWN_PASSES and contains(cold, e)]
    require(generic, "no known native compiler event; absence cannot validate an inactive profiler")
    require(not any(e.get("name") in KNOWN_PASSES and any(overlaps(w, e) for w in warm) for e in events), "native compilation overlapped a warm scope")
    custom = [e for e in events if e.get("name") == EVENT]
    for event in custom:
        require(FIELDS <= set(event.get("args", {})), "custom event metadata missing")
        require(all(math.isfinite(event[k]) for k in ["ts", "dur"]) and event["dur"] >= 0, "invalid event interval")
        require(all(str(event["args"][k]) for k in FIELDS), "empty event identity")
        require(not any(overlaps(w, event) for w in warm), "custom pass event overlapped warm execution")
    selected = [e for e in custom if e["args"]["module"] == module_name]
    if expected == "absent":
        require(not custom, "custom pass events unexpectedly present")
    else:
        require(selected, "custom pass events missing for compiled module")
        if disabled_pass is None:
            require(any(e["args"]["pass"] == "algsimp" for e in selected), "default compile has no custom algsimp event")
    threads = {key: f"thread-{i}" for i, key in enumerate(sorted({(e["pid"], e["tid"]) for e in selected}))}
    groups = defaultdict(list)
    for event in selected:
        args = event["args"]
        require(contains(cold, event), "custom event outside cold compilation")
        require(args["pass"] != disabled_pass, "filtered leaf pass emitted a custom event")
        require(args["changed"] in ({"true", "false"} if args["status"] == "OK" else {"unknown"}), "status/changed metadata inconsistent")
        require(any(parent.get("name") == args["pass"] and contains(parent, event, True) for parent in events), "custom event lacks enclosing same-thread native pass scope")
        key = (args["module"], str(args["program_id"]), args["pipeline"], args["pass"], threads[(event["pid"], event["tid"])])
        groups[key].append(event)
    stats = []
    for key, group in sorted(groups.items()):
        stats.append(dict(zip(["module", "program_id", "pipeline", "pass", "thread"], key)) | {
            "count": len(group), "inclusive_total_us": sum(e["dur"] for e in group),
            "span_union_us": union_duration(group), "status_counts": dict(Counter(e["args"]["status"] for e in group)),
            "changed_counts": dict(Counter(e["args"]["changed"] for e in group))})
    return {"expected_custom_events": expected, "disabled_pass": disabled_pass, "compiled_module": module_name,
            "custom_event_count": len(custom), "selected_custom_event_count": len(selected),
            "other_module_custom_event_count": len(custom) - len(selected), "warm_scope_count": len(warm),
            "generic_known_pass_counts": dict(sorted(Counter(e["name"] for e in generic).items())), "groups": stats,
            "timing_scope": "Chrome trace ts/dur microseconds; inclusive and per-group interval union are not total CPU work or device time."}
