#!/usr/bin/env python3
"""Recheck the provenance and bounded claims of the public PJRT source review.

This is source/contract integrity validation, not execution of the PJRT code.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from runtime_chain_index import IDS, EDGES
from verify_research import ROOT, HERE, check, local_path, read_json, sha256

# Conditions below are reviewed source readings. Lexical guards detect drift;
# they do not independently prove C++ behavior, ownership, or device semantics.
CLAIMS = [
    ("dispatch-branches", "Python execute_sharded and cached C++ direct IFRT dispatch are separate paths.",
     ["jax.fastpath-gate", "jax.execute-replicated", "jax.pjit-fast-execute", "jaxlib.execute-sharded", "jaxlib.execute-local"]),
    ("internal-vs-returned-status", "PjRt IFRT requests completion futures internally; fill_status controls exposure in ExecuteResult.",
     ["ifrt.execute-options", "ifrt.pjrt-execute"]),
    ("no-strict-barrier", "IFRT Execute does not require all arguments or outputs to become ready at one shared barrier.",
     ["ifrt.execute-contract"]),
    ("c-event-ownership", "C API outputs and per-device complete events are caller-owned handles; direct submission error does not populate events.",
     ["pjrt.c-execute-args", "pjrt.capi-execute", "pjrt.event-to-future"]),
    ("ready-includes-error", "Event readiness can mean an error; status must be checked separately.",
     ["pjrt.event-ready-contract", "pjrt.event-callback-contract", "pjrt.buffer-track-event"]),
    ("two-completion-scopes", "Buffer readiness and per-device execution completion are different API scopes.",
     ["pjrt.buffer-ready-contract", "pjrt.c-execute-args", "ifrt.array-ready", "jaxlib.token-wait"]),
    ("host-wait-path", "Array wait uses C++ futures; this path does not directly call PJRT_Event_Await.",
     ["jaxlib.array-wait", "jaxlib.buffers-wait", "jaxlib.wait-with-signals", "ifrt.values-ready", "pjrt.buffer-track-event"]),
    ("effect-thread-scope", "effects_barrier waits the calling thread's RuntimeTokenSet, not an arbitrary global barrier.",
     ["jax.effects-barrier", "jax.runtime-tokens"]),
    ("stream-abi-boundary", "IFRT forwards execution_stream_id to C++ PJRT options; this pinned core C API options/adapter has no direct field mapping.",
     ["ifrt.execute-options", "ifrt.pjrt-execute", "pjrt.c-execute-options", "pjrt.capi-execute-args"]),
    ("reference-wrapper-boundary", "Open-source wrapper_impl is a reference implementation, not established as the loaded libtpu implementation.",
     ["pjrt.capi-execute", "pjrt.c-wrapper-execute"]),
    ("library-context-not-kernel", "PJRT profiler linkage uses context type 14 and a short host TraceMeProducer, not a TPU kernel span.",
     ["pjrt.profiler-link", "pjrt.profiler-context"]),
    ("python-interruption-not-device-cancel", "The wait helper checks Python signals but does not call device CancelExecution.",
     ["jaxlib.wait-with-signals"]),
]


def validate_record(record):
    check(record["evidence_level"] == "SOURCE-ONLY" and record["qualifiers"] == [], "source review promoted to runtime")
    check(record["runtime_execution_verified"] is False and record["libtpu_implementation_identified"] is False,
          "unsupported runtime or private implementation claim")
    check(record["wait_scopes"] == {"array": "selected_output_buffers", "token": "joined_execution_status", "effects": "calling_thread_tokens"},
          "completion scopes collapsed")
    check(record["dispatch_routes"] == {
        "python": ["jax.execute-replicated", "jaxlib.execute-sharded", "jaxlib.execute-local", "ifrt.pjrt-execute"],
        "cpp_fastpath": ["jax.pjit-fast-execute", "ifrt.pjrt-execute"]}, "dispatch branch collapsed")
    check(record["c_api_execution_stream_field"] == "absent-in-pinned-core-struct", "unsupported stream field claim")
    entries = record["source_entries"]
    check(len(entries) == len(IDS) and {e["id"] for e in entries} == IDS, "incomplete source roles")
    check({c["id"] for c in record["claims"]} == {c[0] for c in CLAIMS} and len(record["claims"]) == len(CLAIMS), "missing or duplicate reviewed claim")
    for claim in record["claims"]:
        check(claim["source_refs"] and set(claim["source_refs"]) <= IDS, "claim lacks a known source reference")
    check(len(record["call_edges"]) == len(EDGES), "incomplete reviewed branch edges")
    for edge in record["call_edges"]:
        check(edge["caller"] in IDS and edge["callee"] in IDS, "edge outside reviewed scope")
        check((edge["caller"], edge["callee"]) != ("pjrt.capi-execute", "pjrt.c-wrapper-execute"),
              "unproven plugin-to-reference implementation edge")


def collect():
    index = read_json(HERE / "source-index.json")
    entries = [e for e in index["entries"] if e["id"] in IDS]
    edges = [e for e in index["edges"] if e["caller"] in IDS and e["callee"] in IDS]
    # Check all component revisions/cleanliness; source entry points remain on
    # pristine trees, while runtime build inputs have their own patched audits.
    for name, source in index["source_roots"].items():
        path = local_path(source["path"])
        check(subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == source["revision"], "source revision changed: " + name)
        check(not subprocess.check_output(["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain"], text=True), "dirty source: " + name)
    for entry in entries:
        path = local_path(entry["path"])
        check(sha256(path) == entry["source_sha256"], "source bytes differ: " + entry["id"])
        check(path.read_text().splitlines()[entry["line"]-1].startswith(entry["anchor"]), "source anchor differs")
    for edge in edges:
        check(local_path(edge["path"]).read_text().splitlines()[edge["line"]-1].startswith(edge["anchor"]), "source call edge differs")
    by_id = {e["id"]: e for e in entries}
    guards = []
    def guard(ident, text, present=True, whole_file=False):
        e = by_id[ident]
        lines = local_path(e["path"]).read_text().splitlines(keepends=True)
        start = 0 if whole_file else e["line"]-1
        end = len(lines) if whole_file else (e["end_line"] or len(lines))
        body = "".join(lines[start:end])
        check((text in body) == present, "lexical source guard differs: " + ident + ": " + text)
        guards.append({"entry": ident, "path": e["path"], "start_line": start+1, "end_line": end,
                       "text": text, "present": present, "scope_sha256": hashlib.sha256(body.encode()).hexdigest()})
    guard("jax.pjit-fast-execute", "ifrt_executable()->Execute(")
    guard("jax.pjit-fast-execute", "ExecuteSharded(", False)
    guard("ifrt.pjrt-execute", "returned_pjrt_futures.emplace();")
    guard("ifrt.pjrt-execute", "status = JoinFutures(")
    guard("ifrt.pjrt-execute", "if (options.fill_status)")
    guard("ifrt.pjrt-execute", "opts.execution_stream_id = options.execution_stream_id;")
    guard("pjrt.c-execute-options", "execution_stream_id", False, True)
    guard("pjrt.capi-execute", "execution_stream_id", False, True)
    guard("jax.runtime-tokens", "threading.local")
    guard("jaxlib.wait-with-signals", "PyErr_CheckSignals()")
    guard("jaxlib.wait-with-signals", "CancelExecution", False)
    guard("pjrt.buffer-track-event", "PJRT_Event_IsReady")
    guard("pjrt.buffer-track-event", "PJRT_Event_OnReady")
    guard("pjrt.buffer-track-event", "PJRT_Event_Await", False)
    guard("pjrt.profiler-link", "ContextType::kPjrtLibraryCall")
    record = {
        "evidence_level": "SOURCE-ONLY", "qualifiers": [], "runtime_execution_verified": False,
        "libtpu_implementation_identified": False,
        "validation_scope": "Pinned source/anchor/branch provenance and lexical drift guards. Human source review supplies semantics; no native execution or model simulation.",
        "source_roots": index["source_roots"], "source_entries": entries, "call_edges": edges,
        "claims": [{"id": i, "statement": s, "source_refs": refs} for i,s,refs in CLAIMS],
        "lexical_guards": guards,
        "dispatch_routes": {"python": ["jax.execute-replicated", "jaxlib.execute-sharded", "jaxlib.execute-local", "ifrt.pjrt-execute"], "cpp_fastpath": ["jax.pjit-fast-execute", "ifrt.pjrt-execute"]},
        "wait_scopes": {"array": "selected_output_buffers", "token": "joined_execution_status", "effects": "calling_thread_tokens"},
        "c_api_execution_stream_field": "absent-in-pinned-core-struct",
        "limits": ["No libtpu load, TPU execution, LLO or performance evidence.",
                   "Reference wrapper code is not identified as the implementation of the private plugin.",
                   "Ready can include error; callback-registration failure cleanup was not fault-injected or repaired.",
                   "No claim about native completion callback thread or actual stream scheduling."]}
    validate_record(record)
    return record


def selftest(record):
    tests = []
    def reject(name, mutate):
        bad = copy.deepcopy(record); mutate(bad)
        try: validate_record(bad)
        except ValueError: tests.append(name)
        else: raise AssertionError("invalid review accepted: " + name)
    reject("source-as-tpu-runtime", lambda d: d.update(evidence_level="RUN-TPU"))
    reject("reference-as-libtpu", lambda d: d.update(libtpu_implementation_identified=True))
    reject("array-as-global-barrier", lambda d: d["wait_scopes"].update(array="all_devices_and_threads"))
    reject("fastpath-through-python-wrapper", lambda d: d["dispatch_routes"]["cpp_fastpath"].insert(1, "jaxlib.execute-sharded"))
    reject("invented-c-api-stream-field", lambda d: d.update(c_api_execution_stream_field="execution_stream_id"))
    reject("missing-buffer-ready-role", lambda d: d["source_entries"].pop())
    reject("missing-claim", lambda d: d["claims"].pop())
    reject("unknown-source-reference", lambda d: d["claims"][0]["source_refs"].append("libtpu.private"))
    reject("unproven-plugin-implementation-edge", lambda d: d["call_edges"][0].update(caller="pjrt.capi-execute", callee="pjrt.c-wrapper-execute"))
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--check-saved", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    result = collect()
    if args.selftest: result["negative_contract_tests"] = selftest(result)
    if args.check_saved:
        saved = read_json(HERE / "runtime-chain-results.json")
        check(saved == result, "saved review differs from current source checks")
    if args.result:
        args.result.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"entries": len(result["source_entries"]), "edges": len(result["call_edges"]),
                      "claims": len(result["claims"]), "lexical_guards": len(result["lexical_guards"]),
                      "negative_contract_tests": result.get("negative_contract_tests", []), "evidence_level": "SOURCE-ONLY"}))


if __name__ == "__main__":
    main()
