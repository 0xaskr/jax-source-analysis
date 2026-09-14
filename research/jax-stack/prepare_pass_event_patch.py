#!/usr/bin/env python3
"""Prepare and reverse-check a diagnostic patch in disposable file copies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from matmul_probe import ROOT, fingerprint, write_json


RELATIVE = Path("xla/hlo/pass/hlo_pass_pipeline.cc")
OLD = "    auto status_or_changed = RunHelper<HloT>(pass, hlo, execution_threads);\n"
NEW = '''    auto status_or_changed = [&]() {
      std::optional<tsl::profiler::TraceMe> research_trace;
      if (!pass->IsPassPipeline()) {
        research_trace.emplace([&] {
          return tsl::profiler::TraceMeEncode(
              "research_hlo_pass_run",
              {{"pass", pass_name}, {"pipeline", pipeline_name},
               {"module", hlo->name()}, {"program_id", UniqueId(*hlo)}});
        });
      }
      auto result = RunHelper<HloT>(pass, hlo, execution_threads);
      if (research_trace.has_value()) {
        research_trace->AppendMetadata([&] {
          return tsl::profiler::TraceMeEncode(
              {{"status", absl::StatusCodeToString(result.status().code())},
               {"changed", result.ok() ? (*result ? "true" : "false")
                                       : "unknown"}});
        });
      }
      return result;
    }();
'''


def prepare(output):
    source_root = ROOT / "upstream/xla"
    source = source_root / RELATIVE
    before = source.read_text()
    baseline = json.loads((ROOT / "manifests/baseline.json").read_text())
    revision = baseline["repository"]["sources"]["xla"]["git_commit"]
    assert subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip() == revision
    assert not subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"])
    assert before.count(OLD) == 1
    after = before.replace(OLD, NEW)
    base, candidate = output / "base" / RELATIVE, output / "candidate" / RELATIVE
    for target in [base, candidate]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(before)
    # The build wrapper requires byte identity with git diff --binary
    # --full-index. Use the real pinned Git objects, a disposable index and a
    # minimal alternate work tree; never write the original source/index.
    candidate.write_text(after)
    git_dir = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "--absolute-git-dir"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="patch-index-", dir=output) as temporary:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"), GIT_OPTIONAL_LOCKS="0")
        git = ["git", "--git-dir=" + git_dir, "--work-tree=" + str(output / "candidate")]
        subprocess.run([*git, "read-tree", revision], cwd=output / "candidate", env=environment, check=True)
        patch = subprocess.check_output([*git, "diff", "--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv", revision, "--", str(RELATIVE)], cwd=output / "candidate", env=environment)
    assert patch.startswith(("diff --git a/" + str(RELATIVE) + " b/" + str(RELATIVE)).encode())
    (output / "xla-hlo-pass-events.patch").write_bytes(patch)
    candidate.write_text(before)
    commands = []
    with (output / "patch-check.log").open("w") as log:
        def run(*options):
            command = ["git", "apply", *options, "--directory=" + str((output / "candidate").relative_to(ROOT)), str(output / "xla-hlo-pass-events.patch")]
            commands.append(command)
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        run("--check")
        run()
        assert candidate.read_text() == after
        patched_fp = fingerprint(candidate)
        shutil.copy2(candidate, output / "patched-hlo_pass_pipeline.cc")
        run("--reverse", "--check")
        run("--reverse")
        assert candidate.read_bytes() == source.read_bytes() == base.read_bytes()
    write_json(output / "commands.json", commands)
    summary = {"outcome": "patch-prepared-and-reversed", "evidence_level": "SOURCE-ONLY",
               "source_revision": revision, "source_path": str(source.relative_to(ROOT)), "source": fingerprint(source),
               "patch_path": str((output / "xla-hlo-pass-events.patch").relative_to(ROOT)), "patch": fingerprint(output / "xla-hlo-pass-events.patch"),
               "patch_format": "canonical git diff --binary --full-index --no-color --no-ext-diff --no-textconv at pinned HEAD; only target file changed",
               "patched_source": patched_fp, "event_name": "research_hlo_pass_run",
               "event_metadata": ["pass", "pipeline", "module", "program_id", "status", "changed"],
               "scope": "Only enabled leaf pass RunHelper calls plus metadata append; RAII ends before pipeline dump/invariant/error accounting.",
               "checks": {"context_apply": True, "exact_candidate_bytes": True, "reverse_apply": True,
                          "original_bytes_restored": True, "original_source_tree_clean": True},
               "native_compiled": False, "patched_binary_loaded": False, "custom_event_observed": False,
               "limits": ["This is patch preparation, not C++ compilation, binary loading or runtime proof.",
                          "The running source-build clone/config was not modified.",
                          "Host compiler event timing includes instrumentation overhead; it is not device execution time.",
                          "The baseline matching-source wheel must be verified before applying this patch to a build clone."]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        parser.error("Use a new artifacts/jax-stack directory")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    started = datetime.now(timezone.utc).isoformat()
    summary = prepare(output)
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": summary["outcome"], "evidence_level": "SOURCE-ONLY",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture": output.name, "checks": summary["checks"], "native_compiled": False}))


if __name__ == "__main__":
    main()
