#!/usr/bin/env python3
"""Check cached descriptor evidence without promoting it to an action dependency graph."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

from audit_build_dependencies import file_hash
from verify_build_dependencies import inventory
from verify_research import ROOT, HERE, check, local_path, read_json


def check_summary(capture, summary, repositories):
    check(summary["evidence_level"] == "SOURCE-ONLY", "descriptor evidence promoted to runtime")
    check(summary["cache_presence_proves_build_use"] is False and
          summary["complete_action_input_closure"] is False, "cache promoted to complete build inputs")
    check(summary["repository_count"] == len(repositories), "repository count differs")
    check(summary["descriptor_count"] == sum(len(r["descriptors"]) for r in repositories), "descriptor count differs")
    for row in repositories:
        check(row["used_by_build"] is None, "cached repository promoted to target dependency")
        for descriptor in row["descriptors"].values():
            path = local_path(descriptor["snapshot"])
            check(path.is_relative_to(capture) and file_hash(path) == descriptor["sha256"] and
                  path.stat().st_size == descriptor["size_bytes"], "descriptor snapshot differs")
    python = summary["python"]
    check(python["requested_minor"] == "3.12" and python["configured_version"] == "3.12.13", "Python selection differs")
    check(python["per_build_action_process_observed"] is False, "local Python probe promoted to build process observation")
    selection = summary["selection"]
    def text(name):
        item = selection[name]
        path = local_path(item["snapshot"])
        check(file_hash(path) == item["sha256"], "selection snapshot differs")
        return path.read_text()
    check('"3.12": "3.12.13"' in text("minor_mapping"), "minor mapping absent")
    check('HERMETIC_PYTHON_VERSION = "3.12"' in text("python_selection"), "Python request absent")
    check('python_version = "3.12.13"' in text("python_build"), "generated toolchain version absent")
    config = text("build_config")
    check('HERMETIC_PYTHON_VERSION=3.12' in config and '--lockfile_mode=off' in config, "build configuration differs")
    launch = json.loads(text("build_launch"))
    check(not any('override_module=googletest' in arg for arg in launch["argv"]), "production command unexpectedly has test override")
    build = json.loads(text("build_manifest"))
    check(build["status"] == "build-succeeded" and build["build_id"] == summary["build_id"], "terminal build link differs")
    archive = python["archive"]
    check(archive["sha256"] in text("runtime_manifest") and
          python["source_url"].split('/download/')[1] in text("runtime_manifest"), "archive declaration absent")
    return python


def verify(capture):
    manifest = read_json(capture / "manifest.json")
    check("producer" in manifest, "capture lacks producer provenance")
    count = inventory(capture, manifest)
    summary = read_json(capture / "summary.json")
    repositories = read_json(capture / "repositories.json")
    python = check_summary(capture, summary, repositories)
    archive = local_path(python["archive"]["path"])
    check(file_hash(archive) == python["archive"]["sha256"] and archive.stat().st_size == python["archive"]["size_bytes"], "Python archive changed")
    with tarfile.open(archive, "r:gz") as stream:
        for item in python["matched_archive_members"]:
            member = stream.getmember(item["member"])
            check(member.isfile() and member.size == item["size_bytes"], "Python archive member differs")
            digest = hashlib.sha256()
            with stream.extractfile(member) as data:
                while block := data.read(1024 * 1024):
                    digest.update(block)
            path = local_path(item["actual_path"])
            check(digest.hexdigest() == item["sha256"] == file_hash(path) and
                  path.stat().st_size == item["size_bytes"], "Python binary/header differs from archive")
    interpreter = read_json(capture / "interpreter.json")
    command = read_json(capture / "interpreter-command.json")
    check(command["returncode"] == 0 and json.loads(command["stdout"]) == interpreter, "interpreter command evidence differs")
    check(interpreter["version_info"][:3] == [3, 12, 13], "observed interpreter version differs")
    test_entry = next(r for r in repositories if r["repository"] == "googletest+")
    return {"capture": str(capture.relative_to(ROOT)), "manifest_sha256": file_hash(capture / "manifest.json"),
            "artifact_count": count, "evidence_level": "SOURCE-ONLY", "build_id": summary["build_id"],
            "repository_count": summary["repository_count"], "descriptor_count": summary["descriptor_count"],
            "declared_modules": [{k: r[k] for k in ("repository", "declared_module", "declared_version")}
                                 for r in repositories if r["declared_module"]],
            "python": python, "observed_python_version": interpreter["version"],
            "cached_test_override": {k: test_entry[k] for k in ("repository", "symlink", "resolved_on_host", "used_by_build")},
            "complete_action_input_closure": False, "limits": summary["limits"]}


def selftest(capture):
    summary = read_json(capture / "summary.json")
    repositories = read_json(capture / "repositories.json")
    for mode in ("closure", "dependency", "version", "build-process"):
        fake, rows = copy.deepcopy(summary), copy.deepcopy(repositories)
        if mode == "closure": fake["complete_action_input_closure"] = True
        elif mode == "dependency": rows[0]["used_by_build"] = True
        elif mode == "version": fake["python"]["configured_version"] = "3.12.3"
        else: fake["python"]["per_build_action_process_observed"] = True
        try: check_summary(capture, fake, rows)
        except ValueError: pass
        else: raise AssertionError(f"invalid descriptor claim accepted: {mode}")
    fake = copy.deepcopy(read_json(capture / "manifest.json"))
    fake["artifacts"].pop()
    try: inventory(capture, fake)
    except ValueError: pass
    else: raise AssertionError("incomplete descriptor inventory accepted")
    historical = ROOT / "artifacts/jax-stack/external-descriptors-001"
    try: verify(historical)
    except ValueError as error: check("producer provenance" in str(error), "wrong historical rejection")
    else: raise AssertionError("historical capture with missing producer accepted")
    print("selftest: four invalid dependency/version/process claims, incomplete inventory and historical missing producer rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "artifacts/jax-stack/external-descriptors-002")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(args.capture.resolve())
    result = verify(args.capture.resolve())
    if args.write:
        (HERE / "external-descriptor-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("artifact_count", "repository_count", "descriptor_count", "complete_action_input_closure")}))
