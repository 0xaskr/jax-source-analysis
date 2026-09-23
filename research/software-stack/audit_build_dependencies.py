#!/usr/bin/env python3
"""Audit pinned core archives and replay XLA dependency patches off to the side."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import zipfile

from matmul_probe import ROOT, fingerprint, write_json


def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def patch_targets(text):
    targets=set()
    for path in re.findall(r"^(?:---|\+\+\+) ([^\t\n]+)",text,re.M):
        if path=="/dev/null":continue
        parts=PurePosixPath(path).parts
        if len(parts)<2 or parts[0]=="/" or ".." in parts:
            raise ValueError(f"Unsafe or unsupported patch path: {path}")
        # The actual tf_http_archive rule applies strip=1, even when the first
        # component is a repository name rather than a/ or b/.
        targets.add(str(PurePosixPath(*parts[1:])))
    return targets


def extract_selected(archive,prefix,targets,destination):
    base_records={}
    def retain(name,stream):
        if not name.startswith(prefix+"/"):return
        relative=name[len(prefix)+1:]
        if relative not in targets:return
        data=stream.read();path=destination/relative
        if not path.resolve().is_relative_to(destination.resolve()):raise ValueError("Path escaped replay directory")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        base_records[relative]={"sha256":hashlib.sha256(data).hexdigest(),"size_bytes":len(data)}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():continue
                if info.filename.startswith(prefix+"/") and info.filename[len(prefix)+1:] in targets:
                    with z.open(info) as stream:retain(info.filename,stream)
    else:
        with tarfile.open(archive,"r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith(prefix+"/") and member.name[len(prefix)+1:] in targets:
                    with tar.extractfile(member) as stream:retain(member.name,stream)
    return base_records


def collect(output,build_id):
    cache=ROOT/"artifacts/builds/kickoff-cpu-bazel-cache"
    bases=list(cache.glob("*/external"))
    if len(bases)!=1:raise ValueError("Expected exactly one known Bazel output base")
    external=bases[0]
    cas=cache/"cache/repos/v1/content_addressable/sha256"
    baseline=json.loads((ROOT/"manifests/baseline.json").read_text())
    build=json.loads((ROOT/f"manifests/build-fingerprints/{build_id}.json").read_text())
    clone=ROOT/"artifacts/jax-stack/source-build-001/clones/xla"
    assert subprocess.check_output(["git","-C",str(clone),"rev-parse","HEAD"],text=True).strip()==baseline["repository"]["sources"]["xla"]["git_commit"]
    components=[]
    for component,repo in [("llvm","xla++third_party_ext+llvm-raw"),("stablehlo","xla++third_party_ext+stablehlo"),("shardy","xla++third_party_ext+shardy")]:
        declaration=clone/f"third_party/{component}/workspace.bzl";text=declaration.read_text()
        revision=re.search(r"\b"+component.upper()+r'_COMMIT = "([a-f0-9]+)"',text)[1]
        expected=re.search(r"\b"+component.upper()+r'_SHA256 = "([a-f0-9]+)"',text)[1]
        assert revision==baseline["repository"]["sources"][component]["git_commit"]
        archive=cas/expected/"file";assert file_hash(archive)==expected
        print(component,"archive SHA-256 verified",flush=True)
        patch_names=re.findall(r'"//third_party/'+component+r':([^\"]+\.patch)"',text)
        patches=[clone/f"third_party/{component}"/name for name in patch_names]
        targets=set().union(*(patch_targets(p.read_text()) for p in patches))
        replay=output/"replay"/component;replay.mkdir(parents=True)
        prefix=("llvm-project" if component=="llvm" else component)+"-"+revision
        base_records=extract_selected(archive,prefix,targets,replay)
        patch_records=[]
        with (output/f"{component}-patch-replay.log").open("w") as log:
            for patch in patches:
                command=["/usr/bin/git","apply","--directory="+str(replay.relative_to(ROOT)),str(patch)]
                if patch_targets(patch.read_text()):
                    # git apply verifies all supplied context. Unlike GNU patch's
                    # fuzz=0 boundary heuristic it accepts this pinned patch's
                    # asymmetric trailing context. Final bytes must still equal
                    # the actual Bazel-patched source, independently of the engine.
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    assert result.returncode==0,patch
                patch_records.append({"path":str(patch.relative_to(ROOT)),**fingerprint(patch),"strip":1})
        compared=[]
        for relative in sorted(targets):
            reconstructed=replay/relative;actual=external/repo/relative
            if not reconstructed.exists():
                assert not actual.exists(),f"Deleted patch target still present: {relative}"
                compared.append({"path":relative,"deleted":True});continue
            assert actual.resolve().is_relative_to(ROOT),f"Source leaves repository: {actual}"
            assert actual.is_file() and file_hash(actual)==file_hash(reconstructed),f"Patch replay differs: {component}/{relative}"
            fp=fingerprint(reconstructed)
            compared.append({"path":relative,"base":base_records.get(relative),"patched":fp,
                             "changed":base_records.get(relative,{}).get("sha256")!=fp["sha256"]})
        components.append({"component":component,"base_revision":revision,
            "declaration":{"path":str(declaration.relative_to(ROOT)),**fingerprint(declaration)},
            "archive":{"path":str(archive.relative_to(ROOT)),"sha256":expected,"size_bytes":archive.stat().st_size},
            "actual_source_root":str((external/repo).relative_to(ROOT)),"patches":patch_records,
            "replayed_target_count":len(compared),"targets":compared})
        print(component,"patched targets verified",len(compared),flush=True)
    inventory=[];missing=[]
    for directory in sorted(cas.iterdir()):
        if not re.fullmatch(r"[0-9a-f]{64}",directory.name):continue
        path=directory/"file"
        if not path.is_file():missing.append(directory.name);continue
        actual=file_hash(path);assert actual==directory.name,f"Corrupt repository cache object: {directory.name}"
        inventory.append({"sha256":actual,"size_bytes":path.stat().st_size,"path":str(path.relative_to(ROOT))})
    write_json(output/"repository-cache.json",{"verified_objects":inventory,"incomplete_entries":missing,
        "scope":"Snapshot of cached payloads; may include fetched-but-unused dependencies. This is not a complete Bazel action input closure."})
    write_json(output/"external-repository-names.json",sorted(p.name for p in external.iterdir() if p.is_dir()))
    overlay=external/"xla++llvm_extension+llvm-project"
    sample="llvm/lib/IR/Instruction.cpp"
    assert (overlay/sample).resolve()==(external/"xla++third_party_ext+llvm-raw"/sample).resolve()
    summary={"outcome":"pass","evidence_level":"SOURCE-ONLY","build_id":build_id,
        "build_state_at_read":build["status"],"components":components,
        "repository_cache":{"verified_objects":len(inventory),"verified_bytes":sum(i["size_bytes"] for i in inventory),"incomplete_entries":missing},
        "llvm_overlay":{"path":str(overlay.relative_to(ROOT)),"sample":sample,"resolved_sample":str((overlay/sample).resolve().relative_to(ROOT)),"sha256":file_hash(overlay/sample)},
        "patch_tool":{"path":"/usr/bin/git","sha256":file_hash(Path('/usr/bin/git')),"mode":"git apply, complete supplied context; no index updates"},
        "limits":["Only declared patch targets are reconstructed and compared, not every file in the extracted repositories.",
                  "The live compiler source is the pinned archive plus XLA's patch set; pristine Git revision alone is an incomplete identity.",
                  "Cache hashes and repository names do not prove complete offline reproducibility or identify the exact subset consumed by every action.",
                  "The jaxlib wheel is not validated or installed by this source audit."]}
    write_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--build-id",default="kickoff-cpu-source-002")
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/"artifacts/jax-stack") or not re.fullmatch(r"[a-z0-9-]+",args.build_id):parser.error("Invalid output/build id")
    output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,output/"producer.py")
    started=datetime.now(timezone.utc).isoformat()
    try:summary=collect(output,args.build_id)
    except BaseException:
        import traceback
        (output/"failure.log").write_text(traceback.format_exc())
        raise
    write_json(output/"manifest.json",{"capture_id":output.name,"outcome":"pass","evidence_level":"SOURCE-ONLY",
        "started_at":started,"finished_at":datetime.now(timezone.utc).isoformat(),
        "producer":{"argv":[".venv/bin/python","-B","research/software-stack/audit_build_dependencies.py","--output",str(output.relative_to(ROOT)),"--build-id",args.build_id],"source":str((output/"producer.py").relative_to(ROOT)),**fingerprint(output/"producer.py")},
        "artifacts":[{"path":str(p.relative_to(ROOT)),**fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture":output.name,"components":len(summary["components"]),"cache":summary["repository_cache"]}))


if __name__=="__main__":main()
