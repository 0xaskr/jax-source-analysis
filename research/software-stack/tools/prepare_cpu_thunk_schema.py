#!/usr/bin/env python3
"""Snapshot pinned CPU executable proto inputs and a local protobuf analysis tool."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from matmul_probe import ROOT, fingerprint, write_json
from verify_research import check

INPUTS = ["xla/service/cpu/executable.proto", "xla/pjrt/proto/compile_options.proto",
          "xla/python/pjrt_ifrt/executable_metadata.proto"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protoc", type=Path, required=True)
    parser.add_argument("--protobuf-include", type=Path, required=True)
    parser.add_argument("--protobuf-package", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    check(output.is_relative_to(ROOT / "artifacts/jax-stack"), "use a fresh artifacts/jax-stack output")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "producer.py")
    for name in ["google", "protobuf-7.34.0.dist-info"]:
        shutil.copytree(args.protobuf_package / name, output / "python-tool" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    sys.path.insert(0, str(output / "python-tool"))
    import google.protobuf
    from google.protobuf import descriptor_pb2
    from google.protobuf.internal import api_implementation
    check(google.protobuf.__version__ == "7.34.0" and api_implementation.Type() == "python", "unexpected Python protobuf tool")
    protoc = args.protoc.resolve()
    version = subprocess.check_output([str(protoc), "--version"], text=True).strip()
    check(version == "libprotoc 32.1", "unexpected protoc version")
    includes = [ROOT / "upstream/xla", ROOT / "upstream/xla/third_party/tsl", args.protobuf_include.resolve()]
    argv = [str(protoc), *[f"-I{p}" for p in includes], "--include_imports",
            f"--descriptor_set_out={output}/cpu-executable.descriptor.pb", *INPUTS]
    run = subprocess.run(argv, check=True, text=True, capture_output=True)
    write_json(output / "generation.json", {"argv": argv, "returncode": run.returncode,
        "stdout": run.stdout, "stderr": run.stderr, "protoc_version": version,
        "protoc_sha256": fingerprint(protoc)["sha256"]})
    files = descriptor_pb2.FileDescriptorSet.FromString((output / "cpu-executable.descriptor.pb").read_bytes())
    sources = []
    for file in files.file:
        source = next(p / file.name for p in includes if (p / file.name).is_file())
        dest = output / "proto-sources" / file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        sources.append({"proto": file.name, "source_path": str(source.relative_to(ROOT)),
                        "snapshot_path": str(dest.relative_to(ROOT)), "sha256": fingerprint(source)["sha256"]})
    check(len(sources) == 28, "unexpected descriptor closure")
    write_json(output / "source-inputs.json", sources)
    replay = [str(protoc), f"-I{output}/proto-sources", "--include_imports",
              f"--descriptor_set_out={output}/replayed.descriptor.pb", *INPUTS]
    run = subprocess.run(replay, check=True, text=True, capture_output=True)
    check((output / "replayed.descriptor.pb").read_bytes() == (output / "cpu-executable.descriptor.pb").read_bytes(), "independent schema replay differs")
    write_json(output / "replay.json", {"argv": replay, "returncode": run.returncode,
                                        "stderr": run.stderr, "identical_descriptor": True})
    write_json(output / "tooling.json", {"protobuf_python_version": google.protobuf.__version__,
        "protobuf_implementation": api_implementation.Type(), "protoc_version": version, "schema_files": len(sources),
        "scope": "Copied local analysis tool; no installation or mutation of source/runtime environments."})
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass",
        "evidence_level": "SOURCE-ONLY", "qualifiers": [], "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"argv": [sys.executable, "-B", *sys.argv],
                     "source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(f"Prepared {len(sources)} proto inputs and reproducible descriptor bytes")


if __name__ == "__main__":
    main()
