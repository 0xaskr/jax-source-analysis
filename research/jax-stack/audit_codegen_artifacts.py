#!/usr/bin/env python3
"""Bind existing CPU HLO, LLVM IR, ELF symbols and serialized object bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

from matmul_probe import ROOT, fingerprint, write_json
from verify_research import verify as verify_cpu


def elf_header(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected ELF64 little-endian object")
    kind, machine = struct.unpack_from("<HH", data, 16)
    if (kind, machine) != (1, 62):
        raise ValueError("Expected relocatable x86-64 ELF")
    return {"class": "ELF64", "endianness": "little", "e_type": kind, "e_machine": machine}


def functions(text):
    result = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[0].endswith(":") and fields[3:5] == ["FUNC", "GLOBAL"] and fields[6] != "UND":
            result.append({"name": fields[7], "size_bytes": int(fields[2]), "section_index": fields[6]})
    return result


def collect(output):
    from jaxlib import _hlo
    source = ROOT / "artifacts/jax-stack/cpu-matmul-003"
    checked = verify_cpu(source)
    tools = {}
    for tool in ["readelf", "objdump"]:
        path = Path(shutil.which(tool)).resolve()
        tools[tool] = {"path": str(path), **fingerprint(path),
                       "version": subprocess.check_output([str(path), "--version"], text=True).splitlines()[0]}
    write_json(output / "inspection-tools.json", tools)
    input_records = {}
    objects = []
    cases = []
    def record(path):
        relative = str(path.relative_to(ROOT))
        input_records[relative] = {"path": relative, **fingerprint(path)}
        return relative
    for case in checked["cases"]:
        prefix = case["native_module_prefix"]
        hlo_path = source / case["name"] / "optimized-hlo.txt"
        executable_path = source / case["name"] / "executable.bin"
        hlo = _hlo.hlo_module_from_text(hlo_path.read_text())
        instructions = {i.name: i.opcode.name for c in hlo.computations() for i in c.instructions()}
        case_objects = sorted((source / "xla-dump").glob(prefix + ".*.o"))
        for obj in case_objects:
            stem = obj.name.removeprefix(prefix + ".obj-file.").removesuffix(".o")
            before = source / "xla-dump" / (prefix + "." + stem + ".ir-no-opt.ll")
            after = source / "xla-dump" / (prefix + "." + stem + ".ir-with-opt.ll")
            directory = output / (prefix + "." + stem)
            directory.mkdir()
            commands = [[tools["readelf"]["path"], "-hSW", str(obj)],
                        [tools["readelf"]["path"], "-sW", str(obj)],
                        [tools["readelf"]["path"], "-rW", str(obj)],
                        [tools["objdump"]["path"], "-drwC", str(obj)]]
            files = ["elf-header-sections.txt", "elf-symbols.txt", "elf-relocations.txt", "disassembly.txt"]
            for command, filename in zip(commands, files, strict=True):
                (directory / filename).write_text(subprocess.check_output(command, text=True))
            write_json(directory / "commands.json", commands)
            data = obj.read_bytes()
            header = elf_header(data)
            symbols = functions((directory / "elf-symbols.txt").read_text())
            definitions = re.findall(r"^define .*?@([A-Za-z0-9_.$-]+)\(", after.read_text(), re.M)
            assert symbols and set(definitions) == {s["name"] for s in symbols}
            assert all(instructions.get(s["name"]) == "kFusion" for s in symbols)
            module_id = re.search(r"^; ModuleID = '([^']+)'", after.read_text(), re.M)[1]
            assert module_id == re.search(r"^; ModuleID = '([^']+)'", before.read_text(), re.M)[1]
            assert module_id in (directory / "elf-symbols.txt").read_text()
            executable = executable_path.read_bytes()
            assert executable.count(data) == 1, "Object bytes must occur exactly once in its serialized executable"
            objects.append({"case": case["name"], "native_module_prefix": prefix, "llvm_module_id": module_id,
                            "inputs": {"hlo": record(hlo_path), "ir_before": record(before), "ir_after": record(after),
                                       "object": record(obj), "serialized_executable": record(executable_path)},
                            "elf": header, "defined_functions": symbols, "hlo_opcode": "kFusion",
                            "object_occurrences_in_serialized_package": 1, "serialized_package_byte_offset": executable.find(data),
                            "inspection_directory": str(directory.relative_to(ROOT))})
        cases.append({"case": case["name"], "native_module_prefix": prefix, "object_count": len(case_objects)})
    write_json(output / "input-records.json", list(input_records.values()))
    summary = {"outcome": "pass", "evidence_level": "REPLAY-OFFLINE", "qualifiers": checked["qualifiers"],
               "source_capture": "artifacts/jax-stack/cpu-matmul-003", "source_manifest_sha256": checked["manifest_sha256"],
               "source_runtime_evidence_level": "RUN-CPU", "cases": cases, "objects": objects,
               "limits": ["This audit reads existing compiler products; it does not compile or execute code.",
                          "Exact object bytes occur in the captured serialized package; no pickle or executable was loaded by this audit.",
                          "The original producer verified in-process executable reload. This does not measure individual kernel invocation or performance.",
                          "Native products belong to the captured VERSION-SKEW wheel; indexed source APIs are separate source-only evidence.",
                          "No object for a case does not mean no machine code ran: prebuilt library/runtime paths need separate attribution."]}
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
    summary = collect(output)
    write_json(output / "manifest.json", {"capture_id": output.name, "outcome": "pass", "evidence_level": "REPLAY-OFFLINE",
        "qualifiers": summary["qualifiers"], "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"source": str((output / "producer.py").relative_to(ROOT)), **fingerprint(output / "producer.py")},
        "artifacts": [{"path": str(p.relative_to(ROOT)), **fingerprint(p)} for p in sorted(output.rglob("*")) if p.is_file()]})
    print(json.dumps({"capture": output.name, "objects": len(summary["objects"]), "cases": summary["cases"]}))


if __name__ == "__main__":
    main()
