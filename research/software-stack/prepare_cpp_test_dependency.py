#!/usr/bin/env python3
"""Copy a JAX-selected Googletest and apply the pinned XLA test patches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATCH_NAMES = (
    "0001-Add-ASSERT_OK-EXPECT_OK-ASSERT_OK_AND_ASSIGN-macros.patch",
    "googletest.patch",
)


def inventory(root):
    return {
        str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.base_directory.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/jax-stack"):
        raise ValueError("output must be repository-local ignored artifacts")
    module_text = (source / "MODULE.bazel").read_text()
    if 'version = "1.17.0.bcr.2"' not in module_text:
        raise ValueError("expected the JAX-selected Googletest 1.17.0.bcr.2")
    if (source / "googlemock/include/gmock/internal/xla-gmock-macros.h").exists():
        raise ValueError("base is already patched")
    base = inventory(source)
    output.mkdir(parents=True, exist_ok=False)
    copied = output / "googletest"
    shutil.copytree(source, copied, symlinks=True)
    if inventory(copied) != base:
        raise ValueError("dependency copy differs from base")
    # Stop discovery of the outer research repository: otherwise git apply from
    # an ignored nested directory can silently skip every patch path.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(output))
    patches = [ROOT / "upstream/xla/third_party/googletest" / n for n in PATCH_NAMES]
    commands = []
    for patch in patches:
        for option in ("--check", "--verbose"):
            command = ["git", "apply", option, str(patch)]
            result = subprocess.run(command, cwd=copied, env=env, text=True, capture_output=True)
            commands.append({"argv": command, "returncode": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            result.check_returncode()
    patched = inventory(copied)
    changed = {name: value for name, value in patched.items() if base.get(name) != value}
    expected = {"BUILD.bazel", "googlemock/include/gmock/gmock.h",
                "googlemock/include/gmock/internal/xla-gmock-macros.h"}
    if changed.keys() != expected or inventory(source) != base:
        raise ValueError("unexpected patched files or original base changed")
    record = {"base_directory": str(source), "base_files": base,
              "patched_files": changed, "patches": [
                  {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in patches],
              "outcome": "patched-and-verified", "production_dependencies_modified": False}
    (output / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"--override_module=googletest={copied}")


if __name__ == "__main__":
    main()
