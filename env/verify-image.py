#!/usr/bin/env python3
"""Verify the image's tools before it can become a reusable environment."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def verify(lock):
    for name, path in (("python", lock["python"]["path"]),
                       ("git", lock["git"]["path"]),
                       ("uv", "/usr/local/bin/uv")):
        data = Path(path).read_bytes()
        expected = lock[name]
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"{name}: binary SHA-256 does not match the environment lock")
        if "size_bytes" in expected and len(data) != expected["size_bytes"]:
            raise RuntimeError(f"{name}: binary size does not match the environment lock")
    if sys.version.split()[0] != lock["python"]["version"]:
        raise RuntimeError("Python version mismatch")
    for path in lock["clang"]["paths"]:
        output = subprocess.check_output([path, "--version"], text=True)
        if not re.search(r"\bclang version " + re.escape(lock["clang"]["version"]) + r"\b", output):
            raise RuntimeError(f"{path}: Clang version mismatch")
    print("image tools verified against environment.lock.json")


if __name__ == "__main__":
    verify(json.loads(Path(__file__).with_name("environment.lock.json").read_text()))
