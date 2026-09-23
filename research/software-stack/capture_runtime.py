#!/usr/bin/env python3
"""Explicit unpatched source-build identity for CPU revalidation captures."""

import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def preflight(output, path):
    shutil.copy2(__file__, output / "runtime-helper.py")
    if path is None:
        return None
    from pass_events_probe import build_gate, load_baseline_module
    path = path.resolve()
    shutil.copy2(Path(__file__).with_name("pass_events_probe.py"), output / "binding-helper.py")
    build_gate(path, "absent", output / "build-gate-observation.json")
    load_baseline_module()._load_build_validator().load_and_validate_manifest(path)
    return path


def environment(output, path):
    spec = importlib.util.spec_from_file_location("capture_runtime_baseline", ROOT / "tools/capture-baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    observed = baseline.capture_baseline(path)
    if not all(not source["dirty"] for source in observed["repository"]["sources"].values()):
        raise ValueError("A pinned source tree changed during capture")
    if path is not None:
        from pass_events_probe import bind_native
        binding = bind_native(observed, path, "absent")
        (output / "build-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    return observed


def verify_binding(capture, observed, manifest):
    path = manifest.get("jaxlib_build_manifest")
    distribution = observed["runtime"]["jaxlib"]["distribution"]
    if path is None:
        if distribution["kind"] == "build-manifest" or (capture / "build-binding.json").exists():
            raise ValueError("capture omits its build manifest selection")
        return None
    from pass_events_probe import bind_native
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or distribution.get("build_manifest") != str(path.relative_to(ROOT)):
        raise ValueError("capture build manifest differs from environment identity")
    binding = bind_native(observed, path, "absent")
    saved = json.loads((capture / "build-binding.json").read_text())
    if saved != binding:
        raise ValueError("capture native/wheel binding differs")
    return binding


def verify_current_reader(binding):
    """Bind a live native HLO/cost reader to the same accepted source build."""
    if binding is None:
        return None
    from pass_events_probe import bind_native, load_baseline_module
    path = ROOT / binding["build_manifest_path"]
    observed = load_baseline_module().capture_baseline(path)
    current = bind_native(observed, path, "absent")
    if current["build_manifest"] != binding["build_manifest"]:
        raise ValueError("current native reader selected another build")
    return {"source_build_verified": True, "build_id": current["build_id"],
            "native_payloads": {b["wheel_member"]: b["sha256"]
                                for b in current["matched_loaded_payloads"]},
            "scope": "Current verifier process imports and loaded native payloads match the capture's source-built wheel."}
