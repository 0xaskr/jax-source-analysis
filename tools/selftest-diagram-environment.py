#!/usr/bin/env python3
"""Check drawing dependency integrity and CLI availability without downloads."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import diagram_environment as env


class DrawingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="diagram-environment-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock = self.root / "diagrams.lock.json"
        self.lock.write_text(json.dumps({
            "schema_version": 1, "platform": "ubuntu-24.04-amd64",
            "python_abi": "3.12", "font_family": "Noto Sans CJK SC", "packages": [],
        }))
        for name, value in (("ROOT", self.root), ("LOCK", self.lock), ("CACHE", self.root / "cache")):
            context = patch.object(env, name, value)
            context.start()
            self.addCleanup(context.stop)

    def fixture(self):
        target = env.bundle()
        native = target / "root/usr/lib/fixture.so"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"verified native library")
        (target / "manifest.json").write_text(json.dumps({
            "lock_sha256": env.digest(self.lock),
            "files": {"usr/lib/fixture.so": env.digest(native)},
        }))
        return native

    def test_modified_dependency_is_rejected(self):
        native = self.fixture()
        env.verify_bundle()
        native.write_bytes(b"different native library")
        with self.assertRaisesRegex(RuntimeError, "dependency changed"):
            env.verify_bundle()

    def test_changed_lock_requires_a_new_bundle(self):
        self.fixture()
        self.lock.write_text(self.lock.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            env.verify_bundle()

    def test_corrupt_download_is_not_cached(self):
        package = {"name": "fixture", "version": "1", "size_bytes": 5,
                   "url": "https://archive.ubuntu.com/ubuntu/pool/fixture.deb",
                   "sha256": hashlib.sha256(b"valid").hexdigest()}
        with patch.object(env.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 or size mismatch"):
                env.download(package)
        self.assertEqual(list((env.CACHE / "downloads").iterdir()), [])

    def test_child_uses_only_locked_fonts_and_bindings(self):
        self.fixture()
        with patch.dict(env.os.environ, {"PYTHONPATH": "/ambient", "FONTCONFIG_FILE": "/ambient/fonts.conf"}):
            configured = env.environment()
            self.assertNotIn("/ambient", configured["PYTHONPATH"])
            self.assertIn(str(env.bundle()), configured["FONTCONFIG_FILE"])
            self.assertEqual(env.os.environ["FONTCONFIG_FILE"], "/ambient/fonts.conf")


class RendererCliTests(unittest.TestCase):
    def test_help_needs_no_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        for script in ("render_pallas_inner_outer.py", "render_overview_software_stack_components.py",
                       "render_jaxpr_centered_hub.py", "render_software_stack_component_hubs.py",
                       "render_software_stack_component_flows.py", "render_jax_internal_stack.py",
                       "render_jaxpr_to_mlir.py"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, "-B", "-S", str(root / "tools" / script), "--help"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--help", result.stdout)


if __name__ == "__main__":
    unittest.main()
