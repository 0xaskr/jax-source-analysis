#!/usr/bin/env python3
"""Isolated environment-sync contracts; no network, Docker, or system changes."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("environment_sync", REPOSITORY / "tools/sync-environment.py")
ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENV)


def git(root, *arguments):
    result = subprocess.run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
         "-C", str(root), *arguments], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    return result.stdout.strip()


def commit(root):
    git(root, "-c", "user.name=Environment Selftest", "-c",
        "user.email=environment-selftest@example.invalid", "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


class EnvironmentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jax-environment-selftest-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "workspace with spaces"
        self.root.mkdir()
        self.root_patch = patch.object(ENV, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def source_fixture(self, initialize=False):
        origin = self.root.parent / "origin"
        origin.mkdir()
        git(origin, "init", "-q")
        (origin / "source.py").write_text("value = 1\n")
        (origin / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        git(origin, "add", ".")
        revision = commit(origin)
        git(self.root, "init", "-q")
        relative = "upstream/jax"
        (self.root / ".gitmodules").write_text(
            f'[submodule "{relative}"]\n\tpath = {relative}\n\turl = {origin.as_uri()}\n')
        (self.root / "upstream-sources.lock").write_text(
            f"core|{relative}|{origin.as_uri()}|{revision}|fixture\n")
        (self.root / ".gitignore").write_text("artifacts/\n")
        git(self.root, "add", ".")
        git(self.root, "update-index", "--add", "--cacheinfo", f"160000,{revision},{relative}")
        commit(self.root)
        lock = {"source_paths": [relative]}
        if initialize:
            self.synchronize_fixture(lock)
        return lock, self.root / relative

    def synchronize_fixture(self, lock):
        original = ENV.git
        def local_git(*arguments, **kwargs):
            return original("-c", "protocol.file.allow=always", *arguments, **kwargs)
        with patch.object(ENV, "git", local_git):
            ENV.sync_sources(lock)

    def test_sync_preserves_index_user_files_and_repeats(self):
        lock, source = self.source_fixture()
        index = (self.root / ".git/index").read_bytes()
        (self.root / "user-notes.txt").write_text("preserve me\n")
        self.synchronize_fixture(lock)
        self.synchronize_fixture(lock)
        self.assertEqual((self.root / ".git/index").read_bytes(), index)
        self.assertEqual((self.root / "user-notes.txt").read_text(), "preserve me\n")
        self.assertTrue((source / "source.py").is_file())

    def test_dirty_source_is_rejected_without_checkout(self):
        lock, source = self.source_fixture(initialize=True)
        (source / "source.py").write_text("user edit\n")
        with self.assertRaisesRegex(ENV.EnvironmentError, "local changes"):
            ENV.sync_sources(lock)
        self.assertEqual((source / "source.py").read_text(), "user edit\n")

    def test_resume_clone_before_first_checkout(self):
        lock, source = self.source_fixture()
        source.parent.mkdir()
        git(self.root, "clone", "--no-checkout", (self.root.parent / "origin").as_uri(), str(source))
        self.assertTrue(ENV.unpopulated_source("upstream/jax"))
        self.synchronize_fixture(lock)
        self.assertEqual((source / "source.py").read_text(), "value = 1\n")

    def test_nonempty_uninitialized_directory_is_preserved(self):
        lock, source = self.source_fixture()
        source.mkdir(parents=True)
        (source / "user.txt").write_text("keep")
        with self.assertRaisesRegex(ENV.EnvironmentError, "not empty"):
            ENV.sync_sources(lock)
        self.assertEqual((source / "user.txt").read_text(), "keep")

    def test_staged_gitlink_mismatch_is_rejected(self):
        lock, _ = self.source_fixture()
        git(self.root, "update-index", "--cacheinfo", "160000," + "1" * 40 + ",upstream/jax")
        with self.assertRaisesRegex(ENV.EnvironmentError, "index gitlink"):
            ENV.sync_sources(lock)

    def test_local_remote_override_is_rejected(self):
        lock, _ = self.source_fixture()
        git(self.root, "config", "submodule.upstream/jax.url", "https://example.invalid/elsewhere.git")
        with self.assertRaisesRegex(ENV.EnvironmentError, "local submodule URL"):
            ENV.sync_sources(lock)

    def test_multiple_origin_urls_are_rejected(self):
        lock, source = self.source_fixture(initialize=True)
        expected = git(source, "config", "--get", "remote.origin.url")
        git(source, "config", "remote.origin.url", "https://example.invalid/elsewhere.git")
        git(source, "config", "--add", "remote.origin.url", expected)
        with self.assertRaisesRegex(ENV.EnvironmentError, "source origin URL"):
            ENV.sync_sources(lock)

    def test_source_symlink_is_rejected(self):
        lock, source = self.source_fixture()
        source.parent.mkdir()
        source.symlink_to(self.root.parent / "origin", target_is_directory=True)
        with self.assertRaisesRegex(ENV.EnvironmentError, "symlink"):
            ENV.inspect_sources(lock)

    def test_download_hash_failure_preserves_previous_file(self):
        target = self.root / "tool"
        target.write_bytes(b"existing")
        identity = {"sha256": hashlib.sha256(b"expected").hexdigest()}
        with self.assertRaisesRegex(ENV.EnvironmentError, "SHA-256 mismatch"):
            ENV.write_verified(target, b"corrupt", identity)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_download_cannot_escape_via_cache_symlink(self):
        (self.root / "artifacts").symlink_to(self.root.parent, target_is_directory=True)
        data = b"verified"
        with self.assertRaisesRegex(ENV.EnvironmentError, "symlink"):
            ENV.write_verified(self.root / "artifacts/tool", data,
                               {"sha256": hashlib.sha256(data).hexdigest()})

    def test_bytecode_quarantine_preserves_source_and_other_files(self):
        lock, source = self.source_fixture(initialize=True)
        cache = source / "__pycache__"
        cache.mkdir()
        (cache / "source.pyc").write_bytes(b"cache")
        (cache / "keep.txt").write_bytes(b"user data")
        ENV.quarantine_bytecode(lock)
        self.assertFalse((cache / "source.pyc").exists())
        self.assertEqual((cache / "keep.txt").read_bytes(), b"user data")
        backups = list((self.root / "artifacts/environment/bytecode").rglob("source.pyc"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"cache")

    def test_environment_cannot_redirect_uv_or_inject_python(self):
        with patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "/tmp/other-env",
                                     "UV_PYTHON": "/tmp/other-python",
                                     "PYTHONPATH": "/tmp/injected"}):
            result = ENV.process_environment(self.root / "bin/uv")
        self.assertEqual(result["UV_PROJECT_ENVIRONMENT"], str(self.root / ".venv"))
        self.assertNotIn("UV_PYTHON", result)
        self.assertNotIn("PYTHONPATH", result)
        self.assertFalse((self.root / "artifacts").exists())

    def test_check_rejects_bytecode_without_cleaning_it(self):
        lock, source = self.source_fixture(initialize=True)
        bytecode = source / "generated.pyc"
        bytecode.write_bytes(b"cache")
        with self.assertRaisesRegex(ENV.EnvironmentError, "bytecode files"):
            ENV.verify_workspace(lock, self.root / "uv")
        self.assertEqual(bytecode.read_bytes(), b"cache")

    def test_failed_command_stops_execution(self):
        with self.assertRaises(subprocess.CalledProcessError):
            ENV.run(["/bin/sh", "-c", "exit 17"])

    @unittest.skipUnless(Path("/usr/lib/apt/apt-helper").exists(), "Ubuntu APT helper required")
    def test_apt_helper_requires_the_selected_sha256(self):
        source = self.root / "package.deb"
        source.write_bytes(b"immutable package fixture\n")
        good = hashlib.sha256(source.read_bytes()).hexdigest()
        for digest in (good, "0" * 64):
            result = subprocess.run(
                ["/usr/lib/apt/apt-helper", "download-file", source.as_uri(),
                 str(self.root / digest), "SHA256:" + digest], capture_output=True,
            )
            self.assertEqual(result.returncode == 0, digest == good)

    def test_concurrent_sync_is_rejected(self):
        with ENV.sync_lock():
            with self.assertRaisesRegex(ENV.EnvironmentError, "another environment"):
                with ENV.sync_lock():
                    self.fail("second synchronization obtained the lock")

    def test_image_recipe_change_rejects_stale_image(self):
        (self.root / "env").mkdir()
        for name in ("Dockerfile", "environment.lock.json", "verify-image.py",
                     "install-packages.sh", "fetch-apt-package.sh"):
            (self.root / "env" / name).write_text("fixture\n")
        output = json.dumps([{"Id": "sha256:" + "0" * 64,
                              "Config": {"Labels": {"org.jax-source-analysis.environment": "old"}}}])
        with patch.object(ENV, "run", return_value=output):
            with self.assertRaisesRegex(ENV.EnvironmentError, "build recipe"):
                ENV.docker_image()


if __name__ == "__main__":
    unittest.main(verbosity=2)
