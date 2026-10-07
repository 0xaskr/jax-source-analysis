#!/usr/bin/env python3
"""Offline integrity checks for diagram source caches and absent checkouts."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from component_diagram_sources import source_metadata, ANCHORS, PINS

ROOT=Path(__file__).resolve().parents[1]


class PinnedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('upstream-sources.lock','tools/component_diagram_sources.json','tools/overview_flow_sources.json'):
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        self.repo,self.path,*_=ANCHORS['PjRtBuffer']
        self.cached=self.root/'artifacts/environment/overview-sources'/PINS[self.repo]/self.path
        source=ROOT/'artifacts/environment/overview-sources'/PINS[self.repo]/self.path
        if not source.exists():self.skipTest('Populate the pinned diagram source cache first')
        self.cached.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,self.cached)
        (self.root/'upstream'/self.repo).mkdir(parents=True)

    def read(self):
        return source_metadata(self.root,{'PjRtBuffer'})

    def test_missing_checkout_uses_verified_cache_without_parent_git(self):
        with patch('component_diagram_sources.subprocess.check_output',side_effect=AssertionError('Must not read parent repository HEAD')):
            meta=self.read()
        self.assertEqual(meta['source_pins'][self.repo],PINS[self.repo])
        self.assertEqual(len(meta['source_files']),1)

    def test_changed_cache_is_rejected(self):
        self.cached.write_bytes(self.cached.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):self.read()

    def test_missing_evidence_is_reported_as_missing(self):
        self.cached.unlink()
        with self.assertRaisesRegex(ValueError,'Missing pinned source'):self.read()

    def test_modified_local_source_is_rejected(self):
        local=self.root/'upstream'/self.repo/self.path
        local.parent.mkdir(parents=True,exist_ok=True);local.write_text('modified source')
        with self.assertRaisesRegex(ValueError,'Modified source file'):self.read()

    def test_changed_lock_is_rejected(self):
        lock=self.root/'upstream-sources.lock'
        lock.write_text(lock.read_text().replace(PINS[self.repo],'0'*40))
        with self.assertRaisesRegex(ValueError,'lock changed'):self.read()


if __name__=='__main__':unittest.main()
