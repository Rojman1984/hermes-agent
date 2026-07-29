"""Concurrency tests for SeedVault vault writes.

Verifies that concurrent writers (separate processes) don't crash or corrupt
data when writing to the same vault directory simultaneously. This tests the
B4 fix: per-writer unique tmp filenames in _save_manifest() and save_digest().
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.extractor import _utc_now


def _make_seed(seed_id: str, claim: str) -> dict:
    return {
        "id": seed_id,
        "core_claim": claim,
        "chunks": [{"type": "fact", "content": claim}],
        "meristems": [],
        "source_ref": {"session_id": "s1", "profile": "default"},
        "trust_score": 0.8,
        "trust_history": [
            {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
        ],
        "status": "active",
        "superseded_by": [],
        "superseded_at": None,
        "tags": ["env", "test"],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
    }


def _worker_write_seed(vault_dir: str, seed_id: str, claim: str, errors: list):
    """Worker process: create a SeedVault, write a seed, return error if any."""
    try:
        vault = SeedVault(Path(vault_dir))
        seed = _make_seed(seed_id, claim)
        result = vault.write_seed(seed)
        if not result:
            errors.append(f"{seed_id}: write_seed returned False")
    except Exception as e:
        errors.append(f"{seed_id}: {type(e).__name__}: {e}")


def _worker_save_digest(vault_dir: str, digest_data: dict, errors: list):
    """Worker process: create a SeedVault, save a digest, return error if any."""
    try:
        vault = SeedVault(Path(vault_dir))
        vault.save_digest(digest_data)
    except Exception as e:
        errors.append(f"digest-{digest_data.get('compaction_count', '?')}: {type(e).__name__}: {e}")


class TestConcurrentManifestWrites:
    """Test that concurrent _save_manifest() calls don't crash."""

    def test_concurrent_seed_writes_no_crash(self, tmp_path):
        """Spawn 8 processes, each writing a different seed to the same vault.

        Without the B4 fix (unique tmp paths), several processes would crash
        with FileNotFoundError when os.replace() unlinks a shared tmp file.
        With the fix, all writes succeed and all seeds are present.
        """
        vault_dir = str(tmp_path / "shared_vault")
        n_workers = 8
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            seed_id = f"env-concurrent-{i:03d}"
            claim = f"Concurrent seed number {i} from process {os.getpid()}"
            p = multiprocessing.Process(
                target=_worker_write_seed,
                args=(vault_dir, seed_id, claim, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        # Assert no process crashed
        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {n_workers} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Assert all seeds landed in the vault
        vault = SeedVault(Path(vault_dir))
        for i in range(n_workers):
            seed_id = f"env-concurrent-{i:03d}"
            seed = vault.get_seed(seed_id)
            assert seed is not None, f"Seed {seed_id} missing from vault"
            assert seed["core_claim"] == f"Concurrent seed number {i} from process {os.getpid()}"

    def test_concurrent_manifest_writes_preserve_integrity(self, tmp_path):
        """Repeated concurrent writes produce a valid JSON manifest."""
        vault_dir = str(tmp_path / "shared_vault_int")
        n_workers = 6
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            seed_id = f"env-integrity-{i:03d}"
            claim = f"Integrity test seed {i}"
            p = multiprocessing.Process(
                target=_worker_write_seed,
                args=(vault_dir, seed_id, claim, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        assert not list(errors), f"Worker errors: {list(errors)}"

        # Manifest must be valid JSON with all seeds accounted for
        manifest_path = Path(vault_dir) / "vault_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Each worker wrote a different seed; all should be in the manifest
        for i in range(n_workers):
            seed_id = f"env-integrity-{i:03d}"
            assert seed_id in manifest["seeds"], f"Seed {seed_id} missing from manifest"


class TestConcurrentDigestWrites:
    """Test that concurrent save_digest() calls don't crash."""

    def test_concurrent_digest_writes_no_crash(self, tmp_path):
        """Spawn 6 processes, each saving a different state digest.

        Without the B4 fix, save_digest() wrote directly to the live file
        (no tmp + replace), so concurrent writers would truncate each other's
        data mid-write. With the fix, all writes succeed via unique tmp + replace.
        """
        vault_dir = str(tmp_path / "shared_vault_digest")
        # Create the vault dir first so SeedVault() init works
        SeedVault(Path(vault_dir))

        n_workers = 6
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            digest_data = {
                "current_task": f"task-{i}",
                "active_seed_ids": [f"seed-{i}"],
                "pending_actions": [],
                "compaction_count": i,
                "last_compaction": _utc_now(),
                "session_lineage": [f"session-{i}"],
            }
            p = multiprocessing.Process(
                target=_worker_save_digest,
                args=(vault_dir, digest_data, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {n_workers} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Digest file must be valid JSON (not truncated/corrupted)
        digest_path = Path(vault_dir) / "state_digest.json"
        with open(digest_path) as f:
            digest = json.load(f)

        # The last writer wins (that's expected for a shared file); verify it's valid
        assert "current_task" in digest
        assert "compaction_count" in digest
        assert isinstance(digest["active_seed_ids"], list)


class TestConcurrentMixedWrites:
    """Test concurrent seed writes + digest writes simultaneously."""

    def test_mixed_concurrent_writes(self, tmp_path):
        """4 seed writers + 4 digest writers, all hitting the same vault dir."""
        vault_dir = str(tmp_path / "shared_vault_mixed")
        SeedVault(Path(vault_dir))  # init dirs

        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        # 4 seed writers
        for i in range(4):
            seed_id = f"env-mixed-{i:03d}"
            claim = f"Mixed test seed {i}"
            p = multiprocessing.Process(
                target=_worker_write_seed,
                args=(vault_dir, seed_id, claim, errors),
            )
            procs.append(p)
            p.start()

        # 4 digest writers
        for i in range(4):
            digest_data = {
                "current_task": f"mixed-task-{i}",
                "active_seed_ids": [],
                "pending_actions": [],
                "compaction_count": i,
                "last_compaction": _utc_now(),
                "session_lineage": [],
            }
            p = multiprocessing.Process(
                target=_worker_save_digest,
                args=(vault_dir, digest_data, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {len(procs)} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Verify all 4 seeds are present
        vault = SeedVault(Path(vault_dir))
        for i in range(4):
            seed_id = f"env-mixed-{i:03d}"
            assert vault.get_seed(seed_id) is not None, f"Seed {seed_id} missing"

        # Verify digest is valid JSON
        digest_path = Path(vault_dir) / "state_digest.json"
        with open(digest_path) as f:
            digest = json.load(f)
        assert "current_task" in digest