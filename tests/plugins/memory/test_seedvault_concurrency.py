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


def _worker_archive_seed(vault_dir: str, seed_id: str, errors: list):
    """Worker process: create a SeedVault, archive a seed, return error if any."""
    try:
        vault = SeedVault(Path(vault_dir))
        result = vault.archive_seed(seed_id)
        if not result:
            errors.append(f"archive-{seed_id}: archive_seed returned False")
    except Exception as e:
        errors.append(f"archive-{seed_id}: {type(e).__name__}: {e}")


def _worker_commit_seed(vault_dir: str, seed_id: str, claim: str, tag: str, errors: list):
    """Worker process: create a vault + CommitGate, commit a seed."""
    try:
        from plugins.memory.seedvault.validator import CommitGate
        vault = SeedVault(Path(vault_dir))
        gate = CommitGate(vault)
        seed = _make_seed(seed_id, claim)
        seed["tags"] = [tag]
        ok, reason = gate.commit(seed)
        if not ok:
            errors.append(f"commit-{seed_id}: rejected: {reason}")
    except Exception as e:
        errors.append(f"commit-{seed_id}: {type(e).__name__}: {e}")


class TestConcurrentArchiveWrites:
    """Test concurrent archive_seed() calls — manifest read-modify-write safety."""

    def test_concurrent_archive_no_corruption(self, tmp_path):
        """Pre-populate vault with 6 seeds, then archive all 6 concurrently.

        archive_seed() does a read-modify-write on the manifest (loads it,
        sets status=archived, saves). Without the manifest lock, concurrent
        archivers would clobber each other's status updates.
        """
        vault_dir = str(tmp_path / "shared_vault_archive")
        vault = SeedVault(Path(vault_dir))

        # Pre-populate 6 seeds
        for i in range(6):
            seed_id = f"env-archive-{i:03d}"
            seed = _make_seed(seed_id, f"Archive test seed {i}")
            vault.write_seed(seed)

        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(6):
            seed_id = f"env-archive-{i:03d}"
            p = multiprocessing.Process(
                target=_worker_archive_seed,
                args=(vault_dir, seed_id, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {len(procs)} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Verify all seeds are archived in the manifest
        vault = SeedVault(Path(vault_dir))
        manifest = vault._load_manifest()
        for i in range(6):
            seed_id = f"env-archive-{i:03d}"
            assert seed_id in manifest["seeds"], f"Seed {seed_id} missing from manifest"
            assert manifest["seeds"][seed_id]["status"] == "archived", (
                f"Seed {seed_id} status is {manifest['seeds'][seed_id]['status']}, expected archived"
            )


class TestConcurrentWriteAndArchive:
    """Test concurrent write_seed + archive_seed — mixed manifest mutations."""

    def test_concurrent_write_and_archive_mixed(self, tmp_path):
        """4 new seed writers + 4 archivers archiving pre-existing seeds.

        Both write_seed and archive_seed acquire the manifest lock file.
        Without cross-process locking, their read-modify-write cycles would
        interleave and lose entries or status changes.
        """
        vault_dir = str(tmp_path / "shared_vault_w_a")
        vault = SeedVault(Path(vault_dir))

        # Pre-populate 4 seeds to be archived
        for i in range(4):
            seed_id = f"env-pre-{i:03d}"
            seed = _make_seed(seed_id, f"Pre-existing seed {i}")
            vault.write_seed(seed)

        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        # 4 archivers (archive pre-existing seeds)
        for i in range(4):
            seed_id = f"env-pre-{i:03d}"
            p = multiprocessing.Process(
                target=_worker_archive_seed,
                args=(vault_dir, seed_id, errors),
            )
            procs.append(p)
            p.start()

        # 4 writers (write new seeds)
        for i in range(4):
            seed_id = f"env-new-{i:03d}"
            claim = f"New concurrent seed {i}"
            p = multiprocessing.Process(
                target=_worker_write_seed,
                args=(vault_dir, seed_id, claim, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {len(procs)} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Verify manifest integrity: all 8 seeds present with correct status
        vault = SeedVault(Path(vault_dir))
        manifest = vault._load_manifest()
        for i in range(4):
            pre_id = f"env-pre-{i:03d}"
            assert pre_id in manifest["seeds"], f"Pre-existing seed {pre_id} missing"
            assert manifest["seeds"][pre_id]["status"] == "archived"
        for i in range(4):
            new_id = f"env-new-{i:03d}"
            assert new_id in manifest["seeds"], f"New seed {new_id} missing from manifest"
            assert manifest["seeds"][new_id]["status"] == "active"


class TestConcurrentCommitGate:
    """Test concurrent commit() calls — end-to-end gate + vault concurrency."""

    def test_concurrent_commits_different_tags(self, tmp_path):
        """6 processes commit seeds with different primary tags concurrently.

        Each commit() calls find_superseded_candidates, find_duplicate,
        write_seed, and potentially supersede_seeds — all touching the
        shared manifest. No crashes, all seeds committed.
        """
        vault_dir = str(tmp_path / "shared_vault_commit")
        SeedVault(Path(vault_dir))  # init dirs

        n_workers = 6
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        unique_claims = [
            "Database runs on port 5432 with SSL enabled",
            "React frontend uses TypeScript and Vite bundler",
            "Kubernetes cluster has three worker nodes total",
            "Authentication via OAuth2 with PKCE flow required",
            "Logs shipped to Elasticsearch on daily rotation",
            "Memory limit set to four gigabytes per container",
        ]
        for i in range(n_workers):
            seed_id = f"env-commit-{i:03d}"
            claim = unique_claims[i]
            tag = f"tag-{i}"  # unique tags → no supersession
            p = multiprocessing.Process(
                target=_worker_commit_seed,
                args=(vault_dir, seed_id, claim, tag, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {n_workers} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # All 6 seeds should be present and active
        vault = SeedVault(Path(vault_dir))
        manifest = vault._load_manifest()
        for i in range(n_workers):
            seed_id = f"env-commit-{i:03d}"
            assert seed_id in manifest["seeds"], f"Seed {seed_id} missing from manifest"
            assert manifest["seeds"][seed_id]["status"] == "active"


class TestHighProcessStress:
    """Stress test with 16 concurrent writers — verify no data loss under load."""

    def test_16_concurrent_writers_no_loss(self, tmp_path):
        """16 processes each write a unique seed to the same vault.

        This is a higher-pressure version of the 8-process test. With 16
        processes contending on the manifest lock, the window for races is
        wider. All 16 seeds must be present in the manifest with no crashes.
        """
        vault_dir = str(tmp_path / "shared_vault_stress")
        n_workers = 16
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            seed_id = f"env-stress-{i:03d}"
            claim = f"Stress test seed {i} with unique claim text {i}"
            p = multiprocessing.Process(
                target=_worker_write_seed,
                args=(vault_dir, seed_id, claim, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=60)

        crashed = [p for p in procs if p.exitcode != 0]
        assert not crashed, f"{len(crashed)} of {n_workers} processes crashed"
        assert not list(errors), f"Worker errors: {list(errors)}"

        # Verify all 16 seeds are in the manifest
        vault = SeedVault(Path(vault_dir))
        manifest = vault._load_manifest()
        assert len(manifest["seeds"]) >= n_workers, (
            f"Manifest has {len(manifest['seeds'])} seeds, expected >= {n_workers}"
        )
        for i in range(n_workers):
            seed_id = f"env-stress-{i:03d}"
            assert seed_id in manifest["seeds"], f"Seed {seed_id} missing from manifest"
            seed = vault.get_seed(seed_id)
            assert seed is not None, f"Seed file {seed_id} missing from disk"