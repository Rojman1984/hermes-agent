"""Tests for Phase 8.1 — schema changes + BlobStore CRUD.

Covers:
- Schema: artifacts array validates, backward compat without artifacts,
  artifact_stale enum value validates.
- Blob write/read round-trip: byte-identical.
- Blob write atomicity: multiprocessing concurrency, 0 crashes, all present.
- Blob exact-match dedup: identical bytes → same hash, single file.
- Blob index integrity: SQLite index valid after concurrent writes.
- Blob size limit: oversized blobs rejected.
- BlobStore composition in SeedVault: vault.blob_store attribute exists.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.blobstore import BlobStore, DEFAULT_MAX_BLOB_BYTES
from plugins.memory.seedvault.extractor import _utc_now


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugins" / "memory" / "seedvault" / "schemas" / "memory_seed.schema.json"
)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _make_minimal_seed() -> dict:
    return {
        "id": "env-ollama-001",
        "core_claim": "Ollama runs on 127.0.0.1:11434",
        "chunks": [{"type": "status", "content": "Ollama is up"}],
        "meristems": [],
        "source_ref": {"session_id": "s1", "profile": "default"},
        "trust_score": 0.8,
        "trust_history": [
            {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
        ],
        "status": "active",
        "superseded_by": [],
        "superseded_at": None,
        "tags": ["env", "ollama"],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
    }


class TestSchemaValidation:
    """Test that the schema accepts the new artifacts field and enum value."""

    def test_seed_with_artifacts_validates(self):
        """A seed with the artifacts array validates against the schema."""
        schema = _load_schema()
        seed = _make_minimal_seed()
        seed["artifacts"] = [
            {
                "blob_hash": "a" * 64,
                "content_type": "bash",
                "language": None,
                "byte_length": 42,
            }
        ]
        # Manual validation: check all required fields present and artifacts shape
        required = schema["required"]
        for field in required:
            assert field in seed, f"Missing required field: {field}"
        # Check artifacts structure
        assert "artifacts" in schema["properties"]
        item_schema = schema["properties"]["artifacts"]["items"]
        for art in seed["artifacts"]:
            for req_field in item_schema["required"]:
                assert req_field in art
            assert art["content_type"] in item_schema["properties"]["content_type"]["enum"]
            assert len(art["blob_hash"]) == 64

    def test_seed_without_artifacts_validates(self):
        """A seed without the artifacts field still validates (backward compat)."""
        schema = _load_schema()
        seed = _make_minimal_seed()
        # artifacts is NOT in the required list
        assert "artifacts" not in schema["required"]
        # All required fields are present
        for field in schema["required"]:
            assert field in seed

    def test_artifact_stale_enum_in_schema(self):
        """The artifact_stale reason is in the trust_history reason enum."""
        schema = _load_schema()
        reasons = schema["properties"]["trust_history"]["items"]["properties"]["reason"]["enum"]
        assert "artifact_stale" in reasons

    def test_artifact_blob_hash_pattern(self):
        """The blob_hash pattern accepts a valid 64-char hex string."""
        schema = _load_schema()
        pattern = schema["properties"]["artifacts"]["items"]["properties"]["blob_hash"]["pattern"]
        import re
        assert re.match(pattern, "a" * 64)
        assert not re.match(pattern, "short")

    def test_artifact_content_types(self):
        """All four content_type values are in the enum."""
        schema = _load_schema()
        types = schema["properties"]["artifacts"]["items"]["properties"]["content_type"]["enum"]
        assert set(types) == {"bash", "code", "diff", "config"}


# ---------------------------------------------------------------------------
# BlobStore CRUD tests
# ---------------------------------------------------------------------------

class TestBlobStore:
    """Test BlobStore write/read/dedup/exists operations."""

    def test_blob_write_read_roundtrip(self, tmp_path):
        """Write bytes, read back, assert byte-identical."""
        store = BlobStore(tmp_path / "vault")
        raw = b"def hello():\n    print('world')\n"
        blob_hash = store.write_blob(raw, "test-seed-001", "code", "python")
        assert blob_hash is not None
        assert len(blob_hash) == 64  # SHA-256 hex

        read_back = store.read_blob(blob_hash)
        assert read_back is not None
        assert read_back == raw  # byte-identical

    def test_blob_exists(self, tmp_path):
        """blob_exists returns True for written blobs, False for unknown."""
        store = BlobStore(tmp_path / "vault")
        raw = b"systemctl restart ollama"
        blob_hash = store.write_blob(raw, "test-seed-002", "bash")
        assert blob_hash is not None
        assert store.blob_exists(blob_hash)  # type: ignore[arg-type]
        assert not store.blob_exists("0" * 64)

    def test_blob_dedup_same_bytes(self, tmp_path):
        """Identical bytes submitted twice → same hash, single blob file."""
        store = BlobStore(tmp_path / "vault")
        raw = b"export API_KEY=secret123"
        h1 = store.write_blob(raw, "seed-a", "bash")
        h2 = store.write_blob(raw, "seed-b", "bash")
        assert h1 is not None and h2 is not None
        assert h1 == h2  # same content → same hash
        assert store.blob_exists(h1)  # type: ignore[arg-type]

        # Only one blob file on disk
        blob_path = store.blobs_dir / h1[:2] / f"{h1}.blob"
        assert blob_path.exists()
        # Count blob files (excluding .db, .wal, .shm)
        blob_files = list(store.blobs_dir.rglob("*.blob"))
        assert len(blob_files) == 1

    def test_blob_dedup_different_bytes(self, tmp_path):
        """Different bytes → different hash, two blob files."""
        store = BlobStore(tmp_path / "vault")
        h1 = store.write_blob(b"command one", "seed-a", "bash")
        h2 = store.write_blob(b"command two", "seed-a", "bash")
        assert h1 is not None and h2 is not None
        assert h1 != h2
        assert store.blob_exists(h1)  # type: ignore[arg-type]
        assert store.blob_exists(h2)  # type: ignore[arg-type]

    def test_blob_metadata(self, tmp_path):
        """get_blob_metadata returns the indexed metadata."""
        store = BlobStore(tmp_path / "vault")
        raw = b"pip install pytest"
        blob_hash = store.write_blob(raw, "env-pip-001", "bash")
        assert blob_hash is not None
        meta = store.get_blob_metadata(blob_hash)  # type: ignore[arg-type]
        assert meta is not None
        assert meta["seed_id"] == "env-pip-001"
        assert meta["content_type"] == "bash"
        assert meta["byte_length"] == len(raw)
        assert "created_at" in meta

    def test_blob_size_limit_rejects(self, tmp_path):
        """Blobs exceeding max_blob_bytes are rejected."""
        store = BlobStore(tmp_path / "vault", max_blob_bytes=100)
        raw = b"x" * 200  # exceeds 100 byte limit
        blob_hash = store.write_blob(raw, "seed-big", "code")
        assert blob_hash is None
        assert store.count() == 0

    def test_blob_empty_bytes_rejected(self, tmp_path):
        """Empty bytes are rejected."""
        store = BlobStore(tmp_path / "vault")
        blob_hash = store.write_blob(b"", "seed-empty", "code")
        assert blob_hash is None

    def test_blob_find_by_seed_id(self, tmp_path):
        """find_by_seed_id returns all blobs for a seed."""
        store = BlobStore(tmp_path / "vault")
        store.write_blob(b"cmd1", "seed-x", "bash")
        store.write_blob(b"cmd2", "seed-x", "bash")
        store.write_blob(b"cmd3", "seed-y", "bash")
        blobs = store.find_by_seed_id("seed-x")
        assert len(blobs) == 2
        for b in blobs:
            assert b["seed_id"] == "seed-x"

    def test_blob_read_missing_returns_none(self, tmp_path):
        """read_blob for a non-existent hash returns None."""
        store = BlobStore(tmp_path / "vault")
        assert store.read_blob("0" * 64) is None

    def test_blob_count(self, tmp_path):
        """count() returns the total number of indexed blobs."""
        store = BlobStore(tmp_path / "vault")
        assert store.count() == 0
        store.write_blob(b"a", "s1", "bash")
        store.write_blob(b"b", "s1", "bash")
        store.write_blob(b"a", "s2", "bash")  # dedup with first
        assert store.count() == 2  # 2 unique blobs

    def test_blob_all_hashes(self, tmp_path):
        """all_hashes() returns every indexed hash."""
        store = BlobStore(tmp_path / "vault")
        h1 = store.write_blob(b"data1", "s1", "code")
        h2 = store.write_blob(b"data2", "s1", "code")
        hashes = store.all_hashes()
        assert set(hashes) == {h1, h2}

    def test_blob_file_on_disk_matches_hash(self, tmp_path):
        """The blob file path matches the content hash and sharding scheme."""
        store = BlobStore(tmp_path / "vault")
        raw = b"test content for sharding"
        blob_hash = store.write_blob(raw, "seed-1", "code", "python")
        assert blob_hash is not None
        expected_path = store.blobs_dir / blob_hash[:2] / f"{blob_hash}.blob"
        assert expected_path.exists()
        with open(expected_path, "rb") as f:
            assert f.read() == raw

    def test_blob_index_db_exists(self, tmp_path):
        """The SQLite index database file exists after writing a blob."""
        store = BlobStore(tmp_path / "vault")
        store.write_blob(b"data", "s1", "code")
        assert store.db_path.exists()

    def test_blob_index_db_wal_mode(self, tmp_path):
        """The SQLite index is in WAL mode."""
        import sqlite3
        store = BlobStore(tmp_path / "vault")
        store.write_blob(b"data", "s1", "code")
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


# ---------------------------------------------------------------------------
# SeedVault + BlobStore composition tests
# ---------------------------------------------------------------------------

class TestSeedVaultBlobStoreComposition:
    """Test that SeedVault properly composes BlobStore."""

    def test_vault_has_blob_store(self, tmp_path):
        """SeedVault has a blob_store attribute after construction."""
        vault = SeedVault(tmp_path / "vault")
        assert hasattr(vault, "blob_store")
        assert isinstance(vault.blob_store, BlobStore)

    def test_vault_blobs_dir_created(self, tmp_path):
        """SeedVault.ensure_dirs() creates the blobs/ directory."""
        vault = SeedVault(tmp_path / "vault")
        assert (tmp_path / "vault" / "blobs").is_dir()

    def test_vault_blob_store_writable(self, tmp_path):
        """Blob written via vault.blob_store is readable."""
        vault = SeedVault(tmp_path / "vault")
        raw = b"git push origin main"
        h = vault.blob_store.write_blob(raw, "git-push-001", "bash")
        assert h is not None
        assert vault.blob_store.read_blob(h) == raw


# ---------------------------------------------------------------------------
# Blob concurrency tests
# ---------------------------------------------------------------------------

def _worker_write_blob(
    vault_dir: str,
    blob_data: bytes,
    seed_id: str,
    content_type: str,
    errors: list,
):
    """Worker process: create a BlobStore, write a blob, check for errors."""
    try:
        store = BlobStore(Path(vault_dir) / "blobs" / "..")
        # Use the vault_dir directly as the vault root
        store = BlobStore(Path(vault_dir))
        h = store.write_blob(blob_data, seed_id, content_type)
        if h is None:
            errors.append(f"{seed_id}: write_blob returned None")
        store.close()
    except Exception as e:
        errors.append(f"{seed_id}: {type(e).__name__}: {e}")


def _worker_write_blob_unique(
    vault_dir: str,
    worker_idx: int,
    errors: list,
):
    """Worker process: write a unique blob to the same vault."""
    try:
        store = BlobStore(Path(vault_dir))
        raw = f"unique content from worker {worker_idx}".encode()
        h = store.write_blob(raw, f"seed-{worker_idx:03d}", "code", "python")
        if h is None:
            errors.append(f"worker-{worker_idx}: write_blob returned None")
        store.close()
    except Exception as e:
        errors.append(f"worker-{worker_idx}: {type(e).__name__}: {e}")


def _worker_write_blob_same(
    vault_dir: str,
    worker_idx: int,
    errors: list,
):
    """Worker process: write the SAME blob content to the same vault (dedup stress)."""
    try:
        store = BlobStore(Path(vault_dir))
        raw = b"shared content for dedup stress test"
        h = store.write_blob(raw, f"seed-shared-{worker_idx:03d}", "bash")
        if h is None:
            errors.append(f"worker-{worker_idx}: write_blob returned None")
        store.close()
    except Exception as e:
        errors.append(f"worker-{worker_idx}: {type(e).__name__}: {e}")


class TestBlobConcurrency:
    """Test concurrent blob writes from multiple processes."""

    def test_concurrent_unique_blobs_no_crash(self, tmp_path):
        """Spawn 8 processes, each writing a UNIQUE blob to the same vault.

        All should succeed, all blobs should be present, no crashes.
        """
        vault_dir = str(tmp_path / "vault")
        # Pre-create the vault so BlobStore.ensure_dirs exists
        SeedVault(Path(vault_dir))

        n_workers = 8
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            p = multiprocessing.Process(
                target=_worker_write_blob_unique,
                args=(vault_dir, i, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        assert len(errors) == 0, f"Worker errors: {list(errors)}"

        # Verify all 8 unique blobs are in the index
        store = BlobStore(Path(vault_dir))
        assert store.count() == n_workers
        store.close()

    def test_concurrent_same_blobs_dedup(self, tmp_path):
        """Spawn 8 processes writing the SAME content — dedup to 1 blob, no crashes."""
        vault_dir = str(tmp_path / "vault")
        SeedVault(Path(vault_dir))

        n_workers = 8
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            p = multiprocessing.Process(
                target=_worker_write_blob_same,
                args=(vault_dir, i, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        assert len(errors) == 0, f"Worker errors: {list(errors)}"

        # All 8 writes of the same content → 1 unique blob in the index
        store = BlobStore(Path(vault_dir))
        assert store.count() == 1
        store.close()

    def test_concurrent_blob_index_integrity(self, tmp_path):
        """After concurrent writes, the SQLite index is valid and all blobs
        are accounted for (index entries match files on disk)."""
        vault_dir = str(tmp_path / "vault")
        SeedVault(Path(vault_dir))

        n_workers = 12
        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        for i in range(n_workers):
            p = multiprocessing.Process(
                target=_worker_write_blob_unique,
                args=(vault_dir, i, errors),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)

        assert len(errors) == 0, f"Worker errors: {list(errors)}"

        # Verify index integrity
        store = BlobStore(Path(vault_dir))
        all_hashes = store.all_hashes()
        assert len(all_hashes) == n_workers

        # Every indexed hash has a corresponding file on disk
        for h in all_hashes:
            blob_path = store.blobs_dir / h[:2] / f"{h}.blob"
            assert blob_path.exists(), f"Blob file missing for hash {h}"
            # Verify the file content matches the hash
            with open(blob_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == h, f"Content hash mismatch for {h}"

        # Every file on disk has a corresponding index entry
        disk_blobs = list(store.blobs_dir.rglob("*.blob"))
        assert len(disk_blobs) == n_workers

        store.close()

    def test_concurrent_mixed_unique_and_same(self, tmp_path):
        """Mix of unique and shared content — all succeed, correct count."""
        vault_dir = str(tmp_path / "vault")
        SeedVault(Path(vault_dir))

        manager = multiprocessing.Manager()
        errors = manager.list()
        procs = []

        # 4 unique writers
        for i in range(4):
            p = multiprocessing.Process(
                target=_worker_write_blob_unique,
                args=(vault_dir, i, errors),
            )
            procs.append(p)

        # 4 same-content writers (dedup to 1)
        for i in range(4, 8):
            p = multiprocessing.Process(
                target=_worker_write_blob_same,
                args=(vault_dir, i, errors),
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)

        assert len(errors) == 0, f"Worker errors: {list(errors)}"

        # 4 unique + 1 dedup = 5 total blobs
        store = BlobStore(Path(vault_dir))
        assert store.count() == 5
        store.close()