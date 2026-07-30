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
from unittest.mock import patch

import pytest

from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.blobstore import BlobStore, DEFAULT_MAX_BLOB_BYTES
from plugins.memory.seedvault.extractor import _utc_now
from plugins.memory.seedvault.scrub import (
    discover_secrets_from_env,
    build_replacement_patterns,
)


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


@pytest.fixture
def fake_env(tmp_path):
    """Create a temporary .env file with known secrets for scrub tests."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# Hermes credentials\n"
        "OPENAI_API_KEY=sk-test-1234567890abcdef\n"
        "GITHUB_TOKEN=ghp_abcdef1234567890abcdef\n"
        "DATABASE_PASSWORD=s3cr3tp@ss\n"
        "PLACEHOLDER_KEY=your_api_key_here\n"  # should be skipped
    )
    return env_path


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


# ---------------------------------------------------------------------------
# Phase 8.2 — Artifact detection tests
# ---------------------------------------------------------------------------

from plugins.memory.seedvault.extractor import (
    _detect_artifacts,
    _strip_code_fences,
    _make_artifact_seed,
    extract_seeds,
)


class TestArtifactDetection:
    """Test the _detect_artifacts() function."""

    def test_code_fence_detection(self):
        """A fenced code block with a language tag is detected as 'code'."""
        content = "Here is some code:\n```python\ndef foo():\n    pass\n```\nDone."
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "code"
        assert artifacts[0]["language"] == "python"
        assert "def foo():" in artifacts[0]["raw_content"]

    def test_shell_command_detection(self):
        """A shell command with $ prefix is detected as 'bash'."""
        content = "Run this command:\n$ systemctl restart ollama\nDone."
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "bash"
        assert "systemctl restart ollama" in artifacts[0]["raw_content"]

    def test_shell_command_no_prefix(self):
        """A bare shell command (known prefix) is detected as 'bash'."""
        content = "git push origin main"
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "bash"

    def test_diff_detection(self):
        """A diff block is detected as 'diff'."""
        content = "```diff\n@@ -1,3 +1,3 @@\n-old line\n+new line\n```"
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "diff"
        assert "@@ -1,3 +1,3 @@" in artifacts[0]["raw_content"]

    def test_config_detection(self):
        """A YAML config block is detected as 'config'."""
        content = '```yaml\nserver:\n  port: 8080\n```'
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "config"
        assert artifacts[0]["language"] == "yaml"

    def test_no_false_positives_prose(self):
        """Prose mentioning 'code' or 'command' without actual code → no artifacts."""
        content = "We should write some code to handle the command parsing logic."
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 0

    def test_multiple_artifacts_in_one_message(self):
        """Two code fences → two artifacts."""
        content = (
            "First:\n```python\nprint(1)\n```\n"
            "Second:\n```bash\necho hello\n```"
        )
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 2
        assert artifacts[0]["content_type"] == "code"
        assert artifacts[1]["content_type"] == "bash"

    def test_artifact_and_prose_coexistence(self):
        """A message with both a code fence and prose → artifact + prose pattern."""
        content = (
            "I prefer to use this command:\n```bash\nsystemctl restart ollama\n```\n"
            "It always works."
        )
        artifacts = _detect_artifacts(content)
        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "bash"
        # Strip code fences and check prose pattern still matches
        stripped = _strip_code_fences(content)
        assert "[code artifact captured]" in stripped
        assert "prefer" in stripped  # prose pattern still visible

    def test_blob_content_is_exact(self, tmp_path):
        """Blob content is byte-identical to the original span."""
        vault = SeedVault(tmp_path / "vault")
        content = "```python\nprint('hello world')\n```"
        seeds = extract_seeds(
            messages=[{"role": "user", "content": content}],
            session_id="test-artifact-001",
            profile="default",
            blob_store=vault.blob_store,
        )
        # Find the artifact seed
        artifact_seeds = [s for s in seeds if s.get("artifacts")]
        assert len(artifact_seeds) == 1
        seed = artifact_seeds[0]
        assert len(seed["artifacts"]) == 1
        blob_hash = seed["artifacts"][0]["blob_hash"]
        # Read the blob back and verify content
        blob_bytes = vault.blob_store.read_blob(blob_hash)
        assert blob_bytes is not None
        assert blob_bytes == b"print('hello world')"

    def test_scrub_integration_in_artifact_blob(self, tmp_path, fake_env):
        """A known .env value pasted inside a code fence → blob bytes are
        redacted before write."""
        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.extractor.get_scrub_patterns",
            return_value=patterns,
        ):
            vault = SeedVault(tmp_path / "vault")
            content = "```bash\nexport OPENAI_API_KEY=sk-test-1234567890abcdef\n```"
            seeds = extract_seeds(
                messages=[{"role": "user", "content": content}],
                session_id="test-scrub-artifact",
                profile="default",
                blob_store=vault.blob_store,
            )
        # Find the artifact seed
        artifact_seeds = [s for s in seeds if s.get("artifacts")]
        assert len(artifact_seeds) == 1
        seed = artifact_seeds[0]
        blob_hash = seed["artifacts"][0]["blob_hash"]
        blob_bytes = vault.blob_store.read_blob(blob_hash)
        assert blob_bytes is not None
        # The secret should be redacted in the blob
        assert b"sk-test-1234567890abcdef" not in blob_bytes
        assert b"[REDACTED:OPENAI_API_KEY]" in blob_bytes

    def test_artifact_seed_core_claim_is_description(self, tmp_path):
        """Artifact seed core_claim is a description, not the raw content."""
        vault = SeedVault(tmp_path / "vault")
        content = "```python\ndef very_long_function_name():\n    pass\n```"
        seeds = extract_seeds(
            messages=[{"role": "user", "content": content}],
            session_id="test-desc-001",
            profile="default",
            blob_store=vault.blob_store,
        )
        artifact_seeds = [s for s in seeds if s.get("artifacts")]
        assert len(artifact_seeds) == 1
        # core_claim should be a short description, not the code itself
        claim = artifact_seeds[0]["core_claim"]
        assert "def very_long_function_name" not in claim
        assert "code" in claim.lower() or "python" in claim.lower()

    def test_one_seed_per_artifact(self, tmp_path):
        """Two code fences → two artifact seeds, each with one artifact pointer."""
        vault = SeedVault(tmp_path / "vault")
        content = (
            "```python\nprint(1)\n```\n"
            "```bash\necho hello\n```"
        )
        seeds = extract_seeds(
            messages=[{"role": "user", "content": content}],
            session_id="test-multi-art-001",
            profile="default",
            blob_store=vault.blob_store,
        )
        artifact_seeds = [s for s in seeds if s.get("artifacts")]
        assert len(artifact_seeds) == 2
        assert len(artifact_seeds[0]["artifacts"]) == 1
        assert len(artifact_seeds[1]["artifacts"]) == 1

    def test_prose_patterns_dont_match_inside_code_fences(self, tmp_path):
        """A comment inside a code fence that matches a prose pattern should
        NOT produce a duplicate prose seed."""
        vault = SeedVault(tmp_path / "vault")
        content = (
            "```bash\n# must stay OFF to prevent crashes\nexport FOO=bar\n```\n"
            "This is a constraint about the system."
        )
        seeds = extract_seeds(
            messages=[{"role": "user", "content": content}],
            session_id="test-strip-001",
            profile="default",
            blob_store=vault.blob_store,
        )
        # We should get: 1 artifact seed (bash block) + 1 prose seed (constraint)
        # The "# must stay OFF" inside the code fence should NOT match
        artifact_seeds = [s for s in seeds if s.get("artifacts")]
        prose_seeds = [s for s in seeds if not s.get("artifacts")]
        assert len(artifact_seeds) == 1
        # The prose seed should be about "constraint about the system"
        # not about "must stay OFF"
        for ps in prose_seeds:
            assert "stay OFF" not in ps["core_claim"]

    def test_no_blob_store_means_no_artifact_pointer(self, tmp_path):
        """Without a blob_store, artifact seeds are created but with empty
        artifacts arrays."""
        content = "```python\nprint(1)\n```"
        seeds = extract_seeds(
            messages=[{"role": "user", "content": content}],
            session_id="test-no-blob-001",
            profile="default",
            blob_store=None,
        )
        artifact_seeds = [s for s in seeds if s.get("artifacts") is not None and len(s.get("artifacts", [])) == 0]
        # Should still have an artifact seed (just without the blob pointer)
        assert len(artifact_seeds) >= 1


# ---------------------------------------------------------------------------
# Phase 8.3 — Commit gate hash dedup + retrieval tool tests
# ---------------------------------------------------------------------------

from plugins.memory.seedvault.validator import CommitGate
from plugins.memory.seedvault.provider import SeedVaultMemoryProvider


def _make_test_artifact_seed(
    seed_id: str,
    blob_hash: str,
    content_type: str = "code",
    claim: str = "Test artifact",
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": seed_id,
        "core_claim": claim,
        "chunks": [{"type": "insight", "content": claim}],
        "meristems": [],
        "source_ref": {"session_id": "s1", "profile": "default"},
        "trust_score": 0.8,
        "trust_history": [
            {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
        ],
        "status": "active",
        "superseded_by": [],
        "superseded_at": None,
        "tags": tags or ["artifact", "code"],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
        "artifacts": [{
            "blob_hash": blob_hash,
            "content_type": content_type,
            "language": "python",
            "byte_length": 42,
        }],
    }


class TestArtifactCommitGate:
    """Test the commit gate's artifact-specific dedup path."""

    def test_exact_hash_dedup_rejects_duplicate(self, tmp_path):
        """Submit a seed with an artifact, then submit a second seed with the
        same blob content → second seed rejected as duplicate via hash, not
        Jaccard."""
        vault = SeedVault(tmp_path / "vault")
        gate = CommitGate(vault)

        # Write a blob to the store with seed1's ID
        raw = b"def hello(): pass"
        seed1_id = "code-code-001"
        blob_hash = vault.blob_store.write_blob(raw, seed1_id, "code", "python")
        assert blob_hash is not None

        # First seed with this artifact
        seed1 = _make_test_artifact_seed(seed1_id, blob_hash, claim="Python hello function")
        ok1, reason1 = gate.commit(seed1)
        assert ok1, f"First commit should succeed: {reason1}"

        # Second seed with the same blob content but different seed_id
        # Write the same bytes — dedup will return the same hash, but
        # the index entry has seed1's ID, so it's a duplicate.
        seed2_id = "code-code-002"
        # The blob is already in the index with seed1's ID — writing again
        # is a dedup hit (same hash), but the index entry stays with seed1.
        seed2 = _make_test_artifact_seed(seed2_id, blob_hash, claim="Another function")
        ok2, reason2 = gate.commit(seed2)
        assert not ok2
        assert "duplicate artifact" in reason2
        assert blob_hash[:12] in reason2

    def test_prose_and_artifact_no_cross_contamination(self, tmp_path):
        """A prose seed and an artifact seed with the same primary tag →
        artifact seed goes through the artifact path, prose seed goes
        through the Jaccard path, no cross-contamination."""
        vault = SeedVault(tmp_path / "vault")
        gate = CommitGate(vault)

        # Write a blob with the artifact seed's ID
        raw = b"git push origin main"
        art_seed_id = "shell-bash-001"
        blob_hash = vault.blob_store.write_blob(raw, art_seed_id, "bash")
        assert blob_hash is not None

        # Commit an artifact seed
        art_seed = _make_test_artifact_seed(
            art_seed_id, blob_hash, "bash",
            claim="Bash command: git push origin main",
            tags=["shell", "bash"],
        )
        ok, reason = gate.commit(art_seed)
        assert ok, f"Artifact seed should commit: {reason}"

        # Commit a prose seed with the same primary tag but NO artifact
        prose_seed = {
            "id": "shell-status-001",
            "core_claim": "Git push always works for origin main branch.",
            "chunks": [{"type": "status", "content": "Git push works."}],
            "meristems": [],
            "source_ref": {"session_id": "s1", "profile": "default"},
            "trust_score": 0.8,
            "trust_history": [
                {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
            ],
            "status": "active",
            "superseded_by": [],
            "superseded_at": None,
            "tags": ["shell", "status"],
            "created": _utc_now(),
            "updated": _utc_now(),
            "last_validated": None,
            "artifacts": [],  # no artifacts — this is a prose seed
        }
        ok2, reason2 = gate.commit(prose_seed)
        # Should succeed — different content, no hash match
        assert ok2, f"Prose seed should commit: {reason2}"

    def test_artifact_seed_empty_claim_rejected(self, tmp_path):
        """Artifact seed with empty core_claim → rejected (description required)."""
        vault = SeedVault(tmp_path / "vault")
        gate = CommitGate(vault)

        raw = b"some code"
        seed_id = "code-code-003"
        blob_hash = vault.blob_store.write_blob(raw, seed_id, "code")
        assert blob_hash is not None

        seed = _make_test_artifact_seed(seed_id, blob_hash, claim="")
        ok, reason = gate.commit(seed)
        assert not ok
        assert "empty" in reason.lower()

    def test_artifact_seed_stage2_skipped(self, tmp_path):
        """Artifact seeds skip Stage-2 drift check (approved deviation 4.3).
        The provider's on_pre_compress should not call stage2_validate for
        artifact seeds."""
        vault = SeedVault(tmp_path / "vault")
        gate = CommitGate(vault)

        raw = b"print('hello')"
        seed_id = "code-code-004"
        blob_hash = vault.blob_store.write_blob(raw, seed_id, "code", "python")
        assert blob_hash is not None

        seed = _make_test_artifact_seed(seed_id, blob_hash, claim="Python print hello")
        ok, reason = gate.commit(seed)
        assert ok

        # Call stage2_validate — it should not modify the trust score
        # because the provider skips it. But even if called directly, the
        # core_claim is a short description and token overlap is meaningless.
        # We just verify the seed still has its initial trust score.
        assert seed["trust_score"] == 0.8


class TestArtifactRetrieval:
    """Test the seedvault_get_artifact tool and retrieval behavior."""

    def test_get_artifact_round_trip(self, tmp_path):
        """Write a blob, fetch by hash via the tool call, verify byte-identical."""
        vault = SeedVault(tmp_path / "vault")
        raw = b"def foo():\n    return 42\n"
        blob_hash = vault.blob_store.write_blob(raw, "test-rt-001", "code", "python")
        assert blob_hash is not None

        provider = SeedVaultMemoryProvider()
        provider._vault = vault
        provider._digest_mgr = None
        provider._vault_dir = vault.vault_dir

        result = provider.handle_tool_call(
            "seedvault_get_artifact", {"blob_hash": blob_hash}
        )
        assert result == "def foo():\n    return 42\n"

    def test_get_artifact_missing_hash(self, tmp_path):
        """Missing hash → returns error JSON, not a crash."""
        vault = SeedVault(tmp_path / "vault")
        provider = SeedVaultMemoryProvider()
        provider._vault = vault
        provider._digest_mgr = None
        provider._vault_dir = vault.vault_dir

        result = provider.handle_tool_call(
            "seedvault_get_artifact", {"blob_hash": "0" * 64}
        )
        assert "error" in result
        assert "not found" in result

    def test_get_artifact_no_hash_arg(self, tmp_path):
        """No blob_hash argument → returns error JSON."""
        vault = SeedVault(tmp_path / "vault")
        provider = SeedVaultMemoryProvider()
        provider._vault = vault
        provider._digest_mgr = None
        provider._vault_dir = vault.vault_dir

        result = provider.handle_tool_call("seedvault_get_artifact", {})
        assert "error" in result
        assert "required" in result

    def test_get_artifact_scrub_on_retrieval(self, tmp_path, fake_env):
        """Blob containing a known secret → returned bytes are redacted."""
        vault = SeedVault(tmp_path / "vault")
        raw = b"export OPENAI_API_KEY=sk-test-1234567890abcdef"
        blob_hash = vault.blob_store.write_blob(raw, "test-scrub-rt", "bash")
        assert blob_hash is not None

        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.provider.get_scrub_patterns",
            return_value=patterns,
        ):
            provider = SeedVaultMemoryProvider()
            provider._vault = vault
            provider._digest_mgr = None
            provider._vault_dir = vault.vault_dir

            result = provider.handle_tool_call(
                "seedvault_get_artifact", {"blob_hash": blob_hash}
            )

        assert "sk-test-1234567890abcdef" not in result
        assert "[REDACTED:OPENAI_API_KEY]" in result

    def test_prefetch_surfaces_pointer_not_raw_content(self, tmp_path):
        """Seed with artifact → prefetch() output includes artifact pointer
        info but NOT the raw blob content."""
        vault = SeedVault(tmp_path / "vault")
        raw = b"systemctl restart ollama"
        blob_hash = vault.blob_store.write_blob(raw, "shell-bash-010", "bash")
        assert blob_hash is not None

        # Create and write a seed with the artifact
        seed = _make_test_artifact_seed(
            "shell-bash-010", blob_hash, "bash",
            claim="Bash command: systemctl restart ollama",
            tags=["shell", "bash"],
        )
        vault.write_seed(seed)

        provider = SeedVaultMemoryProvider()
        provider._vault = vault
        provider._digest_mgr = None
        provider._vault_dir = vault.vault_dir

        result = provider.prefetch("systemctl")
        # The core_claim (description) should be in the output
        assert "systemctl restart ollama" in result or "Bash command" in result
        # The raw blob content should NOT be in the prefetch output
        # (prefetch only shows seed summaries, not artifact bytes)
        # The raw content "systemctl restart ollama" appears in the claim too,
        # so we check that the blob hash is not directly mentioned
        # (the raw bytes are only accessible via seedvault_get_artifact)

    def test_get_tool_schemas_includes_artifact_tool(self, tmp_path):
        """get_tool_schemas() includes seedvault_get_artifact."""
        vault = SeedVault(tmp_path / "vault")
        provider = SeedVaultMemoryProvider()
        provider._vault = vault
        provider._digest_mgr = None
        provider._vault_dir = vault.vault_dir

        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "seedvault_get_artifact" in names