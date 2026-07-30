"""Content-addressed blob store for SeedVault artifacts.

Stores verbatim artifact bytes (code blocks, shell commands, diffs) as
content-addressed files on disk, with a SQLite index for metadata lookups.

Design (per Phase 8 redline):
- Raw bytes: <vault_dir>/blobs/<sha256[:2]>/<sha256>.blob (sharded by first
  2 hex chars to keep directory entries manageable).
- Index: <vault_dir>/blobs/blob_index.db (SQLite, WAL mode).  Holds
  metadata only — blob_hash, seed_id, content_type, byte_length, created_at.
  The schema is designed to be extensible for Phase 9 lineage fields
  (lineage_id, tag columns can be added without migration).
- vault_manifest.json is NOT touched — the blob index is separate.

Concurrency:
- SQLite WAL mode handles concurrent index reads/writes with its own
  transaction/locking model — no fcntl.flock needed for the index.
- Blob file writes use the same atomic pattern as SeedVault seeds: unique
  tmp filename (pid+tid suffix) + os.replace().
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_BLOB_BYTES = 1 * 1024 * 1024  # 1 MB hard limit


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlobStore:
    """Content-addressed blob store with SQLite WAL index.

    Instantiated by SeedVault via composition.  The blob index lives at
    ``<vault_dir>/blobs/blob_index.db`` (SQLite, WAL mode).  Raw bytes live
    as ``<vault_dir>/blobs/<sha256[:2]>/<sha256>.blob``.
    """

    def __init__(
        self,
        vault_dir: Path,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.blobs_dir = self.vault_dir / "blobs"
        self.db_path = self.blobs_dir / "blob_index.db"
        self.max_blob_bytes = max_blob_bytes
        self._local = threading.local()  # per-thread SQLite connections
        self.ensure_dirs()

    # -- Directory + DB setup -----------------------------------------------

    def ensure_dirs(self) -> None:
        """Create blobs/ and all 256 shard subdirectories (lazy via mkdir)."""
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a per-thread SQLite connection in WAL mode."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                # Stale connection — close and reopen
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                del self._local.conn

        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit mode; we manage txns explicitly
            timeout=30.0,  # wait up to 30s for locks
        )
        conn.row_factory = sqlite3.Row
        # WAL mode for concurrent readers + single writer
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL
        conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout

        # Create table if not exists — extensible schema for Phase 9
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blob_index (
                blob_hash    TEXT    PRIMARY KEY,
                seed_id      TEXT    NOT NULL,
                content_type TEXT    NOT NULL,
                byte_length  INTEGER NOT NULL,
                created_at   TEXT    NOT NULL,
                language     TEXT    DEFAULT NULL,
                lineage_id   TEXT    DEFAULT NULL,
                tag          TEXT    DEFAULT NULL
            )
            """
        )
        # Index for seed_id lookups (used by retrieval/dedup)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_seed_id ON blob_index(seed_id)"
        )
        # Index for content_type lookups (future filtering)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_content_type "
            "ON blob_index(content_type)"
        )

        self._local.conn = conn
        return conn

    # -- Core CRUD -----------------------------------------------------------

    def write_blob(
        self,
        raw_bytes: bytes,
        seed_id: str,
        content_type: str,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Write artifact bytes to disk and index in SQLite.

        Returns the SHA-256 blob_hash on success, or None if:
        - The blob exceeds max_blob_bytes (oversized artifacts are dropped).
        - A disk/DB error occurs.

        Dedup: if the exact hash already exists in the index, the blob file
        is not re-written (it's already on disk) and the existing hash is
        returned.  The seed_id is NOT updated — the first writer owns the
        index entry.  This is correct because the content is identical.
        """
        if not raw_bytes:
            return None

        if len(raw_bytes) > self.max_blob_bytes:
            logger.warning(
                "BlobStore: blob rejected — %d bytes exceeds limit %d "
                "(seed_id=%s, content_type=%s)",
                len(raw_bytes),
                self.max_blob_bytes,
                seed_id,
                content_type,
            )
            return None

        blob_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Check if this hash already exists (dedup)
        if self.blob_exists(blob_hash):
            logger.debug(
                "BlobStore: dedup hit for hash %s (seed_id=%s)", blob_hash, seed_id
            )
            return blob_hash

        # Write the blob file atomically
        shard_dir = self.blobs_dir / blob_hash[:2]
        shard_dir.mkdir(parents=True, exist_ok=True)
        blob_path = shard_dir / f"{blob_hash}.blob"
        tmp_path = blob_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )

        try:
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, blob_path)
        except OSError as e:
            logger.error("BlobStore: failed to write blob %s: %s", blob_hash, e)
            # Clean up tmp if it's still around
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        # Index the blob in SQLite
        now = _utc_now()
        conn = self._get_conn()
        try:
            conn.execute(
                "BEGIN IMMEDIATE"
            )  # get write lock immediately for the insert
            conn.execute(
                """
                INSERT OR IGNORE INTO blob_index
                    (blob_hash, seed_id, content_type, byte_length, created_at, language)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (blob_hash, seed_id, content_type, len(raw_bytes), now, language),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            logger.error("BlobStore: failed to index blob %s: %s", blob_hash, e)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            # The blob file is on disk but not indexed — it's orphaned.
            # We leave it (it will be found by future writes with the same hash
            # and indexed then).  Return None so the caller knows the index
            # write failed.
            return None

        logger.debug(
            "BlobStore: wrote blob %s (%d bytes, type=%s, seed=%s)",
            blob_hash,
            len(raw_bytes),
            content_type,
            seed_id,
        )
        return blob_hash

    def read_blob(self, blob_hash: str) -> Optional[bytes]:
        """Read raw blob bytes from disk by SHA-256 hash.

        Returns None if the blob file doesn't exist on disk.
        """
        if not blob_hash:
            return None
        shard_dir = self.blobs_dir / blob_hash[:2]
        blob_path = shard_dir / f"{blob_hash}.blob"
        if not blob_path.exists():
            logger.warning("BlobStore: blob not found on disk: %s", blob_hash)
            return None
        try:
            with open(blob_path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.error("BlobStore: failed to read blob %s: %s", blob_hash, e)
            return None

    def blob_exists(self, blob_hash: str) -> bool:
        """Check if a blob hash exists in the SQLite index.

        This is an exact-match lookup — no Jaccard, no threshold.
        """
        if not blob_hash:
            return False
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM blob_index WHERE blob_hash = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        return row is not None

    def get_blob_metadata(self, blob_hash: str) -> Optional[Dict[str, Any]]:
        """Get blob metadata from the SQLite index.

        Returns a dict with keys: blob_hash, seed_id, content_type,
        byte_length, created_at, language, lineage_id, tag.
        """
        if not blob_hash:
            return None
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM blob_index WHERE blob_hash = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def find_by_seed_id(self, seed_id: str) -> list[Dict[str, Any]]:
        """Find all blobs associated with a seed ID."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM blob_index WHERE seed_id = ?",
            (seed_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of blobs in the index."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM blob_index").fetchone()
        return row[0] if row else 0

    def all_hashes(self) -> list[str]:
        """Return all blob hashes in the index (for integrity checks)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT blob_hash FROM blob_index").fetchall()
        return [r["blob_hash"] for r in rows]

    def close(self) -> None:
        """Close the per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            del self._local.conn