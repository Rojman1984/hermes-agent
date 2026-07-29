"""SeedVault memory plugin — MemoryProvider interface.

Structured memory system for Hermes Agent that extracts durable facts
from conversation during compression events and re-injects them into
the system prompt to reduce semantic drift across compaction boundaries.

Adapted from MindSeed concepts (seeds, meristems, state digest, trust
scoring) but operates at RUNTIME on live conversation rather than
build-time on a canonical corpus. The two-stage commit gate is the
compensating control for the loss of build-time determinism.

PARADIGM NOTE: This plugin does NOT inherit MindSeed's determinism
guarantees. Input is unstructured live dialogue, not a curated canonical
corpus. See DESIGN.md for full paradigm documentation.

Config (config.yaml under memory.seedvault:):
  vault_dir: Override vault location (default: <hermes_home>/memory/seedvault)
  extraction.mode: "llm" or "heuristic" (default: "heuristic")
  pruning.stale_days: Days before low-trust seeds archived (default: 7)
  pruning.archive_days: Days before superseded seeds archived (default: 30)
"""

from __future__ import annotations

from .provider import SeedVaultMemoryProvider

__all__ = ["SeedVaultMemoryProvider"]