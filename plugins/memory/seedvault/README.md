# SeedVault Memory System

A MemoryProvider plugin for Hermes Agent that reduces semantic drift across context compression boundaries.

## What It Does

When Hermes compresses conversation history (compaction), context is lost. SeedVault intercepts that moment — it extracts durable facts from messages about to be discarded, stores them as structured "seeds" with provenance and trust scoring, and re-injects active seeds into the system prompt on subsequent turns.

## Architecture

```
plugins/memory/seedvault/
├── __init__.py          # Plugin entry point
├── provider.py         # SeedVaultMemoryProvider (MemoryProvider ABC impl)
├── vault.py             # On-disk vault: manifest, seed CRUD, locking, search, pruning
├── extractor.py         # Seed extraction (LLM-assisted + heuristic fallback)
├── validator.py         # Two-stage commit gate (Stage 1: deterministic, Stage 2: drift check)
├── state_digest.py      # Compaction-surviving state carry-forward
├── schemas/             # JSON schemas for seed and state_digest
│   ├── memory_seed.schema.json
│   └── state_digest.schema.json
└── DESIGN.md            # Full design document
```

Runtime data (default: `~/.hermes/memory/seedvault/`):
```
├── vault_manifest.json   # Seed index with status/tags/trust
├── state_digest.json     # Carry-forward state across compaction
├── seeds/                # Active seed files (one JSON per seed)
└── archive/              # Old superseded seeds (>30 days)
```

## Hook Integration

| Hook | Usage |
|------|-------|
| `on_pre_compress(messages)` | **Core**: extract seeds from messages being compressed, commit through gate, return summary for compression prompt |
| `system_prompt_block()` | Static vault status + state digest (cache-stable) |
| `prefetch(query)` | Keyword/tag search of active seeds, inject as context |
| `sync_turn(user, asst)` | Lightweight state digest update (no extraction) |
| `on_session_switch(new_id)` | Restore state digest after compaction boundary |
| `on_memory_write(action, target, content)` | Mirror built-in MEMORY.md/USER.md writes as seeds |

## Seed Schema

```json
{
  "id": "env-ollama-001",
  "core_claim": "OLLAMA_FLASH_ATTENTION must stay OFF — causes Vulkan ErrorDeviceLost",
  "chunks": [{"type": "constraint", "content": "..."}],
  "meristems": [{"type": "supersedes", "target": "env-ollama-000"}],
  "source_ref": {"session_id": "20260728_120000_abc", "profile": "default"},
  "trust_score": 0.85,
  "trust_history": [
    {"delta": null, "reason": "initial", "value": 0.8, "at": "..."},
    {"delta": 0.1, "reason": "provenance_verified", "at": "..."}
  ],
  "status": "active",
  "superseded_by": [],
  "superseded_at": null,
  "tags": ["env", "constraint"],
  "created": "...",
  "updated": "...",
  "last_validated": "..."
}
```

## Trust Score

Trust score is computed, not self-reported:
- Initial: 0.8 (neutral baseline)
- +0.1 when Stage 2 validates the claim against source messages (provenance_verified)
- -0.2 when Stage 2 detects drift (claim tokens <30% overlap with source)
- -0.3 when superseded by a newer seed
- Floor: 0.0 (auto-archived at <0.3 for >7 days)
- Ceiling: 1.0

Reconciliation: `trust_score == initial_value + sum(deltas)`. The first trust_history entry has `delta: null` and `value: <initial>`, not a phantom +0.8 adjustment.

## Commit Gate (Two-Stage)

**Stage 1 (deterministic, always runs):**
- core_claim non-empty, <=500 chars
- source_ref.session_id required (no orphan seeds)
- No duplicate core_claim (Jaccard >0.7 = reject)
- Meristem targets must resolve to existing seeds (dangling edges dropped)

**Stage 2 (runs after commit, compression events only):**
- Tokenize core_claim, check overlap with source messages
- If <30% overlap: drift detected, trust_score -= 0.2
- If >=30%: validated, trust_score += 0.1

## Supersession

When a new seed is committed with the same primary tag (first tag in `tags[]`) as an existing active seed:
- Old seed: status -> "superseded", superseded_by appends new ID, trust_score -> 0.0
- New seed: gets a `{type: "supersedes", target: old_id}` meristem
- Old seed NOT deleted (audit trail preserved)
- `superseded_by` is a list (multiple children can supersede one parent)

This solves the "stale constraint" problem: if `env-ollama-001` says "flash attention OFF" and later the issue is fixed, a new seed supersedes it and the old one goes quiet.

## Concurrency

- `vault_manifest.json` and `state_digest.json`: fcntl.flock(LOCK_EX) on writes
- Individual seed files: atomic write (tmp + rename)
- v1 limit: single active session per profile (parallel sessions queue on lock)

## Pruning

- Active seeds with trust_score <0.3 for >`stale_days` (default 7) -> archived
- Superseded seeds older than `archive_days` (default 30) -> moved to archive/ directory
- Both thresholds configurable per-profile in config.yaml

## MindSeed vs SeedVault — Paradigm Differences

| Aspect | MindSeed | SeedVault |
|--------|---------|-----------|
| Input | Canonical corpus (curated) | Live conversation (unstructured) |
| Extraction | Build-time (human-authored) | Runtime (LLM-assisted or heuristic) |
| Validation | Triple-metric gate vs reference | Two-stage gate (deterministic + drift check) |
| Delivery | Deterministic (canonical output) | Runtime injection (prefetch + prompt block) |
| Trust | Source authority + gate | Computed trust score with decay |
| Supersession | Manual (re-author seed) | Automatic (primary-tag match triggers supersession) |

**Key limitation**: SeedVault does NOT inherit MindSeed's determinism guarantees. The compensating control is the commit gate — it's explicitly weaker than MindSeed's triple-metric gate because there's no canonical reference to validate against. This is a stated design decision, not an oversight.

## Activation

Add to `config.yaml`:
```yaml
memory:
  provider: seedvault
  seedvault:
    extraction:
      mode: heuristic  # or "llm" (requires model access during compression)
    pruning:
      stale_days: 7
      archive_days: 30
```

## Tools Exposed

- `seedvault_search(query, top_k)` — Search active seeds by keyword/tag
- `seedvault_status()` — Vault summary: seed counts, trust distribution, digest state