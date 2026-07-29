# SeedVault Memory System — Design Document

## Status
Experimental — fork branch `feature/seedvault-memory-system`
Date: 2026-07-28

## Problem Statement

Hermes Agent loses context during compaction, accumulates stateful memory bloat,
and suffers semantic/logical drift across sessions. The existing memory system
(MEMORY.md + USER.md) is flat prose with no provenance, no validation, and no
supersession — entries just accumulate until the char limit forces manual pruning.

## Solution

SeedVault: a structured memory plugin that extracts compact "seeds" from
session context before compaction, stores them in a validated vault, and
re-injects relevant seeds into the system prompt and compression prompt.

The design adapts concepts from the MindSeed knowledge packaging framework
(seeds, meristems, state digest, triple-metric gate) but operates in a
fundamentally different paradigm.

## Paradigm Difference from MindSeed (MUST be documented in docstrings)

MindSeed is a build-time authoring system: a curated canonical corpus is
front-loaded, validated offline, and delivered with near-deterministic
guarantees at runtime. SeedVault is a runtime-extraction system: the "corpus"
is unstructured live dialogue, and seeds are extracted mid-session during
compression events. SeedVault does NOT inherit MindSeed's determinism
guarantees. The commit gate and trust score are compensating controls, not
replacements for build-time validation.

## Architecture

```
plugins/memory/seedvault/
  __init__.py              -- plugin entry point
  provider.py              -- SeedVaultProvider(MemoryProvider)
  schemas/
    memory_seed.schema.json
    state_digest.schema.json
  vault.py                 -- vault management (manifest, CRUD, locking)
  extractor.py             -- seed extraction from session messages (LLM-assisted)
  validator.py             -- commit gate + triple check validation
  state_digest.py          -- state digest read/write/carry-forward

~/.hermes/memory/seedvault/  (runtime data, NOT in repo)
  vault_manifest.json
  state_digest.json
  seeds/
    *.json
  archive/                   -- superseded/archived seeds
```

## Hook Mapping (MemoryProvider ABC)

| Hook | Purpose | When |
|------|---------|------|
| `on_pre_compress(messages)` | Extract seeds from messages being compressed; return active seed summary for compression prompt | Before each compaction |
| `system_prompt_block()` | Return active seeds as structured context | System prompt volatile tier |
| `prefetch(query)` | Return relevant seeds by keyword/tag match | Before each API call |
| `sync_turn(...)` | Update state_digest with current task state | After each turn |
| `on_session_switch(...)` | Write state_digest, update vault manifest across compaction boundary | After compaction completes |
| `on_memory_write(...)` | Mirror built-in memory writes into seed format | On memory tool commits |
| `initialize(...)` | Load vault manifest, state_digest | Agent startup |

## Seed Schema

```json
{
  "id": "env-ollama-001",
  "core_claim": "OLLAMA_FLASH_ATTENTION must stay OFF — causes Vulkan ErrorDeviceLost on AMD iGPU",
  "chunks": [
    {"type": "constraint", "content": "Do not enable flash attention"},
    {"type": "status", "content": "Working model: glm-5.2:cloud via Ollama"}
  ],
  "meristems": [
    {"type": "prerequisite", "target": "env-amd-igpu-001"},
    {"type": "nextstep", "target": "model-fallback-001"}
  ],
  "source_ref": {"session_id": "20260728_212342_2d76c8", "profile": "default"},
  "trust_score": 0.85,
  "trust_history": [
    {"delta": null, "reason": "initial", "value": 0.8, "at": "2026-07-28T22:25:00Z"},
    {"delta": 0.1, "reason": "provenance_verified", "at": "2026-07-28T22:26:00Z"},
    {"delta": -0.05, "reason": "age_decay", "at": "2026-07-29T22:25:00Z"}
  ],
  "status": "active",
  "superseded_by": [],
  "superseded_at": null,
  "tags": ["environment", "ollama", "vulkan", "critical"],
  "created": "2026-07-28T22:25:00Z",
  "updated": "2026-07-28T22:25:00Z",
  "last_validated": "2026-07-28T22:26:00Z"
}
```

### Trust Score Arithmetic

- Initial value: 0.8 (neutral baseline)
- trust_history[0] is always `{"delta": null, "reason": "initial", "value": <initial>}`
- Subsequent entries are deltas from the previous score
- Reconciliation: `score = initial_value + sum(deltas)` — must equal trust_score
- Adjustments:
  - +0.1 provenance_verified (core_claim traced to session DB content)
  - +0.1 explicit_positive (seed surfaced, user message semantically consistent)
  - -0.2 drift_detected (re-extraction can't trace claim to source)
  - -0.3 superseded (a newer seed replaces this one in same domain)
  - -0.05 age_decay (per 24h without positive/negative signal)
- Floor: 0.0. Ceiling: 1.0. Auto-archive at <0.3.

### Supersession

- Trigger: new seed committed with exact primary-tag match (first tag in `tags` array) to an existing active seed
- Old seed: status="superseded", superseded_by appended with new seed ID, superseded_at=timestamp, trust_score drops to 0.0
- New seed: meristem {type: "supersedes", target: old_seed_id} added
- `superseded_by` is a list (multiple children can supersede one parent)
- Old seed NOT deleted (audit trail preserved)
- Superseded seeds excluded from prefetch() and system_prompt_block()

### Commit Gate (Two-Stage)

Stage 1 (deterministic, always runs):
- core_claim non-empty, <= 500 chars
- source_ref.session_id is not null (provenance required)
- No duplicate core_claim (Jaccard similarity > 0.7 = reject)
- Meristem targets resolve to existing seed IDs (dangling edges dropped, not deferred)

Stage 2 (LLM-assisted, compression events only):
- Re-extract claims from same messages, compare to committed seeds
- If core_claim can't be traced to source messages: trust_score -= 0.2
- Seeds below trust_score 0.3 auto-archived

## State Digest Schema

```json
{
  "current_task": "Building SeedVault memory system",
  "active_seed_ids": ["env-ollama-001", "matrix-gateway-001"],
  "pending_actions": ["Implement provider.py", "Write tests"],
  "compaction_count": 2,
  "last_compaction": "2026-07-28T22:20:56Z",
  "session_lineage": ["20260728_212342_2d76c8"]
}
```

## Concurrency

- vault_manifest.json: fcntl.flock(LOCK_EX) on every write
- state_digest.json: fcntl.flock(LOCK_EX) on every write
- Individual seed files: atomic write (write to .tmp, os.rename)
- v1 limit: single active session per profile. Multiple profiles have separate vaults.
- Parallel sessions within same profile queue on the lock.

## Pruning (configurable)

- `memory.seedvault.pruning.stale_days` (default: 7) — active seeds with trust_score < 0.3 for this long get auto-archived
- `memory.seedvault.pruning.archive_days` (default: 30) — superseded seeds older than this move to archive/ subdirectory
- Archived seeds excluded from prefetch() and system_prompt_block() but remain in manifest

## Retrieval (v1)

- Keyword/tag matching (no FAISS/embedding dependency)
- prefetch(query): match query tokens against seed tags + core_claim tokens
- Score by tag overlap weight + token overlap
- Return top-N (configurable, default 5)
- v2 upgrade path: embedding-based retrieval when justified

## Open Items (resolve during implementation)

- [TODO] Explicit positive signal detection for trust score (+0.1 rule)
- [TODO] Age decay scheduling (when does the 24h timer fire?)
- [TODO] Re-extraction prompt design for Stage 2 gate
- [TODO] Integration test with real session DB