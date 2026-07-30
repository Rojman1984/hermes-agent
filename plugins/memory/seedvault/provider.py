"""SeedVault MemoryProvider plugin for Hermes Agent.

A structured memory system that extracts durable facts from conversation
during compression events, stores them as typed seeds with provenance and
trust scoring, and re-injects them into the system prompt to reduce
semantic drift across compaction boundaries.

Adapted from MindSeed concepts (seeds, meristems, state digest) but
operates at RUNTIME on live conversation rather than build-time on a
canonical corpus. The commit gate (validator.py) is the compensating
control for the loss of MindSeed's build-time determinism.

See DESIGN.md for full architecture and paradigm difference documentation.

Configuration (config.yaml under memory.seedvault:):
  vault_dir: Override vault location (default: <hermes_home>/memory/seedvault)
  pruning.stale_days: Days before low-trust seeds are archived (default: 7)
  pruning.archive_days: Days before superseded seeds are moved to archive/ (default: 30)
  extraction.mode: "llm" or "heuristic" (default: "heuristic" — LLM mode requires
    a callable model and adds latency to compression events)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .vault import SeedVault
from .validator import CommitGate
from .state_digest import StateDigestManager
from .extractor import extract_seeds
from .scrub import get_scrub_patterns, scrub_text, scrub_bytes

logger = logging.getLogger(__name__)

DEFAULT_STALE_DAYS = 7
DEFAULT_ARCHIVE_DAYS = 30


class SeedVaultMemoryProvider(MemoryProvider):
    """SeedVault structured memory provider.

    Seeds are extracted from messages being compressed (on_pre_compress),
    pass through a two-stage commit gate, and are stored on disk with
    provenance, trust scoring, and supersession tracking. Active seeds
    are re-injected into the system prompt to reduce drift.
    """

    def __init__(self):
        self._vault: Optional[SeedVault] = None
        self._gate: Optional[CommitGate] = None
        self._digest_mgr: Optional[StateDigestManager] = None
        self._session_id = ""
        self._profile = "default"
        self._hermes_home = ""
        self._llm_caller = None
        self._extraction_mode = "heuristic"
        self._stale_days = DEFAULT_STALE_DAYS
        self._archive_days = DEFAULT_ARCHIVE_DAYS
        self._vault_dir: Optional[Path] = None
        self._turn_count = 0
        self._last_prefetch_result = ""

    # -- Core identity ------------------------------------------------------

    @property
    def name(self) -> str:
        return "seedvault"

    def is_available(self) -> bool:
        """SeedVault is always available — it's local-only, no external deps."""
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "vault_dir",
                "description": "Override vault directory (default: <hermes_home>/memory/seedvault)",
                "required": False,
            },
            {
                "key": "extraction.mode",
                "description": "Extraction mode: 'llm' (model-assisted) or 'heuristic' (pattern-based)",
                "default": "heuristic",
                "choices": ["llm", "heuristic"],
                "required": False,
            },
            {
                "key": "pruning.stale_days",
                "description": f"Days before low-trust seeds are archived (default: {DEFAULT_STALE_DAYS})",
                "default": DEFAULT_STALE_DAYS,
                "required": False,
            },
            {
                "key": "pruning.archive_days",
                "description": f"Days before superseded seeds are moved to archive/ (default: {DEFAULT_ARCHIVE_DAYS})",
                "default": DEFAULT_ARCHIVE_DAYS,
                "required": False,
            },
        ]

    # -- Lifecycle ----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home", os.path.expanduser("~/.hermes")))
        self._profile = str(kwargs.get("agent_identity", "default"))

        # Load config from kwargs or defaults
        # (Hermes passes provider config via memory.provider_config)
        config = kwargs.get("provider_config", {})
        self._extraction_mode = config.get("extraction", {}).get("mode", "heuristic")
        self._stale_days = int(config.get("pruning", {}).get("stale_days", DEFAULT_STALE_DAYS))
        self._archive_days = int(config.get("pruning", {}).get("archive_days", DEFAULT_ARCHIVE_DAYS))

        # Vault directory: config override > hermes_home/memory/seedvault
        vault_dir_str = config.get("vault_dir", "")
        if vault_dir_str:
            self._vault_dir = Path(vault_dir_str)
        else:
            self._vault_dir = Path(self._hermes_home) / "memory" / "seedvault"

        self._vault = SeedVault(self._vault_dir)
        self._vault.ensure_dirs()
        self._gate = CommitGate(self._vault)
        self._digest_mgr = StateDigestManager(self._vault)

        # If an LLM caller was provided (via agent injection), use it
        self._llm_caller = kwargs.get("llm_caller")

        # Run pruning on init (best-effort)
        try:
            pruned = self._vault.prune(self._stale_days, self._archive_days)
            if any(pruned.values()):
                logger.info("SeedVault: init pruning: %s", pruned)
        except Exception as e:
            logger.warning("SeedVault: init pruning failed: %s", e)

        logger.info(
            "SeedVault: initialized (vault=%s, mode=%s, profile=%s, vault_summary=%s)",
            self._vault_dir,
            self._extraction_mode,
            self._profile,
            self._vault.get_manifest_summary(),
        )

    # -- System prompt block ------------------------------------------------

    def system_prompt_block(self) -> str:
        """Return static text for the system prompt describing SeedVault.

        This block is genuinely STATIC — it contains only vault path and
        extraction mode, which are fixed at init time. Live data (seed
        counts, state digest) is deliberately excluded because it changes
        on every seed commit / turn update, which would bust the LLM
        prompt cache when the system prompt is rebuilt after compression.

        Seed-count summaries are available via:
        - prefetch() — injected per-turn where variation is expected
        - seedvault_status tool call — on-demand
        - state digest format_for_prompt() — used in prefetch, not here
        """
        if not self._vault:
            return ""

        lines = [
            "[SeedVault Memory System Active]",
            f"Vault: {self._vault_dir}",
            f"Extraction mode: {self._extraction_mode}",
        ]

        return "\n".join(lines)

    # -- Prefetch -----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant seeds for the upcoming turn.

        Uses keyword/tag matching (v1). Returns formatted seed summaries
        to inject as context before the API call.

        Phase 8a: retrieval-time scrub — the entire output is scrubbed
        for secrets (from .env) immediately before returning, so a seed
        committed before this phase (or before a value existed in .env)
        is caught on every future retrieval, not just once.
        """
        if not self._vault:
            return ""

        lines = ["[SeedVault Retrieved Seeds]"]

        # Include vault summary here (not in system_prompt_block) because
        # prefetch output is per-turn and does not bust the prompt cache.
        summary = self._vault.get_manifest_summary()
        lines.append(
            f"Vault: {summary.get('total', 0)} seeds "
            f"({summary.get('by_status', {})})"
        )

        # State digest — also per-turn, not in the cached system prompt.
        if self._digest_mgr:
            digest_text = self._digest_mgr.format_for_prompt()
            if digest_text:
                lines.append(digest_text)

        results = self._vault.search(query, top_k=5)
        if not results:
            self._last_prefetch_result = "\n".join(lines)
            return "\n".join(lines)

        for seed in results:
            status_marker = ""  # Only active seeds are returned by search()
            lines.append(f"- [{seed['id']}] {seed['core_claim']}")
            if seed.get("tags"):
                lines.append(f"  tags: {', '.join(seed['tags'])}")
            if seed.get("trust_score", 1.0) < 0.6:
                lines.append(f"  trust: {seed['trust_score']:.2f} (low)")
        lines.append(f"({len(results)} seeds matched)")

        result = "\n".join(lines)

        # Phase 8a: retrieval-time scrub — sanitize before injection.
        scrub_patterns = get_scrub_patterns()
        if scrub_patterns:
            result, n = scrub_text(result, scrub_patterns)
            if n > 0:
                logger.debug(
                    "SeedVault: retrieval-time scrub redacted %d secret(s) "
                    "in prefetch output",
                    n,
                )

        self._last_prefetch_result = result
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No background prefetch for v1 — search is fast enough synchronously."""
        pass

    # -- Sync turn ----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Lightweight per-turn update.

        Updates the state digest with current task info. Does NOT extract
        seeds — extraction happens at compression time (on_pre_compress).
        """
        if not self._digest_mgr:
            return

        self._turn_count += 1

        # Derive current task from the latest user message
        task = user_content[:200] if user_content else ""

        # We don't have structured pending_actions at this layer;
        # the state digest will be updated with whatever we know
        try:
            active_ids = self._vault.get_active_seed_ids() if self._vault else []
            self._digest_mgr.update_after_turn(
                current_task=task,
                pending_actions=[],
                active_seed_ids=active_ids,
            )
        except Exception as e:
            logger.debug("SeedVault: sync_turn digest update failed: %s", e)

    # -- Compression hook (the critical one) --------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract seeds from messages about to be compressed.

        This is the core intervention point. Seeds are extracted, validated
        through the commit gate, committed to the vault, and a structured
        summary is returned for injection into the compression prompt itself.

        The compressor receives this summary and preserves it in the
        compressed context, ensuring seeds survive the compaction boundary.
        """
        if not self._vault or not self._gate:
            return ""

        logger.info("SeedVault: on_pre_compress triggered (%d messages to process)",
                     len(messages))

        # Extract seeds
        llm_caller = self._llm_caller if self._extraction_mode == "llm" else None
        raw_seeds = extract_seeds(
            messages=messages,
            session_id=self._session_id,
            profile=self._profile,
            llm_caller=llm_caller,
            blob_store=self._vault.blob_store if self._vault else None,
        )

        if not raw_seeds:
            logger.info("SeedVault: no seeds extracted from compression batch")
            return ""

        # Commit each seed through the gate
        committed: List[Dict[str, Any]] = []
        rejected = 0
        for raw_seed in raw_seeds:
            ok, reason = self._gate.commit(raw_seed)
            if ok:
                committed.append(raw_seed)
                # Stage 2 validation against source messages.
                # Phase 8.3: artifact seeds skip Stage-2 drift check (approved
                # deviation 4.3) — hash equality is a stronger guarantee than
                # token-overlap drift checking for a short description.
                has_artifacts = bool(raw_seed.get("artifacts"))
                if not has_artifacts:
                    self._gate.stage2_validate(raw_seed, messages)
            else:
                rejected += 1
                logger.debug("SeedVault: seed rejected: %s — %s",
                             raw_seed.get("id", "?"), reason)

        if not committed:
            return ""

        # Build summary for the compression prompt
        lines = [
            "[SeedVault — Preserved Memory Seeds]",
            f"Extracted {len(committed)} seed(s) from compressed messages.",
            "These facts have been stored in the seed vault and should be",
            "preserved in the compressed context:",
            "",
        ]

        for seed in committed:
            tags_str = ", ".join(seed.get("tags", []))
            lines.append(f"  [{seed['id']}] (tags: {tags_str}) trust: {seed['trust_score']:.2f}")
            lines.append(f"    {seed['core_claim']}")
            if seed.get("superseded_by") or seed.get("status") != "active":
                lines.append(f"    status: {seed['status']}")
            lines.append("")

        # Update state digest for compaction boundary
        if self._digest_mgr:
            active_ids = self._vault.get_active_seed_ids()
            self._digest_mgr.snapshot_for_compaction(
                current_task="",
                pending_actions=[],
                active_seed_ids=active_ids,
                session_lineage=[self._session_id],
            )

        # Run pruning after compression
        try:
            self._vault.prune(self._stale_days, self._archive_days)
        except Exception as e:
            logger.debug("SeedVault: post-compress pruning failed: %s", e)

        result = "\n".join(lines)
        logger.info("SeedVault: on_pre_compress done — %d committed, %d rejected",
                    len(committed), rejected)
        return result

    # -- Session switch ----------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Handle compaction boundary or session reset.

        On compaction (reset=False, parent_session_id set): the digest
        carries forward. We update the session ID and restore the digest.

        On reset (reset=True): flush per-session state. The vault persists
        but the digest is reset to a fresh state.
        """
        old_session_id = self._session_id
        self._session_id = new_session_id

        if reset:
            # Fresh session — reset digest
            if self._digest_mgr and self._vault:
                fresh = {
                    "current_task": "",
                    "active_seed_ids": [],
                    "pending_actions": [],
                    "compaction_count": 0,
                    "last_compaction": "",
                    "session_lineage": [new_session_id],
                }
                self._digest_mgr.save(fresh)
                logger.info("SeedVault: session reset, digest cleared")
        else:
            # Compaction or resume — restore digest
            if self._digest_mgr:
                digest = self._digest_mgr.restore_after_compaction()
                logger.info("SeedVault: session switch %s -> %s (digest restored, compactions=%d)",
                            old_session_id, new_session_id,
                            digest.get("compaction_count", 0))

    # -- Memory write mirroring --------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes (MEMORY.md/USER.md) into seed format.

        When the built-in memory tool writes a fact, we create a seed for it
        so it benefits from the same search/supersession infrastructure.
        """
        if not self._vault or not self._gate:
            return
        if action == "remove":
            # Don't mirror removals — let supersession handle it
            return

        metadata = metadata or {}
        session_id = metadata.get("session_id", self._session_id)

        # Create a seed from the memory entry
        domain = "user" if target == "user" else "memory"
        seed = {
            "id": "",  # will be generated
            "core_claim": content[:500],
            "chunks": [{
                "type": "preference" if target == "user" else "insight",
                "content": content[:1000],
            }],
            "meristems": [],
            "source_ref": {
                "session_id": session_id,
                "profile": self._profile,
            },
            "trust_score": 0.8,
            "trust_history": [{
                "delta": None,
                "reason": "initial",
                "value": 0.8,
                "at": "",
            }],
            "status": "active",
            "superseded_by": [],
            "superseded_at": None,
            "tags": [domain, "mirrored"],
            "created": "",
            "updated": "",
            "last_validated": None,
        }

        # Use the extractor's ID generator
        from .extractor import _make_seed_id, _utc_now
        seed["id"] = _make_seed_id(domain, "mirrored", hash(content) % 1000)
        now = _utc_now()
        seed["created"] = now
        seed["updated"] = now
        seed["trust_history"][0]["at"] = now

        ok, reason = self._gate.commit(seed)
        if ok:
            logger.debug("SeedVault: mirrored memory write as seed %s", seed["id"])
        else:
            logger.debug("SeedVault: memory mirror rejected: %s", reason)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose seed management tools to the agent."""
        return [
            {
                "name": "seedvault_search",
                "description": "Search the seed vault for durable facts by keyword or tag. Returns matching seeds with trust scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results (default: 5, max: 20).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "seedvault_status",
                "description": "Show vault status: seed counts by status, trust score distribution, and recent activity.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "seedvault_get_artifact",
                "description": "Retrieve the raw content of an artifact (code block, shell command, diff) by its blob hash. The agent calls this when it needs the exact verbatim bytes of a captured artifact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "blob_hash": {
                            "type": "string",
                            "description": "The SHA-256 hash of the artifact blob (from a seed's artifacts array).",
                        },
                    },
                    "required": ["blob_hash"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle tool calls from the agent."""
        import json

        if tool_name == "seedvault_search":
            if not self._vault:
                return json.dumps({"error": "vault not initialized"})
            query = args.get("query", "")
            top_k = min(args.get("top_k", 5), 20)
            results = self._vault.search(query, top_k=top_k)
            output = json.dumps({
                "results": [
                    {
                        "id": s["id"],
                        "core_claim": s["core_claim"],
                        "tags": s.get("tags", []),
                        "trust_score": s.get("trust_score", 0.0),
                        "status": s.get("status", "active"),
                    }
                    for s in results
                ],
                "count": len(results),
            }, indent=2)
            # Phase 8a: retrieval-time scrub on tool output.
            scrub_patterns = get_scrub_patterns()
            if scrub_patterns:
                output, n = scrub_text(output, scrub_patterns)
                if n > 0:
                    logger.debug(
                        "SeedVault: retrieval-time scrub redacted %d secret(s) "
                        "in seedvault_search output",
                        n,
                    )
            return output

        elif tool_name == "seedvault_status":
            if not self._vault:
                return json.dumps({"error": "vault not initialized"})
            summary = self._vault.get_manifest_summary()
            digest = self._digest_mgr.load() if self._digest_mgr else {}
            return json.dumps({
                "vault": str(self._vault_dir),
                "seeds": summary,
                "state_digest": {
                    "compaction_count": digest.get("compaction_count", 0),
                    "active_seed_ids": len(digest.get("active_seed_ids", [])),
                    "current_task": digest.get("current_task", "")[:100],
                },
            }, indent=2)

        elif tool_name == "seedvault_get_artifact":
            # Phase 8.3: retrieve raw artifact bytes by blob hash.
            if not self._vault:
                return json.dumps({"error": "vault not initialized"})
            blob_hash = args.get("blob_hash", "")
            if not blob_hash:
                return json.dumps({"error": "blob_hash is required"})
            blob_bytes = self._vault.blob_store.read_blob(blob_hash)
            if blob_bytes is None:
                return json.dumps({"error": f"artifact not found: {blob_hash[:12]}..."})
            # Phase 8a: scrub-on-retrieval — redact secrets before serving.
            scrub_patterns = get_scrub_patterns()
            if scrub_patterns:
                blob_bytes, n = scrub_bytes(blob_bytes, scrub_patterns)
                if n > 0:
                    logger.debug(
                        "SeedVault: scrubbed %d secret(s) from artifact %s",
                        n,
                        blob_hash[:12],
                    )
            # Return as text (code/config are text). If non-UTF-8, base64.
            try:
                content = blob_bytes.decode("utf-8")
                return content
            except UnicodeDecodeError:
                import base64
                return base64.b64encode(blob_bytes).decode("ascii")

        raise NotImplementedError(f"SeedVault does not handle tool {tool_name}")

    # -- Backup --------------------------------------------------------------

    def backup_paths(self) -> List[str]:
        """Declare vault path for hermes backup."""
        if self._vault_dir:
            return [str(self._vault_dir)]
        return []

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush state and save digest."""
        if self._digest_mgr and self._vault:
            try:
                active_ids = self._vault.get_active_seed_ids()
                digest = self._digest_mgr.load()
                digest["active_seed_ids"] = active_ids
                self._digest_mgr.save(digest)
                logger.info("SeedVault: shutdown complete, digest saved")
            except Exception as e:
                logger.warning("SeedVault: shutdown error: %s", e)