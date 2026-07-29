"""Commit gate and validation for SeedVault seeds.

Two-stage gate adapted from MindSeed's triple-metric gate concept,
but simplified for runtime extraction (no canonical reference corpus).

Stage 1 (deterministic, always runs):
  - core_claim non-empty, <= 500 chars
  - source_ref.session_id is not null (provenance required)
  - No duplicate core_claim (Jaccard >0.7 = reject)
  - Meristem targets resolve to existing seed IDs (dangling dropped)

Stage 2 (LLM-assisted, compression events only):
  - Re-extract claims from same messages, compare to committed seeds
  - If core_claim can't be traced to source messages: trust_score -= 0.2
  - Seeds below trust_score 0.3 auto-archived

PARADIGM NOTE: MindSeed's gate has a canonical reference to validate against.
SeedVault's Stage 2 is weaker — it checks internal consistency (can the claim
be re-derived from the source?) rather than external correctness. This is a
stated limitation, not an oversight. See DESIGN.md.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .vault import SeedVault, _tokenize, _jaccard

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommitGate:
    """Two-stage validation gate for seed commits."""

    def __init__(self, vault: SeedVault):
        self.vault = vault

    def stage1_validate(self, seed: Dict[str, Any], skip_dedup: bool = False) -> tuple[bool, str]:
        """Deterministic validation. Returns (passed, reason).

        If failed, the seed is rejected — it does NOT land in the vault.

        When ``skip_dedup`` is True the Jaccard duplicate check is skipped.
        This is used by :meth:`commit` when supersession candidates exist
        (same primary tag), so that a legitimate "same domain, updated claim"
        seed is not rejected as a duplicate before supersession fires (S1).
        """
        # core_claim checks
        claim = seed.get("core_claim", "")
        if not claim or not claim.strip():
            return False, "core_claim is empty"
        if len(claim) > 500:
            return False, f"core_claim too long ({len(claim)} > 500)"

        # source_ref checks
        source_ref = seed.get("source_ref", {})
        if not source_ref.get("session_id"):
            return False, "source_ref.session_id is null (no provenance)"

        # Duplicate check — skipped when supersession applies (S1)
        if not skip_dedup:
            dup_id = self.vault.find_duplicate(claim, threshold=0.7)
            if dup_id is not None:
                return False, f"duplicate of existing seed {dup_id} (Jaccard >0.7)"

        # Meristem validation — drop dangling edges
        meristems = seed.get("meristems", [])
        validated = self.vault.validate_meristems(meristems)
        seed["meristems"] = validated

        return True, "passed"

    def commit(self, seed: Dict[str, Any]) -> tuple[bool, str]:
        """Run Stage 1 gate and commit if passed.
        
        Handles supersession: if an active seed with the same primary tag exists,
        the new seed supersedes it.
        
        Returns (committed, reason).
        """
        # Check for supersession candidates BEFORE Stage-1 dedup so that a
        # legitimate "same domain, updated claim" seed is not rejected as a
        # duplicate before supersession fires (S1).  When supersession
        # candidates exist we skip the Jaccard dedup gate — the new seed is
        # an update in the same domain, not a duplicate.
        candidates = self.vault.find_superseded_candidates(seed)
        skip_dedup = bool(candidates)

        passed, reason = self.stage1_validate(seed, skip_dedup=skip_dedup)
        if not passed:
            logger.warning("SeedVault: seed rejected by Stage 1 gate: %s", reason)
            return False, reason

        # Initialize trust score if not set
        if "trust_score" not in seed:
            seed["trust_score"] = 0.8
        if not seed.get("trust_history"):
            seed["trust_history"] = [{
                "delta": None,
                "reason": "initial",
                "value": seed["trust_score"],
                "at": _utc_now(),
            }]
        if seed.get("status") is None:
            seed["status"] = "active"
        if seed.get("superseded_by") is None:
            seed["superseded_by"] = []
        if seed.get("superseded_at") is None:
            seed["superseded_at"] = None
        if not seed.get("created"):
            seed["created"] = _utc_now()
        if not seed.get("updated"):
            seed["updated"] = _utc_now()

        # Supersession candidates were already identified before Stage-1.
        # Add supersedes meristems for each.
        if candidates:
            for old_id in candidates:
                seed.setdefault("meristems", []).append({
                    "type": "supersedes",
                    "target": old_id,
                })

        # Write seed
        if not self.vault.write_seed(seed):
            return False, "failed to write seed file"

        # Supersede old seeds
        if candidates:
            self.vault.supersede_seeds(seed["id"], candidates)

        logger.info("SeedVault: committed seed %s (tags=%s, trust=%.2f)",
                    seed["id"], seed.get("tags", []), seed["trust_score"])
        return True, "committed"

    def stage2_validate(self, seed: Dict[str, Any], source_messages: List[Dict[str, Any]]) -> bool:
        """LLM-assisted validation. Checks if core_claim can be traced to source.
        
        This is a lightweight check: tokenize the core_claim and check if its
        tokens appear in the source messages. If <30% of claim tokens appear in
        the source, it's flagged as potential drift.
        
        A full LLM re-extraction would be more robust but adds latency.
        This is the v1 approximation — see DESIGN.md TODO.
        
        Returns True if validated, False if drift detected.
        """
        claim_tokens = _tokenize(seed.get("core_claim", ""))
        if not claim_tokens:
            return False

        # Gather all text from source messages
        source_text = ""
        for msg in source_messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            source_text += " " + str(content)

        source_tokens = _tokenize(source_text)
        if not source_tokens:
            return False

        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        if overlap < 0.3:
            # Drift detected
            now = _utc_now()
            old_score = seed.get("trust_score", 0.8)
            new_score = max(0.0, old_score - 0.2)
            seed["trust_score"] = new_score
            seed["trust_history"].append({
                "delta": -0.2,
                "reason": "drift_detected",
                "value": new_score,
                "at": now,
            })
            # B2: auto-archive if trust dropped below 0.3 threshold.
            # This prevents low-trust seeds from leaking into search/prefetch
            # (B3) — search() already excludes non-active status, so the
            # transition here closes the gap immediately rather than waiting
            # for the delayed prune() path (age > stale_days).
            if new_score < 0.3:
                seed["status"] = "archived"
                seed["trust_history"].append({
                    "delta": 0.0,
                    "reason": "auto_archived",
                    "value": new_score,
                    "at": now,
                })
                seed["last_validated"] = now
                self.vault.write_seed(seed)
                logger.warning(
                    "SeedVault: drift detected for seed %s (overlap=%.2f, "
                    "trust %.2f->%.2f) — auto-archived (below 0.3)",
                    seed["id"], overlap, old_score, new_score,
                )
                return False
            seed["last_validated"] = now
            self.vault.write_seed(seed)
            logger.warning("SeedVault: drift detected for seed %s (overlap=%.2f, trust %.2f->%.2f)",
                          seed["id"], overlap, old_score, new_score)
            return False

        # Validated — bump trust
        now = _utc_now()
        old_score = seed.get("trust_score", 0.8)
        new_score = min(1.0, old_score + 0.1)
        seed["trust_score"] = new_score
        seed["trust_history"].append({
            "delta": 0.1,
            "reason": "provenance_verified",
            "value": new_score,
            "at": now,
        })
        seed["last_validated"] = now
        self.vault.write_seed(seed)
        logger.debug("SeedVault: seed %s validated (overlap=%.2f, trust %.2f->%.2f)",
                    seed["id"], overlap, old_score, new_score)
        return True