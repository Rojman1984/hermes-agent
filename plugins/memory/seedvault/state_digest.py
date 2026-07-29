"""State digest management for SeedVault.

The state digest is a compact JSON carry-forward that survives compaction
boundaries. Written before compaction (on_pre_compress / on_session_switch),
read after compaction to reconstruct task context.

Adapted from MindSeed's state_digest concept. See DESIGN.md for paradigm
differences (runtime extraction vs build-time authoring).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .vault import SeedVault

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateDigestManager:
    """Manages the state digest lifecycle across compaction boundaries."""

    def __init__(self, vault: SeedVault):
        self.vault = vault

    def load(self) -> Dict[str, Any]:
        """Load the current state digest."""
        return self.vault.load_digest()

    def save(self, digest: Dict[str, Any]) -> None:
        """Save the state digest with file locking."""
        self.vault.save_digest(digest)

    def update_after_turn(
        self,
        current_task: str,
        pending_actions: List[str],
        active_seed_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Lightweight update after each turn (sync_turn hook).

        Does NOT touch the compaction counter — just refreshes task state
        and active seed IDs.
        """
        digest = self.load()
        digest["current_task"] = current_task
        digest["pending_actions"] = pending_actions
        if active_seed_ids is not None:
            digest["active_seed_ids"] = active_seed_ids
        self.save(digest)
        return digest

    def snapshot_for_compaction(
        self,
        current_task: str,
        pending_actions: List[str],
        active_seed_ids: List[str],
        session_lineage: List[str],
    ) -> Dict[str, Any]:
        """Create a full digest snapshot before compaction.

        Called from on_pre_compress. Increments compaction_count and
        updates last_compaction timestamp.
        """
        digest = self.load()
        digest["current_task"] = current_task
        digest["pending_actions"] = pending_actions
        digest["active_seed_ids"] = active_seed_ids
        digest["compaction_count"] = digest.get("compaction_count", 0) + 1
        digest["last_compaction"] = _utc_now()

        # Merge session lineage
        for sid in session_lineage:
            if sid not in digest.get("session_lineage", []):
                digest.setdefault("session_lineage", []).append(sid)

        self.save(digest)
        logger.info(
            "SeedVault: state digest snapshot saved (compaction #%d, %d active seeds, lineage=%s)",
            digest["compaction_count"],
            len(active_seed_ids),
            digest["session_lineage"],
        )
        return digest

    def restore_after_compaction(self) -> Dict[str, Any]:
        """Read the digest after compaction to reconstruct context.

        Called from on_session_switch or at session start.
        Returns the digest dict for the provider to inject into prompt.
        """
        digest = self.load()
        logger.info(
            "SeedVault: state digest restored (compaction #%d, task='%s', %d active seeds)",
            digest.get("compaction_count", 0),
            digest.get("current_task", "")[:80],
            len(digest.get("active_seed_ids", [])),
        )
        return digest

    def format_for_prompt(self) -> str:
        """Format the digest as a compact text block for system prompt injection."""
        digest = self.load()
        if not digest.get("current_task") and not digest.get("active_seed_ids"):
            return ""

        lines = ["[SeedVault State Digest]"]
        if digest.get("current_task"):
            lines.append(f"Task: {digest['current_task']}")
        if digest.get("pending_actions"):
            lines.append("Pending:")
            for action in digest["pending_actions"][:10]:
                lines.append(f"  - {action}")
        if digest.get("active_seed_ids"):
            lines.append(f"Active seeds: {', '.join(digest['active_seed_ids'][:20])}")
        if digest.get("compaction_count", 0) > 0:
            lines.append(f"Compactions: {digest['compaction_count']}")
        return "\n".join(lines)