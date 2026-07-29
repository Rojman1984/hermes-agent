"""Seed extraction from conversation messages.

Extracts structured seeds from messages being compressed. Two modes:
1. LLM-assisted extraction (preferred) — uses the agent's model to identify
   extractable facts and structure them as seeds.
2. Heuristic fallback — pattern-based extraction when no LLM is available.

PARADIGM NOTE: Unlike MindSeed's build-time authoring (human-curated seeds
from a canonical corpus), SeedVault extracts at runtime from live dialogue.
The input is unstructured and noisy. The commit gate (validator.py) is the
compensating control, not a replacement for MindSeed's build-time validation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_seed_id(domain: str, topic: str, index: int) -> str:
    """Generate a seed ID in domain-topic-NNN format."""
    domain = re.sub(r"[^a-z0-9]", "", domain.lower())[:20] or "misc"
    topic = re.sub(r"[^a-z0-9]", "", topic.lower())[:20] or "general"
    return f"{domain}-{topic}-{index:03d}"


# -- Heuristic extraction ----------------------------------------------------

# Patterns that indicate durable facts worth seeding
_PATTERNS = [
    (r"(?:must|should|needs? to|require[ds]?)\s+(.+?)(?:[.\n]|$)", "constraint", "env"),
    (r"(?:error|fail|crash|broken|cannot|can\'t)\s+(.+?)(?:[.\n]|$)", "error", "error"),
    (r"(?:decision|decided|chose|chose to)\s+(.+?)(?:[.\n]|$)", "decision", "decision"),
    (r"(?:prefer|preference|always|never)\s+(.+?)(?:[.\n]|$)", "preference", "pref"),
    (r"(?:completed|finished|done|created|installed|configured)\s+(.+?)(?:[.\n]|$)", "status", "status"),
]


def _heuristic_extract(messages: List[Dict[str, Any]], session_id: str, profile: str) -> List[Dict[str, Any]]:
    """Pattern-based seed extraction. Fallback when no LLM available.
    
    Scans user and assistant messages for durable-fact patterns.
    Returns raw seed dicts (pre-gate, pre-validation).
    """
    raw_seeds: List[Dict[str, Any]] = []
    seed_counter = 0

    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c)
                for c in content
            )
        if not isinstance(content, str) or not content.strip():
            continue

        for pattern, chunk_type, domain in _PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Capture the full sentence containing the match, not just the fragment
                # Find sentence boundaries around the match position
                start = content.rfind(". ", 0, match.start())
                start = start + 2 if start >= 0 else max(0, content.rfind("\n", 0, match.start()) + 1)
                end = content.find(". ", match.end())
                if end < 0:
                    end = content.find("\n", match.end())
                if end < 0:
                    end = len(content)
                else:
                    end += 1  # include the period
                claim = content[start:end].strip()[:500]
                if len(claim) < 10:
                    continue

                seed_counter += 1
                seed = {
                    "id": _make_seed_id(domain, chunk_type, seed_counter),
                    "core_claim": claim,
                    "chunks": [{
                        "type": chunk_type,
                        "content": claim,
                    }],
                    "meristems": [],
                    "source_ref": {
                        "session_id": session_id,
                        "profile": profile,
                        "message_id": msg.get("id"),
                    },
                    "trust_score": 0.8,
                    "trust_history": [{
                        "delta": None,
                        "reason": "initial",
                        "value": 0.8,
                        "at": _utc_now(),
                    }],
                    "status": "active",
                    "superseded_by": [],
                    "superseded_at": None,
                    "tags": [domain, chunk_type],
                    "created": _utc_now(),
                    "updated": _utc_now(),
                    "last_validated": None,
                }
                raw_seeds.append(seed)

    logger.info("SeedVault: heuristic extraction found %d raw seeds from %d messages",
                len(raw_seeds), len(messages))
    return raw_seeds


# -- LLM-assisted extraction -------------------------------------------------

_EXTRACTION_PROMPT = """You are a memory extraction system. Analyze the conversation messages below and extract durable facts that should survive compaction.

For each fact worth preserving, output a JSON object with these fields:
- core_claim: The essential fact (max 500 chars). Must be self-contained.
- chunk_type: One of: constraint, status, decision, insight, preference, error
- domain: A short domain tag (e.g., env, project, model, tool, user)
- tags: 1-3 lowercase tags. First tag is the primary domain.

Rules:
- Only extract facts that are DURABLE — things that will matter in future sessions.
- Skip transient state (current file contents, in-progress calculations, temporary observations).
- Skip opinions and speculation. Extract verified facts and explicit decisions only.
- If a fact depends on or relates to another fact, add a meristem edge.
- Output a JSON array of seed objects. If no seeds are found, output [].

Conversation messages:
"""


def _llm_extract(
    messages: List[Dict[str, Any]],
    session_id: str,
    profile: str,
    llm_caller: Any,
) -> List[Dict[str, Any]]:
    """LLM-assisted seed extraction.
    
    llm_caller is a callable that takes (system_prompt, user_content) and
    returns a string (the LLM response). The provider passes this in from
    the agent's configured model.
    """
    # Serialize messages for the LLM
    msg_text = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c)
                for c in content
            )
        msg_text += f"[{i}] {role}: {content}\n\n"

    if not msg_text.strip():
        return []

    try:
        response = llm_caller(_EXTRACTION_PROMPT, msg_text)
    except Exception as e:
        logger.warning("SeedVault: LLM extraction failed: %s, falling back to heuristic", e)
        return []

    # Parse LLM response as JSON array
    response = response.strip()
    # Strip markdown code fences if present
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

    try:
        extracted = json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning("SeedVault: LLM returned invalid JSON: %s", e)
        return []

    if not isinstance(extracted, list):
        logger.warning("SeedVault: LLM returned non-array: %s", type(extracted))
        return []

    # Convert to seed format
    raw_seeds: List[Dict[str, Any]] = []
    for i, item in enumerate(extracted, 1):
        if not isinstance(item, dict):
            continue
        claim = item.get("core_claim", "").strip()
        if not claim:
            continue

        domain = item.get("domain", "misc")
        chunk_type = item.get("chunk_type", "insight")
        tags = item.get("tags", [domain])
        if not tags:
            tags = [domain]
        if chunk_type not in ("constraint", "status", "decision", "insight", "preference", "error"):
            chunk_type = "insight"

        seed = {
            "id": _make_seed_id(domain, chunk_type, i),
            "core_claim": claim[:500],
            "chunks": [{
                "type": chunk_type,
                "content": claim[:1000],
            }],
            "meristems": [],
            "source_ref": {
                "session_id": session_id,
                "profile": profile,
            },
            "trust_score": 0.8,
            "trust_history": [{
                "delta": None,
                "reason": "initial",
                "value": 0.8,
                "at": _utc_now(),
            }],
            "status": "active",
            "superseded_by": [],
            "superseded_at": None,
            "tags": tags,
            "created": _utc_now(),
            "updated": _utc_now(),
            "last_validated": None,
        }
        raw_seeds.append(seed)

    logger.info("SeedVault: LLM extraction found %d raw seeds from %d messages",
                len(raw_seeds), len(messages))
    return raw_seeds


# -- Public API --------------------------------------------------------------

def extract_seeds(
    messages: List[Dict[str, Any]],
    session_id: str,
    profile: str = "default",
    llm_caller: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract seeds from conversation messages.
    
    Uses LLM-assisted extraction if llm_caller is provided, otherwise
    falls back to heuristic pattern matching.
    
    Returns raw seed dicts — they still need to pass the commit gate
    (validator.py) before landing in the vault.
    """
    if llm_caller is not None:
        seeds = _llm_extract(messages, session_id, profile, llm_caller)
        if seeds:
            return seeds
        # Fall through to heuristic if LLM returned nothing

    return _heuristic_extract(messages, session_id, profile)