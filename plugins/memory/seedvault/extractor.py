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
from typing import Any, Dict, List, Optional, Tuple

from .scrub import get_scrub_patterns, scrub_text, scrub_bytes

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


# -- Artifact detection (Phase 8.2) -----------------------------------------

# Known shell command prefixes for detecting shell commands in prose.
# Conservative: false negatives (missed commands) are acceptable; false
# positives (prose flagged as commands) are not.
_SHELL_PREFIXES = frozenset({
    "systemctl", "apt", "pip", "npm", "git", "docker", "curl", "ssh",
    "cd", "ls", "cp", "mv", "rm", "mkdir", "chmod", "chown", "export",
    "source", "sudo", "python3", "python", "node", "brew", "snap",
    "wget", "tar", "unzip", "cat", "echo", "grep", "find", "sed",
    "awk", "head", "tail", "wc", "sort", "uniq", "diff", "patch",
    "make", "cmake", "gcc", "cargo", "rustc", "go", "java", "javac",
})

# Config language tags for fenced blocks
_CONFIG_LANGS = frozenset({"yaml", "yml", "json", "toml", "ini", "env", "conf"})

# Code fence regex: ```lang\n...\n```
_CODE_FENCE_RE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)


def _detect_artifacts(content: str) -> List[Dict[str, Any]]:
    """Detect reproducible artifacts in message content.

    Detects:
    - Fenced code blocks: ```lang ... ``` — captures the exact span
      between fences, including the language tag.
    - Shell command lines: lines starting with `$ ` or lines matching
      a recognizable command-line shape (known prefix + args).
    - Diff blocks: fenced blocks with `diff` language tag or content
      starting with @@/+++/---.

    Returns a list of artifact dicts, each with:
    - raw_content: the exact bytes to store in the blob
    - content_type: "bash", "code", "diff", or "config"
    - language: the fence language tag (str or None)
    - description: a short human-readable description for core_claim

    Does NOT write blobs — the caller is responsible for writing blobs
    via the vault's blob store and attaching artifact pointers to seeds.
    """
    artifacts: List[Dict[str, Any]] = []
    consumed_spans: List[Tuple[int, int]] = []  # ranges already captured by fences

    # 1. Detect fenced code blocks
    for match in _CODE_FENCE_RE.finditer(content):
        lang = match.group(1).lower() or ""
        raw = match.group(2)
        # Strip trailing newline that the fence regex captures
        if raw.endswith("\n"):
            raw = raw[:-1]

        # Classify the content type
        if lang == "diff" or raw.lstrip().startswith(("@@", "+++", "---")):
            content_type = "diff"
            desc = _describe_diff(raw)
        elif lang in ("bash", "sh", "shell", "zsh"):
            content_type = "bash"
            desc = f"Bash command: {_short_desc(raw.strip().splitlines()[0] if raw.strip() else '')}"
        elif lang in _CONFIG_LANGS:
            content_type = "config"
            desc = f"{lang.upper()} config block"
        elif lang:
            content_type = "code"
            desc = f"{lang} code block"
        else:
            # No language tag — check if it looks like a shell command
            first_line = raw.strip().split("\n")[0] if raw.strip() else ""
            if _is_shell_command(first_line):
                content_type = "bash"
                desc = f"Bash command: {_short_desc(first_line)}"
            else:
                content_type = "code"
                desc = f"Code block: {_short_desc(first_line)}"

        artifacts.append({
            "raw_content": raw,
            "content_type": content_type,
            "language": lang or None,
            "description": desc,
        })
        consumed_spans.append((match.start(), match.end()))

    # 2. Detect shell command lines (outside code fences)
    for line in content.splitlines():
        # Skip if this line is inside a code fence we already captured
        line_start = content.index(line) if line in content else -1
        if line_start >= 0:
            inside_fence = any(
                s <= line_start < e for s, e in consumed_spans
            )
            if inside_fence:
                continue

        stripped = line.strip()
        if not stripped:
            continue

        # Shell prompt convention: `$ command`
        if stripped.startswith("$ "):
            cmd = stripped[2:]
            if cmd and not cmd.startswith("#"):  # skip comments
                artifacts.append({
                    "raw_content": cmd,
                    "content_type": "bash",
                    "language": None,
                    "description": f"Bash command: {_short_desc(cmd)}",
                })
        elif _is_shell_command(stripped):
            artifacts.append({
                "raw_content": stripped,
                "content_type": "bash",
                "language": None,
                "description": f"Bash command: {_short_desc(stripped)}",
            })

    return artifacts


def _is_shell_command(line: str) -> bool:
    """Check if a line looks like a shell command.

    Conservative: only matches lines starting with a known command prefix
    followed by arguments. Does not match prose that merely mentions a
    command name.
    """
    # Skip lines that are clearly prose (end with period, are sentences)
    if line.endswith(".") and len(line.split()) > 3:
        return False
    # Skip if it starts with a quote (prose)
    if line.startswith(('"', "'")):
        return False

    parts = line.split()
    if not parts:
        return False
    # Get the base command (handle sudo, env vars)
    cmd_idx = 0
    while cmd_idx < len(parts) and "=" in parts[cmd_idx]:
        cmd_idx += 1
    if cmd_idx >= len(parts):
        return False
    base = parts[cmd_idx]
    # Strip any path prefix
    base_name = base.rsplit("/", 1)[-1]
    return base_name in _SHELL_PREFIXES


def _short_desc(text: str, max_len: int = 60) -> str:
    """Create a short description from the first line of content."""
    first_line = text.strip().split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len - 3] + "..."
    return first_line


def _describe_diff(raw: str) -> str:
    """Create a description for a diff block."""
    first_line = raw.strip().split("\n")[0]
    if first_line.startswith("@@"):
        return f"Diff: {first_line[:50]}"
    return "Diff block"


def _strip_code_fences(content: str) -> str:
    """Replace code fences with a placeholder for pattern matching.

    This prevents prose patterns from matching inside code blocks (e.g.
    a comment `# must stay OFF` inside a code fence matching the
    "constraint" pattern). The original content is preserved for blob
    extraction — this is only used for the pattern-matching pass.
    """
    return _CODE_FENCE_RE.sub("[code artifact captured]", content)


def _make_artifact_seed(
    artifact: Dict[str, Any],
    session_id: str,
    profile: str,
    seed_index: int,
) -> Dict[str, Any]:
    """Create a seed dict from a detected artifact.

    One seed per artifact (per approved design decision 5.7).
    The core_claim is a short description, NOT the raw content.
    The raw content lives in the blob store (written by the caller).
    """
    desc = artifact["description"]
    content_type = artifact["content_type"]
    lang = artifact.get("language") or ""

    # Domain tag from content type
    domain = "artifact"
    if content_type == "bash":
        domain = "shell"
    elif content_type == "code":
        domain = "code"
    elif content_type == "diff":
        domain = "diff"
    elif content_type == "config":
        domain = "config"

    return {
        "id": _make_seed_id(domain, content_type, seed_index),
        "core_claim": desc[:500],
        "chunks": [{
            "type": "insight",
            "content": f"Artifact type: {content_type}, language: {lang or 'none'}",
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
        "tags": [domain, content_type],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
        # Artifacts array will be attached by the caller after blob write
        "artifacts": [],
    }


def _heuristic_extract(
    messages: List[Dict[str, Any]],
    session_id: str,
    profile: str,
    blob_store: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Pattern-based seed extraction. Fallback when no LLM available.
    
    Scans user and assistant messages for durable-fact patterns.
    Also detects reproducible artifacts (code blocks, shell commands, diffs)
    and creates one seed per artifact with a blob pointer.
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

        # Phase 8.2: detect artifacts first
        artifacts = _detect_artifacts(content)
        for artifact in artifacts:
            seed_counter += 1
            seed = _make_artifact_seed(artifact, session_id, profile, seed_counter)
            # Write blob if blob_store is available
            if blob_store is not None:
                raw_bytes = artifact["raw_content"].encode("utf-8")
                # Phase 8a: scrub blob bytes before writing
                scrub_patterns = get_scrub_patterns()
                if scrub_patterns:
                    raw_bytes, scrub_count = scrub_bytes(raw_bytes, scrub_patterns)
                    if scrub_count > 0:
                        logger.debug(
                            "SeedVault: scrubbed %d secret(s) from artifact blob",
                            scrub_count,
                        )
                blob_hash = blob_store.write_blob(
                    raw_bytes=raw_bytes,
                    seed_id=seed["id"],
                    content_type=artifact["content_type"],
                    language=artifact.get("language"),
                )
                if blob_hash is not None:
                    seed["artifacts"] = [{
                        "blob_hash": blob_hash,
                        "content_type": artifact["content_type"],
                        "language": artifact.get("language"),
                        "byte_length": len(raw_bytes),
                    }]
                else:
                    # Blob write failed (oversized or error) — drop artifact
                    # pointer but keep the seed (prose description survives)
                    seed["artifacts"] = []
                    logger.warning(
                        "SeedVault: blob write failed for artifact seed %s, "
                        "committing seed without artifact pointer",
                        seed["id"],
                    )
            raw_seeds.append(seed)

        # Run prose patterns on content with code fences stripped
        # (prevents patterns matching inside code blocks)
        prose_content = _strip_code_fences(content) if artifacts else content

        for pattern, chunk_type, domain in _PATTERNS:
            for match in re.finditer(pattern, prose_content, re.IGNORECASE):
                # Capture the full sentence containing the match
                start = prose_content.rfind(". ", 0, match.start())
                start = start + 2 if start >= 0 else max(0, prose_content.rfind("\n", 0, match.start()) + 1)
                end = prose_content.find(". ", match.end())
                if end < 0:
                    end = prose_content.find("\n", match.end())
                if end < 0:
                    end = len(prose_content)
                else:
                    end += 1  # include the period
                claim = prose_content[start:end].strip()[:500]
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
  Meristem edges are objects with "type" and "target" fields:
  - type: one of "prerequisite", "nextstep", "bridge", "supersedes", "related"
  - target: the seed ID this edge points to (only use IDs from other seeds in this same output array)
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

        # B5: parse meristem edges from LLM response instead of hardcoding [].
        # Sanitize each edge — must be a dict with a non-empty "target" string
        # and a valid "type". Malformed edges are dropped here; dangling edges
        # (target points to a seed not yet in the vault) are dropped later by
        # the commit gate's validate_meristems().
        valid_meristem_types = {"prerequisite", "nextstep", "bridge", "supersedes", "related"}
        raw_meristems = item.get("meristems", [])
        meristems: List[Dict[str, Any]] = []
        if isinstance(raw_meristems, list):
            for edge in raw_meristems:
                if not isinstance(edge, dict):
                    continue
                target = edge.get("target", "")
                edge_type = edge.get("type", "related")
                if not isinstance(target, str) or not target.strip():
                    continue
                if edge_type not in valid_meristem_types:
                    edge_type = "related"
                meristems.append({"type": edge_type, "target": target.strip()})

        seed = {
            "id": _make_seed_id(domain, chunk_type, i),
            "core_claim": claim[:500],
            "chunks": [{
                "type": chunk_type,
                "content": claim[:1000],
            }],
            "meristems": meristems,
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
    blob_store: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract seeds from conversation messages.
    
    Uses LLM-assisted extraction if llm_caller is provided, otherwise
    falls back to heuristic pattern matching.
    
    blob_store: optional BlobStore instance for writing artifact blobs.
    When provided, the heuristic extractor writes detected artifacts
    (code blocks, shell commands, diffs) to the blob store and attaches
    artifact pointers to the seed dicts. (Phase 8.2)
    
    Returns raw seed dicts — they still need to pass the commit gate
    (validator.py) before landing in the vault.
    
    Commit-time scrub: raw message content is scrubbed for secrets
    (from .env) BEFORE reaching either extraction path.  This ensures
    secrets never reach the LLM extraction API call or the heuristic
    pattern matcher.  (Phase 8a)
    """
    # Phase 8a: commit-time scrub — sanitize raw message content before
    # extraction so secrets never enter the seed pipeline.
    scrub_patterns = get_scrub_patterns()
    if scrub_patterns:
        scrubbed_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content", "")
            if isinstance(content, str) and content:
                cleaned, n = scrub_text(content, scrub_patterns)
                if n > 0:
                    msg_copy["content"] = cleaned
                    logger.debug(
                        "SeedVault: commit-time scrub redacted %d secret(s) "
                        "in message role=%s",
                        n,
                        msg.get("role", "?"),
                    )
            scrubbed_messages.append(msg_copy)
        messages = scrubbed_messages

    if llm_caller is not None:
        seeds = _llm_extract(messages, session_id, profile, llm_caller)
        if seeds:
            return seeds
        # Fall through to heuristic if LLM returned nothing

    return _heuristic_extract(messages, session_id, profile, blob_store=blob_store)