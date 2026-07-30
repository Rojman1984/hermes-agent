"""Shared secret scrubbing for SeedVault.

Ports the core detection and replacement logic from
``~/.hermes/skills/security/scrubhistory/scripts/scrub.py`` into an
importable module within the SeedVault plugin.  This is the one shared
implementation — both commit-time (extractor) and retrieval-time
(provider/vault) call sites import from here, not from two copies.

What it does:
- Discovers secrets from the Hermes ``.env`` file (same patterns as
  scrubhistory: ``*_API_KEY|*_TOKEN|*_PASSWORD|*_SECRET|*_ACCESS_KEY|
  *_PRIVATE_KEY|*_CREDENTIAL``, skips placeholder values).
- Builds replacement patterns that replace each secret value with
  ``[REDACTED:LABEL]``.
- Scrubs text and bytes in place.

What it does NOT do:
- Duplicate the full scrub.py CLI (DB scrubbing, log scrubbing, FTS rebuild).
  Those are session-end operations that stay in the scrubhistory skill.
- Pattern/regex library or entropy heuristics — those were dropped from the
  original 8a spec as redundant with iron_proxy / secret_sources.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

REDACTED_LABEL = "REDACTED"

# Patterns for auto-discovering secrets in .env (same as scrubhistory)
SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r'^(?!\s*#)\s*(\w*(?:API_KEY|TOKEN|PASSWORD|SECRET|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)\w*)\s*=\s*(\S+)\s*$',
        re.IGNORECASE,
    ),
]


def discover_secrets_from_env(env_path: Path) -> List[Tuple[str, str]]:
    """Parse .env file and extract (label, value) pairs for non-commented secrets.

    This is a direct port of ``scrubhistory.scrub.discover_secrets_from_env``.
    Same patterns, same placeholder exclusion, same return shape.
    """
    secrets: List[Tuple[str, str]] = []
    if not env_path.exists():
        return secrets
    try:
        text = env_path.read_text()
    except OSError as e:
        logger.warning("SeedVault scrub: could not read %s: %s", env_path, e)
        return secrets
    for line in text.splitlines():
        for pat in SECRET_PATTERNS:
            m = pat.match(line)
            if m and m.group(2) and m.group(2) != "":
                label = m.group(1)
                value = m.group(2)
                # Skip empty or placeholder values
                if value.startswith("your_") or value.startswith("xxxx") or value == "":
                    continue
                secrets.append((label, value))
    return secrets


def build_replacement_patterns(
    secrets: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """Build (escaped_regex, replacement_string, original_value) tuples.

    Each tuple is used by ``scrub_text`` / ``scrub_bytes`` for exact-string
    replacement.  The escaped_regex is ``re.escape(value)`` — the replacement
    uses ``re.sub`` for regex-based replacement (handles edge cases where the
    value contains regex metacharacters).
    """
    patterns: List[Tuple[str, str, str]] = []
    for label, value in secrets:
        escaped = re.escape(value)
        replacement = f"[{REDACTED_LABEL}:{label}]"
        patterns.append((escaped, replacement, value))
    return patterns


def scrub_text(text: str, patterns: List[Tuple[str, str, str]]) -> Tuple[str, int]:
    """Replace all secret occurrences in text.

    Returns (new_text, count).  Uses ``str.replace`` for exact match (same
    as scrubhistory — simpler and faster than regex for literal strings).
    """
    count = 0
    for _escaped, replacement, original in patterns:
        n = text.count(original)
        if n > 0:
            text = text.replace(original, replacement)
            count += n
    return text, count


def scrub_bytes(
    data: bytes, patterns: List[Tuple[str, str, str]]
) -> Tuple[bytes, int]:
    """Replace all secret occurrences in bytes.

    Returns (new_bytes, count).  Decodes as UTF-8 with error replacement,
    scrubs, re-encodes.  Suitable for blob content that is expected to be
    text (code, config, shell commands).
    """
    text = data.decode("utf-8", errors="replace")
    new_text, count = scrub_text(text, patterns)
    if count == 0:
        return data, 0
    return new_text.encode("utf-8"), count


def get_scrub_patterns(env_path: Path | None = None) -> List[Tuple[str, str, str]]:
    """Convenience: discover secrets from .env and build replacement patterns.

    If env_path is None, defaults to ``~/.hermes/.env`` (via HERMES_HOME env).
    Returns an empty list if no .env file exists or no secrets are found —
    callers should handle this gracefully (no scrubbing needed).
    """
    if env_path is None:
        import os

        hermes_home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
        env_path = hermes_home / ".env"
    secrets = discover_secrets_from_env(env_path)
    if not secrets:
        return []
    return build_replacement_patterns(secrets)