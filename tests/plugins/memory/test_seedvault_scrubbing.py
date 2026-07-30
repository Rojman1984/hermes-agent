"""Tests for Phase 8a — commit-time and retrieval-time secret scrubbing.

Tests:
1. Commit-time: a known .env value pasted into a message → redacted before
   reaching the (fake, spy-asserted) LLM extraction call.
2. Retrieval-time, pre-existing gap: commit a seed containing a secret value
   *before* that value exists in .env, then add it to .env, then call
   prefetch() — assert the returned text is redacted.
3. Retrieval-time, repeated exposure: call prefetch() twice for the same
   seed — assert both calls return redacted text.
4. Shared implementation check: both trigger points call the same
   underlying function from the scrub module.
5. Scrub on bytes (for Phase 8 blob integration).
6. No false positives: text without secrets is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.seedvault.scrub import (
    discover_secrets_from_env,
    build_replacement_patterns,
    scrub_text,
    scrub_bytes,
    get_scrub_patterns,
)
from plugins.memory.seedvault.extractor import extract_seeds
from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.extractor import _utc_now


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_env(tmp_path):
    """Create a temporary .env file with known secrets."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# Hermes credentials\n"
        "OPENAI_API_KEY=sk-test-1234567890abcdef\n"
        "GITHUB_TOKEN=ghp_abcdef1234567890abcdef\n"
        "DATABASE_PASSWORD=s3cr3tp@ss\n"
        "PLACEHOLDER_KEY=your_api_key_here\n"  # should be skipped
    )
    return env_path


@pytest.fixture
def tmp_vault(tmp_path):
    """Create a SeedVault with a temporary directory."""
    vault_dir = tmp_path / "memory" / "seedvault"
    return SeedVault(vault_dir)


def _make_seed_with_claim(seed_id: str, claim: str, tags: list[str] | None = None) -> dict:
    return {
        "id": seed_id,
        "core_claim": claim,
        "chunks": [{"type": "fact", "content": claim}],
        "meristems": [],
        "source_ref": {"session_id": "s1", "profile": "default"},
        "trust_score": 0.8,
        "trust_history": [
            {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
        ],
        "status": "active",
        "superseded_by": [],
        "superseded_at": None,
        "tags": tags or ["env", "test"],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
    }


# ---------------------------------------------------------------------------
# Unit tests for the scrub module itself
# ---------------------------------------------------------------------------

class TestScrubModule:
    """Test the shared scrub module functions."""

    def test_discover_secrets_from_env(self, fake_env):
        """discover_secrets_from_env finds non-commented, non-placeholder secrets."""
        secrets = discover_secrets_from_env(fake_env)
        labels = [s[0] for s in secrets]
        assert "OPENAI_API_KEY" in labels
        assert "GITHUB_TOKEN" in labels
        assert "DATABASE_PASSWORD" in labels
        # Placeholder should be skipped
        assert "PLACEHOLDER_KEY" not in labels

    def test_discover_secrets_missing_env(self, tmp_path):
        """discover_secrets_from_env returns empty list for missing file."""
        secrets = discover_secrets_from_env(tmp_path / "nonexistent.env")
        assert secrets == []

    def test_scrub_text_replaces_secret(self, fake_env):
        """scrub_text replaces a known secret with [REDACTED:LABEL]."""
        secrets = discover_secrets_from_env(fake_env)
        patterns = build_replacement_patterns(secrets)
        text = f"The API key is sk-test-1234567890abcdef and it works."
        cleaned, count = scrub_text(text, patterns)
        assert count == 1
        assert "sk-test-1234567890abcdef" not in cleaned
        assert "[REDACTED:OPENAI_API_KEY]" in cleaned

    def test_scrub_text_no_false_positives(self, fake_env):
        """Text without secrets is unchanged."""
        secrets = discover_secrets_from_env(fake_env)
        patterns = build_replacement_patterns(secrets)
        text = "This is a normal message about the weather."
        cleaned, count = scrub_text(text, patterns)
        assert count == 0
        assert cleaned == text

    def test_scrub_text_multiple_secrets(self, fake_env):
        """scrub_text replaces multiple different secrets in one pass."""
        secrets = discover_secrets_from_env(fake_env)
        patterns = build_replacement_patterns(secrets)
        text = (
            "Key: sk-test-1234567890abcdef, "
            "Token: ghp_abcdef1234567890abcdef, "
            "Password: s3cr3tp@ss"
        )
        cleaned, count = scrub_text(text, patterns)
        assert count == 3
        assert "sk-test-1234567890abcdef" not in cleaned
        assert "ghp_abcdef1234567890abcdef" not in cleaned
        assert "s3cr3tp@ss" not in cleaned
        assert "[REDACTED:OPENAI_API_KEY]" in cleaned
        assert "[REDACTED:GITHUB_TOKEN]" in cleaned
        assert "[REDACTED:DATABASE_PASSWORD]" in cleaned

    def test_scrub_bytes(self, fake_env):
        """scrub_bytes replaces secrets in byte content."""
        secrets = discover_secrets_from_env(fake_env)
        patterns = build_replacement_patterns(secrets)
        data = b"export API_KEY=sk-test-1234567890abcdef"
        cleaned, count = scrub_bytes(data, patterns)
        assert count == 1
        assert b"sk-test-1234567890abcdef" not in cleaned
        assert b"[REDACTED:OPENAI_API_KEY]" in cleaned

    def test_scrub_bytes_no_change_when_clean(self, fake_env):
        """scrub_bytes returns original data unchanged when no secrets."""
        secrets = discover_secrets_from_env(fake_env)
        patterns = build_replacement_patterns(secrets)
        data = b"no secrets here"
        cleaned, count = scrub_bytes(data, patterns)
        assert count == 0
        assert cleaned == data

    def test_get_scrub_patterns_empty_when_no_env(self, tmp_path):
        """get_scrub_patterns returns empty list when no .env exists."""
        patterns = get_scrub_patterns(env_path=tmp_path / "nonexistent.env")
        assert patterns == []


# ---------------------------------------------------------------------------
# Commit-time scrub tests
# ---------------------------------------------------------------------------

class TestCommitTimeScrub:
    """Test that secrets are scrubbed before reaching extraction."""

    def test_commit_time_scrub_redacts_secret_in_heuristic_mode(
        self, fake_env, tmp_path
    ):
        """In heuristic mode, a secret in message content is redacted before
        the pattern matcher sees it — the extracted seed's core_claim should
        NOT contain the raw secret."""
        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.extractor.get_scrub_patterns",
            return_value=patterns,
        ):
            messages = [
                {
                    "role": "user",
                    "content": (
                        "The OPENAI_API_KEY must stay set to "
                        "sk-test-1234567890abcdef for the system to work."
                    ),
                }
            ]
            seeds = extract_seeds(
                messages=messages,
                session_id="test-scrub-001",
                profile="default",
            )

        # The extracted seed should NOT contain the raw secret
        for seed in seeds:
            assert "sk-test-1234567890abcdef" not in seed["core_claim"]
            # If the pattern matched, the redacted form should be present
            if "REDACTED" in seed["core_claim"]:
                assert "[REDACTED:OPENAI_API_KEY]" in seed["core_claim"]

    def test_commit_time_scrub_redacts_in_llm_mode(self, fake_env):
        """In LLM mode, a secret in message content is redacted before the
        content reaches the LLM extraction call."""
        # Spy that captures the messages passed to the LLM caller
        captured_messages: list[str] = []

        def fake_llm_caller(system_prompt: str, user_content: str) -> str:
            captured_messages.append(user_content)
            return "[]"  # return empty array — no seeds

        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.extractor.get_scrub_patterns",
            return_value=patterns,
        ):
            messages = [
                {
                    "role": "user",
                    "content": (
                        "I set the GITHUB_TOKEN to ghp_abcdef1234567890abcdef "
                        "for the build."
                    ),
                }
            ]
            extract_seeds(
                messages=messages,
                session_id="test-scrub-llm",
                profile="default",
                llm_caller=fake_llm_caller,
            )

        # The LLM caller should NOT have received the raw secret
        for captured in captured_messages:
            assert "ghp_abcdef1234567890abcdef" not in captured
            assert "[REDACTED:GITHUB_TOKEN]" in captured


# ---------------------------------------------------------------------------
# Retrieval-time scrub tests
# ---------------------------------------------------------------------------

class TestRetrievalTimeScrub:
    """Test that secrets are scrubbed from prefetch/search output."""

    def test_retrieval_time_scrub_redacts_pre_existing_secret(
        self, fake_env, tmp_vault
    ):
        """Commit a seed containing a secret (before scrub was available),
        then scrub at retrieval time — prefetch() should redact it."""
        # Write a seed directly to the vault with a secret in core_claim
        seed = _make_seed_with_claim(
            "env-apikey-001",
            "The API key is sk-test-1234567890abcdef for OpenAI.",
            tags=["env", "apikey"],
        )
        tmp_vault.write_seed(seed)

        # Mock get_scrub_patterns to use our fake .env
        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.provider.get_scrub_patterns",
            return_value=patterns,
        ):
            # Create a minimal provider-like object to call prefetch
            from plugins.memory.seedvault.provider import SeedVaultMemoryProvider
            provider = SeedVaultMemoryProvider()
            provider._vault = tmp_vault
            provider._digest_mgr = None
            provider._vault_dir = tmp_vault.vault_dir

            result = provider.prefetch("API key")

        # The raw secret should NOT be in the output
        assert "sk-test-1234567890abcdef" not in result
        assert "[REDACTED:OPENAI_API_KEY]" in result

    def test_retrieval_time_scrub_repeated_exposure(
        self, fake_env, tmp_vault
    ):
        """Call prefetch() twice — both calls should return redacted text."""
        seed = _make_seed_with_claim(
            "env-token-001",
            "Use GITHUB_TOKEN=ghp_abcdef1234567890abcdef for CI.",
            tags=["env", "token"],
        )
        tmp_vault.write_seed(seed)

        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.provider.get_scrub_patterns",
            return_value=patterns,
        ):
            from plugins.memory.seedvault.provider import SeedVaultMemoryProvider
            provider = SeedVaultMemoryProvider()
            provider._vault = tmp_vault
            provider._digest_mgr = None
            provider._vault_dir = tmp_vault.vault_dir

            result1 = provider.prefetch("GITHUB_TOKEN")
            result2 = provider.prefetch("GITHUB_TOKEN")

        assert "ghp_abcdef1234567890abcdef" not in result1
        assert "ghp_abcdef1234567890abcdef" not in result2
        assert "[REDACTED:GITHUB_TOKEN]" in result1
        assert "[REDACTED:GITHUB_TOKEN]" in result2

    def test_retrieval_time_scrub_search_tool_output(
        self, fake_env, tmp_vault
    ):
        """seedvault_search tool output is also scrubbed."""
        seed = _make_seed_with_claim(
            "env-pass-001",
            "DATABASE_PASSWORD is s3cr3tp@ss for prod.",
            tags=["env", "password"],
        )
        tmp_vault.write_seed(seed)

        patterns = build_replacement_patterns(discover_secrets_from_env(fake_env))
        with patch(
            "plugins.memory.seedvault.provider.get_scrub_patterns",
            return_value=patterns,
        ):
            from plugins.memory.seedvault.provider import SeedVaultMemoryProvider
            provider = SeedVaultMemoryProvider()
            provider._vault = tmp_vault
            provider._digest_mgr = None
            provider._vault_dir = tmp_vault.vault_dir

            result = provider.handle_tool_call(
                "seedvault_search", {"query": "password"}
            )

        assert "s3cr3tp@ss" not in result
        assert "[REDACTED:DATABASE_PASSWORD]" in result

    def test_retrieval_time_scrub_no_false_positives(self, tmp_vault):
        """Seeds without secrets are unaffected by scrub."""
        seed = _make_seed_with_claim(
            "env-ollama-001",
            "Ollama runs on 127.0.0.1:11434.",
            tags=["env", "ollama"],
        )
        tmp_vault.write_seed(seed)

        # No .env → no patterns → no scrubbing
        with patch(
            "plugins.memory.seedvault.provider.get_scrub_patterns",
            return_value=[],
        ):
            from plugins.memory.seedvault.provider import SeedVaultMemoryProvider
            provider = SeedVaultMemoryProvider()
            provider._vault = tmp_vault
            provider._digest_mgr = None
            provider._vault_dir = tmp_vault.vault_dir

            result = provider.prefetch("Ollama")

        assert "127.0.0.1:11434" in result  # not redacted (not a secret)


# ---------------------------------------------------------------------------
# Shared implementation check
# ---------------------------------------------------------------------------

class TestSharedImplementation:
    """Verify both trigger points use the same underlying scrub function."""

    def test_commit_and_retrieval_use_same_scrub_text(self):
        """Both extractor and provider import scrub_text from the same module."""
        from plugins.memory.seedvault.extractor import scrub_text as ext_scrub
        from plugins.memory.seedvault.provider import scrub_text as prov_scrub
        from plugins.memory.seedvault.scrub import scrub_text as src_scrub

        assert ext_scrub is src_scrub
        assert prov_scrub is src_scrub

    def test_commit_and_retrieval_use_same_get_scrub_patterns(self):
        """Both extractor and provider import get_scrub_patterns from the same module."""
        from plugins.memory.seedvault.extractor import get_scrub_patterns as ext_gsp
        from plugins.memory.seedvault.provider import get_scrub_patterns as prov_gsp
        from plugins.memory.seedvault.scrub import get_scrub_patterns as src_gsp

        assert ext_gsp is src_gsp
        assert prov_gsp is src_gsp