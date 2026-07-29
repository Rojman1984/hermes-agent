"""Tests for SeedVault memory provider plugin."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from plugins.memory.seedvault import SeedVaultMemoryProvider
from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.extractor import (
    _make_seed_id,
    _utc_now,
    _heuristic_extract as heuristic_extract,
)
from plugins.memory.seedvault.validator import CommitGate
from plugins.memory.seedvault.state_digest import StateDigestManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    """Create a SeedVault with a temporary directory."""
    vault_dir = tmp_path / "memory" / "seedvault"
    # SeedVault.ensure_dirs creates subdirectories
    return SeedVault(vault_dir)


@pytest.fixture
def tmp_provider(tmp_path):
    """Create a fully initialized SeedVaultMemoryProvider in a temp dir."""
    provider = SeedVaultMemoryProvider()
    provider.initialize("test-session-001", hermes_home=str(tmp_path))
    return provider


def _make_seed(
    seed_id="env-test-001",
    core_claim="Test claim about environment.",
    tags=None,
    trust_score=0.8,
    status="active",
):
    return {
        "id": seed_id,
        "core_claim": core_claim,
        "chunks": [{"type": "fact", "content": core_claim}],
        "meristems": [],
        "source_ref": {"session_id": "s1", "profile": "default"},
        "trust_score": trust_score,
        "trust_history": [
            {"delta": None, "reason": "initial", "value": 0.8, "at": _utc_now()}
        ],
        "status": status,
        "superseded_by": [],
        "superseded_at": None,
        "tags": tags or ["env", "test"],
        "created": _utc_now(),
        "updated": _utc_now(),
        "last_validated": None,
    }


# ---------------------------------------------------------------------------
# Vault tests
# ---------------------------------------------------------------------------

class TestSeedVault:
    def test_vault_creates_directories(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault = SeedVault(vault_dir)
        assert vault_dir.exists()
        assert (vault_dir / "seeds").exists()
        assert (vault_dir / "archive").exists()

    def test_manifest_initialized_empty(self, tmp_vault):
        assert tmp_vault._manifest["seeds"] == {}
        # version is an int in v1
        assert tmp_vault._manifest["version"] == 1

    def test_add_and_get_seed(self, tmp_vault):
        seed = _make_seed()
        ok = tmp_vault.write_seed(seed)
        assert ok
        retrieved = tmp_vault.get_seed("env-test-001")
        assert retrieved is not None
        assert retrieved["core_claim"] == "Test claim about environment."

    def test_seed_persisted_to_disk(self, tmp_vault):
        seed = _make_seed()
        tmp_vault.write_seed(seed)
        seed_path = tmp_vault.seeds_dir / "env-test-001.json"
        assert seed_path.exists()
        on_disk = json.loads(seed_path.read_text())
        assert on_disk["id"] == "env-test-001"

    def test_manifest_updated_after_write(self, tmp_vault):
        seed = _make_seed()
        tmp_vault.write_seed(seed)
        assert "env-test-001" in tmp_vault._manifest["seeds"]
        meta = tmp_vault._manifest["seeds"]["env-test-001"]
        assert meta["status"] == "active"
        assert meta["trust_score"] == 0.8

    def test_update_seed_status(self, tmp_vault):
        seed = _make_seed()
        tmp_vault.write_seed(seed)
        ok = tmp_vault.update_seed("env-test-001", {"status": "archived"})
        assert ok
        assert tmp_vault._manifest["seeds"]["env-test-001"]["status"] == "archived"
        on_disk = tmp_vault.get_seed("env-test-001")
        assert on_disk["status"] == "archived"

    def test_update_nonexistent_seed_returns_false(self, tmp_vault):
        ok = tmp_vault.update_seed("nope-999", {"status": "archived"})
        assert not ok

    def test_list_seeds(self, tmp_vault):
        tmp_vault.write_seed(_make_seed("a-test-001", tags=["a"]))
        tmp_vault.write_seed(_make_seed("b-test-002", tags=["b"]))
        seeds = tmp_vault.list_seeds()
        assert len(seeds) == 2

    def test_list_active_seeds_only(self, tmp_vault):
        tmp_vault.write_seed(_make_seed("a-test-001", tags=["a"], status="active"))
        tmp_vault.write_seed(_make_seed("b-test-002", tags=["b"], status="superseded"))
        active = tmp_vault.list_seeds(status="active")
        assert len(active) == 1
        assert active[0]["id"] == "a-test-001"

    def test_search_finds_by_tag(self, tmp_vault):
        tmp_vault.write_seed(_make_seed("env-ollama-001",
            core_claim="OLLAMA_FLASH_ATTENTION must stay OFF.",
            tags=["env", "ollama"]))
        results = tmp_vault.search("ollama", top_k=5)
        assert len(results) == 1
        assert "OLLAMA" in results[0]["core_claim"]

    def test_search_prefix_matching(self, tmp_vault):
        """Query 'experiment' should match claim containing 'experimenting'."""
        tmp_vault.write_seed(_make_seed("pref-test-001",
            core_claim="I prefer to fork repos before experimenting on live installs.",
            tags=["pref", "fork"]))
        results = tmp_vault.search("fork experiment", top_k=5)
        assert len(results) >= 1

    def test_search_excludes_superseded(self, tmp_vault):
        tmp_vault.write_seed(_make_seed("env-old-001",
            core_claim="Old environment fact.",
            tags=["env"], status="superseded"))
        results = tmp_vault.search("environment", top_k=5)
        assert len(results) == 0

    def test_get_active_seed_ids(self, tmp_vault):
        tmp_vault.write_seed(_make_seed("a-test-001", tags=["a"]))
        tmp_vault.write_seed(_make_seed("b-test-002", tags=["b"], status="superseded"))
        ids = tmp_vault.get_active_seed_ids()
        assert "a-test-001" in ids
        assert "b-test-002" not in ids

    def test_validate_meristems_drops_dangling(self, tmp_vault):
        # Use different primary tags so supersession doesn't fire
        tmp_vault.write_seed(_make_seed("env-real-001",
            core_claim="The OLLAMA flash attention setting causes device loss on AMD graphics.",
            tags=["env", "ollama"]))
        seed_with_edge = _make_seed("info-edge-002",
            core_claim="The meristem validation logic drops references to nonexistent seeds.",
            tags=["info", "validation"])
        seed_with_edge["meristems"] = [
            {"type": "related", "target": "env-real-001"},  # exists
            {"type": "related", "target": "env-nope-999"},  # dangling
        ]
        # Use commit gate — it validates meristems during stage1
        gate = CommitGate(tmp_vault)
        ok, _ = gate.commit(seed_with_edge)
        assert ok
        stored = tmp_vault.get_seed("info-edge-002")
        # Only the valid meristem should remain (dangling dropped)
        valid_meristems = [m for m in stored["meristems"] if m["type"] != "supersedes"]
        assert len(valid_meristems) == 1
        assert valid_meristems[0]["target"] == "env-real-001"


# ---------------------------------------------------------------------------
# Supersession tests
# ---------------------------------------------------------------------------

class TestSupersession:
    def test_supersession_sets_old_seed_inactive(self, tmp_vault):
        old = _make_seed("pref-old-001", tags=["pref", "preference"],
                        core_claim="User prefers dark mode.")
        # Use commit gate so supersession logic fires
        gate = CommitGate(tmp_vault)
        gate.commit(old)

        new = _make_seed("pref-new-002", tags=["pref", "preference"],
                        core_claim="User now prefers light mode.")
        gate.commit(new)

        old_stored = tmp_vault.get_seed("pref-old-001")
        assert old_stored["status"] == "superseded"
        assert old_stored["trust_score"] == 0.0
        assert "pref-new-002" in old_stored["superseded_by"]

    def test_supersession_only_matches_primary_tag(self, tmp_vault):
        """Seeds with different primary tags don't supersede each other."""
        seed_a = _make_seed("env-a-001", tags=["env", "pref"],
                           core_claim="Environment fact.")
        tmp_vault.write_seed(seed_a)

        seed_b = _make_seed("pref-b-002", tags=["pref", "env"],
                           core_claim="Preference fact.")
        tmp_vault.write_seed(seed_b)

        # seed_b has primary tag "pref", seed_a has primary tag "env"
        # They should NOT supersede each other
        a_stored = tmp_vault.get_seed("env-a-001")
        assert a_stored["status"] == "active"

    def test_multiple_supersession(self, tmp_vault):
        """Multiple seeds can supersede the same parent."""
        parent = _make_seed("pref-parent-001", tags=["pref", "config"],
                           core_claim="The application uses YAML for all configuration files.")
        gate = CommitGate(tmp_vault)
        gate.commit(parent)

        child1 = _make_seed("pref-child1-002", tags=["pref", "config"],
                           core_claim="The application now uses TOML instead of YAML for configuration.")
        gate.commit(child1)

        child2 = _make_seed("pref-child2-003", tags=["pref", "config"],
                           core_claim="The application switched to JSON format for configuration management.")
        gate.commit(child2)

        parent_stored = tmp_vault.get_seed("pref-parent-001")
        assert parent_stored["status"] == "superseded"
        assert "pref-child1-002" in parent_stored["superseded_by"]
        assert "pref-child2-003" in parent_stored["superseded_by"]


# ---------------------------------------------------------------------------
# Trust score tests
# ---------------------------------------------------------------------------

class TestTrustScore:
    def test_initial_trust_history_has_null_delta(self, tmp_vault):
        seed = _make_seed()
        tmp_vault.write_seed(seed)
        stored = tmp_vault.get_seed("env-test-001")
        first = stored["trust_history"][0]
        assert first["delta"] is None
        assert first["value"] == 0.8
        assert first["reason"] == "initial"

    def test_trust_score_adjustment_appends_history(self, tmp_vault):
        seed = _make_seed(trust_score=0.8)
        tmp_vault.write_seed(seed)
        tmp_vault.adjust_trust("env-test-001", +0.1, "provenance_verified")
        stored = tmp_vault.get_seed("env-test-001")
        assert stored["trust_score"] == pytest.approx(0.9)
        assert len(stored["trust_history"]) == 2
        assert stored["trust_history"][1]["delta"] == 0.1

    def test_trust_score_floor(self, tmp_vault):
        seed = _make_seed(trust_score=0.1)
        tmp_vault.write_seed(seed)
        tmp_vault.adjust_trust("env-test-001", -0.3, "drift")
        stored = tmp_vault.get_seed("env-test-001")
        assert stored["trust_score"] == 0.0

    def test_trust_score_ceiling(self, tmp_vault):
        seed = _make_seed(trust_score=0.9)
        tmp_vault.write_seed(seed)
        tmp_vault.adjust_trust("env-test-001", +0.3, "verified")
        stored = tmp_vault.get_seed("env-test-001")
        assert stored["trust_score"] == 1.0


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------

class TestExtractor:
    def test_heuristic_extracts_preference(self):
        messages = [
            {"role": "user", "content": "I prefer to use Python for all scripting tasks."},
            {"role": "assistant", "content": "Got it, I'll use Python."},
        ]
        seeds = heuristic_extract(messages, "s1", "default")
        assert len(seeds) >= 1
        pref_seeds = [s for s in seeds if s["tags"][0] == "pref"]
        assert len(pref_seeds) >= 1

    def test_heuristic_extracts_error(self):
        # Use a message where "error" or "crash" is followed by a space
        messages = [
            {"role": "user", "content": "The Vulkan driver crash happens on AMD iGPU when flash attention is enabled."},
            {"role": "assistant", "content": "I'll keep that setting OFF."},
        ]
        seeds = heuristic_extract(messages, "s1", "default")
        error_seeds = [s for s in seeds if s["tags"][0] == "error"]
        assert len(error_seeds) >= 1

    def test_heuristic_extracts_decision(self):
        messages = [
            {"role": "assistant", "content": "I decided to use Conduit as the homeserver for Matrix."},
        ]
        seeds = heuristic_extract(messages, "s1", "default")
        decision_seeds = [s for s in seeds if s["tags"][0] == "decision"]
        assert len(decision_seeds) >= 1

    def test_heuristic_core_claim_is_full_sentence(self):
        messages = [
            {"role": "user", "content": "I prefer to fork repos before experimenting rather than working on live installs."},
        ]
        seeds = heuristic_extract(messages, "s1", "default")
        assert len(seeds) >= 1
        claim = seeds[0]["core_claim"]
        # Should be a full sentence, not just a fragment
        assert len(claim) > 20
        assert claim.endswith(".") or claim.endswith("!")

    def test_seed_id_generation(self):
        sid = _make_seed_id("env", "ollama", 1)
        assert sid == "env-ollama-001"

    def test_seed_id_zero_padded(self):
        sid = _make_seed_id("env", "test", 42)
        assert sid == "env-test-042"

    def test_heuristic_sets_provenance(self):
        messages = [
            {"role": "user", "content": "I prefer concise responses from the assistant."},
        ]
        seeds = heuristic_extract(messages, "test-session-42", "default")
        assert len(seeds) >= 1
        assert seeds[0]["source_ref"]["session_id"] == "test-session-42"

    def test_heuristic_empty_messages(self):
        seeds = heuristic_extract([], "s1", "default")
        assert seeds == []


# ---------------------------------------------------------------------------
# Commit gate tests
# ---------------------------------------------------------------------------

class TestCommitGate:
    def test_gate_accepts_valid_seed(self, tmp_vault):
        gate = CommitGate(tmp_vault)
        seed = _make_seed()
        ok, reason = gate.commit(seed)
        assert ok
        assert reason == "committed"

    def test_gate_rejects_empty_core_claim(self, tmp_vault):
        gate = CommitGate(tmp_vault)
        seed = _make_seed(core_claim="")
        ok, reason = gate.commit(seed)
        assert not ok
        assert "empty" in reason.lower()

    def test_gate_rejects_missing_provenance(self, tmp_vault):
        gate = CommitGate(tmp_vault)
        seed = _make_seed()
        seed["source_ref"] = {}
        ok, reason = gate.commit(seed)
        assert not ok
        assert "provenance" in reason.lower()

    def test_gate_rejects_duplicate(self, tmp_vault):
        gate = CommitGate(tmp_vault)
        seed1 = _make_seed("env-dup-001", core_claim="OLLAMA_FLASH_ATTENTION must stay OFF to avoid crashes.")
        tmp_vault.write_seed(seed1)
        seed2 = _make_seed("env-dup-002", core_claim="OLLAMA_FLASH_ATTENTION must stay OFF to avoid crashes.")
        ok, reason = gate.commit(seed2)
        assert not ok
        assert "duplicate" in reason.lower()


# ---------------------------------------------------------------------------
# State digest tests
# ---------------------------------------------------------------------------

class TestStateDigest:
    def test_digest_default(self, tmp_vault):
        mgr = StateDigestManager(tmp_vault)
        digest = mgr.load()
        assert digest["compaction_count"] == 0
        assert digest["active_seed_ids"] == []

    def test_digest_save_and_reload(self, tmp_vault):
        mgr = StateDigestManager(tmp_vault)
        digest = mgr.load()
        digest["compaction_count"] = 5
        digest["active_seed_ids"] = ["env-001", "env-002"]
        mgr.save(digest)

        reloaded = StateDigestManager(tmp_vault).load()
        assert reloaded["compaction_count"] == 5
        assert "env-001" in reloaded["active_seed_ids"]


# ---------------------------------------------------------------------------
# Provider integration tests
# ---------------------------------------------------------------------------

class TestProvider:
    def test_provider_name(self, tmp_provider):
        assert tmp_provider.name == "seedvault"

    def test_provider_system_prompt_block(self, tmp_provider):
        block = tmp_provider.system_prompt_block()
        assert "SeedVault" in block

    def test_provider_on_pre_compress(self, tmp_provider):
        messages = [
            {"role": "user", "content": "I prefer to use Python for all scripting tasks."},
            {"role": "assistant", "content": "Got it, I'll use Python."},
        ]
        result = tmp_provider.on_pre_compress(messages)
        assert "SeedVault" in result

    def test_provider_on_pre_compress_extracts_seeds(self, tmp_provider):
        messages = [
            {"role": "user", "content": "OLLAMA_FLASH_ATTENTION must stay OFF to avoid Vulkan crash on AMD iGPU."},
            {"role": "assistant", "content": "I'll keep that setting OFF."},
        ]
        tmp_provider.on_pre_compress(messages)
        seeds = tmp_provider._vault.list_seeds()
        assert len(seeds) >= 1

    def test_provider_prefetch_returns_string(self, tmp_provider):
        result = tmp_provider.prefetch("Python scripting")
        assert isinstance(result, str)

    def test_provider_handle_tool_call_status(self, tmp_provider):
        result = tmp_provider.handle_tool_call("seedvault_status", {})
        assert "vault" in result

    def test_provider_handle_tool_call_search(self, tmp_provider):
        # Add a seed first
        tmp_provider.on_pre_compress([
            {"role": "user", "content": "I prefer to use Python for scripting."},
            {"role": "assistant", "content": "Noted."},
        ])
        result = tmp_provider.handle_tool_call("seedvault_search", {"query": "Python"})
        assert "results" in result

    def test_provider_on_memory_write_mirrors(self, tmp_provider):
        tmp_provider.on_memory_write(
            "add", "user", "User prefers concise responses.",
            metadata={"session_id": "s1"}
        )
        mirrored = [s for s in tmp_provider._vault.list_seeds()
                    if "mirrored" in s.get("tags", [])]
        assert len(mirrored) >= 1

    def test_provider_on_session_switch(self, tmp_provider):
        tmp_provider.on_pre_compress([
            {"role": "user", "content": "I prefer to use Python for scripting."},
            {"role": "assistant", "content": "Noted."},
        ])
        tmp_provider.on_session_switch("new-session-002",
                                        parent_session_id="test-session-001",
                                        reset=False)
        digest = tmp_provider._digest_mgr.load()
        assert digest["compaction_count"] >= 1
        assert "test-session-001" in digest["session_lineage"]

    def test_provider_get_tool_schemas(self, tmp_provider):
        schemas = tmp_provider.get_tool_schemas()
        assert len(schemas) >= 2
        names = [s["name"] for s in schemas]
        assert "seedvault_search" in names
        assert "seedvault_status" in names