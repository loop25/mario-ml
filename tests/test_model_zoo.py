"""Tests for the Model Zoo export/import system."""

import json
import os
import tempfile
import zipfile

import pytest

from src.model_zoo.agent_package import export_agent, import_agent, list_exported_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_dir(tmp_path, meta_overrides=None, add_checkpoint=True, add_profile=False):
    """Create a minimal model directory with metadata + optional files."""
    model_dir = os.path.join(str(tmp_path), "models", "ppo")
    os.makedirs(model_dir, exist_ok=True)

    meta = {
        "algorithm": "ppo",
        "game_id": "snake",
        "episode": 500,
        "best_reward": 42.0,
    }
    if meta_overrides:
        meta.update(meta_overrides)

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    if add_checkpoint:
        # Create a dummy checkpoint (small zip)
        ckpt_path = os.path.join(model_dir, "best_model.zip")
        with zipfile.ZipFile(ckpt_path, "w") as zf:
            zf.writestr("weights.bin", b"fake-weights" * 10)

    if add_profile:
        with open(os.path.join(model_dir, "profile.json"), "w") as f:
            json.dump({"personality": "aggressive"}, f)

    return model_dir, meta


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_agent_file(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        out = os.path.join(str(tmp_path), "out.agent")
        result = export_agent(model_dir, output_path=out)
        assert result == out
        assert os.path.isfile(out)

    def test_agent_file_is_valid_zip(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        out = os.path.join(str(tmp_path), "out.agent")
        export_agent(model_dir, output_path=out)
        assert zipfile.is_zipfile(out)

    def test_agent_contains_metadata(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        out = os.path.join(str(tmp_path), "out.agent")
        export_agent(model_dir, output_path=out)
        with zipfile.ZipFile(out, "r") as zf:
            assert "agent_package/metadata.json" in zf.namelist()
            meta = json.loads(zf.read("agent_package/metadata.json"))
            assert meta["game_id"] == "snake"

    def test_agent_contains_checkpoint(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        out = os.path.join(str(tmp_path), "out.agent")
        export_agent(model_dir, output_path=out)
        with zipfile.ZipFile(out, "r") as zf:
            assert "agent_package/checkpoints/best_model.zip" in zf.namelist()

    def test_agent_contains_profile_when_present(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path, add_profile=True)
        out = os.path.join(str(tmp_path), "out.agent")
        export_agent(model_dir, output_path=out)
        with zipfile.ZipFile(out, "r") as zf:
            assert "agent_package/profile.json" in zf.namelist()

    def test_export_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export_agent(os.path.join(str(tmp_path), "nonexistent"))

    def test_export_missing_metadata_raises(self, tmp_path):
        empty_dir = os.path.join(str(tmp_path), "empty")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError):
            export_agent(empty_dir)

    def test_export_auto_generates_path(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = export_agent(model_dir)
            assert result.endswith(".agent")
            assert os.path.isfile(result)
        finally:
            os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_extracts(self, tmp_path):
        model_dir, meta = _make_model_dir(tmp_path)
        agent_file = os.path.join(str(tmp_path), "test.agent")
        export_agent(model_dir, output_path=agent_file)

        dest_models = os.path.join(str(tmp_path), "imported_models")
        dest = import_agent(agent_file, models_dir=dest_models)

        assert os.path.isdir(dest)
        assert os.path.isfile(os.path.join(dest, "metadata.json"))
        assert os.path.isfile(os.path.join(dest, "best_model.zip"))

    def test_import_metadata_matches(self, tmp_path):
        model_dir, orig_meta = _make_model_dir(tmp_path)
        agent_file = os.path.join(str(tmp_path), "test.agent")
        export_agent(model_dir, output_path=agent_file)

        dest_models = os.path.join(str(tmp_path), "imported_models")
        dest = import_agent(agent_file, models_dir=dest_models)

        with open(os.path.join(dest, "metadata.json")) as f:
            imported_meta = json.load(f)
        assert imported_meta["game_id"] == orig_meta["game_id"]
        assert imported_meta["algorithm"] == orig_meta["algorithm"]

    def test_import_invalid_file_raises(self, tmp_path):
        bad_file = os.path.join(str(tmp_path), "bad.agent")
        with open(bad_file, "w") as f:
            f.write("not a zip")
        with pytest.raises(Exception):
            import_agent(bad_file)

    def test_import_missing_metadata_raises(self, tmp_path):
        # Create a valid zip but without agent_package/metadata.json
        bad_agent = os.path.join(str(tmp_path), "nometadata.agent")
        with zipfile.ZipFile(bad_agent, "w") as zf:
            zf.writestr("agent_package/some_file.txt", "hello")
        with pytest.raises(ValueError, match="missing metadata"):
            import_agent(bad_agent)

    def test_import_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_agent(os.path.join(str(tmp_path), "ghost.agent"))


# ---------------------------------------------------------------------------
# list_exported_agents tests
# ---------------------------------------------------------------------------

class TestListExported:
    def test_list_exported_agents(self, tmp_path):
        model_dir, _ = _make_model_dir(tmp_path)
        export_dir = os.path.join(str(tmp_path), "exports")
        os.makedirs(export_dir, exist_ok=True)
        agent_file = os.path.join(export_dir, "test.agent")
        export_agent(model_dir, output_path=agent_file)

        agents = list_exported_agents(export_dir)
        assert len(agents) == 1
        assert agents[0]["name"] == "test.agent"
        assert agents[0]["game_id"] == "snake"
        assert agents[0]["algorithm"] == "ppo"
        assert agents[0]["file_size"] > 0
        assert "path" in agents[0]

    def test_list_empty_dir(self, tmp_path):
        empty = os.path.join(str(tmp_path), "empty_exports")
        os.makedirs(empty)
        assert list_exported_agents(empty) == []

    def test_list_nonexistent_dir(self):
        assert list_exported_agents("/tmp/does_not_exist_model_zoo_test") == []


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip(self, tmp_path):
        model_dir, orig_meta = _make_model_dir(tmp_path, add_profile=True)
        agent_file = os.path.join(str(tmp_path), "roundtrip.agent")
        export_agent(model_dir, output_path=agent_file)

        dest_models = os.path.join(str(tmp_path), "reimported")
        dest = import_agent(agent_file, models_dir=dest_models)

        # Metadata round-trips
        with open(os.path.join(dest, "metadata.json")) as f:
            rt_meta = json.load(f)
        assert rt_meta == orig_meta

        # Profile round-trips
        with open(os.path.join(dest, "profile.json")) as f:
            rt_profile = json.load(f)
        assert rt_profile == {"personality": "aggressive"}

        # Checkpoint round-trips
        assert os.path.isfile(os.path.join(dest, "best_model.zip"))
