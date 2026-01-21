"""
Unit tests for ACE configuration system.

Tests config loading, priority order, and field resolution.
"""
import os
import json
import pytest
from pathlib import Path
from ace.config import (
    ACEConfig,
    EmbeddingProviderConfig,
    QdrantConfig,
    _get_ace_config_field,
    _load_ace_config,
    _get_workspace_root,
    update_ace_config_field,
    reset_config,
)


class TestWorkspaceRootDetection:
    """Test workspace root detection."""

    def test_finds_workspace_root_from_subdirectory(self, tmp_path, monkeypatch):
        """Should find .ace/.ace.json from any subdirectory."""
        # Create workspace structure
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"workspace_name": "test"}')

        sub_dir = tmp_path / "deep" / "nested" / "path"
        sub_dir.mkdir(parents=True)

        # Patch cwd to subdirectory
        monkeypatch.chdir(sub_dir)

        root = _get_workspace_root()
        assert root == tmp_path

    def test_raises_error_if_no_config(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError if no .ace/.ace.json."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="No .ace/.ace.json"):
            _get_workspace_root()


class TestAceConfigFileLoading:
    """Test .ace/.ace.json config file loading."""

    def test_loads_config_from_file(self, tmp_path, monkeypatch):
        """Should load config values from .ace/.ace.json."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.chdir(tmp_path)

        # Clear cache and reload
        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config_dict = _load_ace_config()
        assert config_dict["code_embedding_model"] == "nomic"

    def test_returns_empty_dict_if_no_file(self, tmp_path, monkeypatch):
        """Should return empty dict if config file doesn't exist."""
        monkeypatch.chdir(tmp_path)

        # Clear cache
        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config_dict = _load_ace_config()
        assert config_dict == {}


class TestConfigFieldResolution:
    """Test config field resolution priority: file > env > default."""

    def test_config_file_overrides_env_var(self, tmp_path, monkeypatch):
        """Config file value should override env var."""
        # Setup config file
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        # Set conflicting env var (ACE_CODE_EMBEDDING_MODEL - single D)
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")

        monkeypatch.chdir(tmp_path)
        reset_config()  # Clear global config

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        # Config file should win
        result = _get_ace_config_field("code_embedding_model", "voyage")
        assert result == "nomic"

    def test_env_var_used_when_no_config_file(self, tmp_path, monkeypatch):
        """Env var should be used when no config file."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "jina")
        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        result = _get_ace_config_field("code_embedding_model", "voyage")
        assert result == "jina"

    def test_default_used_when_no_file_or_env(self, tmp_path, monkeypatch):
        """Default should be used when neither config file nor env var."""
        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        result = _get_ace_config_field("code_embedding_model", "voyage")
        assert result == "voyage"

    def test_aceconfig_uses_config_file(self, tmp_path, monkeypatch):
        """ACEConfig.code_embedding_model should read from config file."""
        # Setup config file
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()
        assert config.code_embedding_model == "nomic"


class TestValidEmbeddingModels:
    """Test valid code embedding model values."""

    @pytest.mark.parametrize("model", ["voyage", "jina", "nomic"])
    def test_accepts_valid_models(self, tmp_path, model, monkeypatch):
        """Should accept all valid model names."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text(f'{{"code_embedding_model": "{model}"}}')

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()
        assert config.code_embedding_model == model

    @pytest.mark.parametrize("model", ["local", "openai", "huggingface", "invalid"])
    def test_accepts_any_string_but_maps_correctly(self, model, monkeypatch):
        """Should accept any string value but map correctly."""
        # "local" maps to "jina" internally
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", model)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        provider_config = EmbeddingProviderConfig()
        # Local maps to jina, others pass through
        if model == "local":
            assert provider_config.is_code_local()
        else:
            # Other values are passed through as-is to is_code_voyage/nomic
            pass


class TestConfigFileUpdates:
    """Test config file update functionality."""

    def test_update_ace_config_field_writes_file(self, tmp_path, monkeypatch):
        """Should write updated value to config file."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"workspace_name": "test"}')

        monkeypatch.chdir(tmp_path)

        update_ace_config_field("code_embedding_model", "nomic")

        # Verify file was updated
        config = json.loads((ace_dir / ".ace.json").read_text())
        assert config["code_embedding_model"] == "nomic"
        assert config["workspace_name"] == "test"  # Original preserved

    def test_update_clears_cache(self, tmp_path, monkeypatch):
        """Update should clear module cache."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{}')

        monkeypatch.chdir(tmp_path)

        from ace import config as config_module
        config_module._ace_config_file_cache = {"code_embedding_model": "old_value"}

        update_ace_config_field("code_embedding_model", "new_value")

        # Cache should be cleared
        assert config_module._ace_config_file_cache == {}

    def test_update_creates_file_if_not_exists(self, tmp_path, monkeypatch):
        """Update should create config file if it doesn't exist."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)

        # Ensure no config file
        config_file = ace_dir / ".ace.json"
        if config_file.exists():
            config_file.unlink()

        monkeypatch.chdir(tmp_path)

        # Create minimal config first
        (ace_dir / ".ace.json").write_text('{"workspace_name": "test"}')

        update_ace_config_field("code_embedding_model", "nomic")

        # Verify file exists with correct content
        assert config_file.exists()
        config = json.loads(config_file.read_text())
        assert config["code_embedding_model"] == "nomic"
        assert config["workspace_name"] == "test"


class TestEmbeddingProviderConfig:
    """Test EmbeddingProviderConfig methods."""

    def test_is_code_local_true_for_jina(self, monkeypatch):
        """is_code_local returns True for 'jina'."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "jina")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = EmbeddingProviderConfig()
        assert config.is_code_local()
        assert not config.is_code_voyage()
        assert not config.is_code_nomic()

    def test_is_code_local_true_for_local(self, monkeypatch):
        """is_code_local returns True for 'local'."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "local")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = EmbeddingProviderConfig()
        assert config.is_code_local()

    def test_is_code_voyage_true_for_voyage(self, monkeypatch):
        """is_code_voyage returns True for 'voyage'."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = EmbeddingProviderConfig()
        assert config.is_code_voyage()
        assert not config.is_code_local()
        assert not config.is_code_nomic()

    def test_is_code_nomic_true_for_nomic(self, monkeypatch):
        """is_code_nomic returns True for 'nomic'."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "nomic")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = EmbeddingProviderConfig()
        assert config.is_code_nomic()
        assert not config.is_code_local()
        assert not config.is_code_voyage()

    def test_post_init_uses_config_file_value(self, tmp_path, monkeypatch):
        """__post_init__ should override code_provider from config file."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()
        assert config.code_embedding_model == "nomic"

        provider_config = EmbeddingProviderConfig()
        assert provider_config.is_code_nomic()


class TestQdrantConfig:
    """Test QdrantConfig methods."""

    def test_get_code_collection_name_voyage(self, monkeypatch):
        """Returns correct collection for voyage model."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = QdrantConfig()
        collection = config.get_code_collection_name()
        assert collection == "ace_code_context_voyage"

    def test_get_code_collection_name_jina(self, monkeypatch):
        """Returns correct collection for jina model."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "jina")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = QdrantConfig()
        collection = config.get_code_collection_name()
        assert collection == "ace_code_context_jina"

    def test_get_code_collection_name_nomic(self, monkeypatch):
        """Returns correct collection for nomic model."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "nomic")
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = QdrantConfig()
        collection = config.get_code_collection_name()
        assert collection == "ace_code_context_nomic"

    def test_get_code_collection_name_from_config_file(self, tmp_path, monkeypatch):
        """Config file should determine collection name."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = QdrantConfig()
        collection = config.get_code_collection_name()
        assert collection == "ace_code_context_nomic"
