"""
Integration tests for config system with CodeIndexer and CodeRetrieval.

Tests that config file correctly propagates to indexer and retrieval.
"""
import os
import json
import pytest
from pathlib import Path
from ace.config import (
    ACEConfig,
    QdrantConfig,
    get_embedding_provider_config,
    reset_config,
)
from ace.code_indexer import CodeIndexer
from ace.code_retrieval import CodeRetrieval


class TestConfigPropagationToIndexer:
    """Test config propagation to CodeIndexer."""

    def test_indexer_uses_config_file_model(self, tmp_path, monkeypatch):
        """CodeIndexer should use model from config file."""
        # Setup config file with nomic
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        # Clear env to ensure config file is used
        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        # Create indexer (should not require VOYAGE_API_KEY)
        indexer = CodeIndexer(workspace_path=str(tmp_path))

        # Verify nomic settings were applied
        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_nomic()
        assert indexer.embedding_dim == 3584  # nomic dimension
        assert indexer.collection_name == "ace_code_context_nomic"

    def test_indexer_uses_voyage_when_config_set(self, tmp_path, monkeypatch):
        """CodeIndexer should use Voyage when config file specifies."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "voyage"}')

        # Set required env var
        monkeypatch.setenv("VOYAGE_API_KEY", "test_key")

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))

        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_voyage()
        assert indexer.embedding_dim == 1024  # voyage dimension
        assert indexer.collection_name == "ace_code_context_voyage"

    def test_indexer_uses_jina_when_config_set(self, tmp_path, monkeypatch):
        """CodeIndexer should use Jina when config file specifies."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "jina"}')

        # Clear env to ensure config file is used
        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))

        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_local()
        assert indexer.embedding_dim == 768  # jina dimension
        assert indexer.collection_name == "ace_code_context_jina"

    def test_indexer_respects_config_file_over_env(self, tmp_path, monkeypatch):
        """Config file should override env var for indexer."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        # Set conflicting env var
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")
        monkeypatch.setenv("VOYAGE_API_KEY", "test_key")

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))

        # Config file should win
        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_nomic()
        assert indexer.embedding_dim == 3584  # nomic dimension


class TestConfigPropagationToRetrieval:
    """Test config propagation to CodeRetrieval."""

    def test_retrieval_uses_config_file_model(self, tmp_path, monkeypatch):
        """CodeRetrieval should use model from config file."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "jina"}')

        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        # Create retriever (should use local jina)
        retriever = CodeRetrieval()

        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_local()
        assert retriever.collection_name == "ace_code_context_jina"

    def test_retrieval_uses_nomic_when_config_set(self, tmp_path, monkeypatch):
        """CodeRetrieval should use nomic when config file specifies."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        retriever = CodeRetrieval()

        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_nomic()
        assert retriever.collection_name == "ace_code_context_nomic"

    def test_retrieval_uses_voyage_when_config_set(self, tmp_path, monkeypatch):
        """CodeRetrieval should use voyage when config file specifies."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "voyage"}')

        monkeypatch.setenv("VOYAGE_API_KEY", "test_key")

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        retriever = CodeRetrieval()

        provider_config = get_embedding_provider_config()
        assert provider_config.is_code_voyage()
        assert retriever.collection_name == "ace_code_context_voyage"


class TestQdrantCollectionResolution:
    """Test QdrantConfig.get_code_collection_name()."""

    @pytest.mark.parametrize("model,expected", [
        ("voyage", "ace_code_context_voyage"),
        ("jina", "ace_code_context_jina"),
        ("nomic", "ace_code_context_nomic"),
    ])
    def test_returns_correct_collection_for_model(self, model, expected, monkeypatch):
        """Should return correct collection for each model."""
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", model)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = QdrantConfig()
        collection = config.get_code_collection_name()
        assert collection == expected

    def test_config_file_overrides_collection_name(self, tmp_path, monkeypatch):
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


class TestConfigFilePriority:
    """Test config file > env var > default priority."""

    def test_config_file_highest_priority(self, tmp_path, monkeypatch):
        """Config file should have highest priority."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        # Set conflicting env var
        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "voyage")
        monkeypatch.setenv("VOYAGE_API_KEY", "test_key")

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()
        provider_config = get_embedding_provider_config()

        # Config file value should be used
        assert config.code_embedding_model == "nomic"
        assert provider_config.is_code_nomic()

    def test_env_var_second_priority(self, tmp_path, monkeypatch):
        """Env var should be used when no config file."""
        # No config file - only env var
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("ACE_CODE_EMBEDDING_MODEL", "jina")
        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()
        provider_config = get_embedding_provider_config()

        # Env var value should be used
        assert config.code_embedding_model == "jina"
        assert provider_config.is_code_local()

    def test_default_lowest_priority(self, tmp_path, monkeypatch):
        """Default should be used when neither config file nor env var."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)

        # Clear env var
        monkeypatch.delenv("ACE_CODE_EMBEDDING_MODEL", raising=False)

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        config = ACEConfig()

        # Default value should be used
        assert config.code_embedding_model == "voyage"


class TestEmbeddingDimensions:
    """Test correct embedding dimensions for each model."""

    def test_nomic_dimension_3584(self, tmp_path, monkeypatch):
        """Nomic should use 3584 dimensions."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "nomic"}')

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))
        assert indexer.embedding_dim == 3584

    def test_voyage_dimension_1024(self, tmp_path, monkeypatch):
        """Voyage should use 1024 dimensions."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "voyage"}')

        monkeypatch.setenv("VOYAGE_API_KEY", "test_key")
        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))
        assert indexer.embedding_dim == 1024

    def test_jina_dimension_768(self, tmp_path, monkeypatch):
        """Jina should use 768 dimensions."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        (ace_dir / ".ace.json").write_text('{"code_embedding_model": "jina"}')

        monkeypatch.chdir(tmp_path)
        reset_config()

        from ace import config as config_module
        config_module._ace_config_file_cache = {}

        indexer = CodeIndexer(workspace_path=str(tmp_path))
        assert indexer.embedding_dim == 768
