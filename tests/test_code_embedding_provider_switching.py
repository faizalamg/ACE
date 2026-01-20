"""
Tests for code embedding provider switching.

TDD tests for the ability to switch between Voyage API (cloud) 
and LM Studio (local) code embeddings.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestCodeEmbeddingProviderSwitching:
    """Tests for switching between Voyage and local code embeddings."""
    
    def test_default_provider_is_voyage(self):
        """Default provider should be Voyage API."""
        from ace.config import get_embedding_provider_config
        
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            if 'ACE_CODE_EMBEDDING_PROVIDER' in os.environ:
                del os.environ['ACE_CODE_EMBEDDING_PROVIDER']
            
            # Reimport to get fresh config
            import importlib
            import ace.config
            importlib.reload(ace.config)
            
            config = ace.config.get_embedding_provider_config()
            # Note: Default is 'voyage' per config.py line 114
            assert config.is_code_voyage() or config.code_provider == 'voyage'
    
    def test_provider_config_respects_env_var(self):
        """Provider config should read from ACE_CODE_EMBEDDING_PROVIDER."""
        from ace.config import EmbeddingProviderConfig
        
        # Test local
        with patch.dict(os.environ, {'ACE_CODE_EMBEDDING_PROVIDER': 'local'}):
            import importlib
            import ace.config
            importlib.reload(ace.config)
            config = ace.config.EmbeddingProviderConfig()
            assert config.is_code_local()
            assert not config.is_code_voyage()
        
        # Test voyage
        with patch.dict(os.environ, {'ACE_CODE_EMBEDDING_PROVIDER': 'voyage'}):
            import importlib
            import ace.config
            importlib.reload(ace.config)
            config = ace.config.EmbeddingProviderConfig()
            assert config.is_code_voyage()
            assert not config.is_code_local()
    
    @patch.dict(os.environ, {'ACE_CODE_EMBEDDING_PROVIDER': 'voyage', 'VOYAGE_API_KEY': 'test-key'})
    @patch('voyageai.Client')
    def test_code_retrieval_uses_voyage_when_configured(self, mock_voyage_client):
        """CodeRetrieval should use Voyage when provider is voyage."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_voyage_client.return_value = mock_client_instance
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [[0.1] * 1024]
        mock_client_instance.embed.return_value = mock_embed_result
        
        from ace.code_retrieval import CodeRetrieval
        
        # Create retrieval with mocked Qdrant
        with patch('qdrant_client.QdrantClient'):
            retrieval = CodeRetrieval()
            embedder = retrieval._get_embedder()
            
            # Call embedder
            result = embedder("def hello(): pass")
            
            # Should have called Voyage
            mock_client_instance.embed.assert_called_once()
            assert len(result) == 1024  # Voyage dimension
    
    @patch.dict(os.environ, {'ACE_CODE_EMBEDDING_PROVIDER': 'local', 'ACE_LOCAL_EMBEDDING_URL': 'http://localhost:1234'})
    @patch('httpx.Client')
    def test_code_retrieval_uses_local_when_configured(self, mock_httpx_client):
        """CodeRetrieval should use LM Studio when provider is local."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_httpx_client.return_value = mock_client_instance
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response
        
        from ace.code_retrieval import CodeRetrieval
        
        # Reload to pick up env var
        import importlib
        import ace.config
        importlib.reload(ace.config)
        
        # Create retrieval with mocked Qdrant
        with patch('qdrant_client.QdrantClient'):
            retrieval = CodeRetrieval()
            embedder = retrieval._get_embedder()
            
            # Call embedder
            result = embedder("def hello(): pass")
            
            # Should have called LM Studio HTTP endpoint
            mock_client_instance.post.assert_called_once()
            assert '/v1/embeddings' in str(mock_client_instance.post.call_args)
            assert len(result) == 768  # Jina dimension
    
    def test_local_embedding_config_values(self):
        """LocalEmbeddingConfig should have correct defaults for code."""
        from ace.config import LocalEmbeddingConfig
        
        config = LocalEmbeddingConfig()
        
        # Check default code model is Jina
        assert 'jina' in config.code_model.lower() or 'code' in config.code_model.lower()
        # Check dimension is 768 (Jina v2 base code)
        assert config.code_dimension == 768
        # Check URL default
        assert 'localhost' in config.url or '127.0.0.1' in config.url


class TestProviderDimensionCompatibility:
    """Tests for embedding dimension handling between providers."""
    
    def test_voyage_dimension_is_1024(self):
        """Voyage code embeddings should be 1024d."""
        from ace.config import VoyageCodeEmbeddingConfig
        
        config = VoyageCodeEmbeddingConfig()
        assert config.dimension == 1024
    
    def test_local_code_dimension_is_768(self):
        """Local Jina code embeddings should be 768d."""
        from ace.config import LocalEmbeddingConfig
        
        config = LocalEmbeddingConfig()
        assert config.code_dimension == 768
    
    def test_different_dimensions_require_separate_collections(self):
        """Different dimensions mean separate Qdrant collections needed."""
        # This is a documentation/awareness test
        # Voyage (1024d) and Jina (768d) cannot share the same collection
        from ace.config import VoyageCodeEmbeddingConfig, LocalEmbeddingConfig
        
        voyage = VoyageCodeEmbeddingConfig()
        local = LocalEmbeddingConfig()
        
        assert voyage.dimension != local.code_dimension, \
            "Voyage and Jina have different dimensions - need separate collections"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
