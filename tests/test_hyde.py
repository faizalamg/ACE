"""Test suite for HyDE (Hypothetical Document Embeddings) implementation."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json

from ace.hyde import HyDEGenerator, HyDEConfig
from ace.hyde_retrieval import HyDEEnhancedRetriever


class TestHyDEGenerator(unittest.TestCase):
    """Test cases for HyDE hypothetical document generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock()
        self.config = HyDEConfig(
            num_hypotheticals=3,
            max_tokens=100,
            temperature=0.7,
            cache_enabled=True
        )
        self.hyde_generator = HyDEGenerator(
            llm_client=self.mock_llm_client,
            config=self.config
        )

    def test_generate_single_hypothetical(self):
        """Test generation of a single hypothetical document."""
        # Arrange
        query = "How to fix memory leak in Python?"
        expected_hypothetical = "Memory leaks in Python are often caused by circular references..."

        mock_response = Mock()
        mock_response.text = expected_hypothetical
        self.mock_llm_client.complete.return_value = mock_response

        # Act
        result = self.hyde_generator._generate_single_hypothetical(query)

        # Assert
        self.assertEqual(result, expected_hypothetical)
        self.mock_llm_client.complete.assert_called_once()

    def test_generate_multiple_hypotheticals(self):
        """Test generation of multiple hypothetical documents."""
        # Arrange
        query = "How to debug authentication errors?"
        hypotheticals = [
            "Authentication errors typically occur when credentials are invalid...",
            "To debug authentication, first check the token expiration...",
            "Common authentication issues include misconfigured API keys..."
        ]

        mock_response = Mock()
        self.mock_llm_client.complete.side_effect = [
            Mock(text=h) for h in hypotheticals
        ]

        # Act
        results = self.hyde_generator.generate_hypotheticals(query, num_docs=3)

        # Assert
        self.assertEqual(len(results), 3)
        self.assertEqual(results, hypotheticals)
        self.assertEqual(self.mock_llm_client.complete.call_count, 3)

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        # Arrange
        query = "How to optimize database queries?"
        hypothetical = "Database query optimization involves indexing..."

        mock_response = Mock(text=hypothetical)
        self.mock_llm_client.complete.return_value = mock_response

        # Act - First call should hit LLM
        result1 = self.hyde_generator.generate_hypotheticals(query, num_docs=1)
        # Second call should use cache
        result2 = self.hyde_generator.generate_hypotheticals(query, num_docs=1)

        # Assert
        self.assertEqual(result1, result2)
        self.assertEqual(self.mock_llm_client.complete.call_count, 1)

    def test_cache_disabled(self):
        """Test behavior when cache is disabled."""
        # Arrange
        config = HyDEConfig(num_hypotheticals=1, cache_enabled=False)
        generator = HyDEGenerator(self.mock_llm_client, config)
        query = "How to fix security vulnerabilities?"

        mock_response = Mock(text="Security fix...")
        self.mock_llm_client.complete.return_value = mock_response

        # Act
        generator.generate_hypotheticals(query, num_docs=1)
        generator.generate_hypotheticals(query, num_docs=1)

        # Assert - Should call LLM twice (no caching)
        self.assertEqual(self.mock_llm_client.complete.call_count, 2)

    def test_async_generation(self):
        """Test async generation of hypotheticals."""
        # Arrange
        query = "How to implement rate limiting?"
        hypotheticals = ["Rate limiting can be...", "Implement rate limiting using..."]

        # Create async mock
        mock_async_llm = MagicMock()
        mock_async_llm.acomplete.return_value.__aiter__ = Mock(
            return_value=iter([Mock(text=h) for h in hypotheticals])
        )

        # This test verifies the async interface exists
        # Full async testing requires pytest-asyncio
        self.assertTrue(hasattr(self.hyde_generator, 'agenerate_hypotheticals'))


class TestHyDEEnhancedRetriever(unittest.TestCase):
    """Test cases for HyDE-enhanced retrieval pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_hyde_generator = Mock()
        self.mock_embedding_client = Mock()
        self.mock_qdrant_client = Mock()

        self.retriever = HyDEEnhancedRetriever(
            hyde_generator=self.mock_hyde_generator,
            embedding_client=self.mock_embedding_client,
            qdrant_url="http://localhost:6333",
            collection_name="test_collection"
        )

    @patch('ace.hyde_retrieval.httpx.Client')
    def test_retrieve_with_hyde_enabled(self, mock_httpx):
        """Test retrieval with HyDE expansion enabled."""
        # Arrange
        query = "memory leak issue"
        hypotheticals = [
            "Memory leaks occur when objects are not properly released...",
            "To fix memory leaks, use profiling tools like memory_profiler...",
            "Common causes of memory leaks include circular references..."
        ]

        # Mock hypothetical generation
        self.mock_hyde_generator.generate_hypotheticals.return_value = hypotheticals

        # Mock embedding generation for each hypothetical
        mock_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        self.mock_embedding_client.get_embedding.side_effect = mock_embeddings

        # Mock Qdrant search results
        mock_search_response = {
            "result": [
                {
                    "id": "bullet_1",
                    "score": 0.92,
                    "payload": {
                        "bullet_id": "bullet_1",
                        "content": "Use memory profiling to detect leaks",
                        "section": "debugging",
                        "task_types": ["debugging"],
                        "trigger_patterns": ["memory", "leak"]
                    }
                }
            ]
        }

        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_search_response
        mock_httpx.return_value.__enter__.return_value.post.return_value = mock_http_response

        # Act
        results = self.retriever.retrieve(query, use_hyde=True, limit=5)

        # Assert
        self.mock_hyde_generator.generate_hypotheticals.assert_called_once_with(query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Use memory profiling to detect leaks")

    def test_retrieve_without_hyde(self):
        """Test retrieval without HyDE (baseline behavior)."""
        # Arrange
        query = "memory leak issue"

        # Mock direct embedding
        mock_embedding = [0.1] * 768
        self.mock_embedding_client.get_embedding.return_value = mock_embedding

        # Act
        with patch('ace.hyde_retrieval.httpx.Client') as mock_httpx:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": []}
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            results = self.retriever.retrieve(query, use_hyde=False, limit=5)

        # Assert - Should NOT call HyDE generator
        self.mock_hyde_generator.generate_hypotheticals.assert_not_called()

    def test_query_classification(self):
        """Test automatic HyDE enablement for ambiguous queries."""
        # Arrange
        short_query = "fix bug"  # Ambiguous, should trigger HyDE
        specific_query = "ImportError: cannot import name 'Playbook' from ace.playbook"  # Specific

        # Act
        should_use_hyde_short = self.retriever._should_use_hyde(short_query)
        should_use_hyde_specific = self.retriever._should_use_hyde(specific_query)

        # Assert
        self.assertTrue(should_use_hyde_short)  # Short/ambiguous -> use HyDE
        self.assertFalse(should_use_hyde_specific)  # Specific/long -> skip HyDE

    def test_embedding_averaging(self):
        """Test that hypothetical embeddings are averaged correctly."""
        # Arrange
        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        # Act
        avg_embedding = self.retriever._average_embeddings(embeddings)

        # Assert
        expected = [4.0, 5.0, 6.0]  # Mean of each dimension
        self.assertEqual(avg_embedding, expected)


class TestHyDEConfig(unittest.TestCase):
    """Test configuration validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HyDEConfig()

        self.assertEqual(config.num_hypotheticals, 3)
        self.assertGreater(config.temperature, 0)
        self.assertTrue(config.cache_enabled)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HyDEConfig(
            num_hypotheticals=5,
            max_tokens=200,
            temperature=0.9,
            cache_enabled=False,
            api_key="test-key"
        )

        self.assertEqual(config.num_hypotheticals, 5)
        self.assertEqual(config.max_tokens, 200)
        self.assertEqual(config.temperature, 0.9)
        self.assertFalse(config.cache_enabled)
        self.assertEqual(config.api_key, "test-key")


if __name__ == '__main__':
    unittest.main()
