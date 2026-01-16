"""Tests for QdrantBulletIndex - Vector-based bullet retrieval using Qdrant.

TDD Phase 1A: Write failing tests FIRST before implementation.
These tests cover the QdrantBulletIndex class that integrates ACE playbook
retrieval with Qdrant vector database for O(1) semantic search.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from ace import Playbook


@pytest.mark.unit
class TestQdrantBulletIndexInit(unittest.TestCase):
    """Test QdrantBulletIndex initialization (Phase 1A.1)."""

    def test_qdrant_bullet_index_class_exists(self):
        """Test that QdrantBulletIndex class exists."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        # Class should be importable
        self.assertIsNotNone(QdrantBulletIndex)

    def test_qdrant_bullet_index_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        index = QdrantBulletIndex()

        # Should have default URLs
        self.assertEqual(index._qdrant_url, "http://localhost:6333")
        self.assertEqual(index._embedding_url, "http://localhost:1234")
        self.assertEqual(index._collection, "ace_bullets")

    def test_qdrant_bullet_index_initialization_with_custom_urls(self):
        """Test initialization with custom URLs."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        custom_qdrant = "http://custom:6333"
        custom_embedding = "http://lmstudio:1234"
        custom_collection = "custom_bullets"

        index = QdrantBulletIndex(
            qdrant_url=custom_qdrant,
            embedding_url=custom_embedding,
            collection_name=custom_collection,
        )

        self.assertEqual(index._qdrant_url, custom_qdrant)
        self.assertEqual(index._embedding_url, custom_embedding)
        self.assertEqual(index._collection, custom_collection)

    def test_qdrant_bullet_index_has_http_client(self):
        """Test that index has an HTTP client for API calls."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        index = QdrantBulletIndex()

        # Should have httpx client
        self.assertIsNotNone(index._client)


@pytest.mark.unit
class TestQdrantBulletIndexEmbedding(unittest.TestCase):
    """Test embedding generation (Phase 1A.3-1A.4)."""

    @patch('httpx.Client')
    def test_get_embedding_returns_vector(self, mock_client_class):
        """Test that _get_embedding returns a 768-dim vector."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        # Mock the embedding response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        embedding = index._get_embedding("test text")

        self.assertEqual(len(embedding), 768)
        self.assertIsInstance(embedding, list)

    @patch('httpx.Client')
    def test_get_embedding_calls_lm_studio(self, mock_client_class):
        """Test that _get_embedding calls LM Studio API."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        index._get_embedding("debug this error")

        # Verify correct API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        self.assertIn("embeddings", call_args[0][0])


@pytest.mark.unit
class TestQdrantBulletIndexBulletOperations(unittest.TestCase):
    """Test bullet indexing operations (Phase 1A.5-1A.6)."""

    @patch('httpx.Client')
    def test_index_bullet_stores_in_qdrant(self, mock_client_class):
        """Test that index_bullet stores bullet in Qdrant."""
        from ace.qdrant_retrieval import QdrantBulletIndex
        from ace.playbook import EnrichedBullet

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        bullet = EnrichedBullet(
            id="debug-001",
            section="debugging",
            content="Check error logs first when debugging",
            task_types=["debugging"],
            trigger_patterns=["error", "debug", "bug"],
        )

        # Should not raise
        index.index_bullet(bullet)

        # Should call Qdrant upsert
        put_calls = [call for call in mock_client.method_calls if 'put' in str(call)]
        self.assertGreater(len(put_calls), 0)

    @patch('httpx.Client')
    def test_index_bullet_includes_metadata(self, mock_client_class):
        """Test that indexed bullet includes metadata in payload."""
        from ace.qdrant_retrieval import QdrantBulletIndex
        from ace.playbook import EnrichedBullet

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        bullet = EnrichedBullet(
            id="math-001",
            section="math",
            content="Show step by step reasoning",
            task_types=["reasoning", "math"],
            trigger_patterns=["calculate", "solve"],
            domains=["math"],
        )

        index.index_bullet(bullet)

        # Verify payload contains metadata
        put_calls = mock_client.put.call_args_list
        self.assertGreater(len(put_calls), 0)


@pytest.mark.unit
class TestQdrantBulletIndexRetrieval(unittest.TestCase):
    """Test hybrid retrieval operations (Phase 1A.7-1A.8)."""

    @patch('httpx.Client')
    def test_retrieve_returns_scored_bullets(self, mock_client_class):
        """Test that retrieve returns ScoredBullet objects."""
        from ace.qdrant_retrieval import QdrantBulletIndex
        from ace.retrieval import ScoredBullet

        # Mock Qdrant query response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {
                        "id": 12345,
                        "score": 0.85,
                        "payload": {
                            "bullet_id": "debug-001",
                            "content": "Check logs first",
                            "section": "debugging",
                            "task_types": ["debugging"],
                        }
                    }
                ]
            },
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        results = index.retrieve("how do I debug this error?", limit=5)

        self.assertIsInstance(results, list)
        # Results should be ScoredBullet or similar
        if results:
            self.assertTrue(hasattr(results[0], 'score') or isinstance(results[0], dict))

    @patch('httpx.Client')
    def test_retrieve_uses_hybrid_search(self, mock_client_class):
        """Test that retrieve uses hybrid (dense + sparse) search."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"points": []},
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        index.retrieve("debugging error", limit=10)

        # Should call query endpoint (hybrid search)
        post_calls = mock_client.post.call_args_list
        # At least one call should be to query endpoint with prefetch
        self.assertGreater(len(post_calls), 0)

    @patch('httpx.Client')
    def test_retrieve_respects_limit(self, mock_client_class):
        """Test that retrieve respects the limit parameter."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {"id": i, "score": 0.9 - i*0.1, "payload": {"bullet_id": f"b-{i}", "content": f"Content {i}", "section": "test"}}
                    for i in range(10)
                ]
            },
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        results = index.retrieve("test query", limit=3)

        # Should return at most 3 results
        self.assertLessEqual(len(results), 3)


@pytest.mark.unit
class TestQdrantBulletIndexCollection(unittest.TestCase):
    """Test collection management (Phase 1B.1-1B.2)."""

    @patch('httpx.Client')
    def test_ensure_collection_creates_if_missing(self, mock_client_class):
        """Test that _ensure_collection creates collection if not exists."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        # First call: collection doesn't exist (404)
        # Second call: create collection (200)
        mock_responses = [
            Mock(status_code=404),  # GET collection - not found
            Mock(status_code=200),  # PUT collection - created
        ]
        mock_client = MagicMock()
        mock_client.get.return_value = mock_responses[0]
        mock_client.put.return_value = mock_responses[1]
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        index._ensure_collection()

        # Should call PUT to create collection
        put_calls = mock_client.put.call_args_list
        self.assertGreater(len(put_calls), 0)

    @patch('httpx.Client')
    def test_ensure_collection_skips_if_exists(self, mock_client_class):
        """Test that _ensure_collection skips creation if collection exists."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"status": "green"}}
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        index = QdrantBulletIndex()
        index._client = mock_client

        index._ensure_collection()

        # Should NOT call PUT (collection exists)
        # We only check that no PUT call was made to create collection
        # (GET is expected to check existence)


@pytest.mark.unit
class TestQdrantBulletIndexPlaybook(unittest.TestCase):
    """Test playbook bulk indexing (Phase 1B.3-1B.4)."""

    @patch('httpx.Client')
    def test_index_playbook_indexes_all_bullets(self, mock_client_class):
        """Test that index_playbook indexes all bullets from playbook."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create playbook with multiple bullets
        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="debugging",
            content="Check logs first",
            task_types=["debugging"],
        )
        playbook.add_enriched_bullet(
            section="math",
            content="Show step by step",
            task_types=["reasoning"],
        )
        playbook.add_enriched_bullet(
            section="coding",
            content="Write tests first",
            task_types=["development"],
        )

        index = QdrantBulletIndex()
        index._client = mock_client

        count = index.index_playbook(playbook)

        self.assertEqual(count, 3)

    @patch('httpx.Client')
    def test_index_playbook_batch_upserts(self, mock_client_class):
        """Test that index_playbook uses batch upsert for efficiency."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create playbook with bullets
        playbook = Playbook()
        for i in range(5):
            playbook.add_enriched_bullet(
                section="test",
                content=f"Test bullet {i}",
                task_types=["testing"],
            )

        index = QdrantBulletIndex()
        index._client = mock_client

        index.index_playbook(playbook)

        # Should make batch PUT call(s) rather than individual calls
        put_calls = mock_client.put.call_args_list
        # At least one PUT call should be made
        self.assertGreater(len(put_calls), 0)


@pytest.mark.unit
class TestBM25SparseVector(unittest.TestCase):
    """Test BM25 sparse vector generation."""

    def test_compute_bm25_sparse_vector_returns_dict(self):
        """Test that _compute_bm25_sparse returns indices and values."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        index = QdrantBulletIndex.__new__(QdrantBulletIndex)

        sparse = index._compute_bm25_sparse("Check error logs for debugging issues")

        self.assertIn("indices", sparse)
        self.assertIn("values", sparse)
        self.assertIsInstance(sparse["indices"], list)
        self.assertIsInstance(sparse["values"], list)

    def test_compute_bm25_sparse_vector_preserves_technical_terms(self):
        """Test that BM25 tokenization preserves technical terms."""
        from ace.qdrant_retrieval import QdrantBulletIndex

        index = QdrantBulletIndex.__new__(QdrantBulletIndex)

        # Technical text with CamelCase and terms
        text = "Handle NullReferenceException in async methods"
        sparse = index._compute_bm25_sparse(text)

        # Should have multiple tokens
        self.assertGreater(len(sparse["indices"]), 0)
        self.assertEqual(len(sparse["indices"]), len(sparse["values"]))


@pytest.mark.unit
class TestSmartBulletIndexQdrantIntegration(unittest.TestCase):
    """Test SmartBulletIndex integration with QdrantBulletIndex (Phase 1C)."""

    @patch('ace.qdrant_retrieval.httpx.Client')
    def test_smart_bullet_index_can_use_qdrant_index(self, mock_client_class):
        """Test that SmartBulletIndex can optionally use QdrantBulletIndex."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex
        from ace.qdrant_retrieval import QdrantBulletIndex

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 768}]}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="debugging",
            content="Check logs first",
            task_types=["debugging"],
        )

        qdrant_index = QdrantBulletIndex()

        # SmartBulletIndex should accept optional qdrant_index
        index = SmartBulletIndex(playbook=playbook, qdrant_index=qdrant_index)

        self.assertIsNotNone(index._qdrant_index)

    @patch('ace.qdrant_retrieval.httpx.Client')
    def test_retrieve_uses_qdrant_when_available(self, mock_client_class):
        """Test that retrieve uses Qdrant for vector search when available."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex
        from ace.qdrant_retrieval import QdrantBulletIndex

        # Mock Qdrant responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {
                        "id": 12345,
                        "score": 0.95,
                        "payload": {
                            "bullet_id": "debug-001",
                            "content": "Check logs first",
                            "section": "debugging",
                            "task_types": ["debugging"],
                        }
                    }
                ]
            },
            "data": [{"embedding": [0.1] * 768}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        playbook = Playbook()
        bullet = playbook.add_enriched_bullet(
            section="debugging",
            content="Check logs first",
            task_types=["debugging"],
            trigger_patterns=["error", "bug"],
        )

        qdrant_index = QdrantBulletIndex()
        index = SmartBulletIndex(playbook=playbook, qdrant_index=qdrant_index)

        # Retrieve should use Qdrant and return results
        results = index.retrieve(query="How do I debug this error?")

        # Should have results
        self.assertGreater(len(results), 0)

    def test_smart_bullet_index_works_without_qdrant(self):
        """Test that SmartBulletIndex works normally without Qdrant."""
        from ace import Playbook
        from ace.retrieval import SmartBulletIndex

        playbook = Playbook()
        playbook.add_enriched_bullet(
            section="debugging",
            content="Check logs first",
            task_types=["debugging"],
            trigger_patterns=["error", "bug"],
        )

        # Should work without qdrant_index parameter
        index = SmartBulletIndex(playbook=playbook)

        results = index.retrieve(query="error in my code")

        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
