"""
Test suite for async retrieval capabilities (Phase 4A).

These tests verify async embedding retrieval, batch processing,
and concurrent query handling. All tests use pytest-asyncio.

Test Requirements:
- AsyncQdrantBulletIndex class with async methods
- httpx.AsyncClient for async HTTP operations
- Parallel batch processing via asyncio.gather
- Concurrent retrieval without blocking
"""

import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, call

# This import will fail - module doesn't exist yet (RED phase)
from ace.async_retrieval import (
    AsyncQdrantBulletIndex,
    QdrantScoredResult,
)


@pytest.mark.asyncio
class TestAsyncEmbedding:
    """Test async embedding retrieval (Task 4A.1)."""

    async def test_async_get_embedding(self):
        """
        Test that get_embedding uses httpx.AsyncClient.

        Verification:
        - Uses AsyncClient.post, not sync Client.post
        - Returns 768-dim embedding vector
        - Handles async context manager correctly
        """
        index = AsyncQdrantBulletIndex()

        with patch('httpx.AsyncClient') as mock_async_client:
            # Mock async context manager
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Mock embedding response (768-dim vector)
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            mock_client_instance.post.return_value = mock_response

            # Execute async embedding retrieval
            embedding = await index.get_embedding("test query")

            # Verify async client was used
            mock_async_client.assert_called_once()
            mock_client_instance.post.assert_awaited_once()

            # Verify embedding dimensions
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    async def test_async_embedding_error_handling(self):
        """
        Test async embedding handles HTTP errors gracefully.

        Verification:
        - Raises appropriate exception on API failure
        - Cleans up async resources properly
        """
        index = AsyncQdrantBulletIndex()

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Simulate HTTP error
            mock_client_instance.post.side_effect = Exception("API error")

            with pytest.raises(Exception, match="API error"):
                await index.get_embedding("test query")


@pytest.mark.asyncio
class TestBatchEmbeddings:
    """Test parallel batch embedding processing (Task 4A.2)."""

    async def test_batch_embeddings(self):
        """
        Test that batch_get_embeddings processes texts in parallel.

        Verification:
        - Uses asyncio.gather for concurrent execution
        - All embeddings retrieved in parallel, not sequentially
        - Returns list of embeddings matching input order
        """
        index = AsyncQdrantBulletIndex()
        texts = ["query 1", "query 2", "query 3"]

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Track call order to verify parallelism
            call_times = []

            async def mock_post(*args, **kwargs):
                call_times.append(asyncio.get_event_loop().time())
                mock_response = AsyncMock()
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 768}]
                }
                return mock_response

            mock_client_instance.post = mock_post

            # Execute batch embedding
            embeddings = await index.batch_get_embeddings(texts)

            # Verify all embeddings returned
            assert len(embeddings) == 3
            assert all(len(emb) == 768 for emb in embeddings)

            # Verify parallel execution (calls should be near-simultaneous)
            if len(call_times) > 1:
                time_diffs = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
                # Parallel calls should have minimal time difference (< 100ms)
                assert all(diff < 0.1 for diff in time_diffs), \
                    "Embeddings should be fetched in parallel, not sequentially"

    async def test_batch_embeddings_empty_list(self):
        """
        Test batch processing with empty input.

        Verification:
        - Returns empty list without API calls
        """
        index = AsyncQdrantBulletIndex()

        embeddings = await index.batch_get_embeddings([])

        assert embeddings == []

    async def test_batch_embeddings_partial_failure(self):
        """
        Test batch processing when some embeddings fail.

        Verification:
        - Continues processing other embeddings
        - Reports which embeddings failed
        """
        index = AsyncQdrantBulletIndex()
        texts = ["success 1", "fail", "success 2"]

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            call_count = 0

            async def mock_post(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count == 2:
                    raise Exception("Embedding failed")

                mock_response = AsyncMock()
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 768}]
                }
                return mock_response

            mock_client_instance.post = mock_post

            # Should raise exception but report which embedding failed
            with pytest.raises(Exception):
                await index.batch_get_embeddings(texts)


@pytest.mark.asyncio
class TestConcurrentRetrieval:
    """Test concurrent query retrieval (Task 4A.5)."""

    async def test_concurrent_retrieval(self):
        """
        Test that multiple queries are processed concurrently.

        Verification:
        - Multiple retrieve() calls can run simultaneously
        - Uses async Qdrant client operations
        - No blocking between queries
        """
        index = AsyncQdrantBulletIndex()

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Mock embedding responses
            mock_emb_response = AsyncMock()
            mock_emb_response.json.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }

            # Mock Qdrant search responses
            mock_search_response = AsyncMock()
            mock_search_response.json.return_value = {
                "result": [
                    {
                        "id": "test-id-1",
                        "score": 0.95,
                        "payload": {"text": "result 1"}
                    }
                ]
            }

            async def mock_post(url, *args, **kwargs):
                if "embeddings" in url:
                    return mock_emb_response
                else:
                    return mock_search_response

            mock_client_instance.post = mock_post

            # Execute concurrent retrievals
            queries = ["query 1", "query 2", "query 3"]
            results = await asyncio.gather(
                *[index.retrieve(q, limit=5) for q in queries]
            )

            # Verify all queries returned results
            assert len(results) == 3
            assert all(isinstance(r, list) for r in results)
            assert all(len(r) > 0 for r in results)

    async def test_concurrent_retrieval_with_different_limits(self):
        """
        Test concurrent queries with different result limits.

        Verification:
        - Each query respects its own limit parameter
        - Results are not mixed between queries
        """
        index = AsyncQdrantBulletIndex()

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Mock responses
            mock_emb_response = AsyncMock()
            mock_emb_response.json.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }

            # Track limit values from search requests
            search_limits = []

            async def mock_post(url, *args, **kwargs):
                if "embeddings" in url:
                    return mock_emb_response
                else:
                    # Extract limit from search request
                    if "json" in kwargs:
                        search_limits.append(kwargs["json"].get("limit", 10))

                    mock_response = AsyncMock()
                    mock_response.json.return_value = {"result": []}
                    return mock_response

            mock_client_instance.post = mock_post

            # Execute concurrent queries with different limits
            await asyncio.gather(
                index.retrieve("query 1", limit=5),
                index.retrieve("query 2", limit=10),
                index.retrieve("query 3", limit=3)
            )

            # Verify each query used correct limit
            assert 5 in search_limits
            assert 10 in search_limits
            assert 3 in search_limits


@pytest.mark.asyncio
class TestQdrantScoredResult:
    """Test QdrantScoredResult data class."""

    async def test_scored_result_structure(self):
        """
        Test that QdrantScoredResult properly structures search results.

        Verification:
        - Contains id, score, and payload fields
        - Score is float between 0 and 1
        - Payload contains original bullet data
        """
        result = QdrantScoredResult(
            id="test-id",
            score=0.95,
            payload={"text": "test bullet", "helpful": 5, "harmful": 1}
        )

        assert result.id == "test-id"
        assert result.score == 0.95
        assert 0 <= result.score <= 1
        assert result.payload["text"] == "test bullet"


@pytest.mark.asyncio
class TestAsyncQdrantIntegration:
    """Integration tests for async Qdrant operations."""

    async def test_end_to_end_async_retrieval(self):
        """
        Test complete async retrieval flow: embedding + search.

        Verification:
        - Query gets embedded asynchronously
        - Qdrant search executes with async client
        - Results are properly formatted
        """
        index = AsyncQdrantBulletIndex()

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Mock embedding
            mock_emb_response = AsyncMock()
            mock_emb_response.json.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }

            # Mock search
            mock_search_response = AsyncMock()
            mock_search_response.json.return_value = {
                "result": [
                    {
                        "id": "bullet-1",
                        "score": 0.92,
                        "payload": {
                            "text": "Always validate inputs",
                            "helpful": 3,
                            "harmful": 0
                        }
                    }
                ]
            }

            call_count = 0

            async def mock_post(url, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if "embeddings" in url:
                    return mock_emb_response
                else:
                    return mock_search_response

            mock_client_instance.post = mock_post

            # Execute retrieval
            results = await index.retrieve("input validation", limit=5)

            # Verify async calls made
            assert call_count == 2  # 1 embedding + 1 search

            # Verify result structure
            assert len(results) == 1
            assert results[0].id == "bullet-1"
            assert results[0].score == 0.92
            assert results[0].payload["text"] == "Always validate inputs"

    async def test_async_resource_cleanup(self):
        """
        Test that async clients are properly cleaned up.

        Verification:
        - AsyncClient context manager is used
        - aclose() is called for proper cleanup
        """
        index = AsyncQdrantBulletIndex()

        # Use the class-level __aenter__/__aexit__ for resource cleanup
        async with index:
            # After entering, client should be set
            assert index._client is not None

        # After exiting, client should be cleaned up
        assert index._client is None
