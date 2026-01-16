"""Tests for async parallel processing in adaptation loops.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import asyncio
import unittest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestAsyncAdaptationBasic(unittest.TestCase):
    """Test basic async adaptation functionality."""

    def test_async_offline_adapter_exists(self):
        """Test that AsyncOfflineAdapter class exists."""
        from ace.async_adaptation import AsyncOfflineAdapter

        # Just check import works
        self.assertIsNotNone(AsyncOfflineAdapter)

    def test_async_offline_adapter_has_async_run(self):
        """Test that AsyncOfflineAdapter has async run method."""
        from ace.async_adaptation import AsyncOfflineAdapter

        adapter = AsyncOfflineAdapter.__new__(AsyncOfflineAdapter)
        self.assertTrue(asyncio.iscoroutinefunction(adapter.run))


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncOfflineAdapter(unittest.IsolatedAsyncioTestCase):
    """Test AsyncOfflineAdapter parallel processing."""

    async def test_parallel_sample_processing(self):
        """Test that samples can be processed in parallel."""
        from ace import Playbook, Sample
        from ace.async_adaptation import AsyncOfflineAdapter

        # Create mock components
        playbook = Playbook()
        generator = MagicMock()
        reflector = MagicMock()
        curator = MagicMock()
        environment = MagicMock()

        # Setup async mocks for LLM calls
        generator.generate = AsyncMock(return_value=MagicMock(answer="test"))
        reflector.reflect = AsyncMock(return_value=MagicMock(classifications=[]))
        curator.curate = AsyncMock(return_value=MagicMock(operations=[]))
        environment.evaluate = MagicMock(return_value=MagicMock(feedback="ok", correct=True))

        adapter = AsyncOfflineAdapter(playbook, generator, reflector, curator)

        samples = [Sample(question=f"Q{i}") for i in range(5)]

        # Run with parallelism
        results = await adapter.run(samples, environment, epochs=1, max_parallel=3)

        # All samples should be processed
        self.assertEqual(len(results), 5)

    async def test_max_parallel_limits_concurrency(self):
        """Test that max_parallel limits concurrent operations."""
        from ace import Playbook, Sample
        from ace.async_adaptation import AsyncOfflineAdapter

        playbook = Playbook()
        generator = MagicMock()
        reflector = MagicMock()
        curator = MagicMock()
        environment = MagicMock()

        # Track concurrent operations
        concurrent_count = 0
        max_concurrent = 0

        async def mock_generate(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return MagicMock(answer="test")

        generator.generate = mock_generate
        reflector.reflect = AsyncMock(return_value=MagicMock(classifications=[]))
        curator.curate = AsyncMock(return_value=MagicMock(operations=[]))
        environment.evaluate = MagicMock(return_value=MagicMock(feedback="ok", correct=True))

        adapter = AsyncOfflineAdapter(playbook, generator, reflector, curator)

        samples = [Sample(question=f"Q{i}") for i in range(10)]

        await adapter.run(samples, environment, epochs=1, max_parallel=2)

        # Max concurrent should not exceed max_parallel
        self.assertLessEqual(max_concurrent, 2)

    async def test_batch_processing(self):
        """Test batch processing of samples."""
        from ace import Playbook, Sample
        from ace.async_adaptation import AsyncOfflineAdapter

        playbook = Playbook()
        generator = MagicMock()
        reflector = MagicMock()
        curator = MagicMock()
        environment = MagicMock()

        generator.generate = AsyncMock(return_value=MagicMock(answer="test"))
        reflector.reflect = AsyncMock(return_value=MagicMock(classifications=[]))
        curator.curate = AsyncMock(return_value=MagicMock(operations=[]))
        environment.evaluate = MagicMock(return_value=MagicMock(feedback="ok", correct=True))

        adapter = AsyncOfflineAdapter(playbook, generator, reflector, curator)

        samples = [Sample(question=f"Q{i}") for i in range(20)]

        # Process in batches of 5
        results = await adapter.run(samples, environment, epochs=1, batch_size=5)

        self.assertEqual(len(results), 20)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncOnlineAdapter(unittest.IsolatedAsyncioTestCase):
    """Test AsyncOnlineAdapter for streaming scenarios."""

    async def test_async_online_adapter_exists(self):
        """Test that AsyncOnlineAdapter class exists."""
        from ace.async_adaptation import AsyncOnlineAdapter

        self.assertIsNotNone(AsyncOnlineAdapter)

    async def test_online_step_async(self):
        """Test single step processing is async."""
        from ace import Playbook, Sample
        from ace.async_adaptation import AsyncOnlineAdapter

        playbook = Playbook()
        generator = MagicMock()
        reflector = MagicMock()
        curator = MagicMock()
        environment = MagicMock()

        generator.generate = AsyncMock(return_value=MagicMock(answer="test"))
        reflector.reflect = AsyncMock(return_value=MagicMock(classifications=[]))
        curator.curate = AsyncMock(return_value=MagicMock(operations=[]))
        environment.evaluate = MagicMock(return_value=MagicMock(feedback="ok", correct=True))

        adapter = AsyncOnlineAdapter(playbook, generator, reflector, curator)

        sample = Sample(question="Test question")
        result = await adapter.step(sample, environment)

        self.assertIsNotNone(result)


@pytest.mark.unit
class TestAdapterFactoryMethods(unittest.TestCase):
    """Test factory methods for creating async-enabled adapters."""

    def test_async_adapter_from_components(self):
        """Test that AsyncOfflineAdapter can be created directly with components."""
        from ace import Playbook
        from ace.async_adaptation import AsyncOfflineAdapter

        playbook = Playbook()
        generator = MagicMock()
        reflector = MagicMock()
        curator = MagicMock()

        # Create async adapter directly
        async_adapter = AsyncOfflineAdapter(playbook, generator, reflector, curator)
        self.assertIsInstance(async_adapter, AsyncOfflineAdapter)
        self.assertEqual(async_adapter._playbook, playbook)


if __name__ == "__main__":
    unittest.main()
