"""Async adaptation loops for parallel sample processing.

This module provides async versions of OfflineAdapter and OnlineAdapter
that enable parallel processing of samples for improved throughput.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .adaptation import (
        OfflineAdapter,
        OnlineAdapter,
        Sample,
        TaskEnvironment,
        AdapterStepResult,
    )
    from .playbook import Playbook
    from .roles import Generator, Reflector, Curator


@dataclass
class AsyncAdapterStepResult:
    """Result of a single async adaptation step."""

    sample: "Sample"
    answer: str
    feedback: str
    correct: bool
    playbook_updated: bool


class AsyncOfflineAdapter:
    """Async version of OfflineAdapter for parallel sample processing.

    Enables concurrent processing of multiple samples while respecting
    max_parallel limits to avoid overwhelming LLM APIs.

    Example:
        >>> adapter = AsyncOfflineAdapter(playbook, generator, reflector, curator)
        >>> results = await adapter.run(samples, environment, epochs=3, max_parallel=5)
    """

    def __init__(
        self,
        playbook: "Playbook",
        generator: "Generator",
        reflector: "Reflector",
        curator: "Curator",
    ) -> None:
        """Initialize async adapter with components.

        Args:
            playbook: Playbook to update during adaptation
            generator: Generator role for answer generation
            reflector: Reflector role for feedback analysis
            curator: Curator role for playbook updates
        """
        self._playbook = playbook
        self._generator = generator
        self._reflector = reflector
        self._curator = curator
        self._semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    def from_sync(cls, adapter: "OfflineAdapter") -> "AsyncOfflineAdapter":
        """Create async adapter from sync adapter.

        Args:
            adapter: Sync OfflineAdapter instance

        Returns:
            AsyncOfflineAdapter with same components
        """
        return cls(
            playbook=adapter._playbook,
            generator=adapter._generator,
            reflector=adapter._reflector,
            curator=adapter._curator,
        )

    async def run(
        self,
        samples: List["Sample"],
        environment: "TaskEnvironment",
        epochs: int = 1,
        max_parallel: int = 5,
        batch_size: Optional[int] = None,
    ) -> List[AsyncAdapterStepResult]:
        """Run adaptation loop with parallel processing.

        Args:
            samples: List of samples to process
            environment: Task environment for evaluation
            epochs: Number of passes over samples
            max_parallel: Maximum concurrent sample processing
            batch_size: Optional batch size (defaults to max_parallel)

        Returns:
            List of results for all processed samples
        """
        self._semaphore = asyncio.Semaphore(max_parallel)
        batch_size = batch_size or max_parallel

        all_results: List[AsyncAdapterStepResult] = []

        for epoch in range(epochs):
            # Process in batches
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                batch_results = await self._process_batch(batch, environment)
                all_results.extend(batch_results)

        return all_results

    async def _process_batch(
        self,
        samples: List["Sample"],
        environment: "TaskEnvironment",
    ) -> List[AsyncAdapterStepResult]:
        """Process a batch of samples in parallel.

        Args:
            samples: Batch of samples
            environment: Task environment

        Returns:
            Results for all samples in batch
        """
        tasks = [self._process_sample(sample, environment) for sample in samples]
        return await asyncio.gather(*tasks)

    async def _process_sample(
        self,
        sample: "Sample",
        environment: "TaskEnvironment",
    ) -> AsyncAdapterStepResult:
        """Process a single sample with semaphore-limited concurrency.

        Args:
            sample: Sample to process
            environment: Task environment

        Returns:
            Result for this sample
        """
        assert self._semaphore is not None

        async with self._semaphore:
            # Generate answer
            gen_output = await self._call_async_or_sync(
                self._generator.generate,
                sample.question,
                self._playbook.as_prompt() if hasattr(self._playbook, 'as_prompt') else str(self._playbook),
            )
            answer = gen_output.answer if hasattr(gen_output, 'answer') else str(gen_output)

            # Evaluate
            env_result = environment.evaluate(sample.question, answer, sample.ground_truth)
            feedback = env_result.feedback if hasattr(env_result, 'feedback') else str(env_result)
            correct = env_result.correct if hasattr(env_result, 'correct') else False

            # Reflect
            reflect_output = await self._call_async_or_sync(
                self._reflector.reflect,
                sample.question,
                answer,
                feedback,
                self._playbook.bullets(),
            )

            # Curate
            curator_output = await self._call_async_or_sync(
                self._curator.curate,
                sample.question,
                answer,
                feedback,
                reflect_output.classifications if hasattr(reflect_output, 'classifications') else [],
            )

            # Apply delta
            playbook_updated = False
            if hasattr(curator_output, 'operations') and curator_output.operations:
                from .delta import DeltaBatch
                delta = DeltaBatch(reasoning="async", operations=curator_output.operations)
                self._playbook.apply_delta(delta)
                playbook_updated = True

            return AsyncAdapterStepResult(
                sample=sample,
                answer=answer,
                feedback=feedback,
                correct=correct,
                playbook_updated=playbook_updated,
            )

    async def _call_async_or_sync(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call function, handling both async and sync versions.

        Args:
            func: Function to call (may be async or sync)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class AsyncOnlineAdapter:
    """Async version of OnlineAdapter for streaming scenarios.

    Processes samples one at a time but with async LLM calls,
    enabling non-blocking I/O during online learning.

    Example:
        >>> adapter = AsyncOnlineAdapter(playbook, generator, reflector, curator)
        >>> result = await adapter.step(sample, environment)
    """

    def __init__(
        self,
        playbook: "Playbook",
        generator: "Generator",
        reflector: "Reflector",
        curator: "Curator",
    ) -> None:
        """Initialize async online adapter.

        Args:
            playbook: Playbook to update
            generator: Generator role
            reflector: Reflector role
            curator: Curator role
        """
        self._playbook = playbook
        self._generator = generator
        self._reflector = reflector
        self._curator = curator

    @classmethod
    def from_sync(cls, adapter: "OnlineAdapter") -> "AsyncOnlineAdapter":
        """Create async adapter from sync adapter.

        Args:
            adapter: Sync OnlineAdapter instance

        Returns:
            AsyncOnlineAdapter with same components
        """
        return cls(
            playbook=adapter._playbook,
            generator=adapter._generator,
            reflector=adapter._reflector,
            curator=adapter._curator,
        )

    async def step(
        self,
        sample: "Sample",
        environment: "TaskEnvironment",
    ) -> AsyncAdapterStepResult:
        """Process a single sample asynchronously.

        Args:
            sample: Sample to process
            environment: Task environment

        Returns:
            Result for this sample
        """
        # Generate
        gen_output = await self._call_async_or_sync(
            self._generator.generate,
            sample.question,
            self._playbook.as_prompt() if hasattr(self._playbook, 'as_prompt') else str(self._playbook),
        )
        answer = gen_output.answer if hasattr(gen_output, 'answer') else str(gen_output)

        # Evaluate
        env_result = environment.evaluate(sample.question, answer, sample.ground_truth)
        feedback = env_result.feedback if hasattr(env_result, 'feedback') else str(env_result)
        correct = env_result.correct if hasattr(env_result, 'correct') else False

        # Reflect
        reflect_output = await self._call_async_or_sync(
            self._reflector.reflect,
            sample.question,
            answer,
            feedback,
            self._playbook.bullets(),
        )

        # Curate
        curator_output = await self._call_async_or_sync(
            self._curator.curate,
            sample.question,
            answer,
            feedback,
            reflect_output.classifications if hasattr(reflect_output, 'classifications') else [],
        )

        # Apply delta
        playbook_updated = False
        if hasattr(curator_output, 'operations') and curator_output.operations:
            from .delta import DeltaBatch
            delta = DeltaBatch(reasoning="async_online", operations=curator_output.operations)
            self._playbook.apply_delta(delta)
            playbook_updated = True

        return AsyncAdapterStepResult(
            sample=sample,
            answer=answer,
            feedback=feedback,
            correct=correct,
            playbook_updated=playbook_updated,
        )

    async def run(
        self,
        samples: List["Sample"],
        environment: "TaskEnvironment",
    ) -> List[AsyncAdapterStepResult]:
        """Process samples sequentially (online learning).

        Args:
            samples: Samples to process
            environment: Task environment

        Returns:
            Results for all samples
        """
        results = []
        for sample in samples:
            result = await self.step(sample, environment)
            results.append(result)
        return results

    async def _call_async_or_sync(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call function, handling both async and sync versions."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
