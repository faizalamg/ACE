"""Resilience patterns for robust LLM interactions.

This module provides circuit breaker, retry, and other resilience patterns
for handling transient failures in LLM API calls.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, calls pass through
    OPEN = "open"  # Circuit tripped, calls rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        self.message = message
        super().__init__(self.message)


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for protecting against cascading failures.

    The circuit breaker monitors calls and tracks failures. When failures
    exceed a threshold, the circuit "opens" and rejects further calls
    for a recovery period. After the timeout, it allows a test call
    to check if the service has recovered.

    States:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Circuit tripped, all calls rejected immediately
    - HALF_OPEN: Recovery period, one test call allowed

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        >>>
        >>> @breaker
        >>> def call_llm(prompt):
        ...     return llm_client.complete(prompt)
        >>>
        >>> try:
        ...     result = call_llm("Hello")
        >>> except CircuitOpenError:
        ...     # Circuit is open, use fallback
        ...     result = cached_response
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 1  # successes needed to close from half-open

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _half_open_success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _total_success_count: int = field(default=0, init=False)
    _total_failure_count: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize the lock after dataclass initialization."""
        self._lock = Lock()

    @property
    def state(self) -> str:
        """Get current circuit state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if (
                    self._last_failure_time is not None
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_success_count = 0
            return self._state.value

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    def check(self) -> None:
        """Check if circuit allows a call to pass through.

        Raises:
            CircuitOpenError: If circuit is open
        """
        state = self.state  # This triggers timeout check
        if state == CircuitState.OPEN.value:
            raise CircuitOpenError(
                f"Circuit breaker is open. "
                f"Failures: {self._failure_count}, "
                f"Recovery in: {self._time_until_recovery():.1f}s"
            )

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._total_success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_success_count = 0
            else:
                # In closed state, success resets failure count
                self._failure_count = 0
            self._success_count += 1

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._total_failure_count += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_success_count = 0
            self._last_failure_time = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics.

        Returns:
            Dict with success_count, failure_count, state, etc.
        """
        return {
            "state": self.state,
            "failure_count": self._total_failure_count,
            "success_count": self._total_success_count,
            "consecutive_failures": self._failure_count,
            "last_failure_time": self._last_failure_time,
        }

    def _time_until_recovery(self) -> float:
        """Calculate time until circuit enters half-open state."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use circuit breaker as a decorator.

        Args:
            func: Function to wrap with circuit breaker

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.check()
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding retry behavior with exponential backoff.

    Args:
        config: Retry configuration
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorator function

    Example:
        >>> @with_retry(RetryConfig(max_retries=3))
        >>> def call_api():
        ...     return api_client.call()
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import random

            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (config.exponential_base**attempt),
                            config.max_delay,
                        )
                        if config.jitter:
                            delay *= 0.5 + random.random()
                        time.sleep(delay)
                    else:
                        raise

            # Should never reach here, but for type checker
            raise last_exception  # type: ignore

        return wrapper

    return decorator


def with_circuit_breaker(
    breaker: CircuitBreaker,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding circuit breaker to a function.

    Args:
        breaker: CircuitBreaker instance to use

    Returns:
        Decorator function

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3)
        >>>
        >>> @with_circuit_breaker(breaker)
        >>> def call_api():
        ...     return api_client.call()
    """
    return breaker  # CircuitBreaker is already callable as decorator


class ResilientClient:
    """Base class for resilient LLM clients with built-in circuit breaker.

    Example:
        >>> class MyLLMClient(ResilientClient):
        ...     def _do_complete(self, prompt):
        ...         return api.call(prompt)
        >>>
        >>> client = MyLLMClient()
        >>> result = client.complete("Hello")  # Protected by circuit breaker
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize resilient client.

        Args:
            circuit_breaker: Optional circuit breaker (default created if None)
            retry_config: Optional retry configuration
        """
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._retry_config = retry_config or RetryConfig()

    def complete(self, prompt: str) -> str:
        """Make a resilient LLM call with circuit breaker and retry.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            LLM response

        Raises:
            CircuitOpenError: If circuit breaker is open
        """
        self._circuit_breaker.check()

        @with_retry(self._retry_config)
        def _call() -> str:
            try:
                result = self._do_complete(prompt)
                self._circuit_breaker.record_success()
                return result
            except Exception as e:
                self._circuit_breaker.record_failure()
                raise

        return _call()

    def _do_complete(self, prompt: str) -> str:
        """Actual implementation of the LLM call. Override in subclass.

        Args:
            prompt: The prompt

        Returns:
            Response string
        """
        raise NotImplementedError("Subclass must implement _do_complete")

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return self._circuit_breaker.get_metrics()

    def reset_circuit(self) -> None:
        """Reset the circuit breaker."""
        self._circuit_breaker.reset()
