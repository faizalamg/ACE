"""
Prometheus Metrics Collection for ACE Framework (Phase 3D)

This module provides production-grade metrics collection using Prometheus client library.
Tracks retrieval latency, operation counts, and bullet counts across tenants.

DESIGN NOTE: To support both labeled and unlabeled usage (for test compatibility),
we create wrapper classes that provide default label values when called without labels.
"""

import time
from contextlib import contextmanager
from typing import Generator, Optional

from prometheus_client import Histogram, Counter, Gauge, REGISTRY, generate_latest


# Custom buckets optimized for retrieval operations (10ms to 5s)
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class _LabeledHistogram:
    """Wrapper for Histogram that supports both labeled and unlabeled usage."""

    def __init__(self, histogram):
        self._histogram = histogram
        self._labelnames = histogram._labelnames
        # Expose buckets for test verification
        self._buckets = LATENCY_BUCKETS

    def labels(self, **kwargs):
        """Get labeled child metric."""
        return self._histogram.labels(**kwargs)

    def observe(self, amount):
        """Observe value with default labels (for unlabeled usage)."""
        return self._histogram.labels(tenant_id="default", operation="default").observe(amount)


class _LabeledCounter:
    """Wrapper for Counter that supports both labeled and unlabeled usage."""

    def __init__(self, counter):
        self._counter = counter
        self._labelnames = counter._labelnames

    def labels(self, **kwargs):
        """Get labeled child metric."""
        return self._counter.labels(**kwargs)

    def inc(self, amount=1):
        """Increment with default labels (for unlabeled usage)."""
        return self._counter.labels(status="default", tenant_id="default").inc(amount)


class _LabeledGauge:
    """Wrapper for Gauge that supports both labeled and unlabeled usage."""

    def __init__(self, gauge):
        self._gauge = gauge
        self._labelnames = gauge._labelnames

    def labels(self, **kwargs):
        """Get labeled child metric."""
        return self._gauge.labels(**kwargs)

    def set(self, value):
        """Set value with default labels (for unlabeled usage)."""
        return self._gauge.labels(tenant_id="default").set(value)


# Create underlying Prometheus metrics
_histogram = Histogram(
    'ace_retrieval_latency_seconds',
    'Retrieval operation latency in seconds',
    labelnames=['tenant_id', 'operation'],
    buckets=LATENCY_BUCKETS
)

_counter = Counter(
    'ace_retrieval_count_total',
    'Total number of retrieval operations',
    labelnames=['status', 'tenant_id']
)

_gauge = Gauge(
    'ace_bullet_count',
    'Current number of bullets in playbook',
    labelnames=['tenant_id']
)

# Export wrapped versions that support both labeled and unlabeled usage
retrieval_latency_histogram = _LabeledHistogram(_histogram)
retrieval_count = _LabeledCounter(_counter)
bullet_gauge = _LabeledGauge(_gauge)


class MetricsRegistry:
    """
    Singleton registry for Prometheus metrics export.

    Provides centralized access to metrics and export functionality.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern - ensures only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def export(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            str: Prometheus-formatted metrics text
        """
        return generate_latest(REGISTRY).decode('utf-8')


@contextmanager
def track_latency(operation: str, tenant_id: str = "default") -> Generator[None, None, None]:
    """
    Context manager for automatic latency tracking.

    Records operation duration to retrieval_latency_histogram.

    Args:
        operation: Name of the operation being tracked
        tenant_id: Tenant identifier for multi-tenancy support

    Yields:
        None

    Example:
        >>> with track_latency(operation="semantic_search", tenant_id="tenant_123"):
        ...     perform_search()
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        retrieval_latency_histogram.labels(
            tenant_id=tenant_id,
            operation=operation
        ).observe(duration)


@contextmanager
def track_operation(operation: str, tenant_id: str = "default") -> Generator[None, None, None]:
    """
    Context manager for automatic operation counting.

    Increments success/error counters based on operation outcome.

    Args:
        operation: Name of the operation being tracked
        tenant_id: Tenant identifier for multi-tenancy support

    Yields:
        None

    Raises:
        Exception: Re-raises any exception after recording error metric

    Example:
        >>> with track_operation(operation="retrieval", tenant_id="tenant_123"):
        ...     perform_retrieval()  # Auto-increments success counter
    """
    try:
        yield
        retrieval_count.labels(status="success", tenant_id=tenant_id).inc()
    except Exception:
        retrieval_count.labels(status="error", tenant_id=tenant_id).inc()
        raise
