"""
Distributed Tracing for ACE Framework (Phase 3D)

This module provides OpenTelemetry integration for distributed tracing.
When OpenTelemetry is not installed, importing TracingManager raises ImportError
to allow tests to properly skip. The trace_operation decorator gracefully degrades.
"""

import functools
from typing import Optional, Any

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    TRACING_AVAILABLE = True
    _IMPORT_ERROR = None

except ImportError as e:
    TRACING_AVAILABLE = False
    _IMPORT_ERROR = e

    # Provide no-op Status/StatusCode for graceful degradation
    class Status:
        """No-op Status class when OpenTelemetry not available."""
        def __init__(self, status_code):
            self.status_code = status_code

    class StatusCode:
        """No-op StatusCode enum when OpenTelemetry not available."""
        OK = "ok"
        ERROR = "error"


if TRACING_AVAILABLE:
    class TracingManager:
        """
        Manager for OpenTelemetry distributed tracing.

        Provides centralized span creation and context propagation.
        """

        def __init__(self):
            """Initialize tracing manager with default tracer."""
            self._tracer = trace.get_tracer(__name__)

        def start_span(self, name: str):
            """
            Start a new span for the given operation.

            Args:
                name: Name of the operation being traced

            Returns:
                Span context manager

            Example:
                >>> tracer = TracingManager()
                >>> with tracer.start_span("retrieval_operation") as span:
                ...     span.set_attribute("tenant_id", "tenant_123")
                ...     perform_retrieval()
            """
            return self._tracer.start_as_current_span(name)


    def trace_operation(name: str):
        """
        Decorator for automatic span creation around function calls.

        Args:
            name: Name of the operation being traced

        Returns:
            Decorator function

        Example:
            >>> @trace_operation("compute_similarity")
            ... def compute_similarity(a, b):
            ...     return dot_product(a, b)
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


else:
    # When OpenTelemetry not available, TracingManager access will raise ImportError
    # This is handled by __getattr__ below

    def trace_operation(name: str):
        """No-op tracing decorator when OpenTelemetry not available."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Export public API
__all__ = [
    'TracingManager',
    'trace_operation',
    'TRACING_AVAILABLE',
    'Status',
    'StatusCode'
]


def __getattr__(name):
    """
    Raise ImportError when accessing TracingManager without OpenTelemetry.

    This allows tests to use pytest.skip() pattern:
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")
    """
    if name == 'TracingManager':
        if not TRACING_AVAILABLE:
            raise ImportError(
                "OpenTelemetry is not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            ) from _IMPORT_ERROR
        # This shouldn't happen - TracingManager is defined when TRACING_AVAILABLE
        raise AttributeError(f"TracingManager import failed unexpectedly")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
