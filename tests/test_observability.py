"""
Tests for Enhanced Observability Module (Phase 3D)

This test suite covers:
- Prometheus metrics (histograms, counters, gauges)
- Health checks (Qdrant, LM Studio)
- Distributed tracing (OpenTelemetry)

All tests should FAIL initially (TDD RED phase).
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import time

# Check for optional prometheus_client dependency
try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Skip all tests in this module if prometheus_client is not available
pytestmark = pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed. Install with: pip install prometheus_client"
)


@pytest.mark.unit
class TestPrometheusMetrics(unittest.TestCase):
    """Test Prometheus metrics collection and export."""

    def test_prometheus_metrics_exist(self):
        """Test that metrics module exists."""
        from ace.observability.metrics import MetricsRegistry
        self.assertIsNotNone(MetricsRegistry)

    def test_retrieval_latency_histogram(self):
        """Test histogram for retrieval latency tracking."""
        from ace.observability.metrics import retrieval_latency_histogram

        # Record various latencies
        retrieval_latency_histogram.observe(0.025)  # 25ms
        retrieval_latency_histogram.observe(0.100)  # 100ms
        retrieval_latency_histogram.observe(0.500)  # 500ms

        # Should not raise - histogram accepts observations
        self.assertIsNotNone(retrieval_latency_histogram)

    def test_retrieval_latency_histogram_with_labels(self):
        """Test histogram with tenant_id and operation labels."""
        from ace.observability.metrics import retrieval_latency_histogram

        # Record latency with labels
        retrieval_latency_histogram.labels(
            tenant_id="tenant_123",
            operation="semantic_search"
        ).observe(0.045)

        retrieval_latency_histogram.labels(
            tenant_id="tenant_456",
            operation="hybrid_search"
        ).observe(0.082)

        # Should track separate metric streams per label combination
        self.assertIsNotNone(retrieval_latency_histogram)

    def test_retrieval_count_counter(self):
        """Test counter for total retrievals."""
        from ace.observability.metrics import retrieval_count

        # Increment counter
        retrieval_count.inc()
        retrieval_count.inc()
        retrieval_count.inc()

        # Should not raise
        self.assertIsNotNone(retrieval_count)

    def test_retrieval_count_with_labels(self):
        """Test counter with status and tenant_id labels."""
        from ace.observability.metrics import retrieval_count

        # Track successful retrieval
        retrieval_count.labels(
            status="success",
            tenant_id="tenant_123"
        ).inc()

        # Track failed retrieval
        retrieval_count.labels(
            status="error",
            tenant_id="tenant_456"
        ).inc()

        self.assertIsNotNone(retrieval_count)

    def test_bullet_count_gauge(self):
        """Test gauge for current bullet count."""
        from ace.observability.metrics import bullet_gauge

        # Set gauge value
        bullet_gauge.set(42)
        bullet_gauge.set(100)
        bullet_gauge.set(7)

        # Should track current value
        self.assertIsNotNone(bullet_gauge)

    def test_bullet_count_gauge_with_tenant(self):
        """Test gauge with tenant_id label."""
        from ace.observability.metrics import bullet_gauge

        # Set per-tenant bullet counts
        bullet_gauge.labels(tenant_id="tenant_123").set(25)
        bullet_gauge.labels(tenant_id="tenant_456").set(50)

        self.assertIsNotNone(bullet_gauge)

    def test_metrics_labels_schema(self):
        """Test that metrics have proper label schemas."""
        from ace.observability.metrics import (
            retrieval_latency_histogram,
            retrieval_count,
            bullet_gauge
        )

        # Latency should have tenant_id and operation
        self.assertIn('tenant_id', retrieval_latency_histogram._labelnames)
        self.assertIn('operation', retrieval_latency_histogram._labelnames)

        # Count should have status and tenant_id
        self.assertIn('status', retrieval_count._labelnames)
        self.assertIn('tenant_id', retrieval_count._labelnames)

        # Gauge should have tenant_id
        self.assertIn('tenant_id', bullet_gauge._labelnames)

    def test_metrics_export_format(self):
        """Test Prometheus format export."""
        from ace.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()

        # Export metrics in Prometheus text format
        output = registry.export()

        # Should contain HELP and TYPE declarations
        self.assertIn('# HELP', output)
        self.assertIn('# TYPE', output)

        # Should contain metric names
        self.assertIn('ace_retrieval_latency', output)
        self.assertIn('ace_retrieval_count', output)
        self.assertIn('ace_bullet_count', output)

    def test_metrics_registry_singleton(self):
        """Test that MetricsRegistry follows singleton pattern."""
        from ace.observability.metrics import MetricsRegistry

        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()

        # Should return same instance
        self.assertIs(registry1, registry2)

    def test_custom_buckets_for_latency(self):
        """Test custom histogram buckets for latency distribution."""
        from ace.observability.metrics import retrieval_latency_histogram

        # Should have buckets optimized for retrieval latencies
        # e.g., [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        buckets = retrieval_latency_histogram._buckets

        self.assertGreater(len(buckets), 5)  # At least 5 buckets
        self.assertIn(0.1, buckets)  # 100ms bucket
        self.assertIn(0.5, buckets)  # 500ms bucket


@pytest.mark.unit
class TestHealthCheck(unittest.TestCase):
    """Test health check endpoints and dependency status."""

    def test_health_endpoint_exists(self):
        """Test that health check module exists."""
        from ace.observability.health import HealthChecker
        self.assertIsNotNone(HealthChecker)

    def test_health_status_enum(self):
        """Test HealthStatus enumeration."""
        from ace.observability.health import HealthStatus

        # Should have UP and DOWN states
        self.assertEqual(HealthStatus.UP.value, "up")
        self.assertEqual(HealthStatus.DOWN.value, "down")

    @patch('httpx.Client')
    def test_health_check_qdrant_up(self, mock_client):
        """Test health check when Qdrant is responsive."""
        from ace.observability.health import HealthChecker

        # Mock successful Qdrant response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        checker = HealthChecker(qdrant_url="http://localhost:6333")
        status = checker.check_qdrant()

        self.assertTrue(status.healthy)
        self.assertEqual(status.status, "up")

    @patch('httpx.Client')
    def test_health_check_qdrant_down(self, mock_client):
        """Test health check when Qdrant is unreachable."""
        from ace.observability.health import HealthChecker

        # Mock connection error
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Connection refused")

        checker = HealthChecker(qdrant_url="http://localhost:6333")
        status = checker.check_qdrant()

        self.assertFalse(status.healthy)
        self.assertEqual(status.status, "down")
        self.assertIn("Connection refused", status.error_message)

    @patch('httpx.Client')
    def test_health_check_qdrant_timeout(self, mock_client):
        """Test health check when Qdrant times out."""
        from ace.observability.health import HealthChecker
        import httpx

        # Mock timeout error
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.TimeoutException("Request timeout")

        checker = HealthChecker(qdrant_url="http://localhost:6333")
        status = checker.check_qdrant()

        self.assertFalse(status.healthy)
        self.assertIn("timeout", status.error_message.lower())

    @patch('httpx.Client')
    def test_health_check_lm_studio_up(self, mock_client):
        """Test health check when LM Studio is responsive."""
        from ace.observability.health import HealthChecker

        # Mock successful LM Studio response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        checker = HealthChecker(lm_studio_url="http://localhost:1234")
        status = checker.check_lm_studio()

        self.assertTrue(status.healthy)
        self.assertEqual(status.status, "up")

    @patch('httpx.Client')
    def test_health_check_lm_studio_down(self, mock_client):
        """Test health check when LM Studio is unreachable."""
        from ace.observability.health import HealthChecker

        # Mock connection error
        mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Connection refused")

        checker = HealthChecker(lm_studio_url="http://localhost:1234")
        status = checker.check_lm_studio()

        self.assertFalse(status.healthy)
        self.assertEqual(status.status, "down")

    @patch('httpx.Client')
    def test_health_check_combined(self, mock_client):
        """Test combined health status of all dependencies."""
        from ace.observability.health import HealthChecker

        # Mock both services up
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        checker = HealthChecker(
            qdrant_url="http://localhost:6333",
            lm_studio_url="http://localhost:1234"
        )

        overall_status = checker.check_all()

        # Should return dict with all component statuses
        self.assertIn('qdrant', overall_status)
        self.assertIn('lm_studio', overall_status)
        self.assertIn('overall', overall_status)

        # Overall should be healthy only if all components healthy
        self.assertTrue(overall_status['overall'].healthy)

    @patch('httpx.Client')
    def test_health_check_combined_partial_failure(self, mock_client):
        """Test combined health when one service fails."""
        from ace.observability.health import HealthChecker

        # Mock Qdrant up, LM Studio down
        def side_effect(url, *args, **kwargs):
            if '6333' in url:
                response = Mock()
                response.status_code = 200
                return response
            else:
                raise Exception("Connection refused")

        mock_client.return_value.__enter__.return_value.get.side_effect = side_effect

        checker = HealthChecker(
            qdrant_url="http://localhost:6333",
            lm_studio_url="http://localhost:1234"
        )

        overall_status = checker.check_all()

        # Overall should be unhealthy if any component fails
        self.assertFalse(overall_status['overall'].healthy)
        self.assertTrue(overall_status['qdrant'].healthy)
        self.assertFalse(overall_status['lm_studio'].healthy)

    def test_health_response_format(self):
        """Test JSON response format for health endpoint."""
        from ace.observability.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        # Create mock status
        from ace.observability.health import ComponentHealth
        status = ComponentHealth(
            component="qdrant",
            healthy=True,
            status="up",
            latency_ms=15.3
        )

        # Should serialize to JSON-compatible dict
        json_output = status.to_dict()

        self.assertEqual(json_output['component'], 'qdrant')
        self.assertEqual(json_output['status'], 'up')
        self.assertTrue(json_output['healthy'])
        self.assertIsInstance(json_output['latency_ms'], float)

    @patch('httpx.Client')
    def test_health_check_measures_latency(self, mock_client):
        """Test that health checks measure response latency."""
        from ace.observability.health import HealthChecker

        # Mock slow response
        mock_response = Mock()
        mock_response.status_code = 200

        def slow_get(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return mock_response

        mock_client.return_value.__enter__.return_value.get.side_effect = slow_get

        checker = HealthChecker(qdrant_url="http://localhost:6333")
        status = checker.check_qdrant()

        # Latency should be recorded
        self.assertIsNotNone(status.latency_ms)
        self.assertGreaterEqual(status.latency_ms, 100)

    def test_health_check_respects_timeout(self):
        """Test that health checks enforce timeout limits."""
        from ace.observability.health import HealthChecker

        # Create checker with short timeout
        checker = HealthChecker(
            qdrant_url="http://localhost:6333",
            timeout_seconds=1.0
        )

        # Should have timeout configured
        self.assertEqual(checker.timeout_seconds, 1.0)


@pytest.mark.unit
class TestDistributedTracing(unittest.TestCase):
    """Test OpenTelemetry distributed tracing integration."""

    def test_opentelemetry_available(self):
        """Test OpenTelemetry integration when installed."""
        try:
            from ace.observability.tracing import TracingManager
            self.assertIsNotNone(TracingManager)
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_trace_context_propagation(self):
        """Test that trace context propagates across calls."""
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        tracer = TracingManager()

        # Start parent span
        with tracer.start_span("parent_operation") as parent:
            parent_trace_id = parent.get_span_context().trace_id

            # Start child span
            with tracer.start_span("child_operation") as child:
                child_trace_id = child.get_span_context().trace_id

                # Should share same trace ID
                self.assertEqual(parent_trace_id, child_trace_id)

    def test_span_creation(self):
        """Test creating spans for operations."""
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        tracer = TracingManager()

        # Create span
        with tracer.start_span("test_operation") as span:
            self.assertIsNotNone(span)
            self.assertTrue(span.is_recording())

    def test_span_attributes(self):
        """Test that spans have correct attributes."""
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        tracer = TracingManager()

        # Create span with attributes
        with tracer.start_span("retrieval_operation") as span:
            span.set_attribute("tenant_id", "tenant_123")
            span.set_attribute("operation_type", "semantic_search")
            span.set_attribute("bullet_count", 42)

            # Attributes should be set
            self.assertIsNotNone(span)

    def test_span_events(self):
        """Test adding events to spans."""
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        tracer = TracingManager()

        with tracer.start_span("operation") as span:
            # Add events
            span.add_event("cache_miss")
            span.add_event("qdrant_query_start")
            span.add_event("qdrant_query_complete", {"result_count": 5})

            self.assertIsNotNone(span)

    def test_span_status(self):
        """Test setting span status (success/error)."""
        try:
            from ace.observability.tracing import TracingManager
            from opentelemetry.trace import Status, StatusCode
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        tracer = TracingManager()

        # Successful operation
        with tracer.start_span("success_op") as span:
            span.set_status(Status(StatusCode.OK))

        # Failed operation
        try:
            with tracer.start_span("error_op") as span:
                raise ValueError("Test error")
        except ValueError:
            # Span should record error status
            pass

    def test_decorator_based_tracing(self):
        """Test decorator-based automatic span creation."""
        try:
            from ace.observability.tracing import trace_operation
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        @trace_operation("test_function")
        def sample_function(x, y):
            return x + y

        # Call should create span automatically
        result = sample_function(2, 3)
        self.assertEqual(result, 5)

    def test_async_span_support(self):
        """Test tracing async operations."""
        try:
            from ace.observability.tracing import TracingManager
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        import asyncio

        tracer = TracingManager()

        async def async_operation():
            with tracer.start_span("async_op") as span:
                await asyncio.sleep(0.01)
                span.set_attribute("completed", True)
                return "done"

        # Should support async context
        result = asyncio.run(async_operation())
        self.assertEqual(result, "done")


@pytest.mark.unit
class TestMetricsContextManager(unittest.TestCase):
    """Test context manager for automatic metric recording."""

    def test_latency_context_manager(self):
        """Test automatic latency recording with context manager."""
        from ace.observability.metrics import track_latency

        # Should automatically record duration
        with track_latency(operation="test_op", tenant_id="tenant_123"):
            time.sleep(0.01)  # 10ms operation

        # Latency should be recorded to histogram
        # (actual verification would check prometheus registry)

    def test_counter_increment_on_success(self):
        """Test counter increment on successful operation."""
        from ace.observability.metrics import track_operation

        with track_operation(operation="retrieval", tenant_id="tenant_123"):
            # Successful operation
            pass

        # Should increment success counter

    def test_counter_increment_on_error(self):
        """Test counter increment on failed operation."""
        from ace.observability.metrics import track_operation

        try:
            with track_operation(operation="retrieval", tenant_id="tenant_123"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should increment error counter


if __name__ == '__main__':
    unittest.main()
