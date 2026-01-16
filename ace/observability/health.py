"""
Health Check System for ACE Framework (Phase 3D)

This module provides health checks for external dependencies (Qdrant, LM Studio).
Tracks component availability, latency, and error states.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import httpx


class HealthStatus(Enum):
    """Health status enumeration."""
    UP = "up"
    DOWN = "down"


@dataclass
class ComponentHealth:
    """
    Health status for a single component.

    Attributes:
        component: Name of the component (e.g., "qdrant", "lm_studio")
        healthy: True if component is operational, False otherwise
        status: String representation of health ("up" or "down")
        latency_ms: Response latency in milliseconds (optional)
        error_message: Error details if unhealthy (optional)
    """

    component: str
    healthy: bool
    status: str
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dict with component health information
        """
        return {
            "component": self.component,
            "healthy": self.healthy,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message
        }


class HealthChecker:
    """
    Health checker for ACE framework dependencies.

    Performs HTTP-based health checks with latency measurement and timeout enforcement.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        lm_studio_url: Optional[str] = None,
        timeout_seconds: float = 5.0
    ):
        """
        Initialize health checker.

        Args:
            qdrant_url: Qdrant server URL (e.g., "http://localhost:6333")
            lm_studio_url: LM Studio server URL (e.g., "http://localhost:1234")
            timeout_seconds: HTTP request timeout in seconds
        """
        self.qdrant_url = qdrant_url
        self.lm_studio_url = lm_studio_url
        self.timeout_seconds = timeout_seconds

    def check_qdrant(self) -> ComponentHealth:
        """
        Check Qdrant vector database health.

        Sends GET request to Qdrant health endpoint and measures latency.

        Returns:
            ComponentHealth: Health status with latency and error details
        """
        if not self.qdrant_url:
            return ComponentHealth(
                component="qdrant",
                healthy=False,
                status="down",
                error_message="Qdrant URL not configured"
            )

        start_time = time.time()

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(f"{self.qdrant_url}/")
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return ComponentHealth(
                        component="qdrant",
                        healthy=True,
                        status="up",
                        latency_ms=latency_ms
                    )
                else:
                    return ComponentHealth(
                        component="qdrant",
                        healthy=False,
                        status="down",
                        latency_ms=latency_ms,
                        error_message=f"HTTP {response.status_code}"
                    )

        except httpx.TimeoutException as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="qdrant",
                healthy=False,
                status="down",
                latency_ms=latency_ms,
                error_message=f"Request timeout: {str(e)}"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="qdrant",
                healthy=False,
                status="down",
                latency_ms=latency_ms,
                error_message=str(e)
            )

    def check_lm_studio(self) -> ComponentHealth:
        """
        Check LM Studio server health.

        Sends GET request to LM Studio endpoint and measures latency.

        Returns:
            ComponentHealth: Health status with latency and error details
        """
        if not self.lm_studio_url:
            return ComponentHealth(
                component="lm_studio",
                healthy=False,
                status="down",
                error_message="LM Studio URL not configured"
            )

        start_time = time.time()

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                # Try /v1/models endpoint (standard OpenAI-compatible endpoint)
                response = client.get(f"{self.lm_studio_url}/v1/models")
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return ComponentHealth(
                        component="lm_studio",
                        healthy=True,
                        status="up",
                        latency_ms=latency_ms
                    )
                else:
                    # Fallback: try root endpoint
                    try:
                        response = client.get(f"{self.lm_studio_url}/")
                        latency_ms = (time.time() - start_time) * 1000

                        if response.status_code == 200:
                            return ComponentHealth(
                                component="lm_studio",
                                healthy=True,
                                status="up",
                                latency_ms=latency_ms
                            )
                    except Exception:
                        pass

                    return ComponentHealth(
                        component="lm_studio",
                        healthy=False,
                        status="down",
                        latency_ms=latency_ms,
                        error_message=f"HTTP {response.status_code}"
                    )

        except httpx.TimeoutException as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="lm_studio",
                healthy=False,
                status="down",
                latency_ms=latency_ms,
                error_message=f"Request timeout: {str(e)}"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="lm_studio",
                healthy=False,
                status="down",
                latency_ms=latency_ms,
                error_message=str(e)
            )

    def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Check all configured dependencies.

        Performs health checks for all configured services and computes overall health.
        Overall health is UP only if all components are healthy.

        Returns:
            Dict mapping component names to ComponentHealth objects.
            Includes special "overall" key with aggregate health status.

        Example:
            >>> checker = HealthChecker(qdrant_url="...", lm_studio_url="...")
            >>> status = checker.check_all()
            >>> print(status['overall'].healthy)  # True only if all components UP
        """
        results = {}

        # Check Qdrant if configured
        if self.qdrant_url:
            results['qdrant'] = self.check_qdrant()

        # Check LM Studio if configured
        if self.lm_studio_url:
            results['lm_studio'] = self.check_lm_studio()

        # Compute overall health (all components must be healthy)
        all_healthy = all(
            component.healthy
            for component in results.values()
        )

        results['overall'] = ComponentHealth(
            component="overall",
            healthy=all_healthy,
            status="up" if all_healthy else "down"
        )

        return results
