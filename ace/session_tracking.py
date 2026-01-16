"""
Session Outcome Tracking Infrastructure.

Tracks bullet effectiveness per session type with TTL-based cleanup.
Part of Phase 2A: ACE RAG Optimization - Session Outcome Tracker.
"""

import json
from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime, timedelta


@dataclass
class SessionOutcome:
    """Records outcome statistics for a specific bullet in a session context.

    Attributes:
        uses: Total number of times this bullet was used
        worked: Number of times it contributed to success
        failed: Number of times it contributed to failure
        last_updated: Timestamp of most recent update
    """

    uses: int = 0
    worked: int = 0
    failed: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class SessionOutcomeTracker:
    """Tracks bullet effectiveness per session type with automatic expiration.

    Maintains per-session statistics for bullets to enable context-aware
    bullet selection. Session data expires after TTL period.

    Args:
        ttl_hours: Time-to-live in hours for session data (default: 24)

    Example:
        >>> tracker = SessionOutcomeTracker(ttl_hours=24)
        >>> tracker.track("browser_automation", "bullet_123", "worked")
        >>> effectiveness = tracker.get_session_effectiveness("browser_automation", "bullet_123")
        >>> print(f"Effectiveness: {effectiveness:.2%}")
    """

    def __init__(self, ttl_hours: int = 24):
        """Initialize tracker with specified TTL.

        Args:
            ttl_hours: Time-to-live in hours for session data
        """
        self._outcomes: Dict[str, SessionOutcome] = {}  # key = "session_type:bullet_id"
        self._ttl = timedelta(hours=ttl_hours)

    def track(self, session_type: str, bullet_id: str, outcome: str) -> None:
        """Track outcome for a bullet in a specific session context.

        Args:
            session_type: Type of session (e.g., "browser_automation")
            bullet_id: Unique identifier for the bullet
            outcome: Result - "worked" or "failed"

        Raises:
            ValueError: If outcome is not "worked" or "failed"
        """
        # Automatic cleanup on every track call
        self.cleanup_expired()

        key = f"{session_type}:{bullet_id}"

        # Create new outcome record if doesn't exist
        if key not in self._outcomes:
            self._outcomes[key] = SessionOutcome()

        # Get the outcome record
        record = self._outcomes[key]

        # Increment counters
        record.uses += 1
        if outcome == "worked":
            record.worked += 1
        elif outcome == "failed":
            record.failed += 1

        # Update timestamp
        record.last_updated = datetime.now()

    def get_session_effectiveness(
        self, session_type: str, bullet_id: str, default: float = 0.5
    ) -> float:
        """Calculate effectiveness of a bullet in a specific session context.

        Args:
            session_type: Type of session
            bullet_id: Unique identifier for the bullet
            default: Default effectiveness if no data exists (0.0-1.0)

        Returns:
            Effectiveness ratio (worked / total_outcomes), or default if no data

        Example:
            >>> effectiveness = tracker.get_session_effectiveness("api_calls", "bullet_456")
            >>> if effectiveness > 0.7:
            ...     print("High-performing bullet in API context")
        """
        key = f"{session_type}:{bullet_id}"

        # Return default if no data exists
        if key not in self._outcomes:
            return default

        record = self._outcomes[key]
        total = record.worked + record.failed

        # Return default if no outcome data (only uses tracked)
        if total == 0:
            return default

        # Calculate effectiveness as worked / total
        return record.worked / total

    def cleanup_expired(self) -> None:
        """Remove session data older than TTL.

        Automatically called on every track() operation to maintain
        data freshness and prevent unbounded memory growth.

        Example:
            >>> tracker = SessionOutcomeTracker(ttl_hours=1)
            >>> # ... track some data ...
            >>> tracker.cleanup_expired()  # Remove entries older than 1 hour
        """
        now = datetime.now()
        expired_keys = [
            key
            for key, outcome in self._outcomes.items()
            if (now - outcome.last_updated) > self._ttl
        ]

        for key in expired_keys:
            del self._outcomes[key]

    def save_to_file(self, filepath: str) -> None:
        """Save session outcomes to JSON file.

        Args:
            filepath: Path to JSON file for persistence

        Example:
            >>> tracker = SessionOutcomeTracker()
            >>> tracker.track("browser", "bullet_001", "worked")
            >>> tracker.save_to_file("session_data.json")
        """
        data = {
            "ttl_hours": self._ttl.total_seconds() / 3600,
            "outcomes": {
                key: {
                    "uses": outcome.uses,
                    "worked": outcome.worked,
                    "failed": outcome.failed,
                    "last_updated": outcome.last_updated.isoformat()
                }
                for key, outcome in self._outcomes.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "SessionOutcomeTracker":
        """Load session outcomes from JSON file.

        Args:
            filepath: Path to JSON file containing persisted data

        Returns:
            SessionOutcomeTracker instance with loaded data

        Example:
            >>> tracker = SessionOutcomeTracker.load_from_file("session_data.json")
            >>> effectiveness = tracker.get_session_effectiveness("browser", "bullet_001")
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        tracker = cls(ttl_hours=int(data.get("ttl_hours", 24)))
        for key, outcome_data in data.get("outcomes", {}).items():
            tracker._outcomes[key] = SessionOutcome(
                uses=outcome_data["uses"],
                worked=outcome_data["worked"],
                failed=outcome_data["failed"],
                last_updated=datetime.fromisoformat(outcome_data["last_updated"])
            )
        return tracker
