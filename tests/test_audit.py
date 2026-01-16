"""Tests for AuditLogger - Enterprise audit logging for ACE operations.

TDD Phase 3B: Write failing tests FIRST before implementation.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestAuditLoggerInit(unittest.TestCase):
    """Test AuditLogger initialization."""

    def test_audit_logger_class_exists(self):
        """Test that AuditLogger class exists."""
        from ace.audit import AuditLogger

        self.assertIsNotNone(AuditLogger)

    def test_audit_logger_default_initialization(self):
        """Test default initialization."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            self.assertIsNotNone(logger)
            self.assertEqual(logger._log_dir, Path(tmpdir))

    def test_audit_logger_creates_log_directory(self):
        """Test that logger creates log directory if not exists."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit_logs"
            logger = AuditLogger(log_dir=str(log_path))

            self.assertTrue(log_path.exists())


@pytest.mark.unit
class TestAuditLogEntry(unittest.TestCase):
    """Test audit log entry creation."""

    def test_log_retrieval_operation(self):
        """Test logging a retrieval operation."""
        from ace.audit import AuditLogger, AuditEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_retrieval(
                query="How do I debug this error?",
                results_count=5,
                latency_ms=42.5,
                user_id="user-123",
            )

            self.assertIsInstance(entry, AuditEntry)
            self.assertEqual(entry.operation, "retrieval")
            self.assertEqual(entry.query, "How do I debug this error?")
            self.assertEqual(entry.results_count, 5)
            self.assertEqual(entry.latency_ms, 42.5)
            self.assertEqual(entry.user_id, "user-123")

    def test_log_entry_has_timestamp(self):
        """Test that log entries have timestamps."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_retrieval(
                query="test query",
                results_count=1,
                latency_ms=10.0,
            )

            self.assertIsNotNone(entry.timestamp)
            # Should be ISO format string
            datetime.fromisoformat(entry.timestamp)

    def test_log_entry_has_unique_id(self):
        """Test that each log entry has a unique ID."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry1 = logger.log_retrieval(query="q1", results_count=1, latency_ms=10.0)
            entry2 = logger.log_retrieval(query="q2", results_count=1, latency_ms=10.0)

            self.assertNotEqual(entry1.id, entry2.id)


@pytest.mark.unit
class TestAuditLogPersistence(unittest.TestCase):
    """Test audit log persistence to files."""

    def test_log_entry_written_to_file(self):
        """Test that log entries are written to files."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            logger.log_retrieval(
                query="test query",
                results_count=3,
                latency_ms=25.0,
                user_id="test-user",
            )

            # Find log file
            log_files = list(Path(tmpdir).glob("*.jsonl"))
            self.assertGreater(len(log_files), 0)

            # Read and verify
            with open(log_files[0], "r") as f:
                line = f.readline()
                data = json.loads(line)

            self.assertEqual(data["query"], "test query")
            self.assertEqual(data["results_count"], 3)

    def test_log_entries_appended_to_daily_file(self):
        """Test that entries are appended to daily log file."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            # Log multiple entries
            logger.log_retrieval(query="q1", results_count=1, latency_ms=10.0)
            logger.log_retrieval(query="q2", results_count=2, latency_ms=20.0)
            logger.log_retrieval(query="q3", results_count=3, latency_ms=30.0)

            # Should have one daily file with 3 lines
            log_files = list(Path(tmpdir).glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)

            with open(log_files[0], "r") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)


@pytest.mark.unit
class TestAuditQueryWithResults(unittest.TestCase):
    """Test auditing queries with full result details."""

    def test_audit_query_with_results(self):
        """Test logging query with result bullet IDs."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_retrieval(
                query="debugging help",
                results_count=2,
                latency_ms=15.0,
                result_ids=["bullet-001", "bullet-002"],
                scores=[0.95, 0.87],
            )

            self.assertEqual(entry.result_ids, ["bullet-001", "bullet-002"])
            self.assertEqual(entry.scores, [0.95, 0.87])

    def test_audit_query_with_filters(self):
        """Test logging query with filter parameters."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_retrieval(
                query="test",
                results_count=5,
                latency_ms=20.0,
                filters={"task_type": "debugging", "domain": "software"},
            )

            self.assertEqual(entry.filters, {"task_type": "debugging", "domain": "software"})


@pytest.mark.unit
class TestAuditExport(unittest.TestCase):
    """Test audit log export functionality."""

    def test_export_to_json(self):
        """Test exporting audit logs to JSON."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            # Create some entries
            logger.log_retrieval(query="q1", results_count=1, latency_ms=10.0)
            logger.log_retrieval(query="q2", results_count=2, latency_ms=20.0)

            # Export
            export_path = Path(tmpdir) / "export.json"
            logger.export_json(str(export_path))

            self.assertTrue(export_path.exists())

            with open(export_path, "r") as f:
                data = json.load(f)

            self.assertEqual(len(data), 2)

    def test_export_to_csv(self):
        """Test exporting audit logs to CSV."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            # Create some entries
            logger.log_retrieval(query="q1", results_count=1, latency_ms=10.0)
            logger.log_retrieval(query="q2", results_count=2, latency_ms=20.0)

            # Export
            export_path = Path(tmpdir) / "export.csv"
            logger.export_csv(str(export_path))

            self.assertTrue(export_path.exists())

            with open(export_path, "r") as f:
                lines = f.readlines()

            # Header + 2 data rows
            self.assertEqual(len(lines), 3)

    def test_export_with_date_range(self):
        """Test exporting with date range filter."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            # Create entries
            logger.log_retrieval(query="q1", results_count=1, latency_ms=10.0)

            # Export with date range
            export_path = Path(tmpdir) / "export.json"
            today = datetime.now().strftime("%Y-%m-%d")
            logger.export_json(
                str(export_path),
                start_date=today,
                end_date=today,
            )

            self.assertTrue(export_path.exists())


@pytest.mark.unit
class TestAuditLogOtherOperations(unittest.TestCase):
    """Test logging other operations besides retrieval."""

    def test_log_index_operation(self):
        """Test logging an index operation."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_index(
                bullet_id="bullet-001",
                action="create",
                user_id="admin",
            )

            self.assertEqual(entry.operation, "index")
            self.assertEqual(entry.bullet_id, "bullet-001")
            self.assertEqual(entry.action, "create")

    def test_log_playbook_operation(self):
        """Test logging a playbook operation."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            entry = logger.log_playbook(
                playbook_id="playbook-001",
                action="load",
                bullet_count=50,
                user_id="user-123",
            )

            self.assertEqual(entry.operation, "playbook")
            self.assertEqual(entry.playbook_id, "playbook-001")
            self.assertEqual(entry.action, "load")
            self.assertEqual(entry.bullet_count, 50)


@pytest.mark.unit
class TestAuditMetrics(unittest.TestCase):
    """Test audit metrics aggregation."""

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        from ace.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_dir=tmpdir)

            # Log multiple operations
            logger.log_retrieval(query="q1", results_count=5, latency_ms=10.0)
            logger.log_retrieval(query="q2", results_count=3, latency_ms=20.0)
            logger.log_retrieval(query="q3", results_count=7, latency_ms=30.0)

            metrics = logger.get_metrics()

            self.assertEqual(metrics["total_queries"], 3)
            self.assertEqual(metrics["avg_latency_ms"], 20.0)
            self.assertEqual(metrics["avg_results_count"], 5.0)


if __name__ == "__main__":
    unittest.main()
