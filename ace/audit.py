"""Enterprise audit logging for ACE operations.

Provides comprehensive logging of:
- Retrieval operations (queries, latency, results)
- Index operations (bullet creation, updates)
- Playbook operations (loading, saving)

Logs are written to daily JSONL files for efficient storage and analysis.
"""

import csv
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AuditEntry:
    """Single audit log entry."""

    id: str
    timestamp: str
    operation: str  # "retrieval", "index", "playbook"
    query: Optional[str] = None
    results_count: Optional[int] = None
    latency_ms: Optional[float] = None
    user_id: Optional[str] = None
    result_ids: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    filters: Optional[Dict] = None
    bullet_id: Optional[str] = None
    action: Optional[str] = None
    playbook_id: Optional[str] = None
    bullet_count: Optional[int] = None


class AuditLogger:
    """Logger for ACE operations with JSONL persistence."""

    def __init__(self, log_dir: str):
        """Initialize logger with log directory.

        Args:
            log_dir: Directory path for log files
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to daily JSONL file.

        Args:
            entry: Audit entry to write
        """
        # Daily log file format: YYYY-MM-DD.jsonl
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self._log_dir / f"{today}.jsonl"

        # Append entry as JSON line
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_retrieval(
        self,
        query: str,
        results_count: int,
        latency_ms: float,
        user_id: Optional[str] = None,
        result_ids: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        filters: Optional[Dict] = None,
    ) -> AuditEntry:
        """Log a retrieval operation.

        Args:
            query: Search query
            results_count: Number of results returned
            latency_ms: Query latency in milliseconds
            user_id: Optional user identifier
            result_ids: Optional list of result bullet IDs
            scores: Optional list of result scores
            filters: Optional filter parameters

        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            operation="retrieval",
            query=query,
            results_count=results_count,
            latency_ms=latency_ms,
            user_id=user_id,
            result_ids=result_ids,
            scores=scores,
            filters=filters,
        )
        self._write_entry(entry)
        return entry

    def log_index(
        self,
        bullet_id: str,
        action: str,
        user_id: Optional[str] = None,
    ) -> AuditEntry:
        """Log an index operation.

        Args:
            bullet_id: Bullet identifier
            action: Operation action (e.g., "create", "update", "delete")
            user_id: Optional user identifier

        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            operation="index",
            bullet_id=bullet_id,
            action=action,
            user_id=user_id,
        )
        self._write_entry(entry)
        return entry

    def log_playbook(
        self,
        playbook_id: str,
        action: str,
        bullet_count: int,
        user_id: Optional[str] = None,
    ) -> AuditEntry:
        """Log a playbook operation.

        Args:
            playbook_id: Playbook identifier
            action: Operation action (e.g., "load", "save")
            bullet_count: Number of bullets in playbook
            user_id: Optional user identifier

        Returns:
            Created audit entry
        """
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            operation="playbook",
            playbook_id=playbook_id,
            action=action,
            bullet_count=bullet_count,
            user_id=user_id,
        )
        self._write_entry(entry)
        return entry

    def _read_all_entries(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict]:
        """Read all entries from JSONL files.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            List of entry dictionaries
        """
        entries = []

        for log_file in sorted(self._log_dir.glob("*.jsonl")):
            # Check date range if specified
            if start_date or end_date:
                file_date = log_file.stem  # YYYY-MM-DD

                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue

            # Read JSONL file
            with open(log_file, "r") as f:
                for line in f:
                    entries.append(json.loads(line))

        return entries

    def export_json(
        self,
        path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Export audit logs to JSON file.

        Args:
            path: Output file path
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        """
        entries = self._read_all_entries(start_date, end_date)

        with open(path, "w") as f:
            json.dump(entries, f, indent=2)

    def export_csv(
        self,
        path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Export audit logs to CSV file.

        Args:
            path: Output file path
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        """
        entries = self._read_all_entries(start_date, end_date)

        if not entries:
            return

        # Get all field names from dataclass
        fieldnames = list(asdict(AuditEntry(**entries[0])).keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)

    def get_metrics(self) -> Dict:
        """Get aggregated metrics summary.

        Returns:
            Dictionary with:
            - total_queries: Total number of retrieval operations
            - avg_latency_ms: Average query latency
            - avg_results_count: Average number of results
        """
        entries = self._read_all_entries()

        # Filter to retrieval operations only
        retrievals = [e for e in entries if e.get("operation") == "retrieval"]

        if not retrievals:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "avg_results_count": 0.0,
            }

        total_queries = len(retrievals)
        avg_latency = sum(e["latency_ms"] for e in retrievals) / total_queries
        avg_results = sum(e["results_count"] for e in retrievals) / total_queries

        return {
            "total_queries": total_queries,
            "avg_latency_ms": avg_latency,
            "avg_results_count": avg_results,
        }
