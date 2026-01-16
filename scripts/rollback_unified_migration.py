#!/usr/bin/env python3
"""
Rollback script for ACE unified memory migration.

This script provides safety mechanisms to revert the unified memory migration
if issues are detected. It can:
1. Check if rollback is feasible (old collections still exist)
2. Show migration status (collection counts, timestamps)
3. Delete unified collection and restore to pre-migration state

Usage:
    python scripts/rollback_unified_migration.py --check     # Check rollback feasibility
    python scripts/rollback_unified_migration.py --rollback  # Execute rollback
    python scripts/rollback_unified_migration.py --status    # Show migration status

Safety:
- Confirms before destructive operations
- Logs all operations with timestamps
- Returns non-zero exit code on failure
- Verifies old collections exist before rollback
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("ERROR: httpx not available. Install with: pip install httpx")
    sys.exit(1)


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ace.config import get_qdrant_config

# Configuration from centralized config
_qdrant_config = get_qdrant_config()
QDRANT_URL = _qdrant_config.url
OLD_COLLECTION = _qdrant_config.memories_collection  # ace_memories_hybrid (canonical)
NEW_COLLECTION = "ace_unified"  # Deprecated collection to rollback/delete
TIMEOUT = 60.0


def log(message: str, level: str = "INFO"):
    """Log message with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def get_collection_info(
    collection_name: str,
    qdrant_url: str = QDRANT_URL
) -> Optional[Dict]:
    """
    Get collection information from Qdrant.

    Args:
        collection_name: Name of the collection
        qdrant_url: Qdrant server URL

    Returns:
        Collection info dict or None if collection doesn't exist
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(f"{qdrant_url}/collections/{collection_name}")

            if resp.status_code == 404:
                return None
            elif resp.status_code != 200:
                log(f"Error getting collection {collection_name}: HTTP {resp.status_code}", "ERROR")
                return None

            data = resp.json().get("result", {})
            return {
                "name": collection_name,
                "points_count": data.get("points_count", 0),
                "vectors_count": data.get("vectors_count", 0),
                "indexed_vectors_count": data.get("indexed_vectors_count", 0),
                "status": data.get("status", "unknown"),
            }
    except Exception as e:
        log(f"Exception getting collection {collection_name}: {e}", "ERROR")
        return None


def delete_collection(
    collection_name: str,
    qdrant_url: str = QDRANT_URL
) -> bool:
    """
    Delete a Qdrant collection.

    Args:
        collection_name: Name of collection to delete
        qdrant_url: Qdrant server URL

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.delete(f"{qdrant_url}/collections/{collection_name}")

            if resp.status_code in [200, 404]:
                log(f"Collection {collection_name} deleted successfully")
                return True
            else:
                log(f"Failed to delete collection {collection_name}: HTTP {resp.status_code}", "ERROR")
                return False
    except Exception as e:
        log(f"Exception deleting collection {collection_name}: {e}", "ERROR")
        return False


def check_rollback_feasibility(qdrant_url: str = QDRANT_URL) -> bool:
    """
    Check if rollback is feasible.

    Rollback is feasible if:
    1. Old collection exists
    2. Qdrant is accessible

    Args:
        qdrant_url: Qdrant server URL

    Returns:
        True if rollback is feasible, False otherwise
    """
    log("Checking rollback feasibility...")

    # Check old collection
    old_info = get_collection_info(OLD_COLLECTION, qdrant_url)
    if not old_info:
        log(f"Old collection '{OLD_COLLECTION}' does NOT exist", "ERROR")
        log("Cannot rollback - old data is lost", "ERROR")
        return False

    log(f"Old collection '{OLD_COLLECTION}' exists with {old_info['points_count']} points")

    # Check new collection (optional - can rollback even if it doesn't exist)
    new_info = get_collection_info(NEW_COLLECTION, qdrant_url)
    if new_info:
        log(f"New collection '{NEW_COLLECTION}' exists with {new_info['points_count']} points")
    else:
        log(f"New collection '{NEW_COLLECTION}' does NOT exist (nothing to rollback)")

    log("Rollback is FEASIBLE - old collection is intact", "SUCCESS")
    return True


def show_status(qdrant_url: str = QDRANT_URL) -> int:
    """
    Show migration status.

    Args:
        qdrant_url: Qdrant server URL

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("=" * 70)
    print("ACE Unified Memory Migration Status")
    print("=" * 70)
    print()

    # Old collection status
    old_info = get_collection_info(OLD_COLLECTION, qdrant_url)
    if old_info:
        print(f"Old Collection: {OLD_COLLECTION}")
        print(f"  Status: {old_info['status']}")
        print(f"  Points: {old_info['points_count']}")
        print(f"  Vectors: {old_info['vectors_count']}")
    else:
        print(f"Old Collection: {OLD_COLLECTION}")
        print(f"  Status: DOES NOT EXIST (⚠️  Cannot rollback!)")

    print()

    # New collection status
    new_info = get_collection_info(NEW_COLLECTION, qdrant_url)
    if new_info:
        print(f"New Collection: {NEW_COLLECTION}")
        print(f"  Status: {new_info['status']}")
        print(f"  Points: {new_info['points_count']}")
        print(f"  Vectors: {new_info['vectors_count']}")
    else:
        print(f"New Collection: {NEW_COLLECTION}")
        print(f"  Status: DOES NOT EXIST (Migration not run or rolled back)")

    print()
    print("=" * 70)

    # Determine migration state
    if old_info and new_info:
        print("Migration State: COMPLETED (both collections exist)")
        print("Action: Can rollback to delete new collection")
    elif old_info and not new_info:
        print("Migration State: NOT RUN or ROLLED BACK")
        print("Action: No rollback needed")
    elif not old_info and new_info:
        print("Migration State: OLD DATA LOST (⚠️  Cannot rollback!)")
        print("Action: Rollback not possible - old collection deleted")
    else:
        print("Migration State: NO COLLECTIONS (Clean state)")
        print("Action: No rollback needed")

    print("=" * 70)

    return 0


def execute_rollback(qdrant_url: str = QDRANT_URL, confirm: bool = True) -> int:
    """
    Execute rollback by deleting unified collection.

    Args:
        qdrant_url: Qdrant server URL
        confirm: If True, ask for user confirmation

    Returns:
        Exit code (0 = success, 1 = error)
    """
    log("Starting rollback process...", "WARN")

    # Check feasibility first
    if not check_rollback_feasibility(qdrant_url):
        log("Rollback is NOT feasible - aborting", "ERROR")
        return 1

    # Get collection info
    old_info = get_collection_info(OLD_COLLECTION, qdrant_url)
    new_info = get_collection_info(NEW_COLLECTION, qdrant_url)

    if not new_info:
        log(f"New collection '{NEW_COLLECTION}' does not exist - nothing to rollback")
        return 0

    # Show what will be deleted
    print()
    print("=" * 70)
    print("ROLLBACK PLAN")
    print("=" * 70)
    print(f"Will DELETE: {NEW_COLLECTION} ({new_info['points_count']} points)")
    print(f"Will KEEP:   {OLD_COLLECTION} ({old_info['points_count']} points)")
    print("=" * 70)
    print()

    # Confirm if requested
    if confirm:
        response = input("Are you sure you want to rollback? This will DELETE the unified collection. (yes/no): ")
        if response.lower() != "yes":
            log("Rollback cancelled by user")
            return 0

    # Execute deletion
    log(f"Deleting unified collection '{NEW_COLLECTION}'...", "WARN")
    if not delete_collection(NEW_COLLECTION, qdrant_url):
        log("Failed to delete unified collection", "ERROR")
        return 1

    # Verify deletion
    verify_info = get_collection_info(NEW_COLLECTION, qdrant_url)
    if verify_info:
        log("Unified collection still exists after deletion attempt", "ERROR")
        return 1

    # Verify old collection still exists
    final_old_info = get_collection_info(OLD_COLLECTION, qdrant_url)
    if not final_old_info:
        log("Old collection was lost during rollback!", "ERROR")
        return 1

    print()
    print("=" * 70)
    print("ROLLBACK SUCCESSFUL")
    print("=" * 70)
    print(f"✓ Deleted unified collection: {NEW_COLLECTION}")
    print(f"✓ Preserved old collection: {OLD_COLLECTION} ({final_old_info['points_count']} points)")
    print()
    print("System restored to pre-migration state.")
    print("=" * 70)

    log("Rollback completed successfully", "SUCCESS")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Rollback ACE unified memory migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if rollback is possible
  python scripts/rollback_unified_migration.py --check

  # Show current migration status
  python scripts/rollback_unified_migration.py --status

  # Execute rollback (with confirmation)
  python scripts/rollback_unified_migration.py --rollback

  # Execute rollback without confirmation (dangerous!)
  python scripts/rollback_unified_migration.py --rollback --no-confirm
        """
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--check",
        action="store_true",
        help="Check if rollback is feasible"
    )
    action_group.add_argument(
        "--status",
        action="store_true",
        help="Show migration status"
    )
    action_group.add_argument(
        "--rollback",
        action="store_true",
        help="Execute rollback (delete unified collection)"
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (use with --rollback)"
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant server URL (default: {QDRANT_URL})"
    )

    args = parser.parse_args()

    # Execute requested action
    if args.check:
        feasible = check_rollback_feasibility(args.qdrant_url)
        return 0 if feasible else 1

    elif args.status:
        return show_status(args.qdrant_url)

    elif args.rollback:
        return execute_rollback(
            args.qdrant_url,
            confirm=not args.no_confirm
        )


if __name__ == "__main__":
    sys.exit(main())
