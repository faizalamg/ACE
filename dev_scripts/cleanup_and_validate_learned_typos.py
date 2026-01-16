#!/usr/bin/env python3
"""Clean up learned_typos.json by removing bad corrections and cycle mappings.

This script:
1. Loads the current learned_typos.json
2. Removes low-similarity corrections (< 0.70)
3. Removes cycle mappings (A->B and B->A)
4. Backs up the original file
5. Saves the cleaned version
"""

import json
import shutil
import difflib
from pathlib import Path
from datetime import datetime


def cleanup_learned_typos(
    typos_path: Path,
    min_similarity: float = 0.70,
    backup: bool = True
) -> dict:
    """Clean up learned typos JSON file.

    Args:
        typos_path: Path to learned_typos.json
        min_similarity: Minimum similarity threshold (default 0.70)
        backup: Whether to backup the original file (default True)

    Returns:
        Dictionary with cleanup statistics
    """
    if not typos_path.exists():
        return {
            "success": False,
            "error": f"File not found: {typos_path}"
        }

    # Load original data
    with open(typos_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        original_typos = data.get('typos', {})

    original_count = len(original_typos)

    # Backup original file
    if backup:
        backup_path = typos_path.with_suffix(
            f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        shutil.copy2(typos_path, backup_path)
        print(f"✓ Backed up original to: {backup_path.name}")

    # Clean up low-similarity corrections
    cleaned = {}
    low_similarity_removed = []
    for typo, correction in original_typos.items():
        ratio = difflib.SequenceMatcher(None, typo, correction).ratio()
        if ratio >= min_similarity:
            cleaned[typo] = correction
        else:
            low_similarity_removed.append((typo, correction, ratio))

    # Remove cycle mappings
    cycles = set()
    for typo, correction in cleaned.items():
        if correction in cleaned and cleaned[correction] == typo:
            cycles.add(typo)
            cycles.add(correction)

    final_typos = {k: v for k, v in cleaned.items() if k not in cycles}

    # Save cleaned version
    data['typos'] = final_typos
    data['updated_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    data['count'] = len(final_typos)
    data['cleanup_info'] = {
        'original_count': original_count,
        'removed_low_similarity': len(low_similarity_removed),
        'removed_cycles': len(cycles),
        'final_count': len(final_typos)
    }

    with open(typos_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"LEARNED_TYPOS.JSON CLEANUP SUMMARY")
    print(f"{'='*70}")
    print(f"Original entries: {original_count}")
    print(f"Removed low-similarity (< {min_similarity}): {len(low_similarity_removed)}")
    print(f"Removed cycle mappings: {len(cycles)}")
    print(f"Final entries: {len(final_typos)}")
    print(f"Total removed: {original_count - len(final_typos)}")
    print(f"{'='*70}\n")

    # Show removed entries
    if low_similarity_removed:
        print(f"Low-similarity corrections removed:")
        for typo, correction, ratio in sorted(
            low_similarity_removed,
            key=lambda x: x[2]
        ):
            print(f"  {ratio:.2f} - '{typo}' -> '{correction}'")
        print()

    if cycles:
        print(f"Cycle mappings removed: {sorted(cycles)}")
        print()

    return {
        "success": True,
        "original_count": original_count,
        "low_similarity_removed": len(low_similarity_removed),
        "cycles_removed": len(cycles),
        "final_count": len(final_typos),
        "total_removed": original_count - len(final_typos)
    }


def main():
    """Main entry point."""
    typos_path = Path(__file__).parent / "tenant_data" / "learned_typos.json"

    print("="*70)
    print("CLEANING UP learned_typos.json")
    print("="*70)
    print(f"File: {typos_path}")
    print()

    result = cleanup_learned_typos(
        typos_path=typos_path,
        min_similarity=0.70,
        backup=True
    )

    if result['success']:
        print("✓ Cleanup completed successfully!")
        return 0
    else:
        print(f"✗ Cleanup failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
