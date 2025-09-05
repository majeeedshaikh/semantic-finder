#!/usr/bin/env python3
"""
Restore filenames from a mapping CSV/JSON created by rename_files.py.

Usage:
  python scripts/restore_files.py --map .rename_map/rename-map-20250905-031800.csv
  python scripts/restore_files.py --map .rename_map/rename-map-20250905-031800.json
"""

import argparse
import csv
import json
from pathlib import Path

def load_mapping(map_path: Path):
    if map_path.suffix.lower() == ".json":
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif map_path.suffix.lower() == ".csv":
        rows = []
        with open(map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows
    else:
        raise ValueError("Unsupported mapping format. Use .csv or .json")

def main():
    parser = argparse.ArgumentParser(description="Restore filenames from mapping.")
    parser.add_argument("--map", required=True, help="Path to mapping file (.csv or .json)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    parser.add_argument("--confirm", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args()

    map_path = Path(args.map).resolve()
    if not map_path.exists():
        print(f"Mapping file not found: {map_path}")
        return

    rows = load_mapping(map_path)
    if not rows:
        print("Mapping is empty.")
        return

    # Plan restores
    plans = []
    for row in rows:
        new_path = Path(row["new_path"])
        orig_path = Path(row["original_path"])
        plans.append((new_path, orig_path))

    print(f"Found {len(plans)} rename entries in mapping.")

    # Show a preview
    for src, dst in plans[:20]:
        print(f"{src}  ->  {dst}")
    if len(plans) > 20:
        print(f"... and {len(plans)-20} more")

    if not args.dry_run and not args.confirm:
        ans = input("Proceed with restore? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return

    # Execute
    if not args.dry_run:
        restored = 0
        for src, dst in plans:
            try:
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dst)
                    restored += 1
                else:
                    print(f"SKIP: missing current file {src}")
            except Exception as e:
                print(f"ERROR restoring {src} -> {dst}: {e}")
        print(f"✔ Restored {restored} file(s).")
    else:
        print("Dry-run complete. No changes made.")

if __name__ == "__main__":
    main()
