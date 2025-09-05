#!/usr/bin/env python3
"""
Rename files to random names (words or uuid), preserving extensions.
Creates a reversible mapping (CSV + JSON) in .rename_map/.

Usage:
  python scripts/rename_files.py --root ./data --strategy words --words 3 --dry-run
  python scripts/rename_files.py --root ./data --strategy words --words 2
  python scripts/rename_files.py --root ./data --strategy uuid
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

DEFAULT_EXCLUDES = {
    ".git", ".venv", ".rename_map", "node_modules", "__pycache__", ".DS_Store"
}

# A small fallback wordlist if /usr/share/dict/words doesn't exist.
FALLBACK_WORDS = [
    "oak","river","amber","delta","pixel","crystal","ember","hollow","lumen","marble",
    "quartz","violet","cerulean","willow","cobalt","onyx","cinder","dawn","silk","prism",
    "hazel","cedar","breeze","ridge","meadow","harbor","echo","sable","arbor","nova",
    "orbit","lilac","comet","amber","coral","granite","sage","tundra","obsidian","flare",
    "mint","pearl","drift","maple","bluff","storm","fable","ember","moss","ember"
]

def load_wordlist(limit=5000):
    """Try to load a big system word list; fallback to the small list above."""
    candidates = [
        "/usr/share/dict/words",
        "/usr/dict/words",
    ]
    words = []
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        w = line.strip()
                        # keep only alpha words, lowercase, reasonable length
                        if 2 <= len(w) <= 14 and w.isalpha():
                            words.append(w.lower())
                break
            except Exception:
                pass
    if not words:
        words = FALLBACK_WORDS.copy()
    random.shuffle(words)
    if limit and len(words) > limit:
        words = words[:limit]
    return list(dict.fromkeys(words))  # dedupe preserving order

def random_words_name(words_list, count=3):
    parts = random.choices(words_list, k=count)
    return "-".join(parts)

def uuid_name():
    return str(uuid.uuid4())

def safe_new_name(base_dir: Path, stem: str, ext: str, used: set) -> Path:
    """Ensure the new name doesn't collide in the directory; retry with suffix."""
    attempt = 0
    candidate = base_dir / f"{stem}{ext}"
    while candidate.exists() or str(candidate) in used:
        attempt += 1
        candidate = base_dir / f"{stem}-{attempt}{ext}"
    used.add(str(candidate))
    return candidate

def sha1_of_path(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:8]

def should_exclude(path: Path, excludes: set) -> bool:
    # Exclude if any parent folder matches an exclude token or hidden files
    parts = set(p.name for p in path.parents)
    if parts & excludes:
        return True
    if path.name in excludes:
        return True
    # Skip mapping dir and hidden/system files
    if any(seg.startswith(".") and seg not in {".", ".."} for seg in path.parts if seg):
        # Allow .rename_map through explicit exclude anyway
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Rename files to random names with reversible mapping.")
    parser.add_argument("--root", type=str, required=True, help="Root folder to process (e.g., ./data)")
    parser.add_argument("--strategy", choices=["words", "uuid"], default="words", help="Name generation strategy.")
    parser.add_argument("--words", type=int, default=3, help="If strategy=words, how many words to combine.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without renaming.")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files (default: skip).")
    parser.add_argument("--also-rename-dirs", action="store_true", help="(Advanced) Also rename directories.")
    parser.add_argument("--confirm", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--ext-lower", action="store_true", help="Force extensions to lowercase.")
    parser.add_argument("--extra-exclude", action="append", default=[], help="Extra names to exclude (repeatable).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root does not exist or is not a directory: {root}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    excludes = set(DEFAULT_EXCLUDES) | set(args.extra_exclude)
    # Allow processing .rename_map if user explicitly removes it from excludes
    excludes.add(".rename_map")

    # Prepare mapping dir
    map_dir = root.parent / ".rename_map"
    map_dir.mkdir(exist_ok=True)

    # Prepare name generator
    words_list = []
    if args.strategy == "words":
        words_list = load_wordlist()

    # Collect candidate files
    files = []
    dirs = []
    for cur_root, dirnames, filenames in os.walk(root):
        cur_path = Path(cur_root)

        # Respect excludes (skip entire subtrees early)
        if should_exclude(cur_path, excludes):
            # Prevent descending into excluded directories
            dirnames[:] = []
            continue

        # Directories (for optional renaming)
        for d in list(dirnames):
            dpath = cur_path / d
            if should_exclude(dpath, excludes):
                dirnames.remove(d)
            else:
                dirs.append(dpath)

        # Files
        for fn in filenames:
            fpath = cur_path / fn
            if not args.include_hidden and fn.startswith("."):
                continue
            if should_exclude(fpath, excludes):
                continue
            if fpath.is_file():
                files.append(fpath)

    if not files:
        print("No files found to rename.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to rename under: {root}")

    # Build proposed renames
    proposals = []
    used_targets = set()
    for f in files:
        ext = f.suffix or ""
        ext_use = ext.lower() if args.ext_lower else ext
        parent = f.parent

        if args.strategy == "words":
            # add a tiny hash tail to reduce collision & leak less info
            stem = f"{random_words_name(words_list, args.words)}-{sha1_of_path(f)}"
        else:
            stem = uuid_name()

        target = safe_new_name(parent, stem, ext_use, used_targets)
        proposals.append((f, target))

    # Print a summary
    rename_count = sum(1 for (src, dst) in proposals if src != dst)
    print(f"Planned renames: {rename_count}")
    if args.dry_run:
        for src, dst in proposals[:20]:
            print(f"DRY-RUN: {src}  ->  {dst}")
        if len(proposals) > 20:
            print(f"... and {len(proposals)-20} more")

    # Confirm
    if not args.dry_run and not args.confirm:
        ans = input("Proceed with renaming? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            sys.exit(0)

    # Execute + write mapping
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = map_dir / f"rename-map-{ts}.csv"
    json_path = map_dir / f"rename-map-{ts}.json"

    mapping_rows = []
    for src, dst in proposals:
        row = {
            "original_path": str(src),
            "new_path": str(dst),
            "original_name": src.name,
            "new_name": dst.name,
            "original_ext": src.suffix,
            "new_ext": dst.suffix,
        }
        mapping_rows.append(row)

    if not args.dry_run:
        # Perform renames
        for src, dst in proposals:
            if src == dst:
                continue
            try:
                src.rename(dst)
            except Exception as e:
                print(f"ERROR renaming {src} -> {dst}: {e}")

        # Save mapping (CSV + JSON)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
                writer.writeheader()
                writer.writerows(mapping_rows)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mapping_rows, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ERROR writing mapping files: {e}")
            sys.exit(1)

        print(f"✔ Renamed {rename_count} file(s). Mapping saved:")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
    else:
        print("Dry-run complete. No changes made.")

if __name__ == "__main__":
    main()
