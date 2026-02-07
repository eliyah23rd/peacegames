#!/usr/bin/env python3
"""Ensure a territory adjacency list is symmetric.

For every edge A -> B listed in the JSON, the script guarantees B -> A.
If a neighbor key is missing, it is created with the reverse edge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def load_json(path: Path) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to read JSON from {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"Expected top-level object in {path}.")
    return data


def validate_list(name: str, adj: object) -> list[str]:
    if not isinstance(adj, list):
        raise SystemExit(f"Adjacency for '{name}' must be a list.")
    for item in adj:
        if not isinstance(item, str):
            raise SystemExit(f"Adjacency for '{name}' contains non-string: {item!r}")
    return adj


def ensure_symmetric(data: dict[str, list[str]]) -> tuple[dict[str, list[str]], int, int]:
    added_edges = 0
    added_keys = 0
    missing_keys: dict[str, list[str]] = {}

    for name, adj in data.items():
        adj_list = validate_list(name, adj)
        for neighbor in adj_list:
            if neighbor == name:
                continue
            if neighbor in data:
                target = data[neighbor]
            else:
                target = missing_keys.setdefault(neighbor, [])
                if len(target) == 0 and neighbor not in data:
                    # Count each missing key once.
                    added_keys += 1
            if name not in target:
                target.append(name)
                added_edges += 1

    if missing_keys:
        for key, value in missing_keys.items():
            if key not in data:
                data[key] = value

    return data, added_edges, added_keys


def write_json(path: Path, data: dict[str, list[str]]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure adjacency lists are symmetric.")
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/territory_contiguity.json",
        help="Path to adjacency JSON (default: tests/territory_contiguity.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report fixes without writing changes.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = load_json(path)
    data, added_edges, added_keys = ensure_symmetric(data)

    if added_edges == 0 and added_keys == 0:
        print("No changes needed.")
        return

    print(f"Added {added_edges} reverse edges; added {added_keys} missing keys.")
    if args.dry_run:
        print("Dry run: no file changes written.")
        return

    write_json(path, data)
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
