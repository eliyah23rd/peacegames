#!/usr/bin/env python3
"""Report symmetry issues in a territory adjacency list.

For every edge A -> B listed in the JSON, the script verifies B -> A exists.
It does not modify files; it only reports problems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def find_symmetry_issues(
    data: dict[str, list[str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    missing_keys: list[tuple[str, str]] = []
    missing_reverse: list[tuple[str, str]] = []

    for name, adj in data.items():
        adj_list = validate_list(name, adj)
        for neighbor in adj_list:
            if neighbor == name:
                continue
            if neighbor not in data:
                missing_keys.append((name, neighbor))
                continue
            if name not in data[neighbor]:
                missing_reverse.append((name, neighbor))

    return missing_keys, missing_reverse


def main() -> None:
    parser = argparse.ArgumentParser(description="Report adjacency symmetry issues.")
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/territory_contiguity.json",
        help="Path to adjacency JSON (default: tests/territory_contiguity.json)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = load_json(path)
    missing_keys, missing_reverse = find_symmetry_issues(data)

    if not missing_keys and not missing_reverse:
        print("No symmetry issues found.")
        return

    for name, neighbor in missing_keys:
        print(f"Missing key: '{neighbor}' (referenced by '{name}')")
    for name, neighbor in missing_reverse:
        print(f"Missing reverse edge: '{neighbor}' lacks '{name}'")

    raise SystemExit(
        f"Found {len(missing_keys)} missing keys and {len(missing_reverse)} missing reverse edges."
    )


if __name__ == "__main__":
    main()
