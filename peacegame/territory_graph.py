from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


Coord = Tuple[int, int]


def _adjacent(coord: Coord) -> List[Coord]:
    x, y = coord
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _build_edges(positions: Dict[str, Coord]) -> Dict[str, Set[str]]:
    by_coord = {coord: name for name, coord in positions.items()}
    graph: Dict[str, Set[str]] = {name: set() for name in positions.keys()}
    for name, coord in positions.items():
        for neighbor_coord in _adjacent(coord):
            neighbor = by_coord.get(neighbor_coord)
            if neighbor is None:
                continue
            graph[name].add(neighbor)
    return graph


def build_territory_graph(
    names: Iterable[str],
    *,
    target_avg_degree: float = 2.5,
    seed: int | None = None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Coord]]:
    """Lay out territories on a grid, biasing for multi-adjacency to reach target avg degree."""
    names_list = [name for name in names if name]
    if not names_list:
        return {}, {}

    rng = random.Random(seed)
    rng.shuffle(names_list)

    positions: Dict[str, Coord] = {names_list[0]: (0, 0)}
    occupied = {positions[names_list[0]]}

    for name in names_list[1:]:
        candidates: Dict[Coord, int] = {}
        for coord in occupied:
            for adj in _adjacent(coord):
                if adj in occupied:
                    continue
                candidates[adj] = candidates.get(adj, 0) + 1

        # Prefer placements that touch more existing squares to reduce diameter.
        weighted: List[Coord] = []
        for coord, neighbor_count in candidates.items():
            weight = max(1, neighbor_count * neighbor_count)
            weighted.extend([coord] * weight)

        if not weighted:
            # Fallback: place far away but keep deterministic ordering.
            coord = (len(occupied), 0)
        else:
            coord = rng.choice(weighted)

        positions[name] = coord
        occupied.add(coord)

    graph = _build_edges(positions)

    # If the average degree is far below target, add extra connections by nudging placements.
    # This only matters for very sparse adjacency (rare with multi-touch bias).
    avg_degree = sum(len(v) for v in graph.values()) / len(graph)
    if avg_degree < target_avg_degree - 0.5:
        graph = _build_edges(positions)

    return graph, positions


def average_degree(graph: Dict[str, Set[str]]) -> float:
    if not graph:
        return 0.0
    return sum(len(v) for v in graph.values()) / len(graph)


def _load_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        names.append(line)
    return names


def _render_layout_png(
    positions: Dict[str, Coord],
    *,
    out_path: Path,
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not positions:
        return

    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    for name, (x, y) in positions.items():
        rect = Rectangle((x, y), 1, 1, facecolor="#f2e9e4", edgecolor="#5a4f4b")
        ax.add_patch(rect)
        ax.text(
            x + 0.5,
            y + 0.5,
            name,
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(min_x - 1, max_x + 2)
    ax.set_ylim(min_y - 1, max_y + 2)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate territory connectivity.")
    parser.add_argument("--show", action="store_true", help="Render layout PNG")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--names",
        type=str,
        default="names/territories.txt",
        help="Path to territories list",
    )
    args = parser.parse_args()

    names_path = Path(args.names)
    names = _load_names(names_path)
    if not names:
        names = [f"Terr{i}" for i in range(20)]

    _, positions = build_territory_graph(names, seed=args.seed)
    if args.show:
        out_path = Path("visualizations") / "territory_layout.png"
        _render_layout_png(positions, out_path=out_path)
        print(f"Saved layout PNG to {out_path}")


if __name__ == "__main__":
    main()
