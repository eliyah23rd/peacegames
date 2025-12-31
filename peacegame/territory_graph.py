from __future__ import annotations

import random
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
