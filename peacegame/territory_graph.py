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


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def assign_territories_round_robin(
    agent_names: List[str],
    graph: Dict[str, Set[str]],
    positions: Dict[str, Coord],
    *,
    seed: int | None = None,
) -> Dict[str, Set[str]]:
    """Assign territories by farthest-first seeds, then round-robin adjacency growth."""
    rng = random.Random(seed)
    territories = list(graph.keys())
    if not territories or not agent_names:
        return {agent: set() for agent in agent_names}

    unassigned = set(territories)
    assigned: Dict[str, Set[str]] = {agent: set() for agent in agent_names}

    # Seed selection: farthest-first to spread agents apart.
    seeds: List[str] = []
    seeds.append(rng.choice(territories))
    unassigned.remove(seeds[0])
    for _ in range(1, min(len(agent_names), len(territories))):
        best = None
        best_dist = -1
        for terr in list(unassigned):
            pos = positions[terr]
            min_dist = min(manhattan(pos, positions[s]) for s in seeds)
            if min_dist > best_dist:
                best_dist = min_dist
                best = terr
        if best is None:
            break
        seeds.append(best)
        unassigned.remove(best)

    for agent, terr in zip(agent_names, seeds):
        assigned[agent].add(terr)

    # Round-robin growth by adjacency.
    stalled = 0
    idx = 0
    while unassigned and stalled < len(agent_names):
        agent = agent_names[idx % len(agent_names)]
        idx += 1

        frontier = set()
        for terr in assigned[agent]:
            frontier |= graph.get(terr, set())
        frontier &= unassigned

        if not frontier:
            stalled += 1
            continue

        stalled = 0
        # Prefer territories that open more future options.
        best = None
        best_score = -1
        for terr in frontier:
            score = len(graph.get(terr, set()) & unassigned)
            if score > best_score:
                best_score = score
                best = terr
        if best is None:
            best = rng.choice(list(frontier))
        assigned[agent].add(best)
        unassigned.remove(best)

    return assigned


def is_legal_cession(
    territory: str,
    recipient: str,
    *,
    agent_territories: Dict[str, Set[str]],
    graph: Dict[str, Set[str]],
) -> bool:
    neighbors = graph.get(territory)
    if not neighbors:
        return False
    recipient_terrs = agent_territories.get(recipient, set())
    return any(n in recipient_terrs for n in neighbors)


def compute_legal_cession_lists(
    agent_territories: Dict[str, Set[str]],
    graph: Dict[str, Set[str]],
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    inbound: Dict[str, Dict[str, List[str]]] = {a: {} for a in agent_territories.keys()}
    outbound: Dict[str, Dict[str, List[str]]] = {a: {} for a in agent_territories.keys()}

    for giver, terrs in agent_territories.items():
        for receiver in agent_territories.keys():
            if receiver == giver:
                continue
            legal = [
                terr
                for terr in terrs
                if is_legal_cession(
                    terr,
                    receiver,
                    agent_territories=agent_territories,
                    graph=graph,
                )
            ]
            if legal:
                outbound[giver][receiver] = sorted(legal)
                inbound[receiver][giver] = sorted(legal)

    return inbound, outbound

def build_territory_graph(
    names: Iterable[str],
    *,
    target_avg_degree: float = 2.5,
    seed: int | None = None,
    elongation_bias: float = 4.0,
    shape_mode: str = "elbow",
) -> Tuple[Dict[str, Set[str]], Dict[str, Coord]]:
    """Lay out territories on a grid, biasing for multi-adjacency to reach target avg degree."""
    names_list = [name for name in names if name]
    if not names_list:
        return {}, {}

    rng = random.Random(seed)
    rng.shuffle(names_list)

    positions: Dict[str, Coord] = {names_list[0]: (0, 0)}
    occupied = {positions[names_list[0]]}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    primary_dir = rng.choice(directions)
    if shape_mode == "zigzag":
        dir_sequence = [primary_dir, (-primary_dir[0], -primary_dir[1])]
        turn_steps = [0.4, 0.7]
    elif shape_mode == "arch":
        secondary_dir = rng.choice([d for d in directions if d != primary_dir])
        dir_sequence = [primary_dir, secondary_dir, (-primary_dir[0], -primary_dir[1])]
        turn_steps = [0.35, 0.7]
    else:
        secondary_dir = rng.choice([d for d in directions if d != primary_dir])
        dir_sequence = [primary_dir, secondary_dir]
        turn_steps = [0.45]

    for idx, name in enumerate(names_list[1:], start=1):
        candidates: Dict[Coord, int] = {}
        for coord in occupied:
            for adj in _adjacent(coord):
                if adj in occupied:
                    continue
                candidates[adj] = candidates.get(adj, 0) + 1

        # Prefer placements that touch more existing squares to reduce diameter.
        weighted: List[Coord] = []
        min_x = min(c[0] for c in occupied)
        max_x = max(c[0] for c in occupied)
        min_y = min(c[1] for c in occupied)
        max_y = max(c[1] for c in occupied)
        active_dirs = [dir_sequence[0]]
        if len(turn_steps) == 2 and idx >= int(len(names_list) * turn_steps[1]):
            active_dirs = dir_sequence[:3]
        elif idx >= int(len(names_list) * turn_steps[0]):
            active_dirs = dir_sequence[:2]
        for coord, neighbor_count in candidates.items():
            weight = max(1, neighbor_count * neighbor_count)
            # Bias elongation by expanding along preferred directions.
            for direction in active_dirs:
                dx = coord[0] - max_x if direction == (1, 0) else coord[0] - min_x if direction == (-1, 0) else 0
                dy = coord[1] - max_y if direction == (0, 1) else coord[1] - min_y if direction == (0, -1) else 0
                expands = (dx > 0) or (dy > 0)
                if expands:
                    weight = int(weight * (1 + elongation_bias))
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


def load_territory_names(path: Path) -> List[str]:
    return _load_names(path)


def _render_layout_png(
    positions: Dict[str, Coord],
    *,
    out_path: Path,
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if not positions:
        return

    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    jitter = 0.1
    edge_cache: Dict[Tuple[Coord, Coord], List[Coord]] = {}
    vertex_cache: Dict[Coord, Coord] = {}

    def _edge_key(a: Coord, b: Coord) -> Tuple[Coord, Coord]:
        return (a, b) if a <= b else (b, a)

    def _vertex(coord: Coord) -> Coord:
        if coord not in vertex_cache:
            rng = random.Random(f"v:{coord}")
            vertex_cache[coord] = (
                coord[0] + rng.uniform(-jitter, jitter),
                coord[1] + rng.uniform(-jitter, jitter),
            )
        return vertex_cache[coord]

    def _edge_points(start: Coord, end: Coord) -> List[Coord]:
        key = _edge_key(start, end)
        if key not in edge_cache:
            rng = random.Random(f"{key[0]}:{key[1]}")
            x1, y1 = _vertex(key[0])
            x2, y2 = _vertex(key[1])
            if x1 == x2:
                # Vertical edge: jitter x
                mid1 = (x1 + rng.uniform(-jitter, jitter), y1 + (y2 - y1) * 0.33)
                mid2 = (x1 + rng.uniform(-jitter, jitter), y1 + (y2 - y1) * 0.66)
            else:
                # Horizontal edge: jitter y
                mid1 = (x1 + (x2 - x1) * 0.33, y1 + rng.uniform(-jitter, jitter))
                mid2 = (x1 + (x2 - x1) * 0.66, y1 + rng.uniform(-jitter, jitter))
            edge_cache[key] = [(x1, y1), mid1, mid2, (x2, y2)]
        pts = edge_cache[key]
        if start == key[0] and end == key[1]:
            return pts
        return list(reversed(pts))

    for name, (x, y) in positions.items():
        bottom = _edge_points((x, y), (x + 1, y))
        right = _edge_points((x + 1, y), (x + 1, y + 1))
        top = _edge_points((x + 1, y + 1), (x, y + 1))
        left = _edge_points((x, y + 1), (x, y))
        outline = bottom[:-1] + right[:-1] + top[:-1] + left[:-1]
        poly = Polygon(
            outline,
            closed=True,
            facecolor="#f2e9e4",
            edgecolor="#5a4f4b",
            linewidth=1.2,
        )
        ax.add_patch(poly)
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
    parser.add_argument("--count", type=int, default=None, help="Number of territories to generate")
    parser.add_argument(
        "--shape",
        type=str,
        default="elbow",
        choices=["elbow", "arch", "zigzag"],
        help="Layout bias: elbow (L), arch, or zigzag",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="names/territories.txt",
        help="Path to territories list",
    )
    args = parser.parse_args()

    names_path = Path(args.names)
    names = _load_names(names_path)
    if args.count is not None:
        if len(names) < args.count:
            names.extend([f"Terr{i}" for i in range(len(names), args.count)])
        names = names[: args.count]
    if not names:
        names = [f"Terr{i}" for i in range(20)]

    _, positions = build_territory_graph(
        names,
        seed=args.seed,
        shape_mode=args.shape,
    )
    if args.show:
        out_path = Path("visualizations") / "territory_layout.png"
        _render_layout_png(positions, out_path=out_path)
        print(f"Saved layout PNG to {out_path}")


if __name__ == "__main__":
    main()
