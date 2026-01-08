from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


Coord = Tuple[int, int]

PIE_ORDER = ["lost", "purchases", "upkeep", "damage", "welfare"]
PIE_COLORS = {
    "lost": "#b8b1a6",
    "purchases": "#f0a24b",
    "upkeep": "#d7c24b",
    "damage": "#d46a6a",
    "welfare": "#5db07e",
}


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
    return_capitals: bool = False,
) -> Dict[str, Set[str]] | tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Assign territories by farthest-first seeds, then round-robin adjacency growth."""
    rng = random.Random(seed)
    territories = sorted(graph.keys())
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
        for terr in sorted(unassigned):
            pos = positions[terr]
            min_dist = min(manhattan(pos, positions[s]) for s in seeds)
            if min_dist > best_dist or (min_dist == best_dist and (best is None or terr < best)):
                best_dist = min_dist
                best = terr
        if best is None:
            break
        seeds.append(best)
        unassigned.remove(best)

    capitals: Dict[str, str] = {}
    for agent, terr in zip(agent_names, seeds):
        assigned[agent].add(terr)
        capitals[agent] = terr

    # Round-robin growth by adjacency.
    stalled = 0
    idx = 0
    while unassigned and stalled < len(agent_names):
        agent = agent_names[idx % len(agent_names)]
        idx += 1

        frontier = set()
        for terr in sorted(assigned[agent]):
            frontier |= graph.get(terr, set())
        frontier &= unassigned

        if not frontier:
            stalled += 1
            continue

        stalled = 0
        # Prefer territories that open more future options.
        best = None
        best_score = -1
        for terr in sorted(frontier):
            score = len(graph.get(terr, set()) & unassigned)
            if score > best_score or (score == best_score and (best is None or terr < best)):
                best_score = score
                best = terr
        if best is None:
            best = rng.choice(sorted(frontier))
        assigned[agent].add(best)
        unassigned.remove(best)

    if return_capitals:
        return assigned, capitals
    return assigned


def is_legal_cession(
    territory: str,
    recipient: str,
    *,
    agent_territories: Dict[str, Set[str]],
    graph: Dict[str, Set[str]],
    giver: str | None = None,
    capitals: Dict[str, str] | None = None,
) -> bool:
    if giver is not None and capitals is not None:
        if capitals.get(giver) == territory:
            return False
    neighbors = graph.get(territory)
    if not neighbors:
        return False
    recipient_terrs = agent_territories.get(recipient, set())
    return any(n in recipient_terrs for n in neighbors)


def compute_legal_cession_lists(
    agent_territories: Dict[str, Set[str]],
    graph: Dict[str, Set[str]],
    *,
    capitals: Dict[str, str] | None = None,
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
                    giver=giver,
                    capitals=capitals,
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
    names_list = sorted([name for name in names if name])
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
        for coord in sorted(occupied):
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
        for coord, neighbor_count in sorted(candidates.items()):
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
    if not positions:
        return
    territory_names = sorted(positions.keys())
    owners = [None for _ in territory_names]
    colors = {}
    render_ownership_png(
        territory_names,
        positions,
        owners,
        colors,
        out_path=out_path,
    )


def render_ownership_png(
    territory_names: List[str],
    territory_positions: Dict[str, Coord],
    territory_owners: List[str | None],
    owner_colors: Dict[str, str],
    *,
    territory_resources: Dict[str, Dict[str, int]] | None = None,
    capital_pies: Dict[str, Dict[str, float]] | None = None,
    pie_colors: Dict[str, str] | None = None,
    out_path: Path | None = None,
) -> bytes:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import io
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Wedge
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    if not territory_names:
        return b""

    coords = [territory_positions[name] for name in territory_names]
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    jitter = 0.22
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
            if abs(x1 - x2) < 1e-6:
                # Vertical edge: curve outward in x
                mid1 = (x1 + rng.uniform(-jitter, jitter), y1 + (y2 - y1) * 0.33)
                mid2 = (x1 + rng.uniform(-jitter, jitter), y1 + (y2 - y1) * 0.66)
            else:
                # Horizontal edge: curve outward in y
                mid1 = (x1 + (x2 - x1) * 0.33, y1 + rng.uniform(-jitter, jitter))
                mid2 = (x1 + (x2 - x1) * 0.66, y1 + rng.uniform(-jitter, jitter))
            edge_cache[key] = [(x1, y1), mid1, mid2, (x2, y2)]
        pts = edge_cache[key]
        if start == key[0] and end == key[1]:
            return pts
        return list(reversed(pts))

    icon_cache: Dict[str, any] = {}
    icons_dir = Path(__file__).resolve().parent.parent / "icons"
    pie_colors = pie_colors or PIE_COLORS

    def _draw_pie(name: str, x: float, y: float) -> None:
        if not capital_pies:
            return
        slices = capital_pies.get(name)
        if not slices:
            return
        total = sum(max(float(value), 0.0) for value in slices.values())
        if total <= 0:
            return
        center = (x + 0.25, y + 0.25)
        radius = 0.22
        start = 90.0
        for key in PIE_ORDER:
            value = max(float(slices.get(key, 0.0)), 0.0)
            if value <= 0:
                continue
            angle = 360.0 * (value / total)
            wedge = Wedge(
                center,
                radius,
                start,
                start + angle,
                facecolor=pie_colors.get(key, "#cccccc"),
                edgecolor="#4a423c",
                linewidth=0.4,
                zorder=6,
            )
            ax.add_patch(wedge)
            start += angle
        ax.add_patch(
            Circle(
                center,
                radius,
                fill=False,
                edgecolor="#4a423c",
                linewidth=0.5,
                zorder=6,
            )
        )

    def _draw_resource_icons(name: str, x: float, y: float) -> None:
        if not territory_resources:
            return
        res = territory_resources.get(name, {})
        if not res:
            return
        size = 0.08
        spacing = 0.14
        order = ["energy", "minerals", "food"]
        start_x = x + 0.18
        start_y = y + 0.78
        icons = []
        for rtype in order:
            count = int(res.get(rtype, 0))
            if count <= 0:
                continue
            icons.extend([rtype] * count)
        for idx_icon, rtype in enumerate(icons):
            row = idx_icon // 3
            col = idx_icon % 3
            cx = start_x + col * spacing
            cy = start_y - row * spacing
            icon_path = icons_dir / f"{rtype}.png"
            if icon_path.is_file():
                if rtype not in icon_cache:
                    icon_cache[rtype] = mpimg.imread(icon_path)
                img = icon_cache[rtype]
                ax.imshow(
                    img,
                    extent=(cx - size, cx + size, cy - size, cy + size),
                    interpolation="lanczos",
                    zorder=5,
                )
            else:
                # Fallback to a tiny triangle so the map still renders.
                fallback = [
                    (cx, cy + size),
                    (cx - size, cy - size),
                    (cx + size, cy - size),
                ]
                ax.add_patch(
                    Polygon(
                        fallback,
                        closed=True,
                        facecolor="#dddddd",
                        edgecolor="#4a423c",
                        linewidth=0.6,
                        zorder=5,
                    )
                )

    for idx, name in enumerate(territory_names):
        x, y = territory_positions[name]
        edges = [
            _edge_points((x, y), (x + 1, y)),
            _edge_points((x + 1, y), (x + 1, y + 1)),
            _edge_points((x + 1, y + 1), (x, y + 1)),
            _edge_points((x, y + 1), (x, y)),
        ]

        vertices: List[Coord] = []
        codes: List[int] = []
        start = edges[0][0]
        vertices.append(start)
        codes.append(MplPath.MOVETO)
        for edge in edges:
            vertices.extend(edge[1:])
            codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        vertices.append(start)
        codes.append(MplPath.CLOSEPOLY)
        path = MplPath(vertices, codes)
        owner = territory_owners[idx] if idx < len(territory_owners) else None
        color = owner_colors.get(owner, "#e5e1dc")
        patch = PathPatch(
            path,
            facecolor=color,
            edgecolor="#5a4f4b",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        _draw_resource_icons(name, x, y)
        _draw_pie(name, x, y)
        if len(territory_names) <= 40:
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

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    data = buf.getvalue()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return data


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
