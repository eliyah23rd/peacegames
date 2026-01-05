from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from peacegame.territory_graph import render_ownership_png


def _find_latest_round_data() -> Path | None:
    data_dir = Path(__file__).resolve().parent.parent / "round_data"
    if not data_dir.exists():
        return None
    files = sorted(data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> int:
    data_path = _find_latest_round_data()
    if data_path is None:
        print("No round_data files found.")
        return 1

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    territory_names = payload.get("territory_names", [])
    territory_positions = payload.get("territory_positions", {})
    territory_owners = payload.get("territory_owners", [])
    agents = payload.get("agents", [])
    territory_resources = payload.get("territory_resources", {})

    if not territory_names or not territory_owners:
        print(f"Missing territory data in {data_path.name}")
        return 1

    positions = {
        name: tuple(territory_positions.get(name, (0, 0)))
        for name in territory_names
    }
    owners = territory_owners[0]
    palette = [
        "#34a39a",
        "#e98467",
        "#5f7a75",
        "#f4b073",
        "#93bd8a",
        "#c995a7",
        "#6f879e",
        "#f7c372",
        "#5aa39a",
        "#f58a4b",
    ]
    owner_colors = {
        agent: palette[idx % len(palette)] for idx, agent in enumerate(agents)
    }

    img = render_ownership_png(
        territory_names,
        positions,
        owners,
        owner_colors,
        territory_resources=territory_resources,
    )
    if not img:
        print("Map render returned empty image.")
        return 1

    print(f"Smoke test OK: rendered {len(img)} bytes from {data_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
