#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np


def _select_backend() -> str:
    candidates = [
        ("TkAgg", "tkinter"),
        ("QtAgg", "PyQt6"),
        ("Qt5Agg", "PyQt5"),
        ("MacOSX", None),
    ]
    for backend, module in candidates:
        try:
            if module:
                __import__(module)
            matplotlib.use(backend)
            return backend
        except Exception:
            continue
    raise RuntimeError(
        "No interactive matplotlib backend available. Install tkinter (python3.13-tk), "
        "PyQt6, or PyQt5 to use the seed editor."
    )


_select_backend()

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

import worldmap.draw_terrs_v2 as gen


Coord = Tuple[int, int]


def _load_centers(default_json: Path, overrides: Dict[str, List[int]]) -> List[Coord]:
    if default_json.exists():
        payload = json.loads(default_json.read_text(encoding="utf-8"))
        centers_by_id = {t["id"]: t["center"] for t in payload.get("territories", [])}
    else:
        centers_by_id = {}

    centers = []
    for tid, _name, _region in gen.TERRITORIES:
        if tid in overrides:
            centers.append(tuple(overrides[tid]))
        elif tid in centers_by_id:
            centers.append(tuple(centers_by_id[tid]))
        else:
            centers.append((0, 0))
    if any(c == (0, 0) for c in centers):
        barrier = gen.load_bw(Path("world_outline_1600x800.png"), threshold=200)
        barrier = gen.add_suez_barrier(barrier, width=7)
        land = gen.land_mask(barrier)
        comp_lab, _ = gen.label_components(land)
        centers, _labels = gen.build_layout(land, comp_lab)
    return centers


def _load_overrides(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_overrides(path: Path, centers: List[Coord]) -> None:
    data = {tid: [int(x), int(y)] for (tid, _name, _region), (x, y) in zip(gen.TERRITORIES, centers)}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _compute_labels(
    centers: List[Coord],
    barrier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    open_space = barrier == 255
    land = gen.land_from_seed_components(open_space, centers)
    comp_lab, _ = gen.label_components(land)
    comp_sizes = np.bincount(comp_lab.ravel())
    comp_id_by_seed = gen.seed_component_ids(centers, comp_lab)
    labels = gen.per_component_assignment(
        land,
        centers,
        comp_lab,
        comp_id_by_seed,
        noise_seed=999,
    )
    labels = gen.assign_small_components(
        labels,
        comp_lab,
        comp_sizes,
        centers,
    )
    return labels, land


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive world seed editor")
    parser.add_argument("--outline", default="world_outline_1600x800.png")
    parser.add_argument("--base-json", default="world_territories_32.json")
    parser.add_argument("--overrides", default="seed_overrides.json")
    args = parser.parse_args()

    outline_path = Path(args.outline)
    base_json = Path(args.base_json)
    overrides_path = Path(args.overrides)
    if not outline_path.exists():
        raise FileNotFoundError(f"Missing outline PNG: {outline_path}")

    overrides = _load_overrides(overrides_path)
    centers = _load_centers(base_json, overrides)

    barrier = gen.load_bw(outline_path, threshold=200)
    barrier = gen.add_suez_barrier(barrier, width=7)

    labels, land = _compute_labels(centers, barrier)
    borders = gen.compute_borders(labels, land)
    filled = gen.render_filled_map(barrier, labels, borders)
    label_centers = gen.compute_label_centers(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Drag a seed, release to redraw")
    ax.axis("off")
    image_artist = ax.imshow(filled, interpolation="nearest")

    centers_arr = np.array(centers, dtype=np.int32)
    scatter = ax.scatter(
        centers_arr[:, 0],
        centers_arr[:, 1],
        s=30,
        c="#e6553d",
        zorder=4,
    )
    name_artists = []
    for (x, y), (_tid, name, _region) in zip(label_centers, gen.TERRITORIES):
        name_artists.append(
            ax.text(x + 4, y + 4, name, fontsize=7, color="#222222", zorder=5)
        )

    selected = {"idx": None}

    def _update_borders() -> None:
        nonlocal centers, labels, borders, border_overlay
        labels, land = _compute_labels(centers, barrier)
        borders = gen.compute_borders(labels, land)
        filled = gen.render_filled_map(barrier, labels, borders)
        image_artist.set_data(filled)
        label_centers = gen.compute_label_centers(labels)
        for artist, (x, y) in zip(name_artists, label_centers):
            artist.set_position((x + 4, y + 4))
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        pos = np.array([event.xdata, event.ydata])
        dists = np.sum((centers_arr - pos) ** 2, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < 15 ** 2:
            selected["idx"] = idx

    def on_motion(event):
        if selected["idx"] is None or event.xdata is None or event.ydata is None:
            return
        centers_arr[selected["idx"]] = [int(event.xdata), int(event.ydata)]
        scatter.set_offsets(centers_arr)
        fig.canvas.draw_idle()

    def on_release(event):
        if selected["idx"] is None:
            return
        idx = selected["idx"]
        selected["idx"] = None
        x, y = centers_arr[idx]
        x, y = gen.snap_to_mask(barrier == 255, x, y)
        centers_arr[idx] = [x, y]
        centers[idx] = (int(x), int(y))
        scatter.set_offsets(centers_arr)
        _update_borders()
        _save_overrides(overrides_path, centers)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    plt.show()


if __name__ == "__main__":
    main()
