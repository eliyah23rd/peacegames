#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

for _backend in ("TkAgg", "Qt5Agg", "MacOSX"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

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
    land: np.ndarray,
    comp_lab: np.ndarray,
    comp_sizes: np.ndarray,
) -> np.ndarray:
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
    return labels


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
    land = gen.land_mask(barrier)
    comp_lab, _ = gen.label_components(land)
    comp_sizes = np.bincount(comp_lab.ravel())

    labels = _compute_labels(centers, land, comp_lab, comp_sizes)
    borders = gen.compute_borders(labels, land)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Drag a seed, release to redraw")
    ax.axis("off")
    ax.imshow(barrier, cmap="gray", interpolation="nearest")

    border_overlay = np.zeros((gen.H, gen.W, 4), dtype=np.float32)
    border_overlay[borders] = [0.0, 0.0, 0.0, 1.0]
    border_artist = ax.imshow(border_overlay, interpolation="nearest")

    centers_arr = np.array(centers, dtype=np.int32)
    scatter = ax.scatter(centers_arr[:, 0], centers_arr[:, 1], s=30, c="#e6553d")

    selected = {"idx": None}

    def _update_borders() -> None:
        nonlocal centers, labels, borders, border_overlay
        labels = _compute_labels(centers, land, comp_lab, comp_sizes)
        borders = gen.compute_borders(labels, land)
        border_overlay = np.zeros((gen.H, gen.W, 4), dtype=np.float32)
        border_overlay[borders] = [0.0, 0.0, 0.0, 1.0]
        border_artist.set_data(border_overlay)
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
        x, y = gen.snap_to_mask(land, x, y)
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
