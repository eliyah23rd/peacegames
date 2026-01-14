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
from matplotlib.widgets import Button, TextBox

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

def _load_name_overrides(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _load_label_overrides(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_overrides(path: Path, centers: List[Coord]) -> None:
    data = {tid: [int(x), int(y)] for (tid, _name, _region), (x, y) in zip(gen.TERRITORIES, centers)}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _save_name_overrides(path: Path, names: List[str]) -> None:
    data = {tid: name for (tid, _name, _region), name in zip(gen.TERRITORIES, names)}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _save_label_overrides(path: Path, positions: List[Coord]) -> None:
    data = {
        tid: [int(x), int(y)]
        for (tid, _name, _region), (x, y) in zip(gen.TERRITORIES, positions)
    }
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
    parser.add_argument("--name-overrides", default="name_overrides.json")
    parser.add_argument("--label-overrides", default="label_overrides.json")
    parser.add_argument(
        "--mode",
        choices=["move", "name", "label"],
        default="move",
        help="move=drag seeds, name=edit names only, label=move label boxes",
    )
    parser.add_argument(
        "--label-style",
        choices=["icons", "pie"],
        default="icons",
        help="label mode: initial resource style",
    )
    args = parser.parse_args()

    outline_path = Path(args.outline)
    base_json = Path(args.base_json)
    overrides_path = Path(args.overrides)
    name_overrides_path = Path(args.name_overrides)
    label_overrides_path = Path(args.label_overrides)
    if not outline_path.exists():
        raise FileNotFoundError(f"Missing outline PNG: {outline_path}")

    overrides = _load_overrides(overrides_path)
    name_overrides = _load_name_overrides(name_overrides_path)
    label_overrides = _load_label_overrides(label_overrides_path)
    centers = _load_centers(base_json, overrides)
    names = []
    for tid, name, _region in gen.TERRITORIES:
        names.append(name_overrides.get(tid, name))

    barrier = gen.load_bw(outline_path, threshold=200)
    barrier = gen.add_suez_barrier(barrier, width=7)

    labels, land = _compute_labels(centers, barrier)
    borders = gen.compute_borders(labels, land)
    label_centers = gen.compute_label_centers(labels)
    label_positions = []
    for (tid, _name, _region), (cx, cy) in zip(gen.TERRITORIES, label_centers):
        if tid in label_overrides:
            raw = label_overrides[tid]
            label_positions.append((int(raw[0]), int(raw[1])))
        else:
            label_positions.append((int(cx), int(cy)))
    filled = gen.render_filled_map(barrier, labels, borders)
    icons_dir = Path(__file__).resolve().parent.parent / "icons"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.12)
    if args.mode == "name":
        ax.set_title("Click a seed, edit name, save")
    elif args.mode == "label":
        ax.set_title("Drag a label box, release to save")
    else:
        ax.set_title("Drag a seed, release to redraw")
    ax.axis("off")
    label_style = {"value": args.label_style}
    if args.mode == "label":
        labeled = gen.add_name_labels(
            filled,
            label_positions,
            names,
            icons_dir=icons_dir,
            icon_count_per_resource=gen.ICON_PREVIEW_COUNT,
            resource_style=label_style["value"],
        )
        image_artist = ax.imshow(labeled, interpolation="nearest")
    else:
        image_artist = ax.imshow(filled, interpolation="nearest")

    centers_arr = np.array(centers, dtype=np.int32)
    scatter = ax.scatter(
        centers_arr[:, 0],
        centers_arr[:, 1],
        s=30,
        c="#e6553d",
        zorder=4,
    )
    scatter.set_visible(args.mode in ("move", "name"))
    name_artists = []
    if args.mode != "label":
        for (x, y), name in zip(label_centers, names):
            name_artists.append(
                ax.text(
                    x + 4,
                    y + 4,
                    name,
                    fontsize=7,
                    color="#222222",
                    zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
                )
            )

    label_layouts = []
    label_font = None
    if args.mode == "label":
        label_font = gen.load_label_font(14)
        label_layouts = gen.build_label_layouts(
            label_positions,
            names,
            font=label_font,
            icon_size=14,
            icon_count_per_resource=gen.ICON_PREVIEW_COUNT,
            resource_style=label_style["value"],
        )

    selected = {"idx": None, "moved": False}

    name_box = None
    save_button = None
    toggle_button = None
    if args.mode == "name":
        name_ax = fig.add_axes([0.1, 0.02, 0.6, 0.06])
        name_box = TextBox(name_ax, "Territory name", initial="")
        save_ax = fig.add_axes([0.72, 0.02, 0.18, 0.06])
        save_button = Button(save_ax, "Save name")
    elif args.mode == "label":
        toggle_ax = fig.add_axes([0.72, 0.02, 0.22, 0.06])
        toggle_button = Button(toggle_ax, f"Style: {label_style['value']}")

    def _update_borders() -> None:
        nonlocal centers, labels, borders, names
        labels, land = _compute_labels(centers, barrier)
        borders = gen.compute_borders(labels, land)
        filled = gen.render_filled_map(barrier, labels, borders)
        image_artist.set_data(filled)
        label_centers = gen.compute_label_centers(labels)
        for idx, (artist, (x, y)) in enumerate(zip(name_artists, label_centers)):
            artist.set_position((x + 4, y + 4))
            artist.set_text(names[idx])
        fig.canvas.draw_idle()

    def _update_label_overlay() -> None:
        nonlocal label_layouts
        labeled = gen.add_name_labels(
            filled,
            label_positions,
            names,
            icons_dir=icons_dir,
            icon_count_per_resource=gen.ICON_PREVIEW_COUNT,
            resource_style=label_style["value"],
        )
        image_artist.set_data(labeled)
        label_layouts = gen.build_label_layouts(
            label_positions,
            names,
            font=label_font,
            icon_size=14,
            icon_count_per_resource=gen.ICON_PREVIEW_COUNT,
            resource_style=label_style["value"],
        )
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        pos = np.array([event.xdata, event.ydata])
        if args.mode == "label":
            selected["idx"] = None
            selected["moved"] = False
            for idx, layout in enumerate(label_layouts):
                if not layout:
                    continue
                left, top, right, bottom = layout["box"]
                if left <= event.xdata <= right and top <= event.ydata <= bottom:
                    selected["idx"] = idx
                    selected["moved"] = False
                    break
            return
        dists = np.sum((centers_arr - pos) ** 2, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < 15 ** 2:
            selected["idx"] = idx
            selected["moved"] = False
            if name_box is not None:
                name_box.set_val(names[idx])
        else:
            selected["idx"] = None
            selected["moved"] = False
            if name_box is not None:
                name_box.set_val("")

    def on_motion(event):
        if args.mode not in ("move", "label"):
            return
        if selected["idx"] is None or event.xdata is None or event.ydata is None:
            return
        selected["moved"] = True
        if args.mode == "move":
            centers_arr[selected["idx"]] = [int(event.xdata), int(event.ydata)]
            scatter.set_offsets(centers_arr)
            fig.canvas.draw_idle()
        else:
            label_positions[selected["idx"]] = (int(event.xdata), int(event.ydata))
            _update_label_overlay()

    def on_release(event):
        if args.mode == "label":
            if selected["idx"] is None:
                return
            idx = selected["idx"]
            x, y = label_positions[idx]
            x = int(min(max(x, 0), gen.W - 1))
            y = int(min(max(y, 0), gen.H - 1))
            label_positions[idx] = (x, y)
            _update_label_overlay()
            _save_label_overrides(label_overrides_path, label_positions)
            return
        if args.mode != "move":
            return
        if selected["idx"] is None:
            return
        idx = selected["idx"]
        x, y = centers_arr[idx]
        x, y = gen.snap_to_mask(barrier == 255, x, y)
        centers_arr[idx] = [x, y]
        centers[idx] = (int(x), int(y))
        scatter.set_offsets(centers_arr)
        _update_borders()
        _save_overrides(overrides_path, centers)

    def on_save_name(_event):
        if args.mode != "name" or name_box is None:
            return
        idx = selected["idx"]
        if idx is None:
            return
        text = name_box.text.strip()
        if not text:
            return
        names[idx] = text
        name_artists[idx].set_text(text)
        _save_name_overrides(name_overrides_path, names)
        fig.canvas.draw_idle()

    def on_toggle_style(_event):
        if args.mode != "label":
            return
        label_style["value"] = "pie" if label_style["value"] == "icons" else "icons"
        if toggle_button is not None:
            toggle_button.label.set_text(f"Style: {label_style['value']}")
        _update_label_overlay()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    if save_button is not None:
        save_button.on_clicked(on_save_name)
    if toggle_button is not None:
        toggle_button.on_clicked(on_toggle_style)

    plt.show()


if __name__ == "__main__":
    main()
