from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import io

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class WorldMapState:
    names: list[str]
    ids: list[str]
    centers: dict[str, tuple[int, int]]
    label_positions: list[tuple[int, int]]
    barrier: np.ndarray
    labels: np.ndarray
    borders: np.ndarray
    notice_box: tuple[int, int, int, int] | None
    notice_board: Image.Image | None


def _load_worldmap_module():
    try:
        from worldmap import draw_terrs_v2 as wm  # type: ignore[import-not-found]
        return wm
    except Exception:
        import importlib.util

        root = Path(__file__).resolve().parent.parent
        path = root / "worldmap" / "draw_terrs_v2.py"
        spec = importlib.util.spec_from_file_location("draw_terrs_v2", path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load worldmap/draw_terrs_v2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _draw_capital_pies(
    image: Image.Image,
    *,
    capital_pies: Dict[str, Dict[str, float]],
    centers: Dict[str, tuple[int, int]],
    pie_colors: Dict[str, str],
    order: Iterable[str],
) -> None:
    if not capital_pies:
        return
    draw = ImageDraw.Draw(image)
    radius = 18
    for territory, slices in capital_pies.items():
        center = centers.get(territory)
        if center is None:
            continue
        total = sum(max(float(v), 0.0) for v in slices.values())
        if total <= 0:
            continue
        cx, cy = center
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        start = 90.0
        for key in order:
            value = max(float(slices.get(key, 0.0)), 0.0)
            if value <= 0:
                continue
            sweep = 360.0 * (value / total)
            end = start + sweep
            draw.pieslice(bbox, start=start, end=end, fill=pie_colors.get(key, "#cccccc"), outline="#4a423c")
            start = end
        draw.ellipse(bbox, outline="#4a423c", width=1)


@lru_cache(maxsize=1)
def _load_world_map_state() -> WorldMapState:
    wm = _load_worldmap_module()
    root = Path(__file__).resolve().parent.parent
    world_dir = root / "worldmap"
    json_path = world_dir / "world_territories_32.json"
    outline_path = world_dir / "world_outline_1600x800.png"
    seed_overrides_path = world_dir / "seed_overrides.json"
    label_overrides_path = world_dir / "label_overrides.json"
    notice_board_path = world_dir / "notice_board.jpeg"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    territories = payload.get("territories", [])
    id_to_name = {t["id"]: t["name"] for t in territories if "id" in t and "name" in t}
    id_to_center = {t["id"]: t["center"] for t in territories if "id" in t and "center" in t}

    order_ids = [tid for tid, _name, _region in wm.TERRITORIES]
    names = [id_to_name.get(tid, tid) for tid in order_ids]

    centers_by_name = {}
    for tid in order_ids:
        center = id_to_center.get(tid)
        if isinstance(center, list) and len(center) == 2:
            centers_by_name[id_to_name.get(tid, tid)] = (int(center[0]), int(center[1]))

    barrier = wm.load_bw(outline_path, threshold=200)
    barrier = wm.add_suez_barrier(barrier, width=7)
    open_space = barrier == 255
    land = wm.land_mask(barrier)
    comp_lab, _ = wm.label_components(land)

    centers = None
    if seed_overrides_path.exists():
        overrides = json.loads(seed_overrides_path.read_text(encoding="utf-8"))
        centers = []
        for tid in order_ids:
            raw = overrides.get(tid)
            if raw is None:
                raise KeyError(f"Missing seed override for {tid}")
            x, y = int(raw[0]), int(raw[1])
            x, y = wm.snap_to_mask(open_space, x, y)
            centers.append((x, y))
        land = wm.land_from_seed_components(open_space, centers)
        comp_lab, _ = wm.label_components(land)
        comp_sizes = np.bincount(comp_lab.ravel())
        labels = wm._compute_labels(centers, land, comp_lab, comp_sizes)
    if centers is None:
        centers, labels = wm.build_layout(land, comp_lab)

    borders = wm.compute_borders(labels, land)
    label_centers = wm.compute_label_centers(labels)
    label_positions = wm.build_label_positions(label_centers, label_overrides_path)

    notice_box = None
    notice_board = None
    if notice_board_path.exists():
        board = Image.open(notice_board_path).convert("RGBA")
        board_rgb = board.convert("RGB")
        arr = np.array(board_rgb)
        mask = (arr < 245).any(axis=2)
        if mask.any():
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            board = board.crop((x0, y0, x1, y1))
        notice_board = board
        aspect = board.width / max(board.height, 1)
        notice_ratio = aspect * wm.NOTICE_BOARD_RATIO_MULTIPLIER
        notice_box = wm.compute_notice_board_box(
            land,
            ratio=notice_ratio,
            margin=wm.NOTICE_BOARD_MARGIN,
            ignore_bottom=wm.NOTICE_BOARD_IGNORE_BOTTOM,
        )

    return WorldMapState(
        names=names,
        ids=order_ids,
        centers=centers_by_name,
        label_positions=label_positions,
        barrier=barrier,
        labels=labels,
        borders=borders,
        notice_box=notice_box,
        notice_board=notice_board,
    )


def _draw_notice_legend(
    image: Image.Image,
    *,
    notice_box: tuple[int, int, int, int] | None,
    owner_colors: Dict[str, str],
    pie_colors: Dict[str, str],
    pie_order: Iterable[str],
    icons_dir: Path,
) -> None:
    if notice_box is None:
        return
    x0, y0, x1, y1 = notice_box
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 0 or box_h <= 0:
        return

    pad = max(6, int(min(box_w, box_h) * 0.05))
    shift_x = max(10, int(box_w * 0.05))
    inner_w = max(1, box_w - pad * 2 - shift_x)
    inner_h = max(1, box_h - pad * 2)

    players = sorted(owner_colors.keys())
    pie_items = list(pie_order)
    resources = [
        ("Energy", "energy.png"),
        ("Minerals", "minerals.png"),
        ("Food", "food.png"),
    ]
    total_lines = len(players) + len(pie_items) + len(resources) + 3
    line_h = max(8, inner_h // max(total_lines, 1))
    font_size = max(8, min(18, line_h))

    wm = _load_worldmap_module()
    font = wm.load_label_font(font_size)
    draw = ImageDraw.Draw(image)
    text_color = "#3b0a57"
    stroke_color = "#f6efe4"
    stroke_width = 1 if font_size < 12 else 2

    def _draw_text(x: int, y: int, text: str) -> None:
        try:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
        except TypeError:
            draw.text((x, y), text, font=font, fill=text_color)
            draw.text((x + 1, y), text, font=font, fill=text_color)

    def _draw_heading(text: str, y: int) -> int:
        _draw_text(x0 + pad + shift_x, y, text)
        return y + line_h

    def _draw_swatch_label(color: str, label: str, y: int) -> int:
        swatch = max(6, line_h - 2)
        swatch_y = y + max(0, (line_h - swatch) // 2)
        draw.rectangle(
            (x0 + pad + shift_x, swatch_y, x0 + pad + shift_x + swatch, swatch_y + swatch),
            fill=color,
            outline=stroke_color,
        )
        _draw_text(x0 + pad + shift_x + swatch + 6, y, label)
        return y + line_h

    def _draw_icon_label(icon_name: str, label: str, y: int) -> int:
        icon_size = max(8, line_h - 2)
        icon_path = icons_dir / icon_name
        if icon_path.exists():
            icon = Image.open(icon_path).convert("RGBA")
            icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            image.paste(
                icon,
                (x0 + pad + shift_x, y + max(0, (line_h - icon_size) // 2)),
                icon,
            )
        else:
            draw.rectangle(
                (x0 + pad + shift_x, y, x0 + pad + shift_x + icon_size, y + icon_size),
                outline=stroke_color,
            )
        _draw_text(x0 + pad + shift_x + icon_size + 6, y, label)
        return y + line_h

    y = y0 + pad
    y = _draw_heading("Players", y)
    for player in players:
        y = _draw_swatch_label(owner_colors.get(player, "#e5e1dc"), player, y)
    y = _draw_heading("Capital Pie", y)
    for key in pie_items:
        label = str(key).replace("_", " ").title()
        y = _draw_swatch_label(pie_colors.get(key, "#cccccc"), label, y)
    y = _draw_heading("Resources", y)
    for label, icon_name in resources:
        y = _draw_icon_label(icon_name, label, y)

def render_world_map_png(
    *,
    owner_by_name: Dict[str, str | None],
    owner_colors: Dict[str, str],
    territory_resources: Dict[str, Dict[str, int]] | None = None,
    capital_pies: Dict[str, Dict[str, float]] | None = None,
    pie_colors: Dict[str, str] | None = None,
    pie_order: Iterable[str] | None = None,
) -> bytes:
    wm = _load_worldmap_module()
    state = _load_world_map_state()
    palette = []
    for name in state.names:
        owner = owner_by_name.get(name)
        color = owner_colors.get(owner, "#e5e1dc")
        palette.append(_hex_to_rgb(color))

    filled = wm.render_filled_map(state.barrier, state.labels, state.borders, palette=palette)
    resource_counts = None
    if territory_resources:
        resource_counts = [
            territory_resources.get(name, {}) for name in state.names
        ]
    labeled = wm.add_name_labels(
        filled,
        state.label_positions,
        state.names,
        icons_dir=Path(__file__).resolve().parent.parent / "icons",
        icon_count_per_resource=3,
        resource_style="icons",
        resource_counts=resource_counts,
    )
    if state.notice_box is not None:
        labeled = wm.overlay_notice_board(
            labeled,
            Path(__file__).resolve().parent.parent / "worldmap" / "notice_board.jpeg",
            state.notice_box,
            board_image=state.notice_board,
        )
    image = Image.fromarray(labeled).convert("RGB")
    _draw_notice_legend(
        image,
        notice_box=state.notice_box,
        owner_colors=owner_colors,
        pie_colors=pie_colors or {},
        pie_order=pie_order or (),
        icons_dir=Path(__file__).resolve().parent.parent / "icons",
    )
    _draw_capital_pies(
        image,
        capital_pies=capital_pies or {},
        centers=state.centers,
        pie_colors=pie_colors or {},
        order=pie_order or (),
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
