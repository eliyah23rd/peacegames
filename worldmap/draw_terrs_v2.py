#!/usr/bin/env python3
import colorsys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

W, H = 1600, 800
MIN_COMPONENT_SIZE = 10000
SEED_TRIES = 5
RELAX_ITERS = 5
NOISE_STRENGTH = 14.0


def _build_palette(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(count):
        hue = (i / max(count, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

# ----------------------------
# 32 territories (more in Africa + N. America; 2 in Antarctica)
# ----------------------------
# Region keys map to component selection + sampling boxes.
TERRITORIES = [
    # North America (8)
    ("alaskaNorth",   "AlaskaNorth",   "N_AMERICA"),
    ("canadaWest",    "CanadaWest",    "N_AMERICA"),
    ("canadaEast",    "CanadaEast",    "N_AMERICA"),
    ("greatPlains",   "GreatPlains",   "N_AMERICA"),
    ("westCoastUS",   "WestCoastUS",   "N_AMERICA"),
    ("centralUS",     "CentralUS",     "N_AMERICA"),
    ("eastCoastUS",   "EastCoastUS",   "N_AMERICA"),
    ("mexicoCentral", "MexicoCentral", "MEX_CENTAM"),

    # South America (3)
    ("andesNorth",    "AndesNorth",    "S_AMERICA"),
    ("brazilSouth",   "BrazilSouth",   "S_AMERICA"),
    ("southernCone",  "SouthernCone",  "S_AMERICA"),

    # Europe (4)
    ("iberiaGaul",     "IberiaGaul",     "EUROPE"),
    ("northEurope",    "NorthEurope",    "EUROPE"),
    ("centralEurope",  "CentralEurope",  "EUROPE"),
    ("eastEurope",     "EastEurope",     "EUROPE"),

    # Africa (7)  <-- increased
    ("northAfrica",   "NorthAfrica",   "AFRICA"),
    ("westAfrica",    "WestAfrica",    "AFRICA"),
    ("centralAfrica", "CentralAfrica", "AFRICA"),
    ("hornAfrica",    "HornAfrica",    "AFRICA"),
    ("eastAfrica",    "EastAfrica",    "AFRICA"),
    ("southAfrica",   "SouthAfrica",   "AFRICA"),

    # Middle East / West Asia (3)
    ("anatoliaLevant", "AnatoliaLevant", "EURASIA"),
    ("arabia",         "Arabia",         "EURASIA"),
    ("iranCaucasus",   "IranCaucasus",   "EURASIA"),

    # Asia (4)  (reduced to make room for more Africa+NAm+Antarctica)
    ("siberia",       "Siberia",       "EURASIA"),
    ("indiaPlus",     "IndiaPlus",     "EURASIA"),
    ("chinaCore",     "ChinaCore",     "EURASIA"),
    ("southeastAsia", "SoutheastAsia", "EURASIA"),

    # Oceania (2)
    ("australiaWest",   "AustraliaWest",   "AUSTRALIA"),
    ("australiaEastNZ", "AustraliaEastNZ", "AUSTRALIA"),

    # Antarctica (2)
    ("antarcticaWest", "AntarcticaWest", "ANTARCTICA"),
    ("antarcticaEast", "AntarcticaEast", "ANTARCTICA"),
]

# Sampling boxes (x0,y0,x1,y1) for the 1600x800 equirectangular outline.
# These don’t need to be perfect; seeds are forced onto land.
REGION_BOX = {
    "N_AMERICA":   ( 60,  60,  740, 450),
    "MEX_CENTAM":  (220, 320,  760, 580),

    "S_AMERICA":   (260, 380,  760, 770),

    "EUROPE":      (740,  60, 1250, 330),

    "AFRICA":      (740, 230, 1220, 770),
    # "EURASIA" here means Europe+Asia landmass AFTER we split Africa away with Suez barrier
    "EURASIA":     (980,  40, 1600, 560),

    "AUSTRALIA":   (1240, 560, 1600, 790),

    "ANTARCTICA":  (   0, 660, 1600, 800),
}

# Sea-link adjacency extras (added on top of land adjacencies)
SEA_LINKS = [
    ("eastCoastUS", "northEurope"),
    ("brazilSouth", "westAfrica"),
    ("northAfrica", "iberiaGaul"),
    ("northAfrica", "anatoliaLevant"),
    ("siberia", "alaskaNorth"),
    ("indonesiaArc", "australiaWest"),
    ("chinaCore", "australiaEastNZ"),
    ("indiaPlus", "southernCone"),  # per your request (long sea lane)
]

# ----------------------------
# Raster / mask helpers
# ----------------------------
def load_bw(path: Path, threshold: int = 200) -> np.ndarray:
    """Return uint8 image: 0=black barriers, 255=white."""
    img = Image.open(path).convert("L")
    a = np.array(img, dtype=np.uint8)
    return np.where(a < threshold, 0, 255).astype(np.uint8)

def ocean_from_edges(barrier: np.ndarray) -> np.ndarray:
    """Flood-fill reachable white from edges => ocean mask."""
    open_space = (barrier == 255)
    seeds = np.zeros_like(open_space, dtype=bool)
    seeds[0, :]  = open_space[0, :]
    seeds[-1, :] = open_space[-1, :]
    seeds[:, 0]  = open_space[:, 0]
    seeds[:, -1] = open_space[:, -1]

    ocean = seeds.copy()
    struct = ndi.generate_binary_structure(2, 1)  # 4-neigh
    while True:
        prev = ocean.sum()
        ocean = ndi.binary_dilation(ocean, structure=struct) & open_space
        if ocean.sum() == prev:
            break
    return ocean

def add_suez_barrier(barrier: np.ndarray, width: int = 7) -> np.ndarray:
    """
    Artificial barrier to split Africa and Eurasia for your flood-fill logic.
    Draw a thick black line near Suez (approx lon 32E, lat 30N in equirect).
    """
    out = barrier.copy()
    # Hand-tuned short polyline from Med to Red Sea in pixel coords:
    pts = [(930, 255), (950, 285), (960, 320), (955, 350), (945, 380)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        rr, cc = line_pixels(x0, y0, x1, y1)
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                yy = np.clip(rr + dy, 0, H - 1)
                xx = np.clip(cc + dx, 0, W - 1)
                out[yy, xx] = 0
    return out

def line_pixels(x0, y0, x1, y1):
    """Bresenham-ish line pixel coords (returns arrays rr,cc)."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)
    return ys, xs

def land_mask(barrier: np.ndarray) -> np.ndarray:
    """Land interior = white not reachable from edges."""
    ocean = ocean_from_edges(barrier)
    return (barrier == 255) & (~ocean)

def label_components(mask: np.ndarray):
    """Connected components (8-neigh)."""
    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(2, 2))
    return lab, n

def component_bbox(lab: np.ndarray, comp_id: int):
    ys, xs = np.where(lab == comp_id)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

def smooth_noise_field(seed: int, sigma: float = 28.0) -> np.ndarray:
    """
    Smooth noise to warp borders into more natural shapes.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    noise = ndi.gaussian_filter(noise, sigma=sigma)
    noise -= noise.min()
    noise /= (noise.max() + 1e-9)
    noise = (noise * 2.0 - 1.0)  # [-1,1]
    return noise

def find_bottom_component(comp_lab: np.ndarray) -> int | None:
    """Return the largest component touching the bottom edge (Antarctica)."""
    bottom_labels = comp_lab[-1, :]
    bottom_labels = bottom_labels[bottom_labels > 0]
    if bottom_labels.size == 0:
        return None
    sizes = np.bincount(comp_lab.ravel())
    best = None
    best_size = -1
    for lbl in bottom_labels:
        lbl_int = int(lbl)
        size = sizes[lbl_int] if lbl_int < len(sizes) else 0
        if size > best_size:
            best_size = size
            best = lbl_int
    return best

def pick_component_in_box(
    comp_lab: np.ndarray,
    box,
    exclude: set[int],
    *,
    min_size: int,
    sizes: np.ndarray,
) -> int | None:
    """Pick the dominant component within a box, excluding given ids."""
    x0, y0, x1, y1 = box
    sub = comp_lab[y0:y1, x0:x1]
    ids, counts = np.unique(sub[sub > 0], return_counts=True)
    best = None
    best_count = -1
    for cid, count in zip(ids, counts):
        cid_int = int(cid)
        if cid_int in exclude:
            continue
        if sizes[cid_int] < min_size:
            continue
        if count > best_count:
            best_count = int(count)
            best = cid_int
    return best

# ----------------------------
# Seed placement with spacing + relaxation
# ----------------------------
def sample_seed_on_mask(mask: np.ndarray, box, rng: random.Random, existing, min_dist: int, tries: int = 20000):
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.where(sub)
    if len(xs) == 0:
        raise RuntimeError(f"No land pixels in sampling box {box}")

    idxs = list(range(len(xs)))
    rng.shuffle(idxs)

    for i in idxs[:tries]:
        x = int(xs[i] + x0)
        y = int(ys[i] + y0)
        ok = True
        for ex, ey in existing:
            if (x - ex) ** 2 + (y - ey) ** 2 < (min_dist ** 2):
                ok = False
                break
        if ok:
            return (x, y)

    # fallback (if min_dist too strict)
    i = idxs[0]
    return (int(xs[i] + x0), int(ys[i] + y0))

def place_all_seeds(
    land: np.ndarray,
    comp_lab: np.ndarray,
    *,
    seed: int = 12345,
    antarctica_comp: int | None = None,
    comp_sizes: np.ndarray | None = None,
    min_component_size: int = MIN_COMPONENT_SIZE,
):
    rng = random.Random(seed)
    centers = []
    sizes = comp_sizes if comp_sizes is not None else np.bincount(comp_lab.ravel())
    for tid, name, region in TERRITORIES:
        box = REGION_BOX[region]
        exclude: set[int] = set()
        if antarctica_comp is not None and region != "ANTARCTICA":
            exclude.add(int(antarctica_comp))
        if region == "ANTARCTICA" and antarctica_comp is not None:
            region_mask = comp_lab == antarctica_comp
        else:
            region_comp = pick_component_in_box(
                comp_lab,
                box,
                exclude,
                min_size=min_component_size,
                sizes=sizes,
            )
            if region_comp is None:
                fallback = None
                fallback_size = -1
                for cid, size in enumerate(sizes):
                    if cid == 0 or cid in exclude or size < min_component_size:
                        continue
                    if size > fallback_size:
                        fallback_size = int(size)
                        fallback = cid
                region_comp = fallback
            region_mask = (comp_lab == region_comp) if region_comp is not None else land

        # region-specific spacing (prevents tiny territories)
        if region in ("N_AMERICA", "AFRICA", "EURASIA"):
            min_dist = 70
        elif region in ("S_AMERICA", "EUROPE"):
            min_dist = 60
        else:
            min_dist = 45
        centers.append(sample_seed_on_mask(region_mask, box, rng, centers, min_dist=min_dist))
    return centers

def assign_labels_warped(land: np.ndarray, centers_xy, noise: np.ndarray, noise_strength: float = 22.0):
    """
    Land-only assignment:
    label[p] = argmin_i (dist(p, seed_i) + noise_strength * noise[p])

    This creates more natural, wavy borders than straight Voronoi edges.
    """
    land_y, land_x = np.where(land)
    best_cost = np.full((H, W), np.inf, dtype=np.float32)
    best_idx  = np.full((H, W), -1, dtype=np.int32)

    # Precompute noise term once
    noise_term = noise_strength * noise

    for i, (sx, sy) in enumerate(centers_xy):
        dx = (land_x - sx).astype(np.float32)
        dy = (land_y - sy).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy)  # (Nland,)
        cost = dist + noise_term[land_y, land_x]

        # update winners for land pixels only
        cur = best_cost[land_y, land_x]
        win = cost < cur
        if np.any(win):
            best_cost[land_y[win], land_x[win]] = cost[win]
            best_idx[land_y[win], land_x[win]] = i

    return best_idx

def snap_to_mask(mask: np.ndarray, x: int, y: int, max_r: int = 40):
    """Find nearest True pixel to (x,y) within a small radius spiral search."""
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    if mask[y, x]:
        return (x, y)
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in (-r, r):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < W and 0 <= yy < H and mask[yy, xx]:
                    return (xx, yy)
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < W and 0 <= yy < H and mask[yy, xx]:
                    return (xx, yy)
    return (x, y)

def lloyd_relax_by_component(
    land: np.ndarray,
    centers_xy,
    comp_lab: np.ndarray,
    iters: int = 4,
    seed: int = 777,
):
    """
    Pixel-grid Lloyd-style relaxation:
    - assign labels
    - move each seed to centroid of its region (snapped to nearest land pixel)
    """
    centers = [list(p) for p in centers_xy]
    noise = smooth_noise_field(seed=seed, sigma=28.0)

    for t in range(iters):
        comp_id_by_seed = seed_component_ids(centers, comp_lab)
        labels = per_component_assignment(
            land,
            centers,
            comp_lab,
            comp_id_by_seed,
            noise_seed=seed + t,
        )

        # compute centroids per label
        valid = labels >= 0
        ys, xs = np.where(valid)
        lab = labels[ys, xs]
        n = len(centers)

        cnt = np.bincount(lab, minlength=n).astype(np.float32)
        sx  = np.bincount(lab, weights=xs.astype(np.float32), minlength=n)
        sy  = np.bincount(lab, weights=ys.astype(np.float32), minlength=n)

        comp_masks = {}
        for i in range(n):
            if cnt[i] < 1:
                continue
            cx = sx[i] / cnt[i]
            cy = sy[i] / cnt[i]
            cid = comp_id_by_seed[i]
            if cid not in comp_masks:
                comp_masks[cid] = comp_lab == cid
            nx, ny = snap_to_mask(
                comp_masks[cid],
                int(round(cx)),
                int(round(cy)),
            )
            centers[i][0], centers[i][1] = nx, ny

    return [(int(x), int(y)) for x, y in centers]

# ----------------------------
# Borders + adjacency
# ----------------------------
def compute_borders(labels: np.ndarray, land: np.ndarray) -> np.ndarray:
    border = np.zeros((H, W), dtype=bool)

    # horizontal diffs
    a = labels[:, :-1]
    b = labels[:, 1:]
    m = land[:, :-1] & land[:, 1:] & (a != b) & (a >= 0) & (b >= 0)
    border[:, :-1] |= m
    border[:, 1:]  |= m

    # vertical diffs
    a = labels[:-1, :]
    b = labels[1:, :]
    m = land[:-1, :] & land[1:, :] & (a != b) & (a >= 0) & (b >= 0)
    border[:-1, :] |= m
    border[1:, :]  |= m

    # thicken + close (seal pinholes)
    struct8 = ndi.generate_binary_structure(2, 2)
    border = ndi.binary_dilation(border, structure=struct8, iterations=2)
    border = ndi.binary_closing(border, structure=struct8, iterations=1)
    return border

def adjacency_from_labels(labels: np.ndarray, land: np.ndarray, ids_by_idx):
    adj = {tid: set() for tid in ids_by_idx.values()}

    # right neighbors
    a = labels[:, :-1]
    b = labels[:, 1:]
    m = land[:, :-1] & land[:, 1:] & (a != b) & (a >= 0) & (b >= 0)
    ys, xs = np.where(m)
    for y, x in zip(ys, xs):
        i = int(a[y, x]); j = int(b[y, x])
        ti = ids_by_idx[i]; tj = ids_by_idx[j]
        adj[ti].add(tj); adj[tj].add(ti)

    # down neighbors
    a = labels[:-1, :]
    b = labels[1:, :]
    m = land[:-1, :] & land[1:, :] & (a != b) & (a >= 0) & (b >= 0)
    ys, xs = np.where(m)
    for y, x in zip(ys, xs):
        i = int(a[y, x]); j = int(b[y, x])
        ti = ids_by_idx[i]; tj = ids_by_idx[j]
        adj[ti].add(tj); adj[tj].add(ti)

    return adj

def add_sea_links(adj):
    for a, b in SEA_LINKS:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

# ----------------------------
# Enforce "no cross-continent borders":
# Assign seeds + labels PER COMPONENT
# ----------------------------
def per_component_assignment(land: np.ndarray, centers_xy, comp_lab: np.ndarray, comp_id_by_seed, noise_seed: int = 999):
    """
    Build labels by solving each connected component separately.
    This ensures borders never cross water or continental-divide barriers.
    """
    labels = np.full((H, W), -1, dtype=np.int32)
    noise = smooth_noise_field(seed=noise_seed, sigma=28.0)

    # group seeds by component id
    seeds_by_comp = {}
    for i, (sx, sy) in enumerate(centers_xy):
        cid = comp_id_by_seed[i]
        seeds_by_comp.setdefault(cid, []).append(i)

    for cid, seed_idxs in seeds_by_comp.items():
        comp_mask = (comp_lab == cid)
        # assign only within this component
        sub_centers = [centers_xy[i] for i in seed_idxs]
        sub_labels = assign_labels_warped(
            comp_mask,
            sub_centers,
            noise=noise,
            noise_strength=NOISE_STRENGTH,
        )

        # remap local 0..k-1 back to global indices
        ys, xs = np.where(comp_mask & (sub_labels >= 0))
        labels[ys, xs] = np.array([seed_idxs[int(sub_labels[y, x])] for y, x in zip(ys, xs)], dtype=np.int32)

    return labels

def seed_component_ids(centers_xy, comp_lab):
    ids = []
    for x, y in centers_xy:
        cid = int(comp_lab[y, x])
        if cid == 0:
            # If a seed somehow landed outside land, this will be 0; caller should fix seeds
            raise RuntimeError("Seed landed outside a land component. Try adjusting REGION_BOX.")
        ids.append(cid)
    return ids

def min_area_repair(land, labels, centers_xy, comp_lab, min_pixels: int = 18000):
    """
    Prevent tiny territories: if a region’s area < min_pixels,
    move its seed to a "deep interior" point (max distance from borders) in its component.
    """
    n = len(centers_xy)
    valid = labels >= 0
    ys, xs = np.where(valid)
    lab = labels[ys, xs]

    areas = np.bincount(lab, minlength=n)
    bad = np.where(areas < min_pixels)[0]
    if len(bad) == 0:
        return centers_xy

    # distance from component edge: larger means "safer interior"
    # We approximate edge by distance to NOT-land within that component.
    new_centers = list(centers_xy)
    occupied = {tuple(c) for c in new_centers}
    min_sep = 18

    for i in bad:
        x0, y0 = new_centers[i]
        cid = int(comp_lab[y0, x0])
        comp_mask = (comp_lab == cid)

        # distance to outside comp_mask
        dist = ndi.distance_transform_edt(comp_mask)
        ys, xs = np.where(comp_mask)
        if len(xs) == 0:
            continue
        scores = dist[ys, xs]
        order = np.argsort(scores)[::-1]
        chosen = None
        old = tuple(new_centers[i])
        if old in occupied:
            occupied.remove(old)
        for idx in order[:5000]:
            cand = (int(xs[idx]), int(ys[idx]))
            if cand in occupied:
                continue
            if any((cand[0] - ox) ** 2 + (cand[1] - oy) ** 2 < min_sep ** 2 for ox, oy in occupied):
                continue
            chosen = cand
            break
        if chosen is None:
            top = order[0]
            chosen = (int(xs[top]), int(ys[top]))
        new_centers[i] = chosen
        occupied.add(chosen)

    return new_centers

def _score_labels(labels: np.ndarray, num_seeds: int) -> float:
    areas = np.array([int(np.sum(labels == i)) for i in range(num_seeds)])
    if areas.min() == 0:
        return float("inf")
    return float(areas.max() / areas.min())

def _compute_labels(
    centers: list[tuple[int, int]],
    land: np.ndarray,
    comp_lab: np.ndarray,
    comp_sizes: np.ndarray,
) -> np.ndarray:
    comp_id_by_seed = seed_component_ids(centers, comp_lab)
    labels = per_component_assignment(
        land,
        centers,
        comp_lab,
        comp_id_by_seed,
        noise_seed=999,
    )
    labels = assign_small_components(
        labels,
        comp_lab,
        comp_sizes,
        centers,
    )
    return labels

def render_filled_map(
    barrier: np.ndarray,
    labels: np.ndarray,
    borders: np.ndarray,
    *,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Return RGB image with colored territories and black borders."""
    if palette is None:
        palette = _build_palette(len(TERRITORIES))
    rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    rgb[barrier == 0] = 0
    for idx, color in enumerate(palette):
        rgb[labels == idx] = color
    rgb[borders] = 0
    return rgb

def add_name_labels(
    image: np.ndarray,
    centers: list[tuple[int, int]],
    names: list[str],
) -> np.ndarray:
    """Draw territory names at seed centers."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for (x, y), name in zip(centers, names):
        draw.text((x + 4, y + 4), name, fill=(0, 0, 0), font=font)
    return np.array(pil)

def build_layout(
    land: np.ndarray,
    comp_lab: np.ndarray,
    *,
    seed_base: int = 12345,
    tries: int = SEED_TRIES,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    comp_sizes = np.bincount(comp_lab.ravel())
    antarctica_comp = find_bottom_component(comp_lab)
    best_score = float("inf")
    best_centers = None
    best_labels = None

    for idx in range(tries):
        seed = seed_base + idx * 97
        centers = place_all_seeds(
            land,
            comp_lab,
            seed=seed,
            antarctica_comp=antarctica_comp,
            comp_sizes=comp_sizes,
        )
        centers = lloyd_relax_by_component(
            land,
            centers,
            comp_lab,
            iters=RELAX_ITERS,
            seed=seed + 13,
        )
        comp_id_by_seed = seed_component_ids(centers, comp_lab)
        labels = per_component_assignment(
            land,
            centers,
            comp_lab,
            comp_id_by_seed,
            noise_seed=seed + 29,
        )
        centers2 = min_area_repair(land, labels, centers, comp_lab, min_pixels=18000)
        if centers2 != centers:
            centers = centers2
            comp_id_by_seed = seed_component_ids(centers, comp_lab)
            labels = per_component_assignment(
                land,
                centers,
                comp_lab,
                comp_id_by_seed,
                noise_seed=seed + 31,
            )
        labels = assign_small_components(
            labels,
            comp_lab,
            comp_sizes,
            centers,
        )
        score = _score_labels(labels, len(centers))
        if score < best_score:
            best_score = score
            best_centers = centers
            best_labels = labels

    if best_centers is None or best_labels is None:
        raise RuntimeError("Failed to generate a layout.")
    return best_centers, best_labels

def assign_small_components(
    labels: np.ndarray,
    comp_lab: np.ndarray,
    comp_sizes: np.ndarray,
    centers_xy,
    *,
    min_component_size: int = MIN_COMPONENT_SIZE,
) -> np.ndarray:
    """Assign small island components to the nearest existing seed."""
    new_labels = labels.copy()
    large = {int(idx) for idx, size in enumerate(comp_sizes) if size >= min_component_size}
    ys, xs = np.where(comp_lab > 0)
    small_components = set(np.unique(comp_lab[ys, xs])) - large
    if not small_components:
        return new_labels

    centers_arr = np.array(centers_xy, dtype=np.int32)
    for cid in small_components:
        mask = comp_lab == cid
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        dx = centers_arr[:, 0] - cx
        dy = centers_arr[:, 1] - cy
        nearest = int(np.argmin(dx * dx + dy * dy))
        new_labels[mask] = nearest
    return new_labels

# ----------------------------
# Main
# ----------------------------
def main():
    in_path = Path("world_outline_1600x800.png")
    if not in_path.exists():
        raise FileNotFoundError("Put world_outline_1600x800.png in this folder first.")
    overrides_path = Path("seed_overrides.json")

    # 1) Read barrier map and split Africa/Eurasia with Suez barrier
    barrier = load_bw(in_path, threshold=200)
    barrier = add_suez_barrier(barrier, width=7)

    # 2) Build land mask (keeps Antarctica for now)
    land = land_mask(barrier)

    # 3) Connected components of land (after Suez barrier split)
    comp_lab, ncomp = label_components(land)
    if ncomp < 5:
        print("Warning: fewer components than expected. Still continuing.")

    # 4) Place seeds (initial), then relax per component to spread them
    centers = None
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        centers = []
        for tid, _name, _region in TERRITORIES:
            if tid not in overrides:
                raise KeyError(f"Missing seed override for {tid}")
            raw = overrides[tid]
            centers.append((int(raw[0]), int(raw[1])))
    if centers is None:
        centers, labels = build_layout(land, comp_lab)
    else:
        comp_sizes = np.bincount(comp_lab.ravel())
        labels = _compute_labels(centers, land, comp_lab, comp_sizes)

    # 5) Borders
    borders = compute_borders(labels, land)

    # 6) Output map with internal borders
    out_map = barrier.copy()
    out_map[borders] = 0
    out_map = np.where(out_map < 200, 0, 255).astype(np.uint8)
    Image.fromarray(out_map, mode="L").convert("RGB").save("world_map_32_internal.png", optimize=True)
    names = [name for _tid, name, _region in TERRITORIES]
    labeled = add_name_labels(
        np.array(Image.fromarray(out_map, mode="L").convert("RGB")),
        centers,
        names,
    )
    Image.fromarray(labeled).save("world_map_32_internal_labeled.png", optimize=True)

    filled = render_filled_map(barrier, labels, borders)
    Image.fromarray(filled).save("world_map_32_filled.png", optimize=True)
    filled_labeled = add_name_labels(filled, centers, names)
    Image.fromarray(filled_labeled).save("world_map_32_filled_labeled.png", optimize=True)

    # 7) Adjacency + JSON
    ids_by_idx = {i: TERRITORIES[i][0] for i in range(len(TERRITORIES))}
    names_by_id = {tid: name for tid, name, _ in TERRITORIES}

    adj = adjacency_from_labels(labels, land, ids_by_idx)
    add_sea_links(adj)

    data = {
        "map_name": "World32_NoBleed_NaturalBorders",
        "image_size": [W, H],
        "territories": []
    }
    for i, (tid, name, _region) in enumerate(TERRITORIES):
        cx, cy = centers[i]
        data["territories"].append({
            "id": tid,
            "name": name,
            "center": [int(cx), int(cy)],
            "adjacent": sorted(adj[tid])
        })

    Path("world_territories_32.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("Wrote: world_map_32_internal.png")
    print("Wrote: world_territories_32.json")

if __name__ == "__main__":
    main()
