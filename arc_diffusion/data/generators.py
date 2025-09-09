"""ARC synthetic task generators with uniqueness enforcement.

This module creates small ARC-like tasks as JSON files. Each task generator
returns an input/output grid pair. A simple uniqueness mechanism ensures
that train and test sets do not contain duplicate pairs and are disjoint.
"""
import json
import os
import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

Grid = List[List[int]]

# -------------------- helpers --------------------
def zeros(h: int, w: int) -> Grid:
    """Create an h x w grid filled with 0s."""
    return [[0 for _ in range(w)] for _ in range(h)]

def draw_rect(grid: Grid, top: int, left: int, height: int, width: int, color: int) -> None:
    """Fill an axis-aligned rectangle on grid in-place with color."""
    for r in range(top, top + height):
        for c in range(left, left + width):
            grid[r][c] = color

def rect_interior(top: int, left: int, height: int, width: int) -> Tuple[int, int, int, int]:
    """Return the interior rectangle (top,left,height,width) excluding 1-cell border.
    If the rectangle is too small, returns zero-sized interior (height or width = 0).
    """
    if height < 3 or width < 3:
        return (top+1, left+1, 0, 0)
    return (top + 1, left + 1, height - 2, width - 2)

def place_non_overlapping_rectangles(h: int, w: int, n: int, min_hw=(3,3), max_trials=500) -> List[Tuple[int,int,int,int]]:
    """Sample up to n rectangles without overlap inside an h x w canvas.

    Rectangles are axis-aligned and must not overlap or touch. The function
    retries up to max_trials candidates and returns the list found so far.
    Each rectangle is returned as (top, left, height, width).
    """
    rects = []
    trials = 0
    while len(rects) < n and trials < max_trials:
        trials += 1
        rh = random.randint(min_hw[0], max(min_hw[0], min(h//2, h - 1)))
        rw = random.randint(min_hw[1], max(min_hw[1], min(w//2, w - 1)))
        top = random.randint(0, h - rh)
        left = random.randint(0, w - rw)
        candidate = (top, left, rh, rw)
        overlap = False
        for (t,l,hh,ww) in rects:
            if not (top+rh < t or t+hh < top or left+rw < l or l+ww < left):
                overlap = True
                break
        if not overlap:
            rects.append(candidate)
    return rects

def to_arc_examples(pairs: List[Tuple[Grid, Grid]]) -> List[Dict[str, Grid]]:
    """Convert list of (input, output) grid pairs to ARC JSON example dicts."""
    return [{"input": pin, "output": pout} for (pin, pout) in pairs]

def save_task_json(path: str, train_pairs: List[Tuple[Grid,Grid]], test_pairs: List[Tuple[Grid,Grid]]) -> None:
    """Write ARC-style JSON with 'train' and 'test' lists of examples."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"train": to_arc_examples(train_pairs), "test": to_arc_examples(test_pairs)}
    with open(path, "w") as f:
        json.dump(obj, f)

# -------------------- Task 1: bb43febb --------------------
def gen_bb43febb_single(h: int = 10, w: int = 10) -> Tuple[Grid, Grid]:
    """Replace interior of each solid color-5 rectangle with color 2."""
    grid = zeros(h, w)
    rects = place_non_overlapping_rectangles(h, w, n=random.randint(1, 3), min_hw=(3,3))
    for (t,l,hh,ww) in rects:
        draw_rect(grid, t, l, hh, ww, 5)
    out = deepcopy(grid)
    for (t,l,hh,ww) in rects:
        it, il, ih, iw = rect_interior(t,l,hh,ww)
        for r in range(it, it+ih):
            for c in range(il, il+iw):
                out[r][c] = 2
    return grid, out

# -------------------- Task 2: c9f8e694 --------------------
def gen_c9f8e694_single(h: int = 12, w: int = 12, label_palette: List[int] = None) -> Tuple[Grid, Grid]:
    """Row label recolor: row's label at col 0 recolors all 5s in that row."""
    if label_palette is None:
        label_palette = [c for c in range(1,10) if c != 5]
    grid = zeros(h, w)
    for r in range(h):
        label = random.choice(label_palette)
        grid[r][0] = label
        k_blocks = random.randint(0, 2)
        for _ in range(k_blocks):
            bw = random.randint(2, max(2, w//3))
            start = random.randint(2, max(2, w - bw))
            for c in range(start, start + bw):
                grid[r][c] = 5
    out = deepcopy(grid)
    for r in range(h):
        label = out[r][0]
        for c in range(1, w):
            if out[r][c] == 5:
                out[r][c] = label
    return grid, out

# -------------------- Task 3: cbded52d (two-of-three) --------------------
TILE_SIZE = 2
SEP_POS = [2, 5]

def make_tiled_base() -> Grid:
    """Create 3x3 grid of 2x2 tiles separated by 0s; tiles pre-filled with 1s."""
    h = 8; w = 8
    g = zeros(h, w)
    for r in range(h):
        for c in range(w):
            if r in SEP_POS or c in SEP_POS:
                g[r][c] = 0
            else:
                g[r][c] = 1
    return g

def tile_rc_to_abs(rtile: int, ctile: int, pos: str) -> Tuple[int,int]:
    """Map tile coordinates and corner name to absolute grid indices."""
    r0 = rtile * (TILE_SIZE + 1)
    c0 = ctile * (TILE_SIZE + 1)
    if pos == "TL": return (r0 + 0, c0 + 0)
    if pos == "TR": return (r0 + 0, c0 + 1)
    if pos == "BL": return (r0 + 1, c0 + 0)
    if pos == "BR": return (r0 + 1, c0 + 1)
    raise ValueError("bad pos")

def gen_cbded52d_single() -> Tuple[Grid, Grid]:
    """
    Two-of-three rule:
    - In each tile-row/column and relative position (TL/TR/BL/BR), if there are EXACTLY TWO hints
      with the same color, fill the remaining third. Singles are ignored.
    """
    g = make_tiled_base()
    positions = ["TL", "TR", "BL", "BR"]
    color_choices = [2,3,4,5,6,7,8,9]
    hints: Dict[Tuple[int,int,str], int] = {}
    row_patterns: Dict[Tuple[int,str], Tuple[Tuple[int,int], int]] = {}
    col_patterns: Dict[Tuple[int,str], Tuple[Tuple[int,int], int]] = {}

    def creates_two_of_three(rt: int, ct: int, pos: str) -> bool:
        row_cols = [cc for (rr,cc,pp),_ in hints.items() if rr == rt and pp == pos]
        col_rows = [rr for (rr,cc,pp),_ in hints.items() if cc == ct and pp == pos]
        return (len(row_cols) + 1) == 2 or (len(col_rows) + 1) == 2

    # Row patterns (exactly two)
    num_row_patterns = random.randint(1, 2)
    tries = 0
    while len(row_patterns) < num_row_patterns and tries < 100:
        tries += 1
        rt = random.randint(0, 2)
        pos = random.choice(positions)
        if (rt, pos) in row_patterns:
            continue
        c1, c2 = random.sample([0,1,2], 2)
        color = random.choice(color_choices)
        ex_colors = [hints.get((rt, c, pos)) for c in (c1, c2)]
        ex_colors = [x for x in ex_colors if x is not None]
        if len(ex_colors) == 2 and ex_colors[0] != ex_colors[1]:
            continue
        if len(ex_colors) == 1:
            color = ex_colors[0]
        hints[(rt, c1, pos)] = color
        hints[(rt, c2, pos)] = color
        row_patterns[(rt, pos)] = ((c1, c2), color)

    # Column patterns (exactly two), color-coupled with crossing row-completions
    num_col_patterns = random.randint(1, 2)
    tries = 0
    while len(col_patterns) < num_col_patterns and tries < 100:
        tries += 1
        ct = random.randint(0, 2)
        pos = random.choice(positions)
        if (ct, pos) in col_patterns:
            continue
        r1, r2 = random.sample([0,1,2], 2)
        color = random.choice(color_choices)
        ex_colors = [hints.get((r, ct, pos)) for r in (r1, r2)]
        ex_colors = [x for x in ex_colors if x is not None]
        if len(ex_colors) == 2 and ex_colors[0] != ex_colors[1]:
            continue
        if len(ex_colors) == 1:
            color = ex_colors[0]
        missing_row = ({0,1,2} - {r1, r2}).pop()
        if (missing_row, pos) in row_patterns:
            (c_pair, row_color) = row_patterns[(missing_row, pos)]
            missing_col = ({0,1,2} - set(c_pair)).pop()
            if missing_col == ct:
                color = row_color
        hints[(r1, ct, pos)] = color
        hints[(r2, ct, pos)] = color
        col_patterns[(ct, pos)] = ((r1, r2), color)

    # Optional singles (ignored by the rule)
    num_singles = random.randint(0, 2)
    tries = 0
    while num_singles > 0 and tries < 200:
        tries += 1
        rt = random.randint(0, 2)
        ct = random.randint(0, 2)
        pos = random.choice(positions)
        if (rt, ct, pos) in hints:
            continue
        if creates_two_of_three(rt, ct, pos):
            continue
        hints[(rt, ct, pos)] = random.choice(color_choices)
        num_singles -= 1

    # INPUT grid
    for (rt, ct, pos), col in hints.items():
        r, c = tile_rc_to_abs(rt, ct, pos)
        g[r][c] = col

    # OUTPUT grid (apply completions)
    out = deepcopy(g)
    # Row-wise
    for rt in range(3):
        for pos in positions:
            entries = [(ct, hints[(rt, ct, pos)]) for ct in range(3) if (rt, ct, pos) in hints]
            if len(entries) == 2 and entries[0][1] == entries[1][1]:
                missing_ct = ({0,1,2} - {entries[0][0], entries[1][0]}).pop()
                r, c = tile_rc_to_abs(rt, missing_ct, pos)
                out[r][c] = entries[0][1]
    # Column-wise (don't overwrite row completions)
    for ct in range(3):
        for pos in positions:
            entries = [(rt, hints[(rt, ct, pos)]) for rt in range(3) if (rt, ct, pos) in hints]
            if len(entries) == 2 and entries[0][1] == entries[1][1]:
                missing_rt = ({0,1,2} - {entries[0][0], entries[1][0]}).pop()
                r, c = tile_rc_to_abs(missing_rt, ct, pos)
                if out[r][c] == 1:
                    out[r][c] = entries[0][1]
    return g, out

# -------------------- Task 4: d4a91cb9 --------------------
def gen_d4a91cb9_single(h: int = 12, w: int = 13) -> Tuple[Grid, Grid]:
    """Draw an L path of 4s from the 8 to the 2 (vertical then horizontal)."""
    g = zeros(h, w)
    r8, c8 = random.randint(0,h-1), random.randint(0,w-1)
    r2, c2 = r8, c8
    while r2 == r8 or c2 == c8:
        r2, c2 = random.randint(0,h-1), random.randint(0,w-1)
    g[r8][c8] = 8
    g[r2][c2] = 2
    out = deepcopy(g)
    step = 1 if r2 > r8 else -1
    for r in range(r8 + step, r2 + step, step):
        if (r, c8) != (r2, c2):
            out[r][c8] = 4
    step = 1 if c2 > c8 else -1
    for c in range(c8 + step, c2 + step, step):
        if (r2, c) != (r2, c2):
            out[r2][c] = 4
    return g, out

# -------------------- Task 5: d6ad076f --------------------
@dataclass
class Rect:
    """Axis-aligned rectangle with top-left, size and color."""
    top: int
    left: int
    height: int
    width: int
    color: int

def rect_bounds_interiors(rect: Rect):
    """Return outer and interior bounds for a rectangle.

    Returns (r0, r1, c0, c1, ri0, ri1, ci0, ci1) where r/c are inclusive indices
    and interior indices exclude the 1-cell border.
    """
    r0, r1 = rect.top, rect.top + rect.height - 1
    c0, c1 = rect.left, rect.left + rect.width - 1
    if rect.height >= 3:
        ri0, ri1 = r0 + 1, r1 - 1
    else:
        ri0, ri1 = r0 + 1, r0
    if rect.width >= 3:
        ci0, ci1 = c0 + 1, c1 - 1
    else:
        ci0, ci1 = c0 + 1, c0
    return (r0, r1, c0, c1, ri0, ri1, ci0, ci1)

def gen_two_rects_gap(h: int, w: int) -> Tuple[Grid, Rect, Rect, str]:
    """Place two non-overlapping rectangles with a strict zero gap between them.

    Returns grid with rectangles drawn, the two Rects, and mode: 'stacked' or 'side'.
    The rectangles' interior overlaps along the orthogonal axis must be non-empty.
    """
    g = zeros(h, w)
    colors = [c for c in range(1,10) if c != 8]
    for _ in range(1000):
        mode = random.choice(["stacked", "side"])
        ha = random.randint(3, max(3, h//2))
        wa = random.randint(3, max(3, w//2))
        hb = random.randint(3, max(3, h//2))
        wb = random.randint(3, max(3, w//2))
        if mode == "stacked":
            max_topA = h - ha - hb - 1
            if max_topA < 0:
                continue
            topA = random.randint(0, max_topA)
            min_topB = topA + ha + 1
            max_topB = h - hb
            if min_topB > max_topB:
                continue
            topB = random.randint(min_topB, max_topB)
            leftA = random.randint(0, w - wa)
            leftB = random.randint(0, w - wb)
            rectA = Rect(topA, leftA, ha, wa, random.choice(colors))
            rectB = Rect(topB, leftB, hb, wb, random.choice([c for c in colors if c != rectA.color]))
            _, _, _, _, _, _, ci0A, ci1A = rect_bounds_interiors(rectA)
            _, _, _, _, _, _, ci0B, ci1B = rect_bounds_interiors(rectB)
            ci0 = max(ci0A, ci0B)
            ci1 = min(ci1A, ci1B)
            gap = rectB.top - (rectA.top + rectA.height - 1) - 1
            if ci0 <= ci1 and gap >= 1:
                draw_rect(g, rectA.top, rectA.left, rectA.height, rectA.width, rectA.color)
                draw_rect(g, rectB.top, rectB.left, rectB.height, rectB.width, rectB.color)
                return g, rectA, rectB, mode
        else:
            max_leftA = w - wa - wb - 1
            if max_leftA < 0:
                continue
            leftA = random.randint(0, max_leftA)
            min_leftB = leftA + wa + 1
            max_leftB = w - wb
            if min_leftB > max_leftB:
                continue
            leftB = random.randint(min_leftB, max_leftB)
            topA = random.randint(0, h - ha)
            topB = random.randint(0, h - hb)
            rectA = Rect(topA, leftA, ha, wa, random.choice(colors))
            rectB = Rect(topB, leftB, hb, wb, random.choice([c for c in colors if c != rectA.color]))
            _, _, _, _, ri0A, ri1A, _, _ = rect_bounds_interiors(rectA)
            _, _, _, _, ri0B, ri1B, _, _ = rect_bounds_interiors(rectB)
            ri0 = max(ri0A, ri0B)
            ri1 = min(ri1A, ri1B)
            gap = rectB.left - (rectA.left + rectA.width - 1) - 1
            if ri0 <= ri1 and gap >= 1:
                draw_rect(g, rectA.top, rectA.left, rectA.height, rectA.width, rectA.color)
                draw_rect(g, rectB.top, rectB.left, rectB.height, rectB.width, rectB.color)
                return g, rectA, rectB, mode
    raise RuntimeError("Failed to place two rectangles with desired relation (tried 1000 times).")

def apply_bridge_eights(grid: Grid, rectA: Rect, rectB: Rect, mode: str) -> Grid:
    """Bridge the gap between two rectangles with color-8 band across the overlap."""
    out = deepcopy(grid)
    if mode == "stacked":
        top_rect = rectA if rectA.top < rectB.top else rectB
        bottom_rect = rectB if rectB.top > rectA.top else rectA
        gap_top = top_rect.top + top_rect.height
        gap_bottom = bottom_rect.top - 1
        ggap = gap_bottom - gap_top + 1
        _, _, _, _, _, _, ci0A, ci1A = rect_bounds_interiors(top_rect)
        _, _, _, _, _, _, ci0B, ci1B = rect_bounds_interiors(bottom_rect)
        ci0 = max(ci0A, ci0B)
        ci1 = min(ci1A, ci1B)
        if ci0 <= ci1 and ggap >= 1:
            # width is always overlap - 2 (minus the borders)
            width = ci1 - ci0 + 1
            cols = list(range(ci0, ci0 + width))
            for r in range(gap_top, gap_top + ggap):
                for c in cols:
                    out[r][c] = 8
    else:
        left_rect = rectA if rectA.left < rectB.left else rectB
        right_rect = rectB if rectB.left > rectA.left else rectA
        gap_left = left_rect.left + left_rect.width
        gap_right = right_rect.left - 1
        ggap = gap_right - gap_left + 1
        _, _, _, _, ri0A, ri1A, _, _ = rect_bounds_interiors(left_rect)
        _, _, _, _, ri0B, ri1B, _, _ = rect_bounds_interiors(right_rect)
        ri0 = max(ri0A, ri0B)
        ri1 = min(ri1A, ri1B)
        if ri0 <= ri1 and ggap >= 1:
            # height is always overlap - 2 (minus the borders)
            height = ri1 - ri0 + 1
            rows = list(range(ri0, ri0 + height))
            for r in rows:
                for c in range(gap_left, gap_left + ggap):
                    out[r][c] = 8
    return out

def gen_d6ad076f_single(h: int = 10, w: int = 10) -> Tuple[Grid, Grid]:
    """Bridge two solid rectangles with 8s across the gap per spec."""
    g, ra, rb, mode = gen_two_rects_gap(h, w)
    out = apply_bridge_eights(g, ra, rb, mode)
    return g, out

# -------------------- Uniqueness machinery --------------------
def serialize_grid(grid: Grid) -> bytes:
    """Compactly serialize a grid to bytes for hashing."""
    h = len(grid); w = len(grid[0]) if h else 0
    b = bytearray()
    b.append(h & 0xFF); b.append(w & 0xFF)
    for row in grid:
        b.extend(row)
    return bytes(b)

def pair_key(inp: Grid, out: Grid) -> str:
    """Compute a stable 128-bit key for an (input, output) pair."""
    m = hashlib.blake2b(digest_size=16)
    m.update(serialize_grid(inp)); m.update(b'|'); m.update(serialize_grid(out))
    return m.hexdigest()

def generate_unique_pairs(gen_single_fn, n_total: int,
                          attempts_per_example: int = 50,
                          progress: bool = True) -> Tuple[List[Tuple[Grid,Grid]], int, int]:
    """
    Try to generate n_total UNIQUE (input,output) pairs.
    Returns (pairs, unique_count, attempts) where len(pairs)==unique_count.
    Stops early if attempts exceed n_total*attempts_per_example.
    """
    keys = set()
    pairs: List[Tuple[Grid,Grid]] = []
    max_attempts = max(1, n_total * attempts_per_example)
    attempts = 0

    def _pbar(iterable):
        if progress and tqdm is not None:
            return tqdm(iterable, total=max_attempts, desc="Generating", leave=False)
        return iterable

    for _ in _pbar(range(max_attempts)):
        attempts += 1
        try:
            inp, out = gen_single_fn()
        except Exception:
            continue
        k = pair_key(inp, out)
        if k not in keys:
            keys.add(k)
            pairs.append((inp, out))
            if len(pairs) >= n_total:
                break
    return pairs, len(pairs), attempts

def make_unique_train_test(gen_single_fn,
                           n_train: int,
                           n_test: int,
                           attempts_per_example: int = 50,
                           strict: bool = True,
                           progress: bool = True) -> Tuple[List[Tuple[Grid,Grid]], List[Tuple[Grid,Grid]], Dict[str,int], bool]:
    """
    Generate UNIQUE (train+test) pairs, then split so train and test are disjoint.
    If strict and not enough unique are found, returns partial=False and stats.
    If not strict, returns as many as possible (train filled first), partial=True.
    """
    need = n_train + n_test
    pairs, unique_count, attempts = generate_unique_pairs(gen_single_fn, need, attempts_per_example, progress)
    stats = {
        "requested_total": need,
        "unique_generated": unique_count,
        "attempts": attempts,
        "duplicates_avoided": attempts - unique_count
    }
    if unique_count < need and strict:
        return [], [], stats, False
    train = pairs[:min(n_train, unique_count)]
    test = pairs[min(n_train, unique_count):min(need, unique_count)]
    return train, test, stats, (unique_count >= need)

# -------------------- Task registry --------------------
TASKS = {
    "bb43febb": lambda: gen_bb43febb_single(10,10),
    "c9f8e694": lambda: gen_c9f8e694_single(10,10),
    "cbded52d": lambda: gen_cbded52d_single(),
    "d4a91cb9": lambda: gen_d4a91cb9_single(10,10),
    "d6ad076f": lambda: gen_d6ad076f_single(10,10),
}
