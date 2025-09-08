"""
ARC synthetic task generators for 5 selected puzzles.

Tasks covered:
1) bb43febb  – Replace the interior of each solid 5-rectangle by 2; keep 5-borders.
2) c9f8e694  – Row label recolor: each row's label at col 0 recolors all 5s in that row.
3) cbd:ed52d – 3x3 grid of 2x2 tiles (separated by 0s); propagate hints by tile-row/column.
4) d4a91cb9  – Draw an L path of 4s from 8 to 2 (vertical first, then horizontal).
5) d6ad076f  – Bridge two solid rectangles with 8s across their gap (see rule below).

Notes for d6ad076f:
- Two solid rectangles (colors != 0,8), non-overlapping, either stacked (vertical gap with horizontal interior overlap)
  or side-by-side (horizontal gap with vertical interior overlap).
- Let g be the gap thickness along the separation axis (strict zeros between rects).
- Let o be the thickness of the *interior overlap* (excluding outer borders) along the orthogonal axis.
- Output adds an 8-rectangle spanning: gap size g along the separation axis, and min(o, g) along the orthogonal axis,
  anchored at the start of the interior overlap band (left for stacked; top for side-by-side).

Everything is done with pure Python lists (no numpy). Adjust counts and seeds in the __main__ block as needed.
"""

import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict

Grid = List[List[int]]

# -------------------- basic helpers --------------------

def zeros(h: int, w: int) -> Grid:
    return [[0 for _ in range(w)] for _ in range(h)]

def draw_rect(grid: Grid, top: int, left: int, height: int, width: int, color: int) -> None:
    """Fill an axis-aligned solid rectangle (inclusive area)."""
    for r in range(top, top + height):
        for c in range(left, left + width):
            grid[r][c] = color

def rect_interior(top: int, left: int, height: int, width: int) -> Tuple[int, int, int, int]:
    """
    Return (top, left, height, width) of the *interior* (excluding borders).
    If interior is empty (height<3 or width<3), returns a zero-sized shape (h or w <= 0).
    """
    if height < 3 or width < 3:
        return (top+1, left+1, 0, 0)
    return (top + 1, left + 1, height - 2, width - 2)

def place_non_overlapping_rectangles(h: int, w: int, n: int, min_hw=(3,3), max_trials=500) -> List[Tuple[int,int,int,int]]:
    """
    Randomly place up to n solid rectangles (top,left,height,width) without overlap.
    Touching edges is not allowed; positive-area overlap is forbidden.
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
        # forbid positive-area overlap
        overlap = False
        for (t,l,hh,ww) in rects:
            if not (top+rh < t or t+hh < top or left+rw < l or l+ww < left):
                overlap = True
                break
        if not overlap:
            rects.append(candidate)
    return rects

def to_arc_examples(pairs: List[Tuple[Grid, Grid]]) -> List[Dict[str, Grid]]:
    return [{"input": pin, "output": pout} for (pin, pout) in pairs]

def save_task_json(path: str, train_pairs: List[Tuple[Grid,Grid]], test_pairs: List[Tuple[Grid,Grid]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"train": to_arc_examples(train_pairs), "test": to_arc_examples(test_pairs)}
    with open(path, "w") as f:
        json.dump(obj, f)

# -------------------- Task 1: bb43febb --------------------
# Rule: One or more solid rectangles of color 5. Replace the interior of each with color 2.

def gen_bb43febb_single(h: int = 10, w: int = 10) -> Tuple[Grid, Grid]:
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

def gen_bb43febb(n_train=20, n_test=5, h: int = 10, w: int = 10, seed: int = 0):
    random.seed(seed)
    train = [gen_bb43febb_single(h, w) for _ in range(n_train)]
    test  = [gen_bb43febb_single(h, w) for _ in range(n_test)]
    return train, test

# -------------------- Task 2: c9f8e694 --------------------
# Rule: Each row has a label (col 0, non-zero and != 5). Replace all 5s in that row by the label color.

def gen_c9f8e694_single(h: int = 12, w: int = 12, label_palette: List[int] = None) -> Tuple[Grid, Grid]:
    if label_palette is None:
        label_palette = [c for c in range(1,10) if c != 5]  # avoid 0 and 5
    grid = zeros(h, w)
    for r in range(h):
        label = random.choice(label_palette)
        grid[r][0] = label
        # place 0..2 separate 5-blocks in the row
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

def gen_c9f8e694(n_train=20, n_test=5, h: int = 12, w: int = 12, seed: int = 1):
    random.seed(seed)
    train = [gen_c9f8e694_single(h, w) for _ in range(n_train)]
    test  = [gen_c9f8e694_single(h, w) for _ in range(n_test)]
    return train, test

# -------------------- Task 3: cbd:ed52d --------------------
# Rule: 3x3 tiles of size 2x2 with zero separators at rows/cols 2 and 5. Base color inside tiles is 1.
# Hints (non-1 colors) placed at TL/TR/BL/BR positions propagate across the entire tile-row and tile-column.

TILE_SIZE = 2
SEP_POS = [2, 5]  # zero rows/cols (separators)

def make_tiled_base() -> Grid:
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
    # rtile, ctile in {0,1,2}. pos ∈ {"TL","TR","BL","BR"}.
    r0 = rtile * (TILE_SIZE + 1)   # +1 for separator row
    c0 = ctile * (TILE_SIZE + 1)   # +1 for separator col
    if pos == "TL": return (r0 + 0, c0 + 0)
    if pos == "TR": return (r0 + 0, c0 + 1)
    if pos == "BL": return (r0 + 1, c0 + 0)
    if pos == "BR": return (r0 + 1, c0 + 1)
    raise ValueError("bad pos")

def gen_cbded52d_single() -> Tuple[Grid, Grid]:
    """
    Generate a (input, output) pair for the cbd:ed52d task using the *two-of-three* rule:

    - Work on a 3x3 board of 2x2 tiles (separated by 0-rows/cols). Inside tiles default to color 1.
    - We place some hints (non-1 colors) at TL/TR/BL/BR positions inside tiles.
    - OUTPUT RULE:
        * Row-wise: if a tile-row has exactly TWO hints at the same relative position and same color,
          fill the remaining third tile at that position with that color. If only one hint, ignore.
        * Column-wise: same rule for tile-columns.
      If a cell would be completed by both row and column rules, this generator couples colors to avoid conflicts.
    """
    g = make_tiled_base()
    positions = ["TL", "TR", "BL", "BR"]
    color_choices = [2,3,4,5,6,7,8,9]  # non-1, non-zero

    # Hints live at (rtile, ctile, pos) -> color
    hints: Dict[Tuple[int,int,str], int] = {}

    # Keep track of the explicit "two-hint patterns" we add so we can couple colors if completions cross.
    row_patterns: Dict[Tuple[int,str], Tuple[Tuple[int,int], int]] = {}  # (rt,pos) -> ((c1,c2), color)
    col_patterns: Dict[Tuple[int,str], Tuple[Tuple[int,int], int]] = {}  # (ct,pos) -> ((r1,r2), color)

    # ---- helpers ----
    def creates_two_of_three(rt: int, ct: int, pos: str) -> bool:
        """Would adding a single hint here accidentally create a 2-of-3 in its row/column?
        We use this to place 'single' hints that are meant to be ignored by the rule."""
        row_cols = [cc for (rr,cc,pp),_ in hints.items() if rr == rt and pp == pos]
        col_rows = [rr for (rr,cc,pp),_ in hints.items() if cc == ct and pp == pos]
        return (len(row_cols) + 1) == 2 or (len(col_rows) + 1) == 2

    # ---- place some row patterns (exactly TWO hints in a row at the same pos) ----
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

        # Unify with any existing hints already placed at those locations
        ex_colors = [hints.get((rt, c, pos)) for c in (c1, c2)]
        ex_colors = [x for x in ex_colors if x is not None]
        if len(ex_colors) == 2 and ex_colors[0] != ex_colors[1]:
            continue  # conflicting colors already there; try another choice
        if len(ex_colors) == 1:
            color = ex_colors[0]

        # Place the two hints
        hints[(rt, c1, pos)] = color
        hints[(rt, c2, pos)] = color
        row_patterns[(rt, pos)] = ((c1, c2), color)

    # ---- place some column patterns (exactly TWO hints in a column at the same pos) ----
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

        # Unify with existing hints at those two locations if present
        ex_colors = [hints.get((r, ct, pos)) for r in (r1, r2)]
        ex_colors = [x for x in ex_colors if x is not None]
        if len(ex_colors) == 2 and ex_colors[0] != ex_colors[1]:
            continue
        if len(ex_colors) == 1:
            color = ex_colors[0]

        # ALSO couple with any crossing row completion to avoid conflicting completions:
        #   If for this column pattern the missing row is r3,
        #   and there exists a row pattern (r3,pos) whose missing column is exactly ct,
        #   then force the column pattern's color to the row pattern's color.
        missing_row = ({0,1,2} - {r1, r2}).pop()
        if (missing_row, pos) in row_patterns:
            (c_pair, row_color) = row_patterns[(missing_row, pos)]
            missing_col = ({0,1,2} - set(c_pair)).pop()
            if missing_col == ct:
                color = row_color  # couple colors to avoid a future conflict

        # Place the two hints
        hints[(r1, ct, pos)] = color
        hints[(r2, ct, pos)] = color
        col_patterns[(ct, pos)] = ((r1, r2), color)

    # ---- optionally place a few single hints that the rule should IGNORE ----
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
            continue  # would accidentally form a 2-of-3; skip
        hints[(rt, ct, pos)] = random.choice(color_choices)
        num_singles -= 1

    # ---- build INPUT grid with hints ----
    for (rt, ct, pos), col in hints.items():
        r, c = tile_rc_to_abs(rt, ct, pos)
        g[r][c] = col

    # ---- build OUTPUT grid: apply two-of-three completions ----
    out = deepcopy(g)

    # Row-wise completion: if exactly TWO hints (same color), fill the missing third
    for rt in range(3):
        for pos in positions:
            entries = [(ct, hints[(rt, ct, pos)]) for ct in range(3) if (rt, ct, pos) in hints]
            if len(entries) == 2 and entries[0][1] == entries[1][1]:
                missing_ct = ({0,1,2} - {entries[0][0], entries[1][0]}).pop()
                r, c = tile_rc_to_abs(rt, missing_ct, pos)
                out[r][c] = entries[0][1]

    # Column-wise completion: same, but don't overwrite a cell already set by the row rule
    for ct in range(3):
        for pos in positions:
            entries = [(rt, hints[(rt, ct, pos)]) for rt in range(3) if (rt, ct, pos) in hints]
            if len(entries) == 2 and entries[0][1] == entries[1][1]:
                missing_rt = ({0,1,2} - {entries[0][0], entries[1][0]}).pop()
                r, c = tile_rc_to_abs(missing_rt, ct, pos)
                if out[r][c] == 1:  # untouched (inside tiles are 1)
                    out[r][c] = entries[0][1]

    return g, out

def gen_cbded52d(n_train=20, n_test=5, seed: int = 2):
    random.seed(seed)
    train = [gen_cbded52d_single() for _ in range(n_train)]
    test  = [gen_cbded52d_single() for _ in range(n_test)]
    return train, test

# -------------------- Task 4: d4a91cb9 --------------------
# Rule: Exactly one 8 and one 2. Draw vertical from 8 to row of 2, then horizontal to 2's column, with 4s.

def gen_d4a91cb9_single(h: int = 12, w: int = 13) -> Tuple[Grid, Grid]:
    g = zeros(h, w)
    r8, c8 = random.randint(0,h-1), random.randint(0,w-1)
    r2, c2 = r8, c8
    while r2 == r8 or c2 == c8:
        r2, c2 = random.randint(0,h-1), random.randint(0,w-1)
    g[r8][c8] = 8
    g[r2][c2] = 2

    out = deepcopy(g)
    # vertical leg
    step = 1 if r2 > r8 else -1
    for r in range(r8 + step, r2 + step, step):
        if (r, c8) != (r2, c2):
            out[r][c8] = 4
    # horizontal leg
    step = 1 if c2 > c8 else -1
    for c in range(c8 + step, c2 + step, step):
        if (r2, c) != (r2, c2):
            out[r2][c] = 4
    return g, out

def gen_d4a91cb9(n_train=20, n_test=5, h: int = 12, w: int = 13, seed: int = 3):
    random.seed(seed)
    train = [gen_d4a91cb9_single(h, w) for _ in range(n_train)]
    test  = [gen_d4a91cb9_single(h, w) for _ in range(n_test)]
    return train, test

# -------------------- Task 5: d6ad076f --------------------
# Rule: Bridge two solid rectangles (colors != 0,8) across their gap with an 8-rectangle:
#       gap size g along separation axis × min(o, g) along orthogonal axis (anchored to band start).

@dataclass
class Rect:
    top: int
    left: int
    height: int
    width: int
    color: int

def rect_bounds_interiors(rect: Rect):
    """
    returns: (r0, r1, c0, c1, ri0, ri1, ci0, ci1)
    where (r0..r1, c0..c1) are full rect bounds inclusive,
    and (ri0..ri1, ci0..ci1) are *interior* bounds exclusive of outer border.
    Interior ranges may be empty (ri0 > ri1 or ci0 > ci1) for thin rectangles.
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
    """
    Create two non-overlapping rectangles (height,width >=3) either stacked (vertical separation)
    with horizontal interior overlap, or side-by-side (horizontal separation) with vertical interior overlap.
    Returns (grid, rectA, rectB, mode) where mode in {"stacked","side"}.
    Robust against infeasible samplings.
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
            # ensure room for a positive vertical gap
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
            # horizontal interior overlap + positive gap
            _, _, _, _, _, _, ci0A, ci1A = rect_bounds_interiors(rectA)
            _, _, _, _, _, _, ci0B, ci1B = rect_bounds_interiors(rectB)
            ci0 = max(ci0A, ci0B)
            ci1 = min(ci1A, ci1B)
            gap = rectB.top - (rectA.top + rectA.height - 1) - 1
            if ci0 <= ci1 and gap >= 1:
                draw_rect(g, rectA.top, rectA.left, rectA.height, rectA.width, rectA.color)
                draw_rect(g, rectB.top, rectB.left, rectB.height, rectB.width, rectB.color)
                return g, rectA, rectB, mode

        else:  # side-by-side
            # ensure room for a positive horizontal gap
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
            # vertical interior overlap + positive gap
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
    out = deepcopy(grid)
    if mode == "stacked":
        # vertical gap between A (top) and B (bottom)
        top_rect = rectA if rectA.top < rectB.top else rectB
        bottom_rect = rectB if rectB.top > rectA.top else rectA
        gap_top = top_rect.top + top_rect.height     # one row below top rect
        gap_bottom = bottom_rect.top - 1             # one row above bottom rect
        g = gap_bottom - gap_top + 1                 # gap height

        _, _, _, _, _, _, ci0A, ci1A = rect_bounds_interiors(top_rect)
        _, _, _, _, _, _, ci0B, ci1B = rect_bounds_interiors(bottom_rect)
        ci0 = max(ci0A, ci0B)
        ci1 = min(ci1A, ci1B)

        if ci0 <= ci1 and g >= 1:
            # width is overlap - 2 for borders
            width = max(ci1 - ci0 + 1, 1)
            cols = list(range(ci0, ci0 + width))  # anchor at *left* of overlap band
            for r in range(gap_top, gap_top + g):
                for c in cols:
                    out[r][c] = 8

    else:  # side-by-side
        left_rect = rectA if rectA.left < rectB.left else rectB
        right_rect = rectB if rectB.left > rectA.left else rectA
        gap_left = left_rect.left + left_rect.width
        gap_right = right_rect.left - 1
        g = gap_right - gap_left + 1                 # gap width

        _, _, _, _, ri0A, ri1A, _, _ = rect_bounds_interiors(left_rect)
        _, _, _, _, ri0B, ri1B, _, _ = rect_bounds_interiors(right_rect)
        ri0 = max(ri0A, ri0B)
        ri1 = min(ri1A, ri1B)

        if ri0 <= ri1 and g >= 1:
            # height is overlap - 2 for borders
            height = max(ri1 - ri0 + 1, 1)
            rows = list(range(ri0, ri0 + height))   # anchor at *top* of overlap band
            for r in rows:
                for c in range(gap_left, gap_left + g):
                    out[r][c] = 8
    return out

def gen_d6ad076f_single(h: int = 10, w: int = 10) -> Tuple[Grid, Grid]:
    g, ra, rb, mode = gen_two_rects_gap(h, w)
    out = apply_bridge_eights(g, ra, rb, mode)
    return g, out

def gen_d6ad076f(n_train=20, n_test=5, h: int = 10, w: int = 10, seed: int = 4):
    random.seed(seed)
    train = [gen_d6ad076f_single(h, w) for _ in range(n_train)]
    test  = [gen_d6ad076f_single(h, w) for _ in range(n_test)]
    return train, test

# -------------------- batch generate & save --------------------

def main():
    # You can tweak these defaults.
    SEED = 2025
    tasks = {
        "bb43febb": (gen_bb43febb, {"n_train": 30, "n_test": 10, "h": 10, "w": 10, "seed": SEED}),
        "c9f8e694": (gen_c9f8e694, {"n_train": 30, "n_test": 10, "h": 10, "w": 10, "seed": SEED+1}),
        "cbded52d": (gen_cbded52d, {"n_train": 30, "n_test": 10, "seed": SEED+2}),
        "d4a91cb9": (gen_d4a91cb9, {"n_train": 30, "n_test": 10, "h": 10, "w": 10, "seed": SEED+3}),
        "d6ad076f": (gen_d6ad076f, {"n_train": 30, "n_test": 10, "h": 10, "w": 10, "seed": SEED+4}),
    }

    base_dir = "arc_synth"
    os.makedirs(base_dir, exist_ok=True)
    written = []
    for tid, (fn, kwargs) in tasks.items():
        train_pairs, test_pairs = fn(**kwargs)
        path = os.path.join(base_dir, f"synth_{tid}.json")
        save_task_json(path, train_pairs, test_pairs)
        written.append(path)

    print("Wrote JSON files:")
    for p in written:
        print(" -", p)

if __name__ == "__main__":
    main()
