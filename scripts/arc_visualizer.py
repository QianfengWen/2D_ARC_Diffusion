"""
ARC Visualizer

Features
- draw_grid(grid): render a single ARC grid with a consistent 10-color palette
- show_pair(inp, out, diff=True): side-by-side + highlight changed cells
- visualize_task_json(path, split='train', ...): show many pairs from ARC-AGI JSON
- save_grid_png / save_pair_png: export PNGs (no axes, pixel-aligned)

Dependencies: matplotlib (no numpy or pillow required)
"""

from __future__ import annotations
import json
import os
from readline import set_pre_input_hook
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

Grid = List[List[int]]

# -------------------- Palette & Colormap --------------------

# Widely used ARC palette (similar to many Kaggle ARC notebooks)
DEFAULT_PALETTE = {
    0: "#000000",  # black (background)
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # cyan
    9: "#85144b",  # maroon
}

def make_arc_cmap(palette: Dict[int, str] = DEFAULT_PALETTE):
    """
    Build a ListedColormap + BoundaryNorm that maps integers 0..9 to colors.
    """
    # Ensure indices 0..9 exist; fallback to black if missing.
    colors = [palette.get(i, "#000000") for i in range(10)]
    cmap = ListedColormap(colors, name="arc_palette", N=10)
    # Boundaries such that each integer i maps to bin centered at i
    boundaries = [i - 0.5 for i in range(11)]  # -0.5..9.5 step 1
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    return cmap, norm

# -------------------- Drawing primitives --------------------

def _hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _draw_grid_lines(ax, h: int, w: int, color=(1,1,1,0.15), lw=0.6):
    """
    Light grid lines over each cell. Uses minor ticks at half-integers.
    """
    # Positions at cell boundaries
    xlines = [x - 0.5 for x in range(w + 1)]
    ylines = [y - 0.5 for y in range(h + 1)]
    ax.set_xticks(xlines, minor=True)
    ax.set_yticks(ylines, minor=True)
    ax.grid(which='minor', color=color, linewidth=lw)

def draw_grid(grid: Grid,
              title: Optional[str] = None,
              ax: Optional[plt.Axes] = None,
              palette: Dict[int, str] = DEFAULT_PALETTE,
              show_grid: bool = True) -> plt.Axes:
    """
    Render a single ARC grid (list-of-lists with ints 0..9).

    Returns the matplotlib Axes to allow further annotation.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    h = len(grid)
    w = len(grid[0]) if h else 0

    cmap, norm = make_arc_cmap(palette)
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest', origin='upper')
    _hide_axes(ax)
    if show_grid:
        _draw_grid_lines(ax, h, w)

    if title:
        ax.set_title(title)
    return ax

def _draw_diff_boxes(ax: plt.Axes, inp: Grid, out: Grid,
                     edge_color: str = "#FFFFFF", lw: float = 1.2):
    """
    Draw a thin rectangle around cells that changed from input to output.
    """
    h = len(inp)
    w = len(inp[0]) if h else 0
    for r in range(h):
        for c in range(w):
            if inp[r][c] != out[r][c]:
                rect = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                 edgecolor=edge_color, linewidth=lw)
                ax.add_patch(rect)

def show_pair(inp: Grid,
              out: Grid,
              titles: Tuple[str, str] = ("input", "output"),
              diff: bool = True,
              palette: Dict[int, str] = DEFAULT_PALETTE,
              figsize: Tuple[float, float] = (6.0, 3.0)):
    """
    Side-by-side visualization of an (input, output) pair.
    If diff=True, draws a white outline around changed output cells.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    draw_grid(inp, titles[0], ax=ax1, palette=palette)
    draw_grid(out, titles[1], ax=ax2, palette=palette)
    if diff:
        _draw_diff_boxes(ax2, inp, out)
    plt.tight_layout()
    return fig

# -------------------- JSON loading & multi-pair layout --------------------

def _load_arc_json(path: str) -> Dict[str, List[Dict[str, Grid]]]:
    with open(path, "r") as f:
        data = json.load(f)
    # Basic validation
    if not isinstance(data, dict):
        raise ValueError("File does not look like ARC-AGI JSON (must be an object).")
    return data

def visualize_task_json(path: str,
                        split: str = "train",
                        max_pairs: Optional[int] = None,
                        shuffle: bool = False,
                        seed: Optional[int] = None,
                        cols: int = 3,
                        diff: bool = True,
                        palette: Dict[int, str] = DEFAULT_PALETTE,
                        suptitle: Optional[str] = None,
                        pair_figsize: Tuple[float, float] = (5.0, 2.6)):
    """
    Visualize multiple (input, output) examples from an ARC-AGI JSON file.

    - split: 'train' or 'test'
    - max_pairs: limit how many pairs to display
    - cols: number of pair-rows per figure row (each pair uses two axes)
    """
    data = _load_arc_json(path)
    pairs = data[split]
    items = [(ex["input"], ex["output"]) for ex in pairs]

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(items)

    if max_pairs is not None:
        items = items[:max_pairs]

    n = len(items)
    if n == 0:
        raise ValueError(f"No examples found in split '{split}'.")

    # Build a single figure with (rows x 2) subplots (input|output per example)
    rows = n
    fig, axes = plt.subplots(rows, 2, figsize=(pair_figsize[0]*1.0, pair_figsize[1]*rows))
    if rows == 1:
        axes = [axes]  # make iterable shape [ (ax_in, ax_out) ]

    for i, (inp, out) in enumerate(items):
        ax_in, ax_out = axes[i]
        draw_grid(inp, "input", ax=ax_in, palette=palette)
        draw_grid(out, "output", ax=ax_out, palette=palette)
        if diff:
            _draw_diff_boxes(ax_out, inp, out)

    if suptitle is None:
        suptitle = f"{os.path.basename(path)} [{split}]"
    fig.suptitle(suptitle, y=1.0)
    fig.tight_layout()
    return fig

# -------------------- PNG export --------------------

def save_grid_png(grid: Grid,
                  filename: str,
                  palette: Dict[int, str] = DEFAULT_PALETTE,
                  cell_size: int = 32,
                  grid_lines: bool = False,
                  dpi: int = 100):
    """
    Save a single grid to PNG (no axes, tight bbox).
    cell_size controls approximate pixel scaling via figure size.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    # Compute figure size in inches: pixels / dpi
    px_w = max(1, w * cell_size)
    px_h = max(1, h * cell_size)
    fig_w = px_w / dpi
    fig_h = px_h / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    draw_grid(grid, ax=ax, palette=palette, show_grid=grid_lines)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_pair_png(inp: Grid,
                  out: Grid,
                  filename: str,
                  palette: Dict[int, str] = DEFAULT_PALETTE,
                  diff: bool = True,
                  cell_size: int = 32,
                  pad_cells: int = 1,
                  dpi: int = 100):
    """
    Save a side-by-side (input|output) composited PNG with optional diff boxes.
    pad_cells adds a small blank spacer (in cells) between input and output.
    """
    h = len(inp)
    w_in = len(inp[0]) if h else 0
    w_out = len(out[0]) if len(out) else 0
    w_pad = pad_cells

    px_w = (w_in + w_pad + w_out) * cell_size
    px_h = max(len(inp), len(out)) * cell_size
    fig_w = px_w / dpi
    fig_h = px_h / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # Left axes for input
    ax1 = fig.add_axes([0, 0, w_in/(w_in + w_pad + w_out), 1])
    draw_grid(inp, ax=ax1, palette=palette, show_grid=False)
    # Right axes for output
    ax2 = fig.add_axes([(w_in + w_pad)/(w_in + w_pad + w_out), 0,
                        w_out/(w_in + w_pad + w_out), 1])
    draw_grid(out, ax=ax2, palette=palette, show_grid=False)
    if diff:
        _draw_diff_boxes(ax2, inp, out)

    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_triplet_png(inp: Grid,
                     pred: Grid,
                     gt: Optional[Grid],
                     filename: str,
                     palette: Dict[int, str] = DEFAULT_PALETTE,
                     cell_size: int = 32,
                     pad_cells: int = 1,
                     dpi: int = 100):
    """
    Save a triplet image: input | pred | gt (if provided). If gt exists, draw diff boxes on pred vs gt.
    """
    h = len(inp)
    w_in = len(inp[0]) if h else 0
    w_pred = len(pred[0]) if len(pred) else 0
    w_gt = len(gt[0]) if (gt is not None and len(gt)) else 0
    cols = 3 if gt is not None else 2

    total_w_cells = w_in + pad_cells + w_pred + (pad_cells + w_gt if gt is not None else 0)
    px_w = total_w_cells * cell_size
    px_h = max(len(inp), len(pred), len(gt) if gt is not None else 0) * cell_size
    fig_w = px_w / dpi
    fig_h = max(1, px_h) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # input
    acc_w = 0.0
    frac_in = w_in / total_w_cells if total_w_cells else 1.0
    ax_in = fig.add_axes([acc_w, 0, frac_in, 1])
    draw_grid(inp, ax=ax_in, palette=palette, show_grid=False)
    acc_w += frac_in
    # pred spacer + pred
    if pad_cells:
        acc_w += pad_cells / total_w_cells
    frac_pred = w_pred / total_w_cells if total_w_cells else 0.0
    ax_pred = fig.add_axes([acc_w, 0, frac_pred, 1])
    draw_grid(pred, ax=ax_pred, palette=palette, show_grid=False)
    # diff boxes use gt if available
    if gt is not None:
        _draw_diff_boxes(ax_pred, gt, pred)
    acc_w += frac_pred
    # gt
    if gt is not None:
        if pad_cells:
            acc_w += pad_cells / total_w_cells
        frac_gt = w_gt / total_w_cells if total_w_cells else 0.0
        ax_gt = fig.add_axes([acc_w, 0, frac_gt, 1])
        draw_grid(gt, ax=ax_gt, palette=palette, show_grid=False)

    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_dataset_images(json_path: str,
                        out_dir: str,
                        split: str = "train",
                        prefix: Optional[str] = None,
                        side_by_side: bool = True,
                        palette: Dict[int, str] = DEFAULT_PALETTE,
                        cell_size: int = 32,
                        dpi: int = 100):
    """
    Batch export PNGs from an ARC-AGI JSON file.
    Creates:
      - <prefix>_<split>_<i>_input.png
      - <prefix>_<split>_<i>_output.png
      - (optional) <prefix>_<split>_<i>_pair.png
    """
    data = _load_arc_json(json_path)
    items = data[split]
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(json_path))[0]

    os.makedirs(out_dir, exist_ok=True)

    for i, ex in enumerate(items):
        inp, out = ex["input"], ex["output"]
        base = f"{prefix}_{split}_{i:03d}"
        if side_by_side:
            save_pair_png(inp, out, os.path.join(out_dir, base + "_pair.png"),
                          palette=palette, diff=True, cell_size=cell_size, dpi=dpi)
        # Also export individual grids (useful for training pipelines)
        # save_grid_png(inp, os.path.join(out_dir, base + "_input.png"),
        #               palette=palette, cell_size=cell_size, grid_lines=False, dpi=dpi)
        # save_grid_png(out, os.path.join(out_dir, base + "_output.png"),
        #               palette=palette, cell_size=cell_size, grid_lines=False, dpi=dpi)


def show_episode(context, query_in, query_out=None, pred_out=None, palette=DEFAULT_PALETTE):
    """
    context: list of 3 dicts [{"input": grid, "output": grid}, ...]
    query_in: grid
    query_out: optional ground-truth grid (to show aside pred)
    pred_out: optional predicted grid
    Layout:
      Row 1-3: each context pair (input|output)
      Row 4  : query input | (pred) | (gt if provided)
    """
    rows = 3 + 1
    cols = 2 if (pred_out is None and query_out is None) else (3 if (pred_out is None or query_out is None) else 4)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.8))
    if rows == 1: axes = [axes]
    # context rows
    for i in range(3):
        cin = context[i]["input"]; cout = context[i]["output"]
        draw_grid(cin, f"ctx{i+1} in", ax=axes[i][0], palette=palette)
        draw_grid(cout, f"ctx{i+1} out", ax=axes[i][1], palette=palette)
    # query row
    draw_grid(query_in, "query in", ax=axes[3][0], palette=palette)
    c = 1
    if pred_out is not None:
        draw_grid(pred_out, "pred", ax=axes[3][c], palette=palette)
        if query_out is not None: _draw_diff_boxes(axes[3][c], query_out, pred_out)
        c += 1
    if query_out is not None:
        draw_grid(query_out, "gt", ax=axes[3][c], palette=palette)
    fig.tight_layout()
    return fig

# -------------------- CLI --------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="ARC Visualizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Visualize examples from a JSON file")
    p_show.add_argument("json", help="Path to ARC-AGI JSON")
    p_show.add_argument("--split", choices=["train","test"], default="train")
    p_show.add_argument("--max", type=int, default=None, help="Max pairs to display")
    p_show.add_argument("--shuffle", action="store_true")
    p_show.add_argument("--seed", type=int, default=None)
    p_show.add_argument("--diff", action="store_true", help="Highlight changed cells in outputs")
    p_show.add_argument("--cols", type=int, default=3, help="(kept for compat; each pair uses a row)")
    p_show.add_argument("--title", type=str, default=None)

    p_save = sub.add_parser("save", help="Export PNGs from a JSON file")
    p_save.add_argument("json", help="Path to ARC-AGI JSON")
    p_save.add_argument("--split", choices=["train","test"], default="train")
    p_save.add_argument("--out", required=True, help="Output directory")
    p_save.add_argument("--prefix", type=str, default=None, help="Prefix for file names")
    p_save.add_argument("--no-pair", dest="pair", action="store_false", help="Do not save side-by-side pair image")
    p_save.add_argument("--cell", type=int, default=32, help="Cell size in pixels")
    p_save.add_argument("--dpi", type=int, default=100)

    # Offline predictions: {"test_episodes_outputs": [{"query_input":..., "pred_output":..., "query_gt":...}, ...]}
    p_save_off = sub.add_parser("save_offline", help="Export PNGs from offline predictions JSON (test_episodes_outputs)")
    p_save_off.add_argument("json", help="Path to predictions JSON")
    p_save_off.add_argument("--out", required=True, help="Output directory")
    p_save_off.add_argument("--prefix", type=str, default=None, help="Prefix for file names")
    p_save_off.add_argument("--cell", type=int, default=32, help="Cell size in pixels")
    p_save_off.add_argument("--dpi", type=int, default=100)
    p_save_off.add_argument("--max", type=int, default=None, help="Max episodes to export")

    # Shard visualizer: episodes_3shot_gXX/train_0000.pt style
    p_save_shard = sub.add_parser("save_shard", help="Export PNGs from an episodic shard .pt file (ctx+query gt)")
    p_save_shard.add_argument("shard", help="Path to shard .pt (contains ctx_in, ctx_out, q_in, q_out_idx)")
    p_save_shard.add_argument("--out", required=True, help="Output directory")
    p_save_shard.add_argument("--prefix", type=str, default=None, help="Prefix for file names")
    p_save_shard.add_argument("--max", type=int, default=None, help="Max episodes to export")
    p_save_shard.add_argument("--offset", type=int, default=0, help="Start index within the shard")

    args = p.parse_args()

    if args.cmd == "show":
        fig = visualize_task_json(args.json, split=args.split, max_pairs=args.max,
                                  shuffle=args.shuffle, seed=args.seed, diff=args.diff,
                                  suptitle=args.title)
        plt.show()

    elif args.cmd == "save":
        save_dataset_images(args.json, args.out, split=args.split,
                            prefix=args.prefix, side_by_side=args.pair,
                            cell_size=args.cell, dpi=args.dpi)
        print(f"Saved PNGs to: {args.out}")

    elif args.cmd == "save_offline":
        with open(args.json, "r") as f:
            data = json.load(f)
        items = data.get("test_episodes_outputs", [])
        if args.prefix is None:
            prefix = os.path.splitext(os.path.basename(args.json))[0]
        else:
            prefix = args.prefix
        os.makedirs(args.out, exist_ok=True)
        if args.max is not None:
            items = items[:args.max]
        for i, ex in enumerate(items):
            q_in = ex["query_input"]
            pred = ex.get("pred_output")
            gt = ex.get("query_gt")
            base = f"{prefix}_test_{i:03d}"
            save_triplet_png(q_in, pred, gt, os.path.join(args.out, base + "_triplet.png"),
                             cell_size=args.cell, dpi=args.dpi)
        print(f"Saved offline prediction PNGs to: {args.out}")

    elif args.cmd == "save_shard":
        import torch
        t = torch.load(args.shard, map_location="cpu")
        N = t["q_in"].shape[0]
        start = max(0, int(args.offset))
        end = N if args.max is None else min(N, start + int(args.max))
        os.makedirs(args.out, exist_ok=True)
        prefix = args.prefix or os.path.splitext(os.path.basename(args.shard))[0]

        def oh_to_idx(grid_oh):
            return grid_oh.argmax(dim=0).tolist()

        for i in range(start, end):
            ctx = []
            for k in range(3):
                cin = oh_to_idx(t["ctx_in"][i, k])
                cout = oh_to_idx(t["ctx_out"][i, k])
                ctx.append({"input": cin, "output": cout})
            q_in = oh_to_idx(t["q_in"][i])
            q_gt = t["q_out_idx"][i].tolist()
            fig = show_episode(ctx, q_in, query_out=q_gt, pred_out=None)
            base = f"{prefix}_{i:05d}"
            fig.savefig(os.path.join(args.out, base + "_episode.png"), dpi=120, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        print(f"Saved shard episode PNGs to: {args.out}")

if __name__ == "__main__":
    _cli()