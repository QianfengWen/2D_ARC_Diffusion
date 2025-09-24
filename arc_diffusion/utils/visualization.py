"""Visualization utilities for ARC grids and tasks.

Extended with helpers to visualize training/inference samples composed of
context pairs and the predicted target for ARC diffusion episodes.
"""

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib import image as mpimg

Grid = List[List[int]]

# Standard ARC color palette
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

# Binary occupancy palette for occupancy mode
OCCUPANCY_PALETTE = {
    0: "#000000",  # black (unoccupied)
    1: "#FFFFFF",  # white (occupied)
}


def make_arc_cmap(palette: Dict[int, str] = DEFAULT_PALETTE):
    """Build a ListedColormap + BoundaryNorm that maps integers 0..9 to colors."""
    # Ensure indices 0..9 exist; fallback to black if missing.
    colors = [palette.get(i, "#000000") for i in range(10)]
    cmap = ListedColormap(colors, name="arc_palette", N=10)
    # Boundaries such that each integer i maps to bin centered at i
    boundaries = [i - 0.5 for i in range(11)]  # -0.5..9.5 step 1
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    return cmap, norm


def _hide_axes(ax):
    """Hide axes ticks and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_grid_lines(ax, h: int, w: int, color=(1,1,1,0.15), lw=0.6):
    """Light grid lines over each cell."""
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
    """Render a single ARC grid."""
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


# ---------- Tensor utilities for training-time visualization ----------

def _to_index_grid(arr) -> np.ndarray:
    """Convert various ARC tensor formats to a 2D integer grid (S,S).

    Accepts:
      - torch.Tensor or np.ndarray with shape (10,S,S) as one-hot/probabilities
      - torch.Tensor or np.ndarray with shape (S,S) as already-indexed ints
    Returns a NumPy array of dtype int64 with values in [0,9].
    """
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu()
        if a.ndim == 3 and a.shape[0] == 10:
            # One-hot/prob: take argmax across channel
            return a.argmax(dim=0).to(torch.int64).numpy()
        elif a.ndim == 2:
            return a.to(torch.int64).numpy()
        else:
            raise ValueError(f"Unsupported tensor shape for grid: {tuple(a.shape)}")
    else:
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[0] == 10:
            return a.argmax(axis=0).astype(np.int64)
        elif a.ndim == 2:
            return a.astype(np.int64)
        else:
            raise ValueError(f"Unsupported array shape for grid: {a.shape}")


def _to_occupancy_grid(arr) -> np.ndarray:
    """Convert ARC tensor/grid to binary occupancy grid (0=unoccupied, 1=occupied).
    
    Args:
        arr: Same formats as _to_index_grid
    Returns:
        Binary grid where 0 = unoccupied (black), 1 = occupied (any non-black)
    """
    index_grid = _to_index_grid(arr)
    return (index_grid > 0).astype(np.int64)


def _draw_grid_from_index(index_grid: np.ndarray,
                          title: Optional[str],
                          ax: plt.Axes,
                          palette: Dict[int, str] = DEFAULT_PALETTE,
                          show_grid: bool = True) -> plt.Axes:
    """Render a (S,S) integer grid using the ARC palette."""
    h, w = index_grid.shape
    cmap, norm = make_arc_cmap(palette)
    ax.imshow(index_grid, cmap=cmap, norm=norm, interpolation='nearest', origin='upper')
    _hide_axes(ax)
    if show_grid:
        _draw_grid_lines(ax, h, w)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def save_episode_ctx_pred(ctx_in, ctx_out, q_in, pred_idx,
                          path: str,
                          palette: Dict[int, str] = DEFAULT_PALETTE,
                          show_query: bool = True,
                          max_context_rows: Optional[int] = None) -> None:
    """Save a PNG visualizing an episode's context pairs and predicted target.

    Layout (rows = K context pairs + 1, cols = 2):
      - For each context k: [ctx k input] [ctx k output]
      - Final row:         [query input (optional)] [predicted target]

    Args:
      ctx_in:  (K,10,S,S) one-hot/prob tensor or array
      ctx_out: (K,10,S,S) one-hot/prob tensor or array
      q_in:    (10,S,S)   one-hot/prob tensor or array
      pred_idx:(S,S)      integer tensor/array with values 0..9
      path:    where to save the PNG
      palette: color palette mapping 0..9
      show_query: if True, include query input in the final row's left tile
      max_context_rows: if set, limit number of context rows shown
    """
    # Convert to CPU numpy
    if isinstance(ctx_in, torch.Tensor):
        ctx_in_np = ctx_in.detach().cpu()
    else:
        ctx_in_np = np.asarray(ctx_in)
    if isinstance(ctx_out, torch.Tensor):
        ctx_out_np = ctx_out.detach().cpu()
    else:
        ctx_out_np = np.asarray(ctx_out)

    K = int(ctx_in_np.shape[0])
    if max_context_rows is not None:
        K = min(K, int(max_context_rows))

    rows = K + 1
    cols = 2

    # Figure size heuristic: scale with rows to keep tiles readable
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    if rows == 1:
        axes = np.reshape(axes, (1, cols))

    # Plot context pairs
    for k in range(K):
        ci = _to_index_grid(ctx_in_np[k])
        co = _to_index_grid(ctx_out_np[k])
        _draw_grid_from_index(ci, title=f"ctx {k+1} in", ax=axes[k, 0], palette=palette)
        _draw_grid_from_index(co, title=f"ctx {k+1} out", ax=axes[k, 1], palette=palette)

    # Final row: query in (optional) and prediction
    if show_query:
        qi = _to_index_grid(q_in)
        _draw_grid_from_index(qi, title="query in", ax=axes[rows-1, 0], palette=palette)
    else:
        # Hide the left tile if not showing query
        axes[rows-1, 0].set_visible(False)

    pred = _to_index_grid(pred_idx)
    _draw_grid_from_index(pred, title="pred", ax=axes[rows-1, 1], palette=palette)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_collage_ctx_pred(ctx_in_batch,
                          ctx_out_batch,
                          q_in_batch,
                          pred_idx_batch,
                          path: str,
                          max_samples: int = 6,
                          cols: int = 3,
                          include_query: bool = True,
                          palette: Dict[int, str] = DEFAULT_PALETTE,
                          dpi: int = 150,
                          mode: str = "baseline") -> None:
    """Save a collage PNG for multiple episodes with mode-specific visualization.

    Each episode is rendered as a block with (K+1) rows and 2+ columns:
      - Baseline/Color modes: 2 columns (ctx k in | ctx k out), (query in | pred)
      - Occupancy mode: 3 columns (ctx k in | ctx k out | binary out), (query in | pred | binary pred)

    Args:
      ctx_in_batch:  (B,K,10,S,S)
      ctx_out_batch: (B,K,10,S,S)
      q_in_batch:    (B,10,S,S)
      pred_idx_batch:(B,S,S)
      mode: "baseline", "occupancy", or "color"
    """
    # Convert to CPU numpy/tensors
    if isinstance(ctx_in_batch, torch.Tensor):
        ctx_in_b = ctx_in_batch.detach().cpu()
    else:
        ctx_in_b = np.asarray(ctx_in_batch)
    if isinstance(ctx_out_batch, torch.Tensor):
        ctx_out_b = ctx_out_batch.detach().cpu()
    else:
        ctx_out_b = np.asarray(ctx_out_batch)
    if isinstance(q_in_batch, torch.Tensor):
        q_in_b = q_in_batch.detach().cpu()
    else:
        q_in_b = np.asarray(q_in_batch)
    if isinstance(pred_idx_batch, torch.Tensor):
        pred_b = pred_idx_batch.detach().cpu()
    else:
        pred_b = np.asarray(pred_idx_batch)

    B = int(ctx_in_b.shape[0])
    K = int(ctx_in_b.shape[1]) if ctx_in_b.ndim >= 5 else 0
    n = min(B, int(max_samples))
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    # Mode-specific layout: occupancy gets 3 columns, others get 2
    block_cols = 3 if mode == "occupancy" else 2
    
    # Heuristic sizing: each tile ~2.2 inch; block height = (K+1)*2.2
    tile = 2.2
    block_w = block_cols * tile
    block_h = (K + 1) * tile
    fig_w = cols * block_w
    fig_h = rows * block_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.3)

    for i in range(n):
        r = i // cols
        c = i % cols
        inner = gridspec.GridSpecFromSubplotSpec(
            K + 1, block_cols, subplot_spec=outer[r, c], wspace=0.1, hspace=0.2
        )

        # Context rows
        for k in range(K):
            ax_l = fig.add_subplot(inner[k, 0])
            ax_r = fig.add_subplot(inner[k, 1])
            ci = _to_index_grid(ctx_in_b[i, k])
            co = _to_index_grid(ctx_out_b[i, k])
            _draw_grid_from_index(ci, title=f"ctx {k+1} in", ax=ax_l, palette=palette)
            _draw_grid_from_index(co, title=f"ctx {k+1} out", ax=ax_r, palette=palette)
            
            # Occupancy mode: add binary view of context output
            if mode == "occupancy":
                ax_bin = fig.add_subplot(inner[k, 2])
                co_bin = _to_occupancy_grid(ctx_out_b[i, k])
                _draw_grid_from_index(co_bin, title=f"ctx {k+1} occ", ax=ax_bin, palette=OCCUPANCY_PALETTE)

        # Final row: query and prediction
        ax_l = fig.add_subplot(inner[K, 0])
        ax_r = fig.add_subplot(inner[K, 1])
        
        if include_query:
            qi = _to_index_grid(q_in_b[i])
            _draw_grid_from_index(qi, title="query in", ax=ax_l, palette=palette)
        else:
            ax_l.set_visible(False)
            
        pr = _to_index_grid(pred_b[i])
        _draw_grid_from_index(pr, title="pred", ax=ax_r, palette=palette)
        
        # Occupancy mode: add binary view of prediction
        if mode == "occupancy":
            ax_bin = fig.add_subplot(inner[K, 2])
            pr_bin = _to_occupancy_grid(pred_b[i])
            _draw_grid_from_index(pr_bin, title="pred occ", ax=ax_bin, palette=OCCUPANCY_PALETTE)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _draw_diff_boxes(ax: plt.Axes, inp: Grid, out: Grid,
                     edge_color: str = "#FFFFFF", lw: float = 1.2):
    """Draw a thin rectangle around cells that changed from input to output."""
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
    """Side-by-side visualization of an (input, output) pair."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    draw_grid(inp, titles[0], ax=ax1, palette=palette)
    draw_grid(out, titles[1], ax=ax2, palette=palette)
    if diff:
        _draw_diff_boxes(ax2, inp, out)
    plt.tight_layout()
    return fig


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
    """Visualize multiple pairs from an ARC task JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("File does not look like ARC-AGI JSON (must be an object).")
    
    examples = data.get(split, [])
    if not examples:
        print(f"No {split} examples found in {path}")
        return None
    
    # Sampling
    if shuffle and seed is not None:
        import random
        random.seed(seed)
        examples = examples.copy()
        random.shuffle(examples)
    
    if max_pairs is not None:
        examples = examples[:max_pairs]
    
    if not examples:
        return None
    
    # Layout calculation
    n = len(examples)
    rows = (n + cols - 1) // cols
    
    # Create figure
    fig_width = cols * pair_figsize[0]
    fig_height = rows * pair_figsize[1]
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot examples
    for i, ex in enumerate(examples):
        row = i // cols
        col = i % cols
        ax = axes[row][col]
        
        # Create subplot with two side-by-side grids
        ax.set_xlim(-0.5, len(ex["input"][0]) + len(ex["output"][0]) + 0.5)
        ax.set_ylim(-0.5, max(len(ex["input"]), len(ex["output"])) - 0.5)
        
        # This is simplified - for a full implementation, 
        # would need to properly handle side-by-side layout
        draw_grid(ex["input"], f"Example {i+1}", ax=ax, palette=palette)
    
    # Hide unused subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].set_visible(False)
    
    if suptitle:
        fig.suptitle(suptitle)
    
    plt.tight_layout()
    return fig


def save_grid_png(grid: Grid, path: str, **kwargs):
    """Save a single grid as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    draw_grid(grid, ax=ax, **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=150, pad_inches=0.1)
    plt.close()


def save_pair_png(inp: Grid, out: Grid, path: str, **kwargs):
    """Save an input-output pair as PNG."""
    fig = show_pair(inp, out, **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def merge_pngs_grid(image_paths: List[str],
                    out_path: str,
                    cols: int = 3,
                    titles: Optional[List[str]] = None,
                    dpi: int = 150) -> None:
    """Merge multiple PNG images into a grid and save as a single PNG.

    - image_paths: list of file paths to PNG images to place in the grid
    - out_path: where to save the merged image
    - cols: number of columns in the grid
    - titles: optional list of titles per image (e.g., task slugs)
    - dpi: dpi for the saved figure
    """
    if not image_paths:
        return
    cols = max(1, int(cols))
    n = len(image_paths)
    rows = (n + cols - 1) // cols

    # Load first image to estimate tile aspect ratio for figure sizing
    try:
        sample_img = mpimg.imread(image_paths[0])
        h, w = sample_img.shape[:2]
        aspect = w / max(1, h)
    except Exception:
        h, w, aspect = 800, 600, 600/800

    # Use a heuristic for per-tile inches; scale by aspect
    base_h_in = 4.0
    base_w_in = base_h_in * aspect
    fig_w = cols * base_w_in
    fig_h = rows * base_h_in

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, path in enumerate(image_paths):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        try:
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis('off')
            if titles and i < len(titles) and titles[i]:
                ax.set_title(titles[i], fontsize=10)
        except Exception:
            ax.axis('off')
    # Hide any unused axes
    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r][c].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
