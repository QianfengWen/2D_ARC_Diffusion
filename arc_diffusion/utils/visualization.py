"""Visualization utilities for ARC grids and tasks."""

import json
import os
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

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
