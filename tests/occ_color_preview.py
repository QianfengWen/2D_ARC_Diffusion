"""Preview generator outputs for occ+color tasks.

Generates 50 unique (input, output) pairs per selected task using the
arc_diffusion.data.generators module, saves each pair PNG, and merges a
collage per task using arc_diffusion.utils.visualization helpers.

Run:
  python -m tests.occ_color_preview  # default out dir under tests/occ_color_previews

Optional args:
  --tasks 3aa6fb7a 1bfc4729   # subset to preview
  --count 50                   # number of samples per task
  --out tests/occ_color_previews
"""
import os
import argparse
from typing import List

from arc_diffusion.data.generators import TASKS, generate_unique_pairs
from arc_diffusion.utils.visualization import save_pair_png, merge_pngs_grid


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preview_tasks(tasks: List[str], count: int, out_dir: str) -> None:
    ensure_dir(out_dir)
    for code in tasks:
        if code not in TASKS:
            print(f"Skip unknown task: {code}")
            continue
        print(f"Generating {count} samples for {code} ...")
        gen_fn = TASKS[code]
        pairs, _, _ = generate_unique_pairs(gen_fn, count, attempts_per_example=200, progress=False)

        task_dir = os.path.join(out_dir, code)
        ensure_dir(task_dir)
        image_paths = []
        for i, (inp, out) in enumerate(pairs):
            img_path = os.path.join(task_dir, f"pair_{i:03d}.png")
            save_pair_png(inp, out, img_path)
            image_paths.append(img_path)

        collage_path = os.path.join(out_dir, f"{code}_collage.png")
        merge_pngs_grid(image_paths, collage_path, cols=5)
        print(f"Saved collage → {collage_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=["3aa6fb7a", "0ca9ddb6", "1bfc4729"],
                        help="Task codes to preview (must be in TASKS)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--out", type=str, default=os.path.join("tests", "occ_color_previews"))
    args = parser.parse_args()

    preview_tasks(args.tasks, args.count, args.out)


if __name__ == "__main__":
    main()
