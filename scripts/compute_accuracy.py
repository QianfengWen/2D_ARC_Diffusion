#!/usr/bin/env python3
"""
./scripts/compute_accuracy.py offline --pred predictions_episodes/preds.json

./scripts/compute_accuracy.py arc --pred predictions/preds_bb43febb.json --ref arc_synth_400/synth_bb43febb.json --split test
"""
import argparse
import json
from typing import List, Dict, Any, Tuple


def _grid_dims(g: List[List[int]]) -> Tuple[int, int]:
    h = len(g)
    w = len(g[0]) if h else 0
    return h, w


def _pixel_and_example_acc(preds: List[List[List[int]]], gts: List[List[List[int]]]) -> Tuple[float, float, int, int, int, int]:
    assert len(preds) == len(gts), "pred/gt length mismatch"
    pixel_correct = 0
    pixel_total = 0
    example_correct = 0
    for p, g in zip(preds, gts):
        ph, pw = _grid_dims(p)
        gh, gw = _grid_dims(g)
        assert (ph, pw) == (gh, gw), f"grid size mismatch: pred {ph}x{pw} vs gt {gh}x{gw}"
        all_equal = True
        for r in range(gh):
            for c in range(gw):
                eq = int(p[r][c] == g[r][c])
                pixel_correct += eq
                if eq == 0:
                    all_equal = False
        pixel_total += gh * gw
        if all_equal:
            example_correct += 1
    n = len(preds)
    pix_acc = pixel_correct / max(1, pixel_total)
    ex_acc = example_correct / max(1, n)
    return pix_acc, ex_acc, pixel_correct, pixel_total, example_correct, n


def compute_offline(pred_json: str) -> Dict[str, Any]:
    with open(pred_json, "r") as f:
        data = json.load(f)
    items = data.get("test_episodes_outputs", [])
    preds = [ex["pred_output"] for ex in items]
    gts = [ex.get("query_gt") for ex in items]
    if any(gt is None for gt in gts):
        raise ValueError("Some entries are missing 'query_gt' – cannot compute accuracy.")
    pix_acc, ex_acc, pc, pt, ec, n = _pixel_and_example_acc(preds, gts)
    return {
        "type": "offline",
        "file": pred_json,
        "num_examples": n,
        "pixel_acc": pix_acc,
        "example_acc": ex_acc,
        "pixel_correct": pc,
        "pixel_total": pt,
        "examples_correct": ec,
    }


def _grid_to_tuple(g: List[List[int]]):
    return tuple(tuple(int(x) for x in row) for row in g)


def compute_arc_pair(pred_json: str, ref_json: str, split: str) -> Dict[str, Any]:
    with open(pred_json, "r") as f:
        p = json.load(f)
    with open(ref_json, "r") as f:
        r = json.load(f)
    pred_items = p.get(split, [])
    ref_items = r.get(split, [])
    # try positional match first
    if len(pred_items) == len(ref_items):
        preds = [ex["output"] for ex in pred_items]
        gts = [ex["output"] for ex in ref_items]
        pix_acc, ex_acc, pc, pt, ec, n = _pixel_and_example_acc(preds, gts)
        return {
            "type": "arc_pair_positional",
            "pred_file": pred_json,
            "ref_file": ref_json,
            "split": split,
            "num_examples": n,
            "pixel_acc": pix_acc,
            "example_acc": ex_acc,
            "pixel_correct": pc,
            "pixel_total": pt,
            "examples_correct": ec,
        }
    # fallback: match by input grid key
    ref_map = {_grid_to_tuple(ex["input"]): ex["output"] for ex in ref_items}
    preds = []
    gts = []
    missing = 0
    for ex in pred_items:
        key = _grid_to_tuple(ex["input"])
        gt = ref_map.get(key)
        if gt is None:
            missing += 1
            continue
        preds.append(ex["output"])
        gts.append(gt)
    if not preds:
        raise ValueError("Could not align any predictions to reference by input grids.")
    pix_acc, ex_acc, pc, pt, ec, n = _pixel_and_example_acc(preds, gts)
    return {
        "type": "arc_pair_by_input",
        "pred_file": pred_json,
        "ref_file": ref_json,
        "split": split,
        "aligned_examples": n,
        "missing_in_ref": missing,
        "pixel_acc": pix_acc,
        "example_acc": ex_acc,
        "pixel_correct": pc,
        "pixel_total": pt,
        "examples_correct": ec,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute prediction accuracy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("offline", help="predictions_episodes JSON with test_episodes_outputs")
    po.add_argument("--pred", required=True)

    pa = sub.add_parser("arc", help="ARC-style JSONs: compare pred file vs reference task JSON")
    pa.add_argument("--pred", required=True)
    pa.add_argument("--ref", required=True)
    pa.add_argument("--split", choices=["train", "test"], default="test")

    args = ap.parse_args()
    if args.cmd == "offline":
        res = compute_offline(args.pred)
    else:
        res = compute_arc_pair(args.pred, args.ref, args.split)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


