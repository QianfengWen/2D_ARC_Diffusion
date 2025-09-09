import os, json, glob, argparse, random, hashlib
from typing import List, Tuple, Dict, Optional
import torch

# ---------- core grid helpers ----------

def pad_to_square_center(grid, size: int):
    h, w = len(grid), len(grid[0])
    assert h <= size and w <= size, f"{h}x{w} > target {size}"
    top = (size - h) // 2
    left = (size - w) // 2
    out = [[0 for _ in range(size)] for _ in range(size)]
    for r in range(h):
        for c in range(w):
            out[top + r][left + c] = grid[r][c]
    return out

def to_one_hot(idx_grid, C=10):
    h, w = len(idx_grid), len(idx_grid[0])
    t = torch.zeros(C, h, w, dtype=torch.float32)
    for r in range(h):
        for c in range(w):
            t[idx_grid[r][c], r, c] = 1.0
    return t

# ---------- episode sampling policies ----------

def choose_context_indices(n: int, policy: str, rng: random.Random, exclude: Optional[int]=None):
    assert n >= 3
    policy = policy.lower()
    pool = list(range(n))
    if exclude is not None and exclude in pool:
        pool.remove(exclude)

    if policy == "first3":
        ctx = [i for i in [0,1,2,3,4] if (exclude is None or i != exclude) and i < n][:3]
        if len(ctx) < 3:
            rest = [i for i in range(n) if i not in ctx and i != exclude]
            ctx += rng.sample(rest, 3-len(ctx))
        return tuple(ctx)

    if policy == "sliding":
        for _ in range(20):
            i = rng.randint(0, max(0, n-3))
            cand = [i, i+1, i+2]
            if exclude in cand:
                continue
            return tuple(cand)
        policy = "random"

    return tuple(rng.sample(pool, 3))

# ---------- dedup keys (treat context order as irrelevant) ----------

def _bytes_grid(g):
    h = len(g); w = len(g[0]) if h else 0
    b = bytearray([h & 0xFF, w & 0xFF])
    for row in g: b.extend(row)
    return bytes(b)

def episode_key(context_triplet: List[Tuple[List[List[int]], List[List[int]]]],
                q_in: List[List[int]],
                q_out: List[List[int]]) -> str:
    """
    Canonicalize context triplet by sorting its pair-bytes so episode duplicates
    with different context order hash the same. Query in/out included.
    """
    parts = []
    for (ci, co) in context_triplet:
        parts.append(_bytes_grid(ci) + b'|' + _bytes_grid(co))
    parts.sort()
    m = hashlib.blake2b(digest_size=16)
    for p in parts: m.update(p)
    m.update(b'||'); m.update(_bytes_grid(q_in)); m.update(b'|'); m.update(_bytes_grid(q_out))
    return m.hexdigest()

# ---------- writer ----------

def flush_shard(split: str, shard_idx: int, buf, out_dir: str):
    if buf["N"] == 0: return None
    path = os.path.join(out_dir, f"{split}_{shard_idx:04d}.pt")
    to_save = {
        "ctx_in":   torch.stack(buf["ctx_in"],   dim=0).contiguous(),
        "ctx_out":  torch.stack(buf["ctx_out"],  dim=0).contiguous(),
        "q_in":     torch.stack(buf["q_in"],     dim=0).contiguous(),
        "q_out_oh": torch.stack(buf["q_out_oh"], dim=0).contiguous(),
        "q_out_idx":torch.stack(buf["q_out_idx"],dim=0).contiguous(),
        "ctx_tid":  torch.stack(buf["ctx_tid"],  dim=0).contiguous(),  # (N,3) long
        "q_tid":    torch.stack(buf["q_tid"],    dim=0).contiguous(),  # (N,)   long
    }
    torch.save(to_save, path)
    meta = {"path": path, "n": buf["N"]}
    # reset
    buf["ctx_in"].clear(); buf["ctx_out"].clear()
    buf["q_in"].clear(); buf["q_out_oh"].clear(); buf["q_out_idx"].clear()
    buf["N"] = 0
    return meta

def make_episodes_for_task(task_path: str,
                           grid_size: int,
                           train_per_task: int,
                           test_per_task: int,
                           ctx_policy: str,
                           rng: random.Random,
                           train_keys: set,
                           test_keys: set,
                           train_buf, test_buf,
                           shard_size: int,
                           task_id: int):
    with open(task_path, "r") as f:
        d = json.load(f)
    train_pairs = [(pad_to_square_center(ex["input"], grid_size),
                    pad_to_square_center(ex["output"], grid_size)) for ex in d.get("train",[])]
    test_pairs  = [(pad_to_square_center(ex["input"], grid_size),
                    pad_to_square_center(ex["output"], grid_size)) for ex in d.get("test",[])]

    # ----- train episodes (query from train) -----
    if len(train_pairs) >= 4 and train_per_task > 0:
        attempts = 0
        while train_buf["N_added_for_task"] < train_per_task and attempts < 20*train_per_task:
            attempts += 1
            n = len(train_pairs)
            q_idx = rng.randrange(n)
            c0,c1,c2 = choose_context_indices(n, ctx_policy, rng, exclude=q_idx)
            ctx_trip = [train_pairs[c0], train_pairs[c1], train_pairs[c2]]
            q_in, q_out = train_pairs[q_idx]
            key = episode_key(ctx_trip, q_in, q_out)
            if key in train_keys: continue
            train_keys.add(key)
            # pack tensors
            S = grid_size
            ctx_in  = torch.stack([to_one_hot(ctx_trip[k][0]) for k in range(3)], dim=0)
            ctx_out = torch.stack([to_one_hot(ctx_trip[k][1]) for k in range(3)], dim=0)
            q_in_oh   = to_one_hot(q_in)
            q_out_oh  = to_one_hot(q_out)
            q_out_idx = torch.tensor(q_out, dtype=torch.long)
            train_buf["ctx_in"].append(ctx_in)
            train_buf["ctx_out"].append(ctx_out)
            train_buf["ctx_tid"].append(torch.tensor([task_id, task_id, task_id], dtype=torch.long))
            train_buf["q_tid"].append(torch.tensor(task_id, dtype=torch.long))
            train_buf["q_in"].append(q_in_oh)
            train_buf["q_out_oh"].append(q_out_oh)
            train_buf["q_out_idx"].append(q_out_idx)
            train_buf["N"] += 1
            train_buf["N_added_for_task"] += 1
            if train_buf["N"] >= shard_size:
                yield "train_flush"
        if train_buf["N_added_for_task"] < train_per_task:
            print(f"[WARN] {os.path.basename(task_path)}: wanted {train_per_task} train episodes, got {train_buf['N_added_for_task']} (dedup limited).")

    # ----- test episodes (query from test; contexts from train) -----
    if len(train_pairs) >= 3 and len(test_pairs) >= 1 and test_per_task > 0:
        attempts = 0
        while test_buf["N_added_for_task"] < test_per_task and attempts < 20*test_per_task:
            attempts += 1
            n = len(train_pairs)
            c0,c1,c2 = choose_context_indices(n, ctx_policy, rng, exclude=None)
            q_idx = rng.randrange(len(test_pairs))
            ctx_trip = [train_pairs[c0], train_pairs[c1], train_pairs[c2]]
            q_in, q_out = test_pairs[q_idx]
            key = episode_key(ctx_trip, q_in, q_out)
            if key in test_keys: continue
            test_keys.add(key)
            S = grid_size
            ctx_in  = torch.stack([to_one_hot(ctx_trip[k][0]) for k in range(3)], dim=0)
            ctx_out = torch.stack([to_one_hot(ctx_trip[k][1]) for k in range(3)], dim=0)
            q_in_oh   = to_one_hot(q_in)
            q_out_oh  = to_one_hot(q_out)
            q_out_idx = torch.tensor(q_out, dtype=torch.long)
            test_buf["ctx_in"].append(ctx_in)
            test_buf["ctx_out"].append(ctx_out)
            test_buf["ctx_tid"].append(torch.tensor([task_id, task_id, task_id], dtype=torch.long))
            test_buf["q_tid"].append(torch.tensor(task_id, dtype=torch.long))
            test_buf["q_in"].append(q_in_oh)
            test_buf["q_out_oh"].append(q_out_oh)
            test_buf["q_out_idx"].append(q_out_idx)
            test_buf["N"] += 1
            test_buf["N_added_for_task"] += 1
            if test_buf["N"] >= shard_size:
                yield "test_flush"
        if test_buf["N_added_for_task"] < test_per_task:
            print(f"[WARN] {os.path.basename(task_path)}: wanted {test_per_task} test episodes, got {test_buf['N_added_for_task']} (dedup limited).")

def main():
    ap = argparse.ArgumentParser(description="Build 3+1 ARC episodes (sharded .pt with one-hot tensors)")
    ap.add_argument("--glob", required=True, help='e.g. "arc_synth_unique/synth_*.json"')
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--ctx_policy", choices=["first3","random","sliding"], default="random")
    ap.add_argument("--train_per_task", type=int, default=50000)
    ap.add_argument("--test_per_task",  type=int, default=5000)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="episodes_3shot_g16")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(args.glob))
    assert paths, f"No files matched: {args.glob}"
    os.makedirs(args.out_dir, exist_ok=True)

    # buffers + meta
    train_buf = {"ctx_in":[], "ctx_out":[], "q_in":[], "q_out_oh":[], "q_out_idx":[],
             "ctx_tid":[], "q_tid":[], "N":0, "N_added_for_task":0}
    test_buf  = {"ctx_in":[], "ctx_out":[], "q_in":[], "q_out_oh":[], "q_out_idx":[],
                "ctx_tid":[], "q_tid":[], "N":0, "N_added_for_task":0}

    train_meta, test_meta = [], []
    train_keys, test_keys = set(), set()
    train_shard_idx = 0; test_shard_idx = 0

    for tid, p in enumerate(paths):
        train_buf["N_added_for_task"] = 0
        test_buf["N_added_for_task"]  = 0
        for signal in make_episodes_for_task(
            p, args.grid_size, args.train_per_task, args.test_per_task,
            args.ctx_policy, rng, train_keys, test_keys,
            train_buf, test_buf, args.shard_size, task_id=tid
        ):
            if signal == "train_flush":
                meta = flush_shard("train", train_shard_idx, train_buf, args.out_dir)
                if meta: train_meta.append(meta); train_shard_idx += 1
            elif signal == "test_flush":
                meta = flush_shard("test",  test_shard_idx,  test_buf,  args.out_dir)
                if meta: test_meta.append(meta); test_shard_idx  += 1

    # final flush
    meta = flush_shard("train", train_shard_idx, train_buf, args.out_dir)
    if meta: train_meta.append(meta)
    meta = flush_shard("test",  test_shard_idx,  test_buf,  args.out_dir)
    if meta: test_meta.append(meta)

    # write meta.json for the training script
    summary = {
        "grid_size": args.grid_size,
        "ctx_policy": args.ctx_policy,
        "train_shards": train_meta,
        "test_shards":  test_meta
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Wrote meta + {len(train_meta)} train shards, {len(test_meta)} test shards to {args.out_dir}")

if __name__ == "__main__":
    main()
