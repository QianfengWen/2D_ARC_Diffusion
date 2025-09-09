"""Dataset classes for ARC episodes."""

import os
import json
import torch
from torch.utils.data import Dataset


class EpisodesPTDataset(Dataset):
    """
    Loads episodes from a directory with meta.json and shard .pt files created by make_arc_episodes_offline.py.
    Each shard file has tensors:
      - ctx_in   : (N,3,10,S,S) float
      - ctx_out  : (N,3,10,S,S) float
      - q_in     : (N,10,S,S)   float
      - q_out_oh : (N,10,S,S)   float
      - q_out_idx: (N,S,S)      long
    """
    def __init__(self, episodes_dir: str, split: str = "train"):
        assert split in ("train","test")
        self.base_dir = os.path.abspath(episodes_dir)
        with open(os.path.join(self.base_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self.S = meta["grid_size"]
        self.shards = meta["train_shards"] if split=="train" else meta["test_shards"]
        assert self.shards, f"No {split} shards in {episodes_dir}"
        # prefix sums for idx -> (shard_id, local_idx)
        self.cum = []
        total = 0
        for s in self.shards:
            total += int(s["n"])
            self.cum.append(total)
        self.total = total
        self._cache = {"id": None, "tensors": None}  # simple single-shard cache

    def __len__(self): 
        return self.total

    def _load_shard(self, shard_id: int):
        raw = self.shards[shard_id]["path"]
        # Resolve shard path relative to episodes_dir to be robust to CWD
        path = raw if os.path.isabs(raw) else os.path.join(self.base_dir, os.path.basename(raw))
        if self._cache["id"] != path:
            tensors = torch.load(path, map_location="cpu")

            # Optional hard check (runs once per shard load)
            if "ctx_tid" in tensors and "q_tid" in tensors:
                ctx_tid = tensors["ctx_tid"]  # (N,3)
                q_tid   = tensors["q_tid"]    # (N,)
                same = (ctx_tid[:, 0] == q_tid) & (ctx_tid[:, 1] == q_tid) & (ctx_tid[:, 2] == q_tid)
                if not bool(same.all()):
                    bad = (~same).nonzero(as_tuple=False).flatten().tolist()[:5]
                    raise RuntimeError(f"Shard {path} has episodes with mixed tasks (first bad idx: {bad})")

            self._cache["id"] = path
            self._cache["tensors"] = tensors
        return self._cache["tensors"]

    def _locate(self, idx: int):
        # binary search cum
        lo, hi = 0, len(self.cum)-1
        while lo < hi:
            mid = (lo+hi)//2
            if idx < self.cum[mid]: hi = mid
            else: lo = mid+1
        shard_id = lo
        prev = 0 if shard_id==0 else self.cum[shard_id-1]
        local = idx - prev
        return shard_id, local

    def __getitem__(self, idx: int):
        shard_id, local = self._locate(idx)
        t = self._load_shard(shard_id)
        return (t["ctx_in"][local], t["ctx_out"][local], t["q_in"][local], t["q_out_oh"][local], t["q_out_idx"][local])
