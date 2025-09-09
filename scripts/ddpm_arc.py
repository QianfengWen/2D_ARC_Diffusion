import os, json, argparse, math, random
from typing import List, Tuple, Dict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

COLOR_CLASSES = 10

# ---------------- Dataset over PT shards ----------------

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
        with open(os.path.join(episodes_dir, "meta.json"), "r") as f:
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

    def __len__(self): return self.total

    def _load_shard(self, shard_id: int):
        path = self.shards[shard_id]["path"]
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

# ---------------- Time embedding ----------------

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

# ---------------- Model (3-shot) ----------------

class ResidualBlock(nn.Module):
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch); self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch); self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, ch))
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class PairEncoder(nn.Module):
    def __init__(self, ctx_dim=256, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(20, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, ctx_dim)
    def forward(self, pair):  # (B,20,S,S)
        h = self.net(pair); h = self.pool(h).flatten(1); return self.fc(h)

class FlatUNet3Shot(nn.Module):
    def __init__(self, model_ch=128, num_blocks=6, t_dim=256, ctx_dim=256):
        super().__init__()
        self.in_conv = nn.Conv2d(20, model_ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(model_ch, t_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(8, model_ch); self.out_conv = nn.Conv2d(model_ch, 10, 3, padding=1)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.ctx_enc = PairEncoder(ctx_dim=ctx_dim, ch=64)
        self.ctx_to_t = nn.Sequential(nn.SiLU(), nn.Linear(ctx_dim, t_dim))
    def forward(self, x_t, q_in, t, ctx_in=None, ctx_out=None):
        B = x_t.size(0)
        t_emb = self.t_proj(timestep_embedding(t, 256))
        if ctx_in is not None and ctx_out is not None:
            K = ctx_in.size(1)
            pairs = torch.cat([ctx_in, ctx_out], dim=2).view(B*K, 20, x_t.size(-2), x_t.size(-1))
            ctx_vecs = self.ctx_enc(pairs).view(B, K, -1)
            ctx_vec  = ctx_vecs.mean(dim=1)
            t_emb = t_emb + self.ctx_to_t(ctx_vec)
        h = self.in_conv(torch.cat([x_t, q_in], dim=1))
        for blk in self.blocks: h = blk(h, t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))  # predict noise

# ---------------- Diffusion ----------------

class DiffusionCfg:
    def __init__(self, timesteps=400, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps; self.beta_start = beta_start; self.beta_end = beta_end

class GaussianDiffusion:
    def __init__(self, cfg: DiffusionCfg, device):
        T = cfg.timesteps; self.T = T; self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float32, device=device)
        alphas = 1 - betas; ac = torch.cumprod(alphas, dim=0); ac_prev = torch.cat([torch.tensor([1.0], device=device), ac[:-1]], dim=0)
        self.betas, self.alphas = betas, alphas
        self.sqrt_ac = torch.sqrt(ac); self.sqrt_om_ac = torch.sqrt(1 - ac)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.post_var = betas * (1 - ac_prev) / (1 - ac)
    def q_sample(self, x0, t, noise):
        return self.sqrt_ac[t][:,None,None,None]*x0 + self.sqrt_om_ac[t][:,None,None,None]*noise
    @torch.no_grad()
    def p_sample(self, model, x_t, q_in, t, ctx_in=None, ctx_out=None):
        eps = model(x_t, q_in, t, ctx_in, ctx_out)
        mean = self.sqrt_recip_alphas[t][:,None,None,None]*(x_t - (self.betas[t][:,None,None,None]/self.sqrt_om_ac[t][:,None,None,None])*eps)
        if (t==0).all(): return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(self.post_var[t][:,None,None,None]) * noise
    @torch.no_grad()
    def sample(self, model, q_in, shape, ctx_in=None, ctx_out=None):
        B, C, S, _ = shape
        x_t = torch.randn(shape, device=q_in.device)
        was_train = model.training; model.eval()
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=q_in.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, q_in, t, ctx_in, ctx_out)
        if was_train: model.train(True)
        return x_t

# ---------------- Train / Eval ----------------

def train_one_epoch(model, diffusion, loader, opt, scaler, device, epoch, log_every=100):
    model.train(); running=0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for it, (ctx_in, ctx_out, q_in, q_out_oh, _) in enumerate(pbar):
        ctx_in = ctx_in.to(device); ctx_out = ctx_out.to(device)
        q_in = q_in.to(device); q_out_oh = q_out_oh.to(device)
        B = q_in.size(0)
        t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(q_out_oh); x_t = diffusion.q_sample(q_out_oh, t, noise)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            eps = model(x_t, q_in, t, ctx_in, ctx_out)
            loss = F.mse_loss(eps, noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        running += loss.item()
        if (it+1)%log_every==0: pbar.set_postfix(loss=running/(it+1))
    return running/max(1,len(loader))

@torch.no_grad()
def evaluate(model, diffusion, loader, device, max_batches=10):
    model.eval()
    pix_correct=0; pix_total=0; prob_correct=0; prob_total=0; seen=0
    for ctx_in, ctx_out, q_in, q_out_oh, q_out_idx in loader:
        ctx_in=ctx_in.to(device); ctx_out=ctx_out.to(device)
        q_in=q_in.to(device); q_out_idx=q_out_idx.to(device)
        B, _, S, _ = q_in.shape
        x0 = diffusion.sample(model, q_in, (B, COLOR_CLASSES, S, S), ctx_in, ctx_out)
        pred = x0.argmax(dim=1)  # (B,S,S)
        pix_correct += (pred==q_out_idx).sum().item()
        pix_total   += q_out_idx.numel()
        prob_correct += (pred.view(B,-1)==q_out_idx.view(B,-1)).all(dim=1).sum().item()
        prob_total   += B
        seen += 1
        if seen>=max_batches: break
    return (pix_correct/max(1,pix_total), prob_correct/max(1,prob_total))

def main():
    ap = argparse.ArgumentParser(description="3-shot DDPM (offline episodic shards)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--episodes_dir", required=True)
    p_tr.add_argument("--batch_size", type=int, default=64)
    p_tr.add_argument("--epochs", type=int, default=50)
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--timesteps", type=int, default=400)
    p_tr.add_argument("--val_batches", type=int, default=8)
    p_tr.add_argument("--save_every", type=int, default=10)
    p_tr.add_argument("--out_dir", type=str, default="checkpoints_3shot_offline")

    p_pr = sub.add_parser("predict")
    p_pr.add_argument("--ckpt", required=True)
    p_pr.add_argument("--episodes_dir", required=True)
    p_pr.add_argument("--out_json", type=str, default="predictions_episodes/preds.json")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if args.cmd == "train":
        ds_tr = EpisodesPTDataset(args.episodes_dir, split="train")
        ds_va = EpisodesPTDataset(args.episodes_dir, split="test")
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = FlatUNet3Shot(model_ch=128, num_blocks=6, t_dim=256, ctx_dim=256).to(device)
        diffusion = GaussianDiffusion(DiffusionCfg(timesteps=args.timesteps), device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

        os.makedirs(args.out_dir, exist_ok=True)
        best_prob = 0.0
        for epoch in range(1, args.epochs+1):
            loss = train_one_epoch(model, diffusion, dl_tr, opt, scaler, device, epoch)
            pix_acc, prob_acc = evaluate(model, diffusion, dl_va, device, max_batches=args.val_batches)
            print(f"[Epoch {epoch}] loss={loss:.4f}  val_pixel_acc={pix_acc:.4f}  val_problem_acc={prob_acc:.4f}")

            if prob_acc > best_prob:
                best_prob = prob_acc
                ck = os.path.join(args.out_dir, "ddpm_3shot_best.pt")
                torch.save({"model": model.state_dict(),
                            "cfg": {"timesteps": args.timesteps, "beta_start":1e-4,"beta_end":2e-2},
                            "grid_size": ds_tr.S}, ck)
                print("  ↑ Saved", ck)
            if epoch % args.save_every == 0:
                ck = os.path.join(args.out_dir, f"ddpm_3shot_epoch{epoch}.pt")
                torch.save({"model": model.state_dict(),
                            "cfg": {"timesteps": args.timesteps, "beta_start":1e-4,"beta_end":2e-2},
                            "grid_size": ds_tr.S}, ck)
                print("  Saved", ck)

    elif args.cmd == "predict":
        with open(os.path.join(args.episodes_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        ds_te = EpisodesPTDataset(args.episodes_dir, split="test")
        dl_te = DataLoader(ds_te, batch_size=1, shuffle=False, num_workers=0)

        ckpt = torch.load(args.ckpt, map_location="cpu")
        model = FlatUNet3Shot(model_ch=128, num_blocks=6, t_dim=256, ctx_dim=256).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        cfg = DiffusionCfg(**ckpt["cfg"])
        diffusion = GaussianDiffusion(cfg, device)

        out = []
        for ctx_in, ctx_out, q_in, q_out_oh, q_out_idx in tqdm(dl_te, desc="Predict"):
            ctx_in=ctx_in.to(device); ctx_out=ctx_out.to(device); q_in=q_in.to(device)
            S = q_in.shape[-1]
            x0 = diffusion.sample(model, q_in, (1, COLOR_CLASSES, S, S), ctx_in, ctx_out)
            pred = x0.argmax(dim=1)[0].cpu().tolist()
            # reconstruct a json episode entry (without storing contexts again)
            out.append({"query_input": q_in[0].argmax(dim=0).cpu().tolist(),
                        "pred_output": pred,
                        "query_gt": q_out_idx[0].cpu().tolist()})
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"test_episodes_outputs": out}, f)
        print("Wrote", args.out_json)

if __name__ == "__main__":
    main()