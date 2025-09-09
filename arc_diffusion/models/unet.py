"""UNet architectures for ARC Diffusion."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


COLOR_CLASSES = 10


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1: 
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, ch))
    
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class PairEncoder(nn.Module):
    """Encode context input-output pairs."""
    
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
        h = self.net(pair)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class FlatUNet3Shot(nn.Module):
    """Flat UNet for 3-shot ARC diffusion (no spatial downsampling)."""
    
    def __init__(self, model_ch=128, num_blocks=6, t_dim=256, ctx_dim=256):
        super().__init__()
        self.model_ch = model_ch
        self.num_blocks = num_blocks
        self.t_dim = t_dim
        self.ctx_dim = ctx_dim
        
        # Input projection: concat of x_t (10 channels) + q_in (10 channels) = 20 channels
        self.in_conv = nn.Conv2d(20, model_ch, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(model_ch, t_dim) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.out_norm = nn.GroupNorm(8, model_ch)
        self.out_conv = nn.Conv2d(model_ch, COLOR_CLASSES, 3, padding=1)
        
        # Time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, t_dim), 
            nn.SiLU(), 
            nn.Linear(t_dim, t_dim)
        )
        
        # Context encoding
        self.ctx_enc = PairEncoder(ctx_dim=ctx_dim, ch=64)
        self.ctx_to_t = nn.Sequential(nn.SiLU(), nn.Linear(ctx_dim, t_dim))
    
    def forward(self, x_t, q_in, t, ctx_in=None, ctx_out=None):
        """Forward pass.
        
        Args:
            x_t: Noisy query target (B, 10, S, S)
            q_in: Query input (B, 10, S, S)  
            t: Timestep (B,)
            ctx_in: Context inputs (B, 3, 10, S, S)
            ctx_out: Context outputs (B, 3, 10, S, S)
            
        Returns:
            Predicted noise (B, 10, S, S)
        """
        B = x_t.size(0)
        
        # Time embedding
        t_emb = self.t_proj(timestep_embedding(t, self.t_dim))
        
        # Context conditioning
        if ctx_in is not None and ctx_out is not None:
            K = ctx_in.size(1)  # Number of context pairs (should be 3)
            # Concatenate input and output for each context pair
            pairs = torch.cat([ctx_in, ctx_out], dim=2)  # (B, 3, 20, S, S)
            pairs = pairs.view(B*K, 20, x_t.size(-2), x_t.size(-1))  # (B*3, 20, S, S)
            
            # Encode each pair and average
            ctx_vecs = self.ctx_enc(pairs).view(B, K, -1)  # (B, 3, ctx_dim)
            ctx_vec = ctx_vecs.mean(dim=1)  # (B, ctx_dim)
            
            # Add context to time embedding
            t_emb = t_emb + self.ctx_to_t(ctx_vec)
        
        # Main processing
        h = self.in_conv(torch.cat([x_t, q_in], dim=1))  # (B, model_ch, S, S)
        
        # Apply residual blocks with time conditioning
        for blk in self.blocks:
            h = blk(h, t_emb)
        
        # Output projection
        return self.out_conv(F.silu(self.out_norm(h)))  # predict noise
