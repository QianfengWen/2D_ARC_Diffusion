"""Diffusion models for ARC tasks."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DiffusionCfg:
    """Configuration for diffusion process."""
    timesteps: int = 400
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusion:
    """DDPM Gaussian diffusion process."""
    
    def __init__(self, cfg: DiffusionCfg, device):
        T = cfg.timesteps
        self.T = T
        self.device = device
        
        # Noise schedule
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float32, device=device)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = torch.cat([torch.tensor([1.0], device=device), ac[:-1]], dim=0)
        
        # Store noise schedule parameters
        self.betas = betas
        self.alphas = alphas
        self.sqrt_ac = torch.sqrt(ac)
        self.sqrt_om_ac = torch.sqrt(1 - ac)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.post_var = betas * (1 - ac_prev) / (1 - ac)
    
    def q_sample(self, x0, t, noise):
        """Forward diffusion: add noise to clean data."""
        return (self.sqrt_ac[t][:, None, None, None] * x0 + 
                self.sqrt_om_ac[t][:, None, None, None] * noise)
    
    @torch.no_grad()
    def p_sample(self, model, x_t, q_in, t, ctx_in=None, ctx_out=None):
        """Single reverse diffusion step."""
        # Predict noise
        eps = model(x_t, q_in, t, ctx_in, ctx_out)
        
        # Compute mean
        mean = (self.sqrt_recip_alphas[t][:, None, None, None] * 
                (x_t - (self.betas[t][:, None, None, None] / 
                        self.sqrt_om_ac[t][:, None, None, None]) * eps))
        
        if (t == 0).all():
            return mean
        
        # Add noise for non-final steps
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(self.post_var[t][:, None, None, None]) * noise
    
    @torch.no_grad()
    def sample(self, model, q_in, shape, ctx_in=None, ctx_out=None):
        """Full sampling process from noise to clean data."""
        B, C, S, _ = shape
        x_t = torch.randn(shape, device=q_in.device)
        
        # Set model to eval mode, restore later
        was_train = model.training
        model.eval()
        
        # Reverse diffusion process
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=q_in.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, q_in, t, ctx_in, ctx_out)
        
        # Restore training mode
        if was_train:
            model.train(True)
            
        return x_t
    
    def compute_loss(self, model, batch):
        """Compute training loss for a batch."""
        ctx_in, ctx_out, q_in, q_out_oh, _ = batch
        B = q_in.size(0)
        
        # Sample timesteps and noise
        t = torch.randint(0, self.T, (B,), device=q_in.device, dtype=torch.long)
        noise = torch.randn_like(q_out_oh)
        
        # Forward diffusion
        x_t = self.q_sample(q_out_oh, t, noise)
        
        # Predict noise
        eps = model(x_t, q_in, t, ctx_in, ctx_out)
        
        # MSE loss
        loss = F.mse_loss(eps, noise)
        return loss
