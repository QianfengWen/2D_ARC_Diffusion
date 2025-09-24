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
    parameterization: str = "predict_noise"  # "predict_noise", "predict_x0", "predict_v"
    mode: str = "baseline"  # "baseline", "occupancy", "color"


class GaussianDiffusion:
    """DDPM Gaussian diffusion process with multiple parameterizations."""
    
    def __init__(self, cfg: DiffusionCfg, device):
        T = cfg.timesteps
        self.T = T
        self.device = device
        self.parameterization = cfg.parameterization
        self.mode = cfg.mode
        
        # Validate parameterization
        valid_params = ["predict_noise", "predict_x0", "predict_v"]
        if self.parameterization not in valid_params:
            raise ValueError(f"parameterization must be one of {valid_params}, got {self.parameterization}")
        
        # Validate mode
        valid_modes = ["baseline", "occupancy", "color"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")
        
        # Validate mode-parameterization compatibility
        if self.mode in ["occupancy", "color"] and self.parameterization != "predict_x0":
            raise ValueError(f"Mode '{self.mode}' only supports 'predict_x0' parameterization, got '{self.parameterization}'")
        
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
        model_output = model(x_t, q_in, t, ctx_in, ctx_out)
        
        # Convert model output to predicted x0 based on parameterization
        if self.parameterization == "predict_noise":
            # Standard DDPM: model predicts noise
            eps = model_output
            x0_pred = (x_t - self.sqrt_om_ac[t][:, None, None, None] * eps) / self.sqrt_ac[t][:, None, None, None]
        elif self.parameterization == "predict_x0":
            # Model directly predicts x0
            x0_pred = model_output
            # Compute corresponding noise
            eps = (x_t - self.sqrt_ac[t][:, None, None, None] * x0_pred) / self.sqrt_om_ac[t][:, None, None, None]
        elif self.parameterization == "predict_v":
            # v-parameterization: v = α_t * ε - σ_t * x0
            v_pred = model_output
            # Convert v to x0: x0 = α_t * x_t - σ_t * v
            x0_pred = self.sqrt_ac[t][:, None, None, None] * x_t - self.sqrt_om_ac[t][:, None, None, None] * v_pred
            # Compute corresponding noise 
            eps = (x_t - self.sqrt_ac[t][:, None, None, None] * x0_pred) / self.sqrt_om_ac[t][:, None, None, None]
        
        # Clamp x0 to valid range for stability (one-hot should be in [0,1])
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
        
        # Compute mean using predicted x0
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
        
        # Get model prediction
        model_output = model(x_t, q_in, t, ctx_in, ctx_out)
        
        # Compute loss based on mode and parameterization
        if self.mode == "baseline":
            # Baseline mode: use parameterization-specific loss
            if self.parameterization == "predict_noise":
                # Standard DDPM: MSE loss on noise prediction
                loss = F.mse_loss(model_output, noise)
            elif self.parameterization == "predict_x0":
                # Direct x0 prediction: categorical cross-entropy loss
                loss = self._compute_x0_loss(model_output, q_out_oh)
            elif self.parameterization == "predict_v":
                # v-parameterization: MSE loss on v-target  
                v_target = self.sqrt_ac[t][:, None, None, None] * noise - self.sqrt_om_ac[t][:, None, None, None] * q_out_oh
                loss = F.mse_loss(model_output, v_target)
                
        elif self.mode == "occupancy":
            # Occupancy mode: BCE on occupancy probabilities (only with predict_x0)
            assert self.parameterization == "predict_x0", "Occupancy mode requires predict_x0"
            # For occupancy mode, we need raw logits before softmax for BCE with logits
            # Get raw model output before softmax is applied
            raw_logits = model(x_t, q_in, t, ctx_in, ctx_out, apply_softmax=False)
            loss = self._compute_occupancy_loss_with_logits(raw_logits, q_out_oh)
            
        elif self.mode == "color":
            # Color mode: masked categorical cross-entropy on occupied pixels (only with predict_x0)
            assert self.parameterization == "predict_x0", "Color mode requires predict_x0"
            loss = self._compute_color_loss(model_output, q_out_oh)
        
        return loss
    
    def _compute_x0_loss(self, x0_pred, x0_target):
        """Compute cross-entropy loss for x0 prediction (baseline mode)."""
        # Reshape for cross-entropy: (B*S*S, 10) and (B*S*S,)
        pred_flat = x0_pred.permute(0, 2, 3, 1).reshape(-1, 10)  # (B*S*S, 10)
        target_flat = x0_target.permute(0, 2, 3, 1).reshape(-1, 10)  # (B*S*S, 10)
        
        # Convert one-hot target to class indices
        target_classes = target_flat.argmax(dim=1)  # (B*S*S,)
        
        # Categorical cross-entropy loss
        loss = F.cross_entropy(pred_flat, target_classes)
        return loss
    
    def _compute_occupancy_loss(self, x0_pred, x0_target):
        """Compute BCE loss for occupancy mode (2B: Occupancy + predict_x0)."""
        # Convert x0 predictions to binary occupancy probabilities
        occupied_pred = x0_pred[:, 1:, :, :].sum(dim=1)      # (B, S, S) - sum of color channels
        unoccupied_pred = x0_pred[:, 0, :, :]                # (B, S, S) - black channel
        
        # Ground truth binary occupancy  
        occupied_target = (x0_target[:, 1:, :, :].sum(dim=1) > 0).float()  # (B, S, S) - 1.0 if occupied, 0.0 if not
        unoccupied_target = x0_target[:, 0, :, :]            # (B, S, S) - 1.0 if unoccupied
        
        # BCE loss on binary occupancy decision
        occupied_loss = F.binary_cross_entropy(occupied_pred, occupied_target)
        unoccupied_loss = F.binary_cross_entropy(unoccupied_pred, unoccupied_target)
        
        # Average the two losses
        loss = (occupied_loss + unoccupied_loss) / 2
        return loss
    
    def _compute_occupancy_loss_with_logits(self, logits, x0_target):
        """Compute BCE loss with logits for occupancy mode (autocast-safe)."""
        # Convert logits to binary occupancy logits
        occupied_logits = logits[:, 1:, :, :].sum(dim=1)     # (B, S, S) - sum of color logits
        unoccupied_logits = logits[:, 0, :, :]               # (B, S, S) - black logits
        
        # Ground truth binary occupancy  
        occupied_target = (x0_target[:, 1:, :, :].sum(dim=1) > 0).float()  # (B, S, S)
        unoccupied_target = x0_target[:, 0, :, :]            # (B, S, S)
        
        # BCE with logits loss (autocast-safe)
        occupied_loss = F.binary_cross_entropy_with_logits(occupied_logits, occupied_target)
        unoccupied_loss = F.binary_cross_entropy_with_logits(unoccupied_logits, unoccupied_target)
        
        # Average the two losses
        loss = (occupied_loss + unoccupied_loss) / 2
        return loss
    
    def _compute_color_loss(self, x0_pred, x0_target):
        """Compute masked categorical cross-entropy for color mode (3B: Color + predict_x0)."""
        # Mask for occupied pixels (where any non-black channel = 1 in ground truth)
        occupied_mask = (x0_target[:, 1:, :, :].sum(dim=1) > 0)  # (B, S, S)
        
        if not occupied_mask.any():
            # If no occupied pixels in batch, return small loss to avoid NaN
            return torch.tensor(0.0, device=x0_pred.device, requires_grad=True)
        
        # Flatten masks and data for indexing
        occupied_mask_flat = occupied_mask.flatten()  # (B*S*S,)
        
        # Reshape predictions and targets
        pred_flat = x0_pred.permute(0, 2, 3, 1).reshape(-1, 10)  # (B*S*S, 10)
        target_flat = x0_target.permute(0, 2, 3, 1).reshape(-1, 10)  # (B*S*S, 10)
        
        # Extract only occupied pixels
        occupied_pred = pred_flat[occupied_mask_flat]    # (N_occupied, 10)  
        occupied_target = target_flat[occupied_mask_flat] # (N_occupied, 10)
        
        # Convert one-hot target to class indices
        target_classes = occupied_target.argmax(dim=1)  # (N_occupied,)
        
        # Categorical cross-entropy loss only on occupied pixels
        loss = F.cross_entropy(occupied_pred, target_classes)
        return loss
