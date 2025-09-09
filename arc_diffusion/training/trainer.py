"""Enhanced trainer with monitoring for ARC diffusion."""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .metrics import evaluate


class LossPlotter:
    """Real-time loss plotting during training."""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.losses = []
        self.steps = []
        
    def update(self, loss: float, step: int):
        """Add a new loss value and update the plot."""
        self.losses.append(loss)
        self.steps.append(step)
        
        # Update plot every 10 steps to avoid too frequent I/O
        if len(self.losses) % 10 == 0:
            self.plot()
    
    def plot(self):
        """Create and save the loss plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, 'b-', alpha=0.7)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory


class DiffusionTrainer:
    """Enhanced trainer for diffusion models."""
    
    def __init__(self, model, diffusion, optimizer, scaler, config, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.device = device
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Track best model
        self.best_prob_acc = 0.0
        self.step_count = 0
        
    def setup_monitoring(self):
        """Setup loss plotting and logging."""
        self.loss_plotter = None
        if self.config.training.plot_losses:
            loss_plot_path = os.path.join(self.config.paths.logs_dir, "loss_plot.png")
            self.loss_plotter = LossPlotter(loss_plot_path)
    
    def train_one_epoch(self, loader, epoch: int) -> float:
        """Train for one epoch with monitoring."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for it, batch in enumerate(pbar):
            # Move batch to device
            batch = [item.to(self.device) for item in batch]
            
            # Mixed precision training
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                loss = self.diffusion.compute_loss(self.model, batch)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Tracking
            running_loss += loss.item()
            num_batches += 1
            self.step_count += 1
            
            # Real-time monitoring
            if self.step_count % self.config.training.log_frequency == 0:
                current_avg_loss = running_loss / num_batches
                pbar.set_postfix(loss=current_avg_loss)
                
                # Update loss plot
                if self.loss_plotter:
                    self.loss_plotter.update(current_avg_loss, self.step_count)
        
        return running_loss / max(1, num_batches)
    
    def save_checkpoint(self, epoch: int, pix_acc: float, prob_acc: float, 
                       is_best: bool = False, is_periodic: bool = False):
        """Save model checkpoint with meaningful naming."""
        os.makedirs(self.config.paths.models_dir, exist_ok=True)
        
        checkpoint = {
            "model": self.model.state_dict(),
            "cfg": {
                "timesteps": self.config.diffusion.params["timesteps"], 
                "beta_start": self.config.diffusion.params["beta_start"],
                "beta_end": self.config.diffusion.params["beta_end"]
            },
            "grid_size": self.config.data.episodes.grid_size,
            "epoch": epoch,
            "pix_acc": pix_acc,
            "prob_acc": prob_acc,
            "step": self.step_count
        }
        
        if is_best:
            path = os.path.join(self.config.paths.models_dir, "best_model.pt")
            torch.save(checkpoint, path)
            print(f"  ↑ Saved best model: {path}")
            
        if is_periodic:
            path = os.path.join(self.config.paths.models_dir, f"epoch_{epoch:03d}.pt")
            torch.save(checkpoint, path)
            print(f"  Saved checkpoint: {path}")
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        print(f"Starting training for {self.config.training.epochs} epochs")
        print(f"Model saved to: {self.config.paths.models_dir}")
        print(f"Logs saved to: {self.config.paths.logs_dir}")
        
        for epoch in range(1, self.config.training.epochs + 1):
            # Training
            train_loss = self.train_one_epoch(train_loader, epoch)
            
            # Validation
            pix_acc, prob_acc = 0.0, 0.0
            if epoch % self.config.training.val_frequency == 0:
                pix_acc, prob_acc = evaluate(
                    self.model, self.diffusion, val_loader, 
                    self.device, max_batches=self.config.training.val_batches
                )
            
            print(f"[Epoch {epoch:3d}] loss={train_loss:.4f}  val_pixel_acc={pix_acc:.4f}  val_problem_acc={prob_acc:.4f}")
            
            # Checkpointing
            is_best = prob_acc > self.best_prob_acc
            if is_best:
                self.best_prob_acc = prob_acc
                
            is_periodic = epoch % self.config.training.save_frequency == 0
            
            if is_best or is_periodic:
                self.save_checkpoint(epoch, pix_acc, prob_acc, is_best, is_periodic)
        
        print(f"Training complete. Best problem accuracy: {self.best_prob_acc:.4f}")
        return self.best_prob_acc
