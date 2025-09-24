"""Enhanced trainer with monitoring for ARC diffusion."""

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .metrics import evaluate
from ..utils.visualization import save_episode_ctx_pred, save_collage_ctx_pred
from ..data.dataset import EpisodesPTDataset


class LossPlotter:
    """Plots per-epoch training loss with validation loss overlay."""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        # Track epoch-wise losses
        self.train_epochs = []
        self.train_losses = []
        self.val_epochs = []
        self.val_losses = []

    def record_train_epoch(self, epoch: int, loss: float):
        self.train_epochs.append(epoch)
        self.train_losses.append(loss)
        self.plot()

    def record_val_epoch(self, epoch: int, val_loss: float):
        self.val_epochs.append(epoch)
        self.val_losses.append(val_loss)
        self.plot()
    
    def plot(self):
        """Create and save the per-epoch loss plot with validation overlay.
        Uses log scale on the Y-axis for clearer visibility across magnitudes.
        """
        plt.figure(figsize=(10, 6))
        if self.train_epochs:
            plt.plot(self.train_epochs, self.train_losses, 'b-o', alpha=0.8, label='Train Loss')
        if self.val_epochs:
            plt.plot(self.val_epochs, self.val_losses, 'r-s', alpha=0.9, label='Val Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        # Use log scale; matplotlib handles positive values. Losses should be > 0.
        try:
            plt.yscale('log')
        except Exception:
            pass
        plt.grid(True, which='both', alpha=0.3)
        if self.train_epochs or self.val_epochs:
            plt.legend()
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory


class AccuracyPlotter:
    """Plots validation pixel and problem accuracy over epochs."""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.epochs = []
        self.pix_acc = []
        self.prob_acc = []

    def update(self, epoch: int, pix_acc: float, prob_acc: float):
        self.epochs.append(epoch)
        self.pix_acc.append(pix_acc)
        self.prob_acc.append(prob_acc)
        self.plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        if self.epochs:
            plt.plot(self.epochs, self.pix_acc, 'g-', marker='o', label='Val Pixel Acc')
            plt.plot(self.epochs, self.prob_acc, 'm-', marker='s', label='Val Problem Acc')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close()


class CombinedPlotter:
    """Combined loss and accuracy plotter."""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        # Loss tracking
        self.train_epochs = []
        self.train_losses = []
        self.val_epochs = []
        self.val_losses = []
        # Accuracy tracking
        self.acc_epochs = []
        self.pix_acc = []
        self.prob_acc = []
        
    def record_train_epoch(self, epoch: int, loss: float):
        self.train_epochs.append(epoch)
        self.train_losses.append(loss)
        self.plot()
        
    def record_val_epoch(self, epoch: int, val_loss: float):
        self.val_epochs.append(epoch)
        self.val_losses.append(val_loss)
        self.plot()
        
    def record_accuracy(self, epoch: int, pix_acc: float, prob_acc: float):
        self.acc_epochs.append(epoch)
        self.pix_acc.append(pix_acc)
        self.prob_acc.append(prob_acc)
        self.plot()
        
    def plot(self):
        """Create combined loss and accuracy plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Loss
        if self.train_epochs:
            ax1.plot(self.train_epochs, self.train_losses, 'b-o', alpha=0.8, label='Train Loss')
        if self.val_epochs:
            ax1.plot(self.val_epochs, self.val_losses, 'r-s', alpha=0.9, label='Val Loss')
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        try:
            ax1.set_yscale('log')
        except Exception:
            pass
        ax1.grid(True, which='both', alpha=0.3)
        if self.train_epochs or self.val_epochs:
            ax1.legend()
            
        # Right plot: Accuracy
        if self.acc_epochs:
            ax2.plot(self.acc_epochs, self.pix_acc, 'g-', marker='o', label='Val Pixel Acc')
            ax2.plot(self.acc_epochs, self.prob_acc, 'm-', marker='s', label='Val Problem Acc')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close()


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
        # Per-task plotting state created on-demand when multi-task metrics present
        self._task_plotters = {}
        
    def setup_monitoring(self):
        """Setup combined loss and accuracy plotting."""
        self.loss_plotter = None
        self.acc_plotter = None
        self.combined_plotter = None
        
        if self.config.training.plot_losses:
            # Use combined plotter instead of separate ones
            combined_plot_path = os.path.join(self.config.paths.logs_dir, "training_progress.png")
            self.combined_plotter = CombinedPlotter(combined_plot_path)
            
            # Keep individual plotters for backward compatibility (but they won't be used)
            loss_plot_path = os.path.join(self.config.paths.logs_dir, "loss_plot.png")
            acc_plot_path = os.path.join(self.config.paths.logs_dir, "val_accuracy.png")
            self.loss_plotter = LossPlotter(loss_plot_path)
            self.acc_plotter = AccuracyPlotter(acc_plot_path)

    def _get_task_slug_map(self, dataset) -> dict:
        """Load task slug list from dataset meta if available."""
        try:
            base = dataset.base_dir
            meta_path = os.path.join(base, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            slugs = meta.get("task_slugs") or []
            names = meta.get("task_names") or []
            return {i: (slugs[i] if i < len(slugs) and slugs[i] else (names[i] if i < len(names) else f"tid{i:02d}"))
                    for i in range(max(len(slugs), len(names)))}
        except Exception:
            return {}

    def _update_per_task_accuracy_plots(self, dataset, epoch: int, per_task: dict):
        """Update per-task accuracy plots, creating plotters on first use.
        Only called when multiple tasks are present.
        """
        slug_map = self._get_task_slug_map(dataset)
        for tid, metrics in per_task.items():
            slug = slug_map.get(tid, f"tid{int(tid):02d}")
            if tid not in self._task_plotters:
                save_path = os.path.join(self.config.paths.logs_dir, f"val_accuracy_task_{slug}.png")
                self._task_plotters[tid] = AccuracyPlotter(save_path)
            self._task_plotters[tid].update(epoch, float(metrics.get("pix_acc", 0.0)), float(metrics.get("prob_acc", 0.0)))
    
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
            
            # Real-time monitoring in progress bar only
            if self.step_count % self.config.training.log_frequency == 0:
                current_avg_loss = running_loss / num_batches
                pbar.set_postfix(loss=current_avg_loss)
        
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
            
            # After training epoch finishes, record train loss point
            if self.combined_plotter:
                self.combined_plotter.record_train_epoch(epoch, train_loss)
            elif self.loss_plotter:  # Fallback to old plotter
                self.loss_plotter.record_train_epoch(epoch, train_loss)

            # Validation (only on scheduled epochs)
            did_val = (epoch % self.config.training.val_frequency == 0)
            pix_acc, prob_acc, val_loss = 0.0, 0.0, None
            if did_val:
                pix_acc, prob_acc, val_loss, per_task = evaluate(
                    self.model, self.diffusion, val_loader,
                    self.device, max_batches=self.config.training.val_batches
                )
                # Update plots
                if self.combined_plotter:
                    if val_loss is not None:
                        self.combined_plotter.record_val_epoch(epoch, val_loss)
                    self.combined_plotter.record_accuracy(epoch, pix_acc, prob_acc)
                else:
                    # Fallback to old plotters
                    if self.loss_plotter and val_loss is not None:
                        self.loss_plotter.record_val_epoch(epoch, val_loss)
                    if self.acc_plotter:
                        self.acc_plotter.update(epoch, pix_acc, prob_acc)

                # If multi-task metrics available, update per-task plots
                if per_task and len(per_task) > 1:
                    try:
                        self._update_per_task_accuracy_plots(val_loader.dataset, epoch, per_task)
                    except Exception as e:
                        print(f"  (warn) per-task accuracy plotting failed: {e}")

                # Visualize first few samples from the first validation batch
                try:
                    self._visualize_validation_samples(val_loader, epoch)
                except Exception as vis_e:
                    print(f"  (warn) validation visualization failed: {vis_e}")

            # Printing: only include validation metrics on validation epochs
            if did_val and val_loss is not None:
                print(
                    f"[Epoch {epoch:3d}] loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_pixel_acc={pix_acc:.4f}  val_problem_acc={prob_acc:.4f}"
                )
            else:
                print(f"[Epoch {epoch:3d}] loss={train_loss:.4f}")
            
            # Checkpointing
            is_best = (pix_acc > 0 or prob_acc > 0) and (prob_acc > self.best_prob_acc)
            if is_best:
                self.best_prob_acc = prob_acc
                
            is_periodic = epoch % self.config.training.save_frequency == 0
            
            if is_best or is_periodic:
                self.save_checkpoint(epoch, pix_acc, prob_acc, is_best, is_periodic)
        
        print(f"Training complete. Best problem accuracy: {self.best_prob_acc:.4f}")
        return self.best_prob_acc

    @torch.no_grad()
    def _visualize_validation_samples(self, loader, epoch: int):
        """Visualize validation samples.

        If `group_by_task` is True, saves one collage per task using a few
        examples per task. Otherwise, falls back to a single collage from the
        first validation batch.
        """
        out_dir = os.path.join(self.config.paths.logs_dir, "val_vis")
        os.makedirs(out_dir, exist_ok=True)

        if not getattr(self.config.visualization, "group_by_task", True):
            # Legacy behavior: just visualize first batch
            try:
                batch = next(iter(loader))
            except StopIteration:
                return
            ctx_in, ctx_out, q_in, q_out_oh, q_out_idx = [x.to(self.device) for x in batch]
            B = q_in.size(0)
            S = q_in.shape[-1]
            x0 = self.diffusion.sample(self.model, q_in, (B, 10, S, S), ctx_in, ctx_out)
            preds = x0.argmax(dim=1)
            collage_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
            save_collage_ctx_pred(
                ctx_in_batch=ctx_in,
                ctx_out_batch=ctx_out,
                q_in_batch=q_in,
                pred_idx_batch=preds,
                path=collage_path,
                max_samples=self.config.visualization.val_samples,
                cols=self.config.visualization.collage_cols,
                include_query=self.config.visualization.include_query,
                dpi=self.config.visualization.save_dpi,
                mode=self.diffusion.mode,
            )
            return

        # Group-by-task visualization
        episodes_dir = loader.dataset.base_dir
        meta_path = os.path.join(episodes_dir, "meta.json")
        task_names = None
        task_slugs = None
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                task_names = meta.get("task_names")
                task_slugs = meta.get("task_slugs")
        except Exception:
            pass

        # Build a lightweight dataset that returns task ids
        ds = EpisodesPTDataset(episodes_dir, split="test", return_tid=True)
        per_task = {}
        samples_needed = getattr(self.config.visualization, "val_samples_per_task", None) or max(1, self.config.visualization.val_samples)
        max_scan = max(1, int(getattr(self.config.visualization, "max_vis_scan", 2000)))

        # Scan through validation examples collecting up to N per task
        added = 0
        for i in range(min(len(ds), max_scan)):
            item = ds[i]
            # item: (ctx_in, ctx_out, q_in, q_out_oh, q_out_idx, q_tid)
            tid = int(item[5].item()) if hasattr(item[5], "item") else int(item[5])
            bucket = per_task.setdefault(tid, [])
            if len(bucket) < samples_needed:
                bucket.append(i)
                added += 1
            # If we know all tasks, early-exit when filled
            if task_names and len(per_task) == len(task_names):
                if all(len(per_task[t]) >= samples_needed for t in per_task):
                    break
        if not per_task:
            return

        # Render one collage per task id, collect paths for merging
        collage_paths = []
        collage_titles = []
        for tid, idxs in per_task.items():
            if not idxs:
                continue
            idxs = idxs[:samples_needed]
            # Gather tensors into a batch
            ctx_in_list, ctx_out_list, q_in_list, pred_list = [], [], [], []
            for idx in idxs:
                ci, co, qi, _, _ , _tid = ds[idx]
                ctx_in_list.append(ci)
                ctx_out_list.append(co)
                q_in_list.append(qi)
            # Stack and move to device
            ctx_in = torch.stack(ctx_in_list, dim=0).to(self.device)
            ctx_out = torch.stack(ctx_out_list, dim=0).to(self.device)
            q_in = torch.stack(q_in_list, dim=0).to(self.device)
            B, _, S, _ = q_in.shape
            x0 = self.diffusion.sample(self.model, q_in, (B, 10, S, S), ctx_in, ctx_out)
            preds = x0.argmax(dim=1).cpu()

            # Name and save
            slug = None
            if task_slugs and tid < len(task_slugs):
                slug = task_slugs[tid]
            elif task_names and tid < len(task_names):
                # fallback to a slugified version of the name if no slug list (rare)
                from ..data.task_names import slugify
                slug = slugify(task_names[tid])
            else:
                slug = f"tid{tid:02d}"

            collage_path = os.path.join(out_dir, f"epoch_{epoch:03d}_task_{slug}.png")
            save_collage_ctx_pred(
                ctx_in_batch=ctx_in.cpu(),
                ctx_out_batch=ctx_out.cpu(),
                q_in_batch=q_in.cpu(),
                pred_idx_batch=preds,
                path=collage_path,
                max_samples=len(idxs),
                cols=self.config.visualization.collage_cols,
                include_query=self.config.visualization.include_query,
                dpi=self.config.visualization.save_dpi,
                mode=self.diffusion.mode,
            )
            collage_paths.append(collage_path)
            collage_titles.append(slug)

        # If multiple tasks, also merge per-task collages into one grid
        try:
            if len(collage_paths) > 1:
                from ..utils.visualization import merge_pngs_grid
                merged_path = os.path.join(out_dir, f"epoch_{epoch:03d}_tasks_collage.png")
                merge_pngs_grid(collage_paths, merged_path, cols=self.config.visualization.collage_cols, titles=collage_titles, dpi=self.config.visualization.save_dpi)
        except Exception as e:
            print(f"  (warn) merging task collages failed: {e}")
