"""Unified pipeline for ARC diffusion training."""

import os
import glob
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .models import MODEL_REGISTRY, DIFFUSION_REGISTRY, DiffusionCfg
from .data.dataset import EpisodesPTDataset
from .training.trainer import DiffusionTrainer
from .utils.io import setup_directories, ensure_dataset_exists
from .utils.visualization import save_episode_ctx_pred, save_collage_ctx_pred


class ARCDiffusionPipeline:
    """Unified pipeline for data generation, training, and evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        
        # Setup directories
        setup_directories(config)
        
        print(f"Pipeline initialized:")
        print(f"  Experiment: {config.experiment.name}")
        print(f"  Device: {self.device}")
        print(f"  Output: {config.paths.output_root}")
    
    def _setup_device(self):
        """Setup compute device."""
        if self.config.hardware.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps") 
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.hardware.device)
        
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        return device
    
    def _episodes_exist(self) -> bool:
        """Check if episode shards already exist."""
        episodes_dir = os.path.join(self.config.paths.data_root, "episodes")
        meta_path = os.path.join(episodes_dir, "meta.json")
        return os.path.exists(meta_path)
    
    def _ensure_episodes(self) -> str:
        """Ensure episode data exists using centralized dataset management."""
        episodes_dir = os.path.join(self.config.paths.data_root, "episodes")
        
        if self._episodes_exist():
            print(f"Found existing episodes: {episodes_dir}")
            return episodes_dir
        
        print("Ensuring dataset exists (checking central storage)...")
        
        # Use centralized dataset management
        # This will either reuse existing dataset or create new one
        synthetic_tasks_dir, episodes_dir = ensure_dataset_exists(self.config)
        
        return episodes_dir
    
    def _create_model(self):
        """Create model and diffusion process."""
        # Get model class
        model_class = MODEL_REGISTRY[self.config.model.architecture]
        
        # Add parameterization to model params
        model_params = self.config.model.params.copy()
        model_params["parameterization"] = self.config.diffusion.parameterization
        
        model = model_class(**model_params).to(self.device)
        
        # Get diffusion process
        diffusion_class = DIFFUSION_REGISTRY[self.config.diffusion.method]
        # Filter params to only include those supported by DiffusionCfg
        supported_params = {
            "timesteps": self.config.diffusion.params.get("timesteps", 400),
            "beta_start": self.config.diffusion.params.get("beta_start", 1e-4),
            "beta_end": self.config.diffusion.params.get("beta_end", 2e-2),
            "parameterization": self.config.diffusion.parameterization,
            "mode": self.config.diffusion.mode,
        }
        diffusion_cfg = DiffusionCfg(**supported_params)
        diffusion = diffusion_class(diffusion_cfg, self.device)
        
        return model, diffusion
    
    def _create_data_loaders(self, episodes_dir: str):
        """Create training and validation data loaders."""
        # Create datasets
        train_dataset = EpisodesPTDataset(episodes_dir, split="train")
        # For validation, include task ids so we can compute per-task metrics
        val_dataset = EpisodesPTDataset(episodes_dir, split="test", return_tid=True)
        
        # DataLoader kwargs
        dl_kwargs = {
            "batch_size": self.config.data.loading.batch_size,
            "pin_memory": self.config.data.loading.pin_memory
        }
        
        if self.config.data.loading.num_workers > 0:
            dl_kwargs.update({
                "num_workers": self.config.data.loading.num_workers,
                "persistent_workers": True,
                "prefetch_factor": self.config.data.loading.prefetch_factor
            })
        else:
            dl_kwargs.update({"num_workers": 0})
        
        # Create loaders
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
        
        print(f"Data loaders created:")
        print(f"  Train: {len(train_dataset)} episodes, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} episodes, {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _create_optimizer(self, model):
        """Create optimizer and scaler."""
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config.training.learning_rate, 
                betas=(0.9, 0.999)
            )
        elif self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.config.training.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Mixed precision scaler
        if self.config.hardware.mixed_precision:
            try:
                scaler = torch.amp.GradScaler(device=self.device.type if self.device.type != "cpu" else None)
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=="cuda"))
        else:
            scaler = None
        
        return optimizer, scaler
    
    def train(self) -> float:
        """Run the complete training pipeline."""
        print("\\n=== Starting Training Pipeline ===")
        
        # Ensure data exists
        episodes_dir = self._ensure_episodes()
        
        # Create model and diffusion
        model, diffusion = self._create_model()
        print(f"Model: {self.config.model.architecture} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(episodes_dir)
        
        # Create optimizer
        optimizer, scaler = self._create_optimizer(model)
        
        # Create trainer
        trainer = DiffusionTrainer(model, diffusion, optimizer, scaler, self.config, self.device)
        
        # Run training
        best_acc = trainer.train(train_loader, val_loader)
        
        print("\\n=== Training Complete ===")
        return best_acc
    
    def predict(self, checkpoint_path: str, episodes_dir: str = None) -> str:
        """Run prediction on test episodes."""
        if episodes_dir is None:
            episodes_dir = self._ensure_episodes()
        
        print(f"\\n=== Running Prediction ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Episodes: {episodes_dir}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Create model
        model, diffusion = self._create_model()
        model.load_state_dict(ckpt["model"])
        model.eval()
        
        # Create test loader
        test_dataset = EpisodesPTDataset(episodes_dir, split="test", return_tid=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Generate predictions
        predictions = []
        vis_dir = os.path.join(self.config.paths.results_dir, "test_vis")
        os.makedirs(vis_dir, exist_ok=True)
        # Accumulate per task id for collages
        per_task_samples = {}
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                if len(batch) == 6:
                    ctx_in, ctx_out, q_in, q_out_oh, q_out_idx, q_tid = batch
                else:
                    ctx_in, ctx_out, q_in, q_out_oh, q_out_idx = batch
                    q_tid = None
                ctx_in = ctx_in.to(self.device)
                ctx_out = ctx_out.to(self.device)
                q_in = q_in.to(self.device)
                
                S = q_in.shape[-1]
                x0 = diffusion.sample(model, q_in, (1, 10, S, S), ctx_in, ctx_out)
                pred = x0.argmax(dim=1)[0].cpu().tolist()
                
                predictions.append({
                    "query_input": q_in[0].argmax(dim=0).cpu().tolist(),
                    "pred_output": pred,
                    "query_gt": q_out_idx[0].cpu().tolist()
                })
                # Accumulate per-task for collage
                try:
                    tid_int = int(q_tid[0].item()) if q_tid is not None else -1
                except Exception:
                    tid_int = -1
                bucket = per_task_samples.setdefault(tid_int, [])
                if len(bucket) < getattr(self.config.visualization, "test_samples_per_task", None) or len(bucket) < self.config.visualization.test_samples:
                    bucket.append((
                        ctx_in[0].cpu(), ctx_out[0].cpu(), q_in[0].cpu(), x0.argmax(dim=1)[0].cpu()
                    ))

        # Save per-task collages for test predictions
        # Try to load task slugs to name files
        task_slugs = None
        try:
            with open(os.path.join(test_dataset.base_dir, "meta.json"), "r") as f:
                meta = json.load(f)
                task_slugs = meta.get("task_slugs")
        except Exception:
            pass
        merged_collage_paths = []
        merged_titles = []
        for tid, samples in per_task_samples.items():
            if not samples:
                continue
            try:
                # Stack into batch tensors
                n = min(len(samples), getattr(self.config.visualization, "test_samples_per_task", None) or self.config.visualization.test_samples)
                ctx_in_b = torch.stack([s[0] for s in samples[:n]], dim=0)  # (B,K,10,S,S)
                ctx_out_b = torch.stack([s[1] for s in samples[:n]], dim=0)
                q_in_b = torch.stack([s[2] for s in samples[:n]], dim=0)
                pred_b = torch.stack([s[3] for s in samples[:n]], dim=0)    # (B,S,S)
                if task_slugs and tid >= 0 and tid < len(task_slugs):
                    slug = task_slugs[tid]
                else:
                    slug = f"tid{tid:02d}" if tid >= 0 else "unknown"
                out_path = os.path.join(vis_dir, f"collage_{slug}.png")
                save_collage_ctx_pred(
                    ctx_in_batch=ctx_in_b,
                    ctx_out_batch=ctx_out_b,
                    q_in_batch=q_in_b,
                    pred_idx_batch=pred_b,
                    path=out_path,
                    max_samples=n,
                    cols=self.config.visualization.collage_cols,
                    include_query=self.config.visualization.include_query,
                    dpi=self.config.visualization.save_dpi,
                    mode=diffusion.mode,
                )
                merged_collage_paths.append(out_path)
                merged_titles.append(slug)
            except Exception as vis_e:
                print(f"(warn) saving test collage for task {tid} failed: {vis_e}")

        # Merge per-task collages into one image if multiple tasks
        try:
            if len(merged_collage_paths) > 1:
                from .utils.visualization import merge_pngs_grid
                merge_pngs_grid(merged_collage_paths, os.path.join(vis_dir, "collage_all_tasks.png"), cols=self.config.visualization.collage_cols, titles=merged_titles, dpi=self.config.visualization.save_dpi)
        except Exception as e:
            print(f"(warn) merging test collages failed: {e}")
        
        # Save predictions
        out_path = os.path.join(self.config.paths.results_dir, "predictions.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"test_episodes_outputs": predictions}, f)
        
        print(f"Saved {len(predictions)} predictions to: {out_path}")
        return out_path
    
    def run(self):
        """Run the complete pipeline: data generation -> training -> evaluation."""
        print(f"\\n=== Running Complete Pipeline ===")
        
        # Save the configuration used
        from .config import save_config
        config_path = os.path.join(self.config.paths.output_root, "config.yaml")
        save_config(self.config, config_path)
        print(f"Saved config to: {config_path}")
        
        # Train model
        best_acc = self.train()
        
        # Find best checkpoint and run prediction
        best_ckpt = os.path.join(self.config.paths.models_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            pred_path = self.predict(best_ckpt)
            print(f"\\n=== Pipeline Complete ===")
            print(f"Best accuracy: {best_acc:.4f}")
            print(f"Predictions: {pred_path}")
            return {"best_accuracy": best_acc, "predictions": pred_path}
        else:
            print(f"\\n=== Training Complete ===")
            print(f"Best accuracy: {best_acc:.4f}")
            return {"best_accuracy": best_acc}
