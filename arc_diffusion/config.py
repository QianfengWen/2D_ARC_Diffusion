"""Configuration loading and validation for ARC Diffusion."""

import os
import yaml
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str = ""  # Will be set by CLI; no default to avoid confusion
    description: str = "Default experiment"


@dataclass
class ModelConfig:
    architecture: str = "flat_unet_3shot"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "model_ch": 128,
        "num_blocks": 6,
        "t_dim": 256,
        "ctx_dim": 256
    })


@dataclass
class DiffusionConfig:
    method: str = "ddpm"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "timesteps": 400,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "schedule": "linear"
    })


@dataclass 
class GenerationConfig:
    tasks: List[str] = field(default_factory=lambda: ["bb43febb"])
    n_train: int = 400
    n_test: int = 100
    seed: int = 42
    attempts_per_example: int = 50

@dataclass
class EpisodesConfig:
    grid_size: int = 10
    ctx_policy: str = "random"
    train_per_task: int = 5000
    test_per_task: int = 500
    shard_size: int = 50000

@dataclass
class LoadingConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass
class DataConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    episodes: EpisodesConfig = field(default_factory=EpisodesConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)


@dataclass
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 2e-4
    optimizer: str = "adamw"
    scheduler: Dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    val_frequency: int = 5
    val_batches: int = 10
    save_frequency: int = 10
    save_top_k: int = 3
    log_frequency: int = 100
    plot_losses: bool = True
    save_samples: bool = False


@dataclass
class PathConfig:
    # Paths are grouped by month/day, with a day folder (DD) and the
    # experiment name applied as the final component. Supported template vars:
    #  - {experiment.name}: name of the experiment (CLI-only; not from config)
    #  - {time.month}: current month in YYYY-MM
    #  - {time.date}: current date in YYYY-MM-DD (not used by defaults)
    #  - {time.day}: current day in DD
    data_root: str = "experiments/{time.month}/{time.day}/{experiment.name}/data"
    output_root: str = "experiments/{time.month}/{time.day}/{experiment.name}"
    models_dir: str = "{output_root}/models"
    logs_dir: str = "{output_root}/logs"
    results_dir: str = "{output_root}/results"


@dataclass
class HardwareConfig:
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class Config:
    """Main configuration class."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    def __post_init__(self):
        """Process paths with template substitution."""
        # Only resolve paths if experiment name is set (will be set by CLI)
        if self.experiment.name:
            self._resolve_paths()
    
    def _resolve_paths(self):
        """Resolve path templates like {experiment.name}."""
        # Safety check: prevent empty experiment names from causing path issues
        if not self.experiment.name or self.experiment.name.strip() == "":
            raise ValueError("Experiment name cannot be empty. Please provide a name via --name argument.")
        
        # Get template variables
        now = datetime.now()
        template_vars = {
            "experiment.name": self.experiment.name,
            "time.month": now.strftime("%Y-%m"),
            "time.date": now.strftime("%Y-%m-%d"),
            "time.day": now.strftime("%d"),
        }
        
        # Resolve data_root and output_root first
        for key, value in template_vars.items():
            self.paths.data_root = self.paths.data_root.replace(f"{{{key}}}", value)
            self.paths.output_root = self.paths.output_root.replace(f"{{{key}}}", value)
        
        # Then resolve other paths that may depend on output_root
        template_vars["output_root"] = self.paths.output_root
        
        for attr in ["models_dir", "logs_dir", "results_dir"]:
            path_value = getattr(self.paths, attr)
            for key, value in template_vars.items():
                path_value = path_value.replace(f"{{{key}}}", value)
            setattr(self.paths, attr, path_value)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Config object with loaded settings
    """
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using defaults")
        return Config()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if not config_dict:
        return Config()
    
    # Create config objects from dict
    experiment = ExperimentConfig(**config_dict.get("experiment", {}))
    model = ModelConfig(**config_dict.get("model", {}))
    diffusion = DiffusionConfig(**config_dict.get("diffusion", {}))
    
    # Handle nested data config
    data_dict = config_dict.get("data", {})
    generation = GenerationConfig(**data_dict.get("generation", {}))
    episodes = EpisodesConfig(**data_dict.get("episodes", {}))
    loading = LoadingConfig(**data_dict.get("loading", {}))
    data = DataConfig(generation=generation, episodes=episodes, loading=loading)
    
    training = TrainingConfig(**config_dict.get("training", {}))
    paths = PathConfig(**config_dict.get("paths", {}))
    hardware = HardwareConfig(**config_dict.get("hardware", {}))
    
    return Config(
        experiment=experiment,
        model=model,
        diffusion=diffusion,
        data=data,
        training=training,
        paths=paths,
        hardware=hardware
    )


def save_config(config: Config, path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert to dict for serialization
    config_dict = {
        "experiment": {
            "name": config.experiment.name,
            "description": config.experiment.description
        },
        "model": {
            "architecture": config.model.architecture,
            "params": config.model.params
        },
        "diffusion": {
            "method": config.diffusion.method,
            "params": config.diffusion.params
        },
        "data": {
            "generation": {
                "tasks": config.data.generation.tasks,
                "n_train": config.data.generation.n_train,
                "n_test": config.data.generation.n_test,
                "seed": config.data.generation.seed,
                "attempts_per_example": config.data.generation.attempts_per_example
            },
            "episodes": {
                "grid_size": config.data.episodes.grid_size,
                "ctx_policy": config.data.episodes.ctx_policy,
                "train_per_task": config.data.episodes.train_per_task,
                "test_per_task": config.data.episodes.test_per_task,
                "shard_size": config.data.episodes.shard_size
            },
            "loading": {
                "batch_size": config.data.loading.batch_size,
                "num_workers": config.data.loading.num_workers,
                "pin_memory": config.data.loading.pin_memory,
                "prefetch_factor": config.data.loading.prefetch_factor
            }
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "val_frequency": config.training.val_frequency,
            "val_batches": config.training.val_batches,
            "save_frequency": config.training.save_frequency,
            "save_top_k": config.training.save_top_k,
            "log_frequency": config.training.log_frequency,
            "plot_losses": config.training.plot_losses,
            "save_samples": config.training.save_samples
        },
        "paths": {
            "data_root": config.paths.data_root,
            "output_root": config.paths.output_root,
            "models_dir": config.paths.models_dir,
            "logs_dir": config.paths.logs_dir,
            "results_dir": config.paths.results_dir
        },
        "hardware": {
            "device": config.hardware.device,
            "mixed_precision": config.hardware.mixed_precision,
            "compile_model": config.hardware.compile_model
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
