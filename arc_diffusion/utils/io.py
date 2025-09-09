"""I/O utilities for ARC diffusion."""

import os
import glob
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ..data.generators import TASKS, make_unique_train_test, save_task_json


def setup_directories(config) -> None:
    """Create necessary directories for experiment."""
    directories = [
        config.paths.data_root,
        config.paths.output_root,
        config.paths.models_dir,
        config.paths.logs_dir,
        config.paths.results_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_synthetic_tasks(config) -> List[str]:
    """Generate synthetic task JSON files (legacy function for backward compatibility).
    
    Returns:
        List of paths to generated task files
    """
    tasks_dir = os.path.join(config.paths.data_root, "synthetic_tasks")
    return generate_synthetic_tasks_in_dir(config, tasks_dir)


def generate_dataset_id(config) -> str:
    """Generate a unique dataset identifier based on config parameters.
    
    The dataset ID encodes all parameters that affect dataset generation:
    - tasks, n_train, n_test, seed, attempts_per_example
    - grid_size, ctx_policy, train_per_task, test_per_task, shard_size
    
    Format: tasks_<task1+task2>_train<n>_test<n>_grid<n>_ctx<policy>_tpt<n>_shard<n>_seed<n>
    Example: tasks_bb43febb+c9f8e694_train400_test100_grid10_ctxrandom_tpt1000_shard50000_seed42
    """
    # Sort tasks for consistent naming
    tasks = sorted(config.data.generation.tasks)
    tasks_str = "+".join(tasks)
    
    dataset_id = (
        f"tasks_{tasks_str}_"
        f"train{config.data.generation.n_train}_"
        f"test{config.data.generation.n_test}_"
        f"grid{config.data.episodes.grid_size}_"
        f"ctx{config.data.episodes.ctx_policy}_"
        f"tpt{config.data.episodes.train_per_task}_"
        f"tet{config.data.episodes.test_per_task}_"
        f"shard{config.data.episodes.shard_size}_"
        f"seed{config.data.generation.seed}"
    )
    
    return dataset_id


def get_central_datasets_dir() -> str:
    """Get the central datasets directory path."""
    # Use project root for central datasets storage
    project_root = Path(__file__).parent.parent.parent
    central_dir = project_root / "datasets"
    central_dir.mkdir(exist_ok=True)
    return str(central_dir)


def find_existing_dataset(dataset_id: str) -> Optional[Tuple[str, str]]:
    """Check if a dataset with the given ID already exists in central storage.
    
    Returns:
        Tuple of (synthetic_tasks_dir, episodes_dir) if found, None otherwise
    """
    central_dir = get_central_datasets_dir()
    dataset_dir = os.path.join(central_dir, dataset_id)
    
    if not os.path.exists(dataset_dir):
        return None
    
    synthetic_tasks_dir = os.path.join(dataset_dir, "synthetic_tasks")
    episodes_dir = os.path.join(dataset_dir, "episodes")
    
    # Check if both directories exist and have required files
    if (os.path.exists(synthetic_tasks_dir) and 
        os.path.exists(episodes_dir) and 
        os.path.exists(os.path.join(episodes_dir, "meta.json"))):
        
        print(f"Found existing dataset: {dataset_id}")
        return synthetic_tasks_dir, episodes_dir
    
    return None


def create_central_dataset(config, dataset_id: str) -> Tuple[str, str]:
    """Create a new dataset in central storage.
    
    Returns:
        Tuple of (synthetic_tasks_dir, episodes_dir) paths
    """
    central_dir = get_central_datasets_dir()
    dataset_dir = os.path.join(central_dir, dataset_id)
    
    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Creating new central dataset: {dataset_id}")
    
    # Generate synthetic tasks in central location
    synthetic_tasks_dir = os.path.join(dataset_dir, "synthetic_tasks")
    task_paths = generate_synthetic_tasks_in_dir(config, synthetic_tasks_dir)
    
    # Generate episodes in central location
    episodes_dir = os.path.join(dataset_dir, "episodes")
    generate_episodes_in_dir(config, task_paths, episodes_dir)
    
    # Save dataset metadata
    metadata = {
        "dataset_id": dataset_id,
        "created_at": os.path.getctime(dataset_dir),
        "config": {
            "tasks": config.data.generation.tasks,
            "n_train": config.data.generation.n_train,
            "n_test": config.data.generation.n_test,
            "seed": config.data.generation.seed,
            "grid_size": config.data.episodes.grid_size,
            "ctx_policy": config.data.episodes.ctx_policy,
            "train_per_task": config.data.episodes.train_per_task,
            "test_per_task": config.data.episodes.test_per_task,
            "shard_size": config.data.episodes.shard_size,
        }
    }
    
    with open(os.path.join(dataset_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return synthetic_tasks_dir, episodes_dir


def copy_dataset_to_experiment(src_synthetic_tasks: str, src_episodes: str, 
                              dst_synthetic_tasks: str, dst_episodes: str) -> None:
    """Copy dataset from central storage to experiment directory."""
    # Copy synthetic tasks
    if os.path.exists(src_synthetic_tasks):
        if os.path.exists(dst_synthetic_tasks):
            shutil.rmtree(dst_synthetic_tasks)
        shutil.copytree(src_synthetic_tasks, dst_synthetic_tasks)
        print(f"Copied synthetic tasks to: {dst_synthetic_tasks}")
    
    # Copy episodes  
    if os.path.exists(src_episodes):
        if os.path.exists(dst_episodes):
            shutil.rmtree(dst_episodes)
        shutil.copytree(src_episodes, dst_episodes)
        print(f"Copied episodes to: {dst_episodes}")


def generate_synthetic_tasks_in_dir(config, tasks_dir: str) -> List[str]:
    """Generate synthetic task JSON files in specified directory.
    
    Returns:
        List of paths to generated task files
    """
    from ..config import save_config
    
    # Setup directories
    os.makedirs(tasks_dir, exist_ok=True)
    
    # Save config used for generation
    config_path = os.path.join(tasks_dir, "generation_config.yaml")
    save_config(config, config_path)
    
    task_paths = []
    for task_name in config.data.generation.tasks:
        if task_name not in TASKS:
            print(f"Warning: Unknown task {task_name}, skipping")
            continue
            
        print(f"Generating task: {task_name}")
        gen_single = TASKS[task_name]
        
        # Generate unique train/test pairs
        train, test, stats, complete = make_unique_train_test(
            gen_single, 
            config.data.generation.n_train, 
            config.data.generation.n_test,
            attempts_per_example=config.data.generation.attempts_per_example,
            strict=True,  # Always require full dataset
            progress=True
        )
        
        print(f"[{task_name}] requested {stats['requested_total']} → unique {stats['unique_generated']} (attempts {stats['attempts']}, duplicates {stats['duplicates_avoided']})")
        
        if not complete:
            print(f"[{task_name}] Not enough unique samples; skipping.")
            continue
        
        # Save task file
        task_path = os.path.join(tasks_dir, f"synth_{task_name}.json")
        save_task_json(task_path, train, test)
        print(f"Wrote: {task_path} (train={len(train)}, test={len(test)})")
        task_paths.append(task_path)
    
    return task_paths


def generate_episodes_in_dir(config, task_paths: List[str], episodes_dir: str) -> None:
    """Generate episodes from task files in specified directory."""
    from ..data.episodes import create_episodes_from_tasks
    
    create_episodes_from_tasks(
        task_paths=task_paths,
        grid_size=config.data.episodes.grid_size,
        train_per_task=config.data.episodes.train_per_task,
        test_per_task=config.data.episodes.test_per_task,
        ctx_policy=config.data.episodes.ctx_policy,
        shard_size=config.data.episodes.shard_size,
        out_dir=episodes_dir,
        seed=config.data.generation.seed
    )


def ensure_dataset_exists(config) -> Tuple[str, str]:
    """Ensure dataset exists, either by reusing existing or creating new.
    
    Returns:
        Tuple of (synthetic_tasks_dir, episodes_dir) in experiment directory
    """
    # Generate dataset ID based on config
    dataset_id = generate_dataset_id(config)
    print(f"Dataset ID: {dataset_id}")
    
    # Check if dataset exists in central storage
    existing = find_existing_dataset(dataset_id)
    
    if existing:
        central_synthetic_tasks, central_episodes = existing
        print(f"Reusing existing dataset from central storage")
    else:
        print(f"Creating new dataset in central storage")
        central_synthetic_tasks, central_episodes = create_central_dataset(config, dataset_id)
    
    # Set up experiment directories
    exp_synthetic_tasks = os.path.join(config.paths.data_root, "synthetic_tasks")
    exp_episodes = os.path.join(config.paths.data_root, "episodes")
    
    # Copy from central storage to experiment directory
    copy_dataset_to_experiment(central_synthetic_tasks, central_episodes,
                              exp_synthetic_tasks, exp_episodes)
    
    return exp_synthetic_tasks, exp_episodes


def find_task_files(config) -> List[str]:
    """Find existing task files or generate them if needed using central dataset management."""
    synthetic_tasks_dir, _ = ensure_dataset_exists(config)
    
    # Get required task files from experiment directory
    required_files = [os.path.join(synthetic_tasks_dir, f"synth_{task}.json") 
                     for task in config.data.generation.tasks]
    
    # Verify all files exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(f"Missing task files after dataset setup: {missing_files}")
    
    print(f"Using task files from: {synthetic_tasks_dir}")
    return required_files
