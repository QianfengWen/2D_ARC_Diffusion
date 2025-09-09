#!/usr/bin/env python3
"""Main entry point for ARC Diffusion pipeline."""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_diffusion.config import load_config
from arc_diffusion.pipeline import ARCDiffusionPipeline


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="ARC Diffusion - Unified pipeline for training diffusion models on ARC tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (name required; will be prefixed with index for today)
  python main.py --name my_set
  
  # Use custom config
  python main.py --config my_config.yaml --name my_set
  
  # Just train (skip data generation if exists)
  python main.py --mode train --name my_set
  
  # Generate data only
  python main.py --mode generate --name my_set
  
  # Run prediction with specific checkpoint (name still required)
  python main.py --mode predict --name eval \
    --checkpoint experiments/2025-09/09/1_my_set/models/best_model.pt
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["pipeline", "train", "generate", "predict"],
        default="pipeline",
        help="Pipeline mode (default: pipeline - runs everything)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for prediction mode"
    )
    
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="Base experiment name (CLI-only). Will be prefixed as 'N_<name>' where N is the next index for today's folder."
    )
    
    parser.add_argument(
        "--episodes-dir",
        type=str,
        help="Episodes directory for prediction (if different from config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        return 1
    
    print(f"Loaded configuration: {args.config}")
    print(f"Mode: {args.mode}")
    
    # Determine today's folder and next index, then prefix the provided base name
    base_name = args.name.strip()
    if not base_name:
        print("Error: --name must be a non-empty string")
        return 1
    
    now = datetime.now()
    month = now.strftime("%Y-%m")
    day = now.strftime("%d")
    project_root = Path(__file__).resolve().parent
    day_dir = project_root / "experiments" / month / day
    day_dir.mkdir(parents=True, exist_ok=True)
    
    max_idx = 0
    for item in day_dir.iterdir():
        if item.is_dir():
            name = item.name
            if "_" in name:
                prefix, _ = name.split("_", 1)
                if prefix.isdigit():
                    try:
                        max_idx = max(max_idx, int(prefix))
                    except ValueError:
                        pass
    next_idx = max_idx + 1
    exp_name = f"{next_idx}_{base_name}"
    
    # Override experiment name and re-resolve paths based on it (paths include month/day)
    config.experiment.name = exp_name
    try:
        # Recompute path templates since name changed post-init
        config._resolve_paths()
    except Exception:
        pass
    print(f"Experiment: {config.experiment.name}")
    print(f"Experiment root: {config.paths.output_root}")
    
    # Create pipeline
    try:
        pipeline = ARCDiffusionPipeline(config)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return 1
    
    # Run based on mode
    try:
        if args.mode == "pipeline":
            results = pipeline.run()
            print("\\nPipeline completed successfully!")
            if "predictions" in results:
                print(f"Results: {results}")
            
        elif args.mode == "train":
            best_acc = pipeline.train()
            print(f"\\nTraining completed! Best accuracy: {best_acc:.4f}")
            
        elif args.mode == "generate":
            episodes_dir = pipeline._ensure_episodes()
            print(f"\\nData generation completed: {episodes_dir}")
            
        elif args.mode == "predict":
            if not args.checkpoint:
                print("Error: --checkpoint required for predict mode")
                return 1
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint not found: {args.checkpoint}")
                return 1
                
            pred_path = pipeline.predict(args.checkpoint, args.episodes_dir)
            print(f"\\nPrediction completed: {pred_path}")
            
    except Exception as e:
        print(f"Error during {args.mode}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
