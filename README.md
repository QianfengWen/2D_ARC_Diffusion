# 2D_ARC_Diffusion

A modular diffusion model for ARC (Abstraction and Reasoning Corpus) tasks with clean architecture and unified pipeline.

## Features

- **Modular Design**: Easily swap UNet architectures and diffusion methods
- **Unified Pipeline**: Single command runs data generation → training → evaluation
- **Smart Data Management**: Automatically detects existing data, generates only if needed
- **Real-time Monitoring**: Loss plotting during training with clean checkpointing
- **Clean Configuration**: All options documented in a single YAML config file
- **Better Organization**: Organized experiments and clean naming conventions

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Run everything; name is required and will be prefixed with the next index for today's folder (DD)
python main.py --name my_set

# Use custom config (still prompts unless --name is provided)
python main.py --config my_config.yaml --name my_experiment
```

### 3. Individual Commands
```bash
# Generate data only
python main.py --mode generate --name my_set

# Train only (uses existing data)
python main.py --mode train

# Predict with specific checkpoint
# Note the time-grouped folder structure under experiments/YYYY-MM/DD and indexed names
python main.py --mode predict --name eval \
  --checkpoint experiments/2025-09/09/1_my_set/models/best_model.pt
```

## Configuration

All settings are in `config.yaml`. Key sections:

```yaml
experiment:
  # Name is set via CLI only; this is just metadata
  description: "optional description"

model:
  architecture: "flat_unet_3shot"  # Currently only option
  params:
    model_ch: 128
    num_blocks: 6

data:
  generation:
    tasks: ["bb43febb"]  # Available: bb43febb, c9f8e694, cbded52d, d4a91cb9, d6ad076f
    n_train: 400
    n_test: 100
  episodes:
    grid_size: 10
    train_per_task: 5000

training:
  epochs: 200
  learning_rate: 2e-4
  plot_losses: true     # Real-time loss plotting
```

## Output Organization

```
experiments/
└── YYYY-MM/
    └── DD/                      # Day folder (01–31)
        └── N_my_set/            # Auto-indexed run name (e.g., 1_my_set, 2_my_set)
            ├── config.yaml      # Config used for this run
            ├── models/          # Checkpoints
            │   ├── best_model.pt
            │   └── epoch_XXX.pt
            ├── logs/            # Training logs & plots
            │   └── loss_plot.png
            ├── results/         # Predictions & evaluation
            │   └── predictions.json
            └── data/
                ├── synthetic_tasks/   # Generated task JSON
                └── episodes/          # Episode shards
```

## Architecture

### Modular Components
- **Models**: `arc_diffusion/models/` - UNet architectures and diffusion methods
- **Data**: `arc_diffusion/data/` - Task generators, episode creation, datasets
- **Training**: `arc_diffusion/training/` - Training loop with monitoring
- **Utils**: `arc_diffusion/utils/` - Visualization and I/O utilities

### Current Implementations
- **UNet**: FlatUNet3Shot (no spatial downsampling, good for small ARC grids)
- **Diffusion**: DDPM with linear noise schedule
- **Tasks**: 5 synthetic ARC tasks with uniqueness enforcement

### Future Extensibility
The modular design allows easy addition of:
- New UNet architectures (hierarchical, attention-based)
- New diffusion methods (DDIM, adaptive sampling)
- New ARC tasks and generators
- Alternative training strategies

## Data Pipeline

1. **Task Generation**: Creates unique input-output pairs for each task type
2. **Episode Creation**: Builds 3-shot episodes (3 context pairs + 1 query)
3. **Training Data**: Sharded PyTorch tensors for efficient training
4. **Smart Caching**: Reuses existing data when parameters haven't changed

## Training Features

- **Mixed Precision**: Automatic mixed precision for faster training
- **Real-time Monitoring**: Loss plots updated during training
- **Smart Checkpointing**: Saves best model + periodic checkpoints
- **Validation**: Pixel and problem-level accuracy metrics
- **Resumable**: Can resume from checkpoints (future feature)

## Migration from Old Structure

The new pipeline produces identical results to the old scripts:
- `scripts/run_generator.sh` → `python main.py --mode generate`
- `scripts/run_make_episodes.sh` + `scripts/run_ddpm.sh train` → `python main.py --mode train`
- `scripts/run_ddpm.sh predict` → `python main.py --mode predict --checkpoint <path>`

## Examples

### Single Task Quick Test
```yaml
experiment:
  name: "quick_test"
data:
  generation:
    tasks: ["bb43febb"]
    n_train: 50
    n_test: 10
  episodes:
    train_per_task: 500
    test_per_task: 50
training:
  epochs: 50
```

### Full Multi-Task Training
```yaml
experiment:
  name: "full_training"
data:
  generation:
    tasks: ["bb43febb", "c9f8e694", "cbded52d", "d4a91cb9", "d6ad076f"]
    n_train: 400
    n_test: 100
  episodes:
    train_per_task: 5000
    test_per_task: 500
training:
  epochs: 500
```

## Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` in config.yaml

**Slow Data Loading**: Adjust `num_workers` and `prefetch_factor` in config.yaml

**Loss Not Decreasing**: Try different `learning_rate` or increase `model_ch`
