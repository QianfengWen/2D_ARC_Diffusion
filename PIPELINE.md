# ARC Diffusion Pipeline — End-to-End Rundown

This document explains the complete pipeline implemented in this repository, in execution order, with detailed notes on where the step is a design choice (with alternatives) versus a natural step that follows directly from the goal. File references are provided for quick navigation.

- Entry point: `main.py`
- Orchestration: `arc_diffusion/pipeline.py`
- Configuration: `arc_diffusion/config.py`, `config.yaml`
- Data: `arc_diffusion/utils/io.py`, `arc_diffusion/data/*`
- Models: `arc_diffusion/models/*`
- Training & Eval: `arc_diffusion/training/*`
- Visualization: `arc_diffusion/utils/visualization.py`


## 0) CLI & Experiment Naming

- What happens
  - Parse CLI: `--config`, `--mode [pipeline|train|generate|predict]`, `--checkpoint`, `--name`.
  - Load YAML config, then apply experiment name from CLI and resolve paths with date-structured folders.
  - Auto-prefix `--name` with the next index under today’s date: `experiments/YYYY-MM/DD/N_name/`.
  - Instantiate the pipeline with the resolved `Config`.
  
- Where in code
  - `main.py`
  - Config load/resolve: `arc_diffusion/config.py`

- Classification
  - Natural step: CLI parsing, config loading, path resolution, and instantiating the pipeline.
  - Design choice: The date-structured experiment layout and auto-indexing pattern. Alternatives include flat run directories or hash-based run ids.

- Alternatives / Notes
  - Different run-ID schemes (timestamp, git SHA, full config hash) or external experiment trackers (W&B, MLflow) can replace or augment the directory layout.


## 1) Directory Setup & Device Selection

- What happens
  - Create experiment directories: data root, output root, models, logs, results.
  - Detect device: `cuda` → `mps` → `cpu` and enable `cudnn.benchmark` on CUDA.

- Where in code
  - Directories: `arc_diffusion/utils/io.py` → `setup_directories`
  - Device: `arc_diffusion/pipeline.py` → `_setup_device`

- Classification
  - Natural step: Ensuring required directories exist; picking an available device.
  - Design choice: Device auto-detection order and enabling `cudnn.benchmark`. Some workloads prefer deterministic cudnn settings or explicit device selection.

- Alternatives / Notes
  - Add seed control for determinism. Optionally support multi-GPU (DDP) or automatic mixed precision policies per device type.


## 2) Dataset Management (Central Reuse + Experiment Copy)

- What happens
  - Compute a dataset ID from config parameters that affect data: tasks, counts, grid size, context policy, shard size, seed, etc.
  - Check central storage at `datasets/<dataset_id>/` for existing synthetic tasks and episode shards.
  - If not found, generate synthetic tasks and episode shards once in central storage; always copy the dataset into the experiment’s `data/` directory for immutability of runs.

- Where in code
  - Dataset ID & orchestration: `arc_diffusion/utils/io.py` → `generate_dataset_id`, `ensure_dataset_exists`, `find_existing_dataset`, `create_central_dataset`, `copy_dataset_to_experiment`

- Classification
  - Natural step: Reusing existing identical datasets and copying them into the run directory.
  - Design choice: The exact dataset-ID schema and the decision to keep a central cache vs. always regenerating. The copy step is a design for reproducibility at run-time.

- Alternatives / Notes
  - Store datasets remotely (S3/GCS) and lazy-download. Use hashes of task JSON and config for ID stability. Support read-only central datasets without per-run copies.


## 3) Synthetic Task Generation (if needed)

- What happens
  - For each configured task code or friendly name, generate unique input/output pairs (ARC-style JSON) with duplicate avoidance.
  - Enforce uniqueness across train+test and split disjointly.

- Where in code
  - Task registry and generators: `arc_diffusion/data/generators.py`
  - Uniqueness logic: `generate_unique_pairs`, `make_unique_train_test`
  - Writing files: `save_task_json`
  - Orchestration in central dir: `arc_diffusion/utils/io.py` → `generate_synthetic_tasks_in_dir`

- Classification
  - Design choice: The built-in synthetic tasks and uniqueness policy. Many other synthetic families are possible; uniqueness could be relaxed or replaced with other balancing constraints.
  - Natural step: Persisting generated tasks into JSON files.

- Alternatives / Notes
  - Use real ARC-AGI tasks or additional synthetic families. Swap uniqueness with coverage-driven sampling, curriculum schedules, or stratified sampling by pattern.


## 4) Episode Sharding (3-shot episodes)

- What happens
  - Build 3-shot episodes from task JSONs by picking a triplet of context input/output pairs and a query pair, then pack tensors:
    - `ctx_in` and `ctx_out`: `(K=3, 10, S, S)`
    - `q_in`: `(10, S, S)`
    - `q_out_oh`, `q_out_idx`: labels as one-hot and index grids
    - `ctx_tid` `(3,)` and `q_tid`: integer task IDs (for metrics and visualization)
  - Write to shard `.pt` files for both train and test with a `meta.json` that records shard files, grid size, and task names/slugs.
  - Deduplicate episodes by hashing the complete episode content (context-set + query in/out).

- Where in code
  - Episode creation: `arc_diffusion/data/episodes.py` → `create_episodes_from_tasks`, `make_episodes_for_task`, `flush_shard`
  - Context selection: `choose_context_indices`

- Classification
  - Design choice: 3-shot format, context selection policy (`first3`, `random`, `sliding`), padding size `S`, and shard size. All can be varied.
  - Natural step: Persisting episodes to efficient tensor shards and writing `meta.json` for loaders.

- Alternatives / Notes
  - Single-shot or K-shot with K≠3. Different padding strategies (no pad, border pad, top-left align). Hard caps per-task; weighted sampling. Include query from train for test (current design uses train contexts + test query).


## 5) Data Loading (Training/Validation/Test)

- What happens
  - Load shards from `episodes/meta.json` and map global indices to shard/local indices with a small LRU-like cache for a single shard.
  - Validation and prediction can request task IDs in the dataset outputs to compute per-task metrics and plots.
  - Construct `torch.utils.data.DataLoader` with batch, workers, pinning, and prefetch options from config.

- Where in code
  - Dataset: `arc_diffusion/data/dataset.py` → `EpisodesPTDataset`
  - Loader creation: `arc_diffusion/pipeline.py` → `_create_data_loaders`

- Classification
  - Natural step: Mapping from shards to a random-access dataset and building data loaders.
  - Design choice: Loader params such as `batch_size`, `num_workers`, `prefetch_factor`, and whether to include `return_tid`.

- Alternatives / Notes
  - Multi-epoch prefetching, caching more than one shard, or memory-mapping. Implement weighted sampling over tasks or curriculum sampling.


## 6) Model Construction

- What happens
  - Select the model class from a registry by `config.model.architecture`, then instantiate and move to device.
  - Current implementation: `FlatUNet3Shot` — a flat UNet without spatial down/up-sampling, with time conditioning and a pair-encoder for context aggregation fed into the time stream.

- Where in code
  - Registry: `arc_diffusion/models/__init__.py` → `MODEL_REGISTRY`
  - UNet: `arc_diffusion/models/unet.py` → `FlatUNet3Shot`, `PairEncoder`, `ResidualBlock`, `timestep_embedding`
  - Pipeline: `arc_diffusion/pipeline.py` → `_create_model`

- Classification
  - Design choice: Architecture selection and hyperparameters (`model_ch`, `num_blocks`, `t_dim`, `ctx_dim`). Many viable alternatives.
  - Natural step: Moving the model to the chosen device; optional cudnn flags.

- Alternatives / Notes
  - Hierarchical UNet (downsample/upsample), attention blocks, transformer hybrids, classifier-free guidance, or conditioning via cross-attention to contexts.


## 7) Diffusion Process

- What happens
  - Select diffusion method from registry and build with `DiffusionCfg` (timesteps and β schedule range). Current method: DDPM with linear β schedule.
  - Training loss: predict noise ε with MSE against the noise injected into the clean `q_out_oh`.
  - Sampling: reverse-time loop from `T-1 → 0` using the DDPM update.

- Where in code
  - Registry: `arc_diffusion/models/__init__.py` → `DIFFUSION_REGISTRY`, `DiffusionCfg`
  - DDPM: `arc_diffusion/models/diffusion.py` → `GaussianDiffusion`
  - Pipeline: `arc_diffusion/pipeline.py` → `_create_model` (diffusion instantiation)

- Classification
  - Design choice: Diffusion family (DDPM vs. DDIM vs. discrete diffusion), noise schedule (`linear` vs. `cosine`), number of timesteps.
  - Natural step: Integrating the diffusion loss and sampler once a method is chosen.

- Alternatives / Notes
  - DDIM for faster sampling, cosine or sigmoid schedules, learnable schedules, or true discrete diffusion (categorical) instead of continuous one-hot diffusion.


## 8) Optimizer, Precision, and Optional Compile

- What happens
  - Build optimizer (`adamw` or `adam`) with configured learning rate.
  - Enable AMP scaler when `mixed_precision` and supported by device.
  - Optional: `torch.compile` (config flag is present; integration point for future optimization).

- Where in code
  - Optimizer/scaler: `arc_diffusion/pipeline.py` → `_create_optimizer`
  - Config: `arc_diffusion/config.py` → `TrainingConfig`, `HardwareConfig`

- Classification
  - Design choice: Optimizer, LR, weight decay (in AdamW defaults), and AMP usage. Compiling the model is also a design choice.
  - Natural step: Wiring the optimizer/scaler into the training loop once chosen.

- Alternatives / Notes
  - Add schedulers (cosine, step, plateau) beyond the placeholder. SAM, Lion optimizers, or per-parameter configs.


## 9) Training Loop with Validation, Plots, and Checkpoints

- What happens
  - Train epochs with AMP autocast; compute diffusion MSE loss; step optimizer with scaler.
  - Periodically validate: sample predictions, compute pixel and problem accuracy, and compute validation loss with the same objective.
  - Plot per-epoch train & val loss (`logs/loss_plot.png`) and validation accuracies (`logs/val_accuracy.png`); optionally per-task accuracy plots when multiple tasks.
  - Save best checkpoint by problem accuracy and periodic epoch checkpoints.
  - Optional validation visualizations: collages of contexts + query + predicted grid, grouped per task and merged into a single grid.

- Where in code
  - Trainer: `arc_diffusion/training/trainer.py` → `DiffusionTrainer`
  - Metrics: `arc_diffusion/training/metrics.py` → `evaluate`
  - Visualization helpers: `arc_diffusion/utils/visualization.py`

- Classification
  - Natural step: Epoch/iteration structure, computing loss, and running backprop.
  - Design choice: Validation frequency and batch cap, metric definitions (pixel vs. problem accuracy), selection of “best” checkpoint, and the exact visualization strategy.

- Alternatives / Notes
  - Early stopping, SWA/EMA, gradient clipping, LR scheduling, richer logging backends, and multi-sample voting at validation.


## 10) End-to-End Pipeline Mode

- What happens
  - Save the resolved config into the run directory.
  - Run training. If best checkpoint exists, run prediction on test episodes and save results.

- Where in code
  - Orchestration: `arc_diffusion/pipeline.py` → `ARCDiffusionPipeline.run`

- Classification
  - Natural step: Persist the exact config used, then train and evaluate.
  - Design choice: Whether to auto-evaluate after training; could make this optional or broaden to multiple evaluation suites.

- Artifacts
  - `models/best_model.pt`, `models/epoch_XXX.pt`
  - `logs/loss_plot.png`, `logs/val_accuracy.png`, and optional per-task accuracy plots
  - `results/predictions.json` and per-task prediction collages under `results/test_vis/`


## 11) Prediction (Standalone Mode)

- What happens
  - Load checkpoint, create model/diffusion, and iterate test episodes.
  - For each episode, sample a prediction and record JSON with query input, predicted output, and ground-truth.
  - Save per-task collages and an optional merged grid of all tasks.

- Where in code
  - `arc_diffusion/pipeline.py` → `ARCDiffusionPipeline.predict`

- Classification
  - Natural step: Restore model and generate predictions for test episodes.
  - Design choice: Output format (JSON schema), visualization layout, and sampling budget (single vs. multiple samples/votes).

- Alternatives / Notes
  - Multi-sample voting or multiple context triplets at inference; additional output formats; side-by-side GT vs. pred collages.


## Configuration Surface (by stage)

- Experiment & paths
  - `experiment.description` (metadata only)
  - `paths.*` templates with dynamic `{time.*}` and `{experiment.name}` substitution

- Data
  - Generation: `tasks` (codes or friendly names), `n_train`, `n_test`, `seed`, `attempts_per_example`
  - Episodes: `grid_size`, `ctx_policy`, `train_per_task`, `test_per_task`, `shard_size`
  - Loading: `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`

- Model & diffusion
  - Model: `architecture` (registry key), `params` (`model_ch`, `num_blocks`, `t_dim`, `ctx_dim`)
  - Diffusion: `method` (registry key), `params` (`timesteps`, `beta_start`, `beta_end`, `schedule`)

- Training & hardware
  - `epochs`, `learning_rate`, `optimizer`, `scheduler` (placeholder), `val_frequency`, `val_batches`, `save_frequency`, `save_top_k`, `log_frequency`, `plot_losses`, `save_samples`
  - Hardware: `device`, `mixed_precision`, `compile_model`

- Visualization
  - Validation/test: `val_samples_per_task`, `test_samples_per_task`, `group_by_task`, `max_vis_scan`, `collage_cols`, `include_query`, `save_dpi`


## Clear “Design Choice” vs “Natural Step” Checklist

- Design choices (swap points)
  - Model architecture and hyperparameters
  - Diffusion method, schedule, and timesteps
  - Synthetic task families; uniqueness and split policy
  - Episode format (3-shot), context selection policy, grid padding size, shard sizes
  - Optimizer, LR, precision policy, potential compile
  - Validation cadence, metrics, “best” model criterion
  - Visualization strategy for validation and prediction
  - Dataset ID schema and central-cache policy

- Natural steps (expected given the task)
  - CLI parsing, config loading, path resolution
  - Creating required directories and moving models to devices
  - Loading episodes, building data loaders
  - Training loop structure (forward, loss, backward, step)
  - Saving checkpoints and serializing predictions/plots


## Repository Flow (Call Graph Summary)

1) CLI → config
   - `main.py` → `load_config` → set `experiment.name` → resolve `paths`
2) Pipeline init
   - `ARCDiffusionPipeline(config)` → `_setup_device` → `setup_directories`
3) Ensure dataset
   - `_ensure_episodes` → `ensure_dataset_exists` → (find or create central) → copy to run dir
4) Train
   - `_create_model` (model + diffusion) → `_create_data_loaders` → `_create_optimizer` → `DiffusionTrainer.train`
   - Inside train: `train_one_epoch` (AMP) → periodic `evaluate` + `_visualize_validation_samples` → `save_checkpoint`
5) Predict (pipeline or standalone)
   - `predict` → load `best_model.pt` → iterate test `EpisodesPTDataset` → `diffusion.sample` → save JSON + collages


## Notes on Extensibility

- Add models: register in `MODEL_REGISTRY` and implement forward compatible with current shapes.
- Add diffusion methods: register in `DIFFUSION_REGISTRY` with `DiffusionCfg`-compatible constructor and `compute_loss`/`sample` contract.
- Add tasks: extend `arc_diffusion/data/generators.py` and update `arc_diffusion/data/task_names.py` for friendly names.
- Adjust episodes: tweak context policy, K, padding scheme, or dedup keys.
- Improve evaluation: richer metrics, test-time augmentation or voting across contexts.


## Outputs & Layout

- Root: `experiments/YYYY-MM/DD/N_name/`
  - `config.yaml` (resolved config used)
  - `models/` → `best_model.pt`, `epoch_XXX.pt`
  - `logs/` → `loss_plot.png`, `val_accuracy.png`, optional per-task plots, `val_vis/`
  - `results/` → `predictions.json`, `test_vis/` collages (per-task and merged)
  - `data/` → `synthetic_tasks/` (JSONs), `episodes/` (shards + `meta.json`)


---

This rundown mirrors the current codebase. The items listed as “Design choices” are modular by intent and can be swapped or extended with minimal surface changes elsewhere in the pipeline.

