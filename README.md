2D_ARC_Diffusion – Quickstart

This project provides synthetic ARC data generation, a 3-shot DDPM baseline, episodic dataset creation, and visualization tools.

Prereqs
- Python 3.10+ with a venv at ./.venv (or set $PYTHON)
- pip install: torch, torchvision, tqdm, matplotlib

1) Generate ARC synth datasets
Run all tasks with uniqueness and disjoint train/test splits:
```bash
./scripts/run_generator.sh
```
Defaults: n_train=400, n_test=100 per task → arc_synth_400/synth_*.json

2) Build offline 3+1 episodic shards
Materialize episodes for faster training:
```bash
./scripts/run_make_episodes.sh \
  GLOB='arc_synth_400/synth_*.json' GRID_SIZE=16 CTX_POLICY=random \
  TRAIN_PER_TASK=5000 TEST_PER_TASK=500 SHARD_SIZE=5000 OUT_DIR=episodes_3shot_g16
```
Tip: quote the glob so the shell doesn’t expand it.

3) Train the 3-shot DDPM (offline)
```bash
./scripts/run_ddpm.sh train EPISODES_DIR=episodes_3shot_g16 \
  BATCH_SIZE=512 EPOCHS=200 TIMESTEPS=400
```
Checkpoints → checkpoints_3shot_offline/

4) Predict on offline episodes
```bash
./scripts/run_ddpm.sh predict \
  CKPT=checkpoints_3shot_offline/ddpm_3shot_best.pt \
  EPISODES_DIR=episodes_3shot_g16 \
  OUT_JSON=predictions_episodes/preds.json
```
Creates a compact JSON with test_episodes_outputs.

5) Visualize
- Dataset JSONs (pairs):
```bash
./scripts/run_visualizer.sh save arc_synth_400/synth_bb43febb.json --out viz_out/bb43febb
```
- Offline predictions (input|pred|gt):
```bash
./scripts/arc_visualizer.py save_offline predictions_episodes/preds.json --out viz_out/preds_offline
```
- Episodic shard previews (3 contexts + query GT):
```bash
./scripts/run_visualizer.sh save_shard episodes_3shot_g16/train_0000.pt --out viz_out/episodes_train_0000 --max 32
```

Notes
- Grid size: choose ≥ max(H,W) of your tasks (common: 10 or 16).
- For large runs, increase TRAIN_PER_TASK/TEST_PER_TASK and SHARD_SIZE.
- All runners pass extra flags through to the underlying Python CLIs.

Design Doc
- See DDPM_ARC_DESIGN.md for a detailed description of the 3-shot DDPM architecture, diffusion process, and training pipeline.


