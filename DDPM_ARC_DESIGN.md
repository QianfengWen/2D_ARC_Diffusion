DDPM for ARC (3-shot) – Design and Training Details

Overview
The goal is to learn grid-to-grid transformations on ARC tasks, where each episode provides 3 context demonstrations (input→output) and a query input. The model predicts the query output by sampling from a conditional diffusion model. We support two pipelines:
1) On-the-fly episodes (contexts re-sampled every step)
2) Offline episodic shards (precomputed tensors for speed)

Data Representation
- Colors: integers 0..9 (C=10)
- Grids: variable sizes H×W per example
- Padding: center-pad each grid to S×S (grid_size) with zeros so batches are uniform
- One-hot: each grid becomes (C, S, S); the output label also stored as indices (S, S)

Episodes (3+1)
An episode e consists of:
- Context triplet: [(c_in1, c_out1), (c_in2, c_out2), (c_in3, c_out3)]
- Query: (q_in, q_out)
For training, we add noise only to the query target (q_out) and predict that noise given q_in and the contexts.

Context Sampling Policies
- first3: use the first three train pairs of a task (classic ARC format)
- random: sample three distinct train pairs uniformly at random
- sliding: choose a random i, take (i, i+1, i+2) as contexts (if possible)
Validation and prediction typically use first3 or randomized voting.

Model Architecture

Core: Flat UNet (no spatial down/up sampling)
- Input to the network at each diffusion step: concat of
  - x_t: noisy query target, shape (B, 10, S, S)
  - q_in: query input (one-hot), shape (B, 10, S, S)
  => concatenated to (B, 20, S, S)
- Residual blocks: N repeated residual blocks with GroupNorm and SiLU
- Time conditioning: sinusoidal embedding t → MLP → added as FiLM-like bias per block
- Context conditioning: encode each (c_in|c_out) pair into a vector via a small CNN (PairEncoder), average the three vectors, then project and add into the time embedding stream
- Output: predict epsilon (noise) with shape (B, 10, S, S)

Why this design
- Flat UNet keeps spatial size fixed and matches ARC’s small grids without pooling artifacts
- FiLM-like conditioning lets time and context influence all blocks
- Concatenating q_in with x_t provides direct alignment between input and target pixels

Diffusion Process

Notation
- T: number of diffusion steps (e.g., 400)
- β_t: noise schedule (linear from β_start to β_end)
- α_t = 1 − β_t
- ᾱ_t = ∏_{s=1..t} α_s

Forward (q_sample)
We corrupt the one-hot target x0 (continuous float in [0,1]) to x_t via:
x_t = sqrt(ᾱ_t) · x0 + sqrt(1 − ᾱ_t) · ε,  where ε ~ N(0, I)

Reverse (p_sample)
The model predicts ε̂(x_t, q_in, t, context) and uses the DDPM update:
μ_t = 1/√α_t · (x_t − β_t/√(1 − ᾱ_t) · ε̂)
If t > 0, sample x_{t−1} ~ N(μ_t, σ_t^2 I), else return μ_t

Sampling (predict)
- Initialize x_T ~ N(0, I)
- For t=T..1: apply p_sample; stop at x_0
- Map x_0 to color indices via argmax over channels

Training Objective
- Epsilon prediction MSE: L = E_{t,ε} [ || ε̂ − ε ||^2 ]
- Only the query target is noised; contexts and query input remain clean and provide conditioning

Validation Metrics
- Pixel accuracy: fraction of matching pixels between argmax(x_0) and q_out
- Problem (example) accuracy: 1 if all pixels match for the example, else 0
We report both; best checkpoint can track problem accuracy.

Prediction Modes
1) ARC pair JSONs: predict output for each input in a task (uses first3 or chosen contexts) and write ARC-style JSON (pairs)
2) Offline episodes: for each episodic test sample (ctx triplet + query), sample and write a compact JSON with query_input, pred_output, query_gt
Optionally, do majority voting across multiple random context triplets (n_ctx_sets > 1).

Pipelines

On-the-fly episodes
- Build episodes per batch from task JSONs; contexts re-sampled each iteration
- Pros: more diversity, less storage
- Cons: slower per-step (JSON → pad → one-hot every batch)

Offline episodic shards
- Precompute episodes to .pt shards (ctx_in, ctx_out, q_in, q_out_oh, q_out_idx)
- Pros: fast IO and training; reproducible; easy to shard across jobs
- Cons: fixed sample set; storage cost

Why padding to S×S
- ARC grids vary (e.g., 8×10). Padding to a fixed S makes batch shapes consistent
- Center padding preserves spatial alignment and symmetry
Choose S ≥ max(H, W) expected in your data (10 or 16 are typical).

Three-shot conditioning vs. single-pair mapping
- Single-pair mapping learns a global rule per family/task offline and applies it
- 3-shot conditioning uses three demonstrations at inference, guiding the model per episode/puzzle
- Our context encoder aggregates the three pairs, exposing higher-level structure

Implementation Pointers (files)
- scripts/ddpm_arc.py: training and prediction CLIs for 3-shot; offline variant loads shards
- scripts/make_arc_episodes_offline.py: builds episodic shards with dedup and sharding
- scripts/run_make_episodes.sh: wrapper to generate shards
- scripts/run_ddpm.sh: train/predict wrappers (offline)
- scripts/arc_visualizer.py: draw pairs, episodes, shards, and offline predictions

Tuning Suggestions
- Increase model_ch or num_blocks for capacity; add attention if needed
- Try cosine β schedule; classifier-free guidance for stronger conditioning
- Use multiple context triplets at predict-time and majority vote
- Balance pixel and example accuracy in model selection depending on goals

Known Limitations
- Context aggregation by mean may lose ordering information
- One-hot continuous diffusion approximates discrete diffusion; true discrete methods may improve results
- No cropping back to original sizes at save-time in pair predictor unless explicitly done

Appendix: Shapes
- ctx_in, ctx_out: (B, 3, 10, S, S)
- q_in, x_t:       (B, 10, S, S)
- model input:     concat(x_t, q_in) → (B, 20, S, S)
- model output:    ε̂: (B, 10, S, S)
- labels:          q_out_oh (B, 10, S, S), q_out_idx (B, S, S)


