# Replay Fine Tune Spec

## Purpose

This document specifies a replay-based fine-tuning pass of the BLT entropy model. It extends the basic fine-tune (see [basic_fine_tune.md](basic_fine_tune.md)) by interleaving batches of Voynich text with batches of text from the model's pre-training distribution. The goal is to adapt the model to Voynich byte patterns **without catastrophically forgetting** the general byte-level knowledge it acquired during pre-training.

**Why replay at all:** The basic fine-tune deliberately overfits the entropy model on ~15 chunks of Voynich text. With full-parameter updates on such a small, highly-distributional-shifted dataset, the model's learned representations for ordinary English byte patterns are overwritten. The resulting entropy signal becomes "distance from Voynich" rather than a calibrated probability distribution — the model assigns near-uniform high entropy to any byte that isn't Voynich PUA. By mixing in batches of pre-training-like text (DCLM) during fine-tuning, the model is pulled back toward its original distribution on each replay step, keeping its general byte knowledge intact while still adapting to Voynich.

**Why not a different regularization technique:** Alternatives like LoRA (low-rank adapters), elastic weight consolidation (EWC), or layer freezing each have tradeoffs. Replay is the simplest and most transparent: no adapter architecture, no Fisher information estimation, no decisions about which layers to freeze. It directly optimizes the quantity we care about (loss on the original distribution remains low) by continuing to train on that distribution. Replay also composes cleanly with those techniques if they are added later.

## Scope

This spec covers a single training variant: the basic fine-tune's training loop, extended with interleaved replay batches drawn from the DCLM streaming sampler. Everything not explicitly changed here inherits from [basic_fine_tune.md](basic_fine_tune.md) — same target model, same loss function, same checkpoint strategy, same ClearML tracking, same folio-level validation split, same Lightning framework.

**What is in scope:**
- A replay data source (DCLM via `hf_data_samp`).
- A batch-mixing strategy controlled by a replay-ratio hyperparameter.
- Evaluation of both Voynich-specific and pre-training-distribution performance.

**What is out of scope:**
- LoRA / adapter-based fine-tuning.
- Elastic weight consolidation or other explicit regularization methods.
- Multi-source replay (mixing DCLM with other datasets). DCLM alone covers the pre-training distribution.
- Curriculum schedules for the replay ratio. The ratio is held constant across training; schedules are a follow-up.

## Pipeline Context

The replay fine-tune sits at the same pipeline position as the basic fine-tune, but pulls from two data sources:

```
voynpy.corpora.vms_unicode (DataFrame)       hf_data_samp.sample(DCLM, ...)
        ↓                                              ↓
vms_uprep.prepare()                          list[str] (DCLM documents)
        ↓                                              ↓
vms_uprep.stack_lines(max_bytes=8192)        vms_uprep.stack_lines(max_bytes=8192)
        ↓                                              ↓
UTF-8 encoding                               UTF-8 encoding
        ↓                                              ↓
Token IDs: [b + 4 for b in raw_bytes]        Token IDs: [b + 4 for b in raw_bytes]
        ↓                                              ↓
┌───────────────────────────────────────────────────────────────┐
│  THIS SPEC: Replay fine-tune entropy model                    │
│                                                                │
│  Interleaved batch stream:                                     │
│    Voynich batches  ←→  DCLM replay batches                    │
│  Ratio: controllable hyperparameter                            │
│                                                                │
│  Loss: cross-entropy on next-byte (same for both sources)      │
│  Checkpoints → data/experiments/                               │
│  Tracking → ClearML (separate metrics per source)              │
└───────────────────────────────────────────────────────────────┘
        ↓
Fine-tuned model checkpoint
```

**Why reuse `hf_data_samp` for replay:** The `hf_data_samp` module (see [hf_data_samp.md](hf_data_samp.md)) already provides streaming DCLM access with UTF-8-safe truncation at 8192 bytes — exactly the format the entropy model expects. Writing a separate DCLM loader would duplicate that logic. `stack_lines()` then handles chunk packing identically for both sources, which keeps the two data paths symmetric and interchangeable at the batch level.

## Replay Data Source

### Source Dataset

**DCLM Baseline 1.0** (`mlfoundations/dclm-baseline-1.0`), accessed via `hf_data_samp.DCLM`.

**Why DCLM:** This is the dataset the BLT entropy model was pre-trained on. Replaying samples from the exact pre-training distribution is the most direct way to preserve pre-training behavior — the model's gradients on DCLM are the same gradients (in expectation) that shaped it originally, so including them in fine-tuning acts as a direct anchor to the pre-trained state.

**Why not a smaller or different corpus:** Alternatives like Wikipedia or a subset of Common Crawl would work but introduce a distribution mismatch with pre-training. The goal of replay is to preserve pre-training knowledge; using pre-training data is the most faithful way to do that.

### Sampling Strategy

Replay samples are drawn via `hf_data_samp.sample(DCLM, n=..., seed=..., max_bytes=8192)`.

**Pre-fetch vs. stream-during-training:** Replay samples are pre-fetched once at the start of training into an in-memory pool, not streamed on-demand during each epoch. A pool of **1000 documents** (configurable) is fetched with a fixed `seed` at run start.

**Why pre-fetch:**
- HF streaming over the network adds variable per-step latency that would dominate training step time.
- A fixed pool makes replay losses directly comparable across epochs and across runs with the same seed.
- 1000 documents × up to 8192 bytes ≈ 8 MB in memory — negligible.
- The pool is large enough that the model sees many distinct DCLM samples over 100 epochs without cycling tightly.

**Why not use the full dataset:** DCLM is ~3.8 TB. Streaming the entire corpus defeats the purpose of a small, controlled fine-tune. A 1000-document pool captures enough distributional breadth to preserve pre-training behavior while keeping runs fast and reproducible.

### Replay Dataset (PyTorch)

A `DCLMReplayDataset` class, structurally symmetric to `VoynichEntropyDataset`:

1. At initialization: call `hf_data_samp.sample(DCLM, n=pool_size, seed=replay_seed, max_bytes=8192)`, then pass the returned list through `vms_uprep.stack_lines(max_bytes=8192)` to produce 8192-byte chunks.
2. `__len__`: number of replay chunks.
3. `__getitem__`: returns a tensor of BLT token IDs for the chunk, using the same `byte + 4` offset as the Voynich dataset.

**Why feed DCLM through `stack_lines()`:** `stack_lines()` packs short text segments into 8192-byte windows and emits structural markers between them. Applying the same packer to both sources means every batch — Voynich or replay — has the same shape and structural marker vocabulary. This keeps gradients on structural markers consistent across sources, so the model's handling of those tokens doesn't drift toward either distribution.

**Why the same token-ID offset:** Replay batches must use the same byte-to-token encoding as Voynich batches. The entropy model has one 260-token vocabulary; using any different encoding for replay would make the replay loss uninterpretable.

## Batch Mixing

### Replay Ratio Hyperparameter

The central hyperparameter of this spec is **`replay_ratio`** — the ratio of replay batches to Voynich batches presented to the model during training.

**Definition:** `replay_ratio = R` means that for every 1 Voynich batch, the model sees R replay batches. Expressed as a fraction of total batches, replay occupies `R / (1 + R)`.

| `replay_ratio` | Voynich fraction | Replay fraction | Interpretation |
|----------------|-----------------:|----------------:|----------------|
| 0.0            | 100%             | 0%              | Equivalent to basic fine-tune (no replay) |
| 0.5            | ~67%             | ~33%            | Mild replay |
| 1.0            | 50%              | 50%             | Balanced |
| 2.0            | ~33%             | ~67%            | Replay-dominant |
| 4.0            | 20%              | 80%             | Heavy replay |

The ratio is a plain float; non-integer values are supported via the stochastic scheduling strategy below.

**Why a continuous ratio rather than discrete modes:** The right balance between Voynich adaptation and pre-training preservation is an empirical question. Exposing `replay_ratio` as a continuous hyperparameter lets us sweep it (e.g., 0.0, 0.25, 0.5, 1.0, 2.0, 4.0) and identify the regime where validation perplexity drops on Voynich without blowing up on DCLM. A coarse discrete enum would force arbitrary cutoffs.

**Why frame it as batches-per-batch rather than as a probability:** Expressing the ratio as "R replay per 1 Voynich" makes integer values interpretable as deterministic schedules (R=2 → every third batch is Voynich) and makes the hyperparameter natural to reason about when planning epochs. A probability (`p_replay`) is equivalent information (`p_replay = R / (1+R)`) but less intuitive at the extremes.

### Batch Scheduling Strategy

Batches are drawn from a combined stream. For a given `replay_ratio = R`, the scheduler produces a sequence of batch-source decisions (each decision: Voynich or replay) with the following properties:

1. **Long-run ratio:** Over a full epoch, the fraction of replay batches converges to `R / (1 + R)`.
2. **Per-batch stochasticity:** For non-integer R, the source of each individual batch is sampled according to `p_replay = R / (1 + R)` using a dedicated `torch.Generator` seeded from the run config.
3. **Integer-R determinism:** When R is an integer, the schedule is deterministic and repeating: one Voynich batch followed by R replay batches, then repeat. For R=0, every batch is Voynich (equivalent to the basic fine-tune). For very large R, every batch is replay with Voynich inserted every R+1 batches.
4. **Independence across runs:** The scheduler seed is separate from the data-split seed and the replay-pool seed, so a given mixing schedule can be reproduced independently of the data it's mixed over.

**Why per-batch stochastic mixing rather than alternating blocks:** Alternating large blocks (e.g., "one full epoch of Voynich, then one full epoch of DCLM") causes the optimizer state (AdamW momentum, learning-rate schedule position) to see long runs of correlated gradients, which can destabilize training. Fine-grained interleaving keeps gradient variance similar across batches and lets the LR scheduler track a single notion of "step" without knowing about sources.

**Why deterministic for integer R:** Integer ratios are the most common experimental configurations (0, 1, 2, 4). Making them deterministic means runs with the same seed produce bit-identical schedules, which simplifies debugging and comparison.

### Epoch Definition

An **epoch** is defined as one pass over all Voynich training chunks. Replay batches are drawn as needed from the replay dataset to satisfy the ratio, cycling through the replay chunks with reshuffling at each cycle.

**Why anchor the epoch to Voynich:** The quantity of interest is how many times the model has seen each Voynich chunk. Anchoring to Voynich keeps the "epoch" count comparable to the basic fine-tune run, which matters for reusing hyperparameters like `epochs=100` and for plotting loss curves on the same x-axis.

**Why reshuffle replay rather than just cycling:** With a pool of ~1000 replay chunks and potentially thousands of replay batches over a 100-epoch run, cycling without reshuffle would repeat the same order each time. Reshuffling at each cycle (seeded deterministically from the run seed + cycle index) gives the model a different sequence of replay exposures while keeping the total pool fixed.

### Step Counting and LR Schedule

Total optimizer steps per epoch = `num_voynich_chunks + ceil(num_voynich_chunks * R)`.

The cosine LR schedule and warmup from the basic fine-tune are preserved, but warmup steps and total steps are recomputed based on the mixed step count so the schedule spans the actual training trajectory.

**Why recompute schedule lengths:** The basic fine-tune's `warmup_steps=50` was chosen relative to ~1200 total steps (12 chunks × 100 epochs). With replay, total steps scale by `(1 + R)`, so a fixed 50-step warmup becomes a smaller fraction of training at high R. Recomputing warmup as a fixed fraction of total steps (e.g., 4%) keeps the schedule shape consistent across ratios.

## Loss Computation and Logging

### Per-Source Loss Tracking

Both Voynich and replay batches use the same cross-entropy loss function on next-byte prediction — architecturally, the model cannot tell which batch is which. However, the Lightning module tracks and logs losses separately per source.

**Metrics logged to ClearML:**
- `train/loss/voynich` — cross-entropy on Voynich batches (per step)
- `train/loss/replay` — cross-entropy on DCLM replay batches (per step)
- `train/loss/combined` — mean cross-entropy over all batches (per step)
- `train/perplexity/voynich`, `train/perplexity/replay`, `train/perplexity/combined`
- `train/lr` — learning rate (to verify schedule)
- `val/loss/voynich` — validation cross-entropy on held-out Voynich folios (per epoch)
- `val/loss/replay` — validation cross-entropy on a held-out replay pool (per epoch)
- `val/perplexity/voynich`, `val/perplexity/replay`
- `replay_ratio_realized` — actual replay-batch fraction observed this epoch (sanity check that the scheduler is producing the configured ratio)

**Why split by source:** The interesting question is whether Voynich loss drops **while** replay loss stays flat. A single combined loss would obscure the tradeoff this spec exists to expose.

### Held-Out Replay Validation

A separate pool of replay chunks is reserved for validation, drawn with a distinct seed from the training pool:

- Training replay pool: `seed = replay_seed`, size 1000.
- Validation replay pool: `seed = replay_seed + 1`, size 100.

**Why a separate validation pool:** Measuring `val/loss/replay` on a pool the model saw during training would conflate memorization with generalization. A distinct pool gives a clean measurement of how well the model still predicts the pre-training distribution. Size 100 is small enough to evaluate in seconds but large enough to average out per-document variance.

## Hyperparameters

All basic-fine-tune hyperparameters carry over unless overridden here. Replay-specific additions:

| Parameter | Default | Rationale |
|-----------|--------:|-----------|
| `replay_ratio` | 1.0 | Balanced starting point. Sweep from this default. |
| `replay_pool_size` | 1000 | ~8 MB memory; enough distributional breadth for 100 epochs without tight cycling. |
| `replay_val_pool_size` | 100 | Fast validation; still stable enough to detect meaningful drift. |
| `replay_seed` | 42 | Separate from `data_split_seed` so replay shuffling is independent of train/val folio assignment. |
| `replay_schedule_seed` | 43 | Separate from replay-pool seed so the batch-mixing schedule can be re-derived without refetching data. |
| `replay_source` | `"DCLM"` | String key referencing a preset in `hf_data_samp` (currently just `DCLM`). Future specs may add others. |
| `warmup_fraction` | 0.04 | Replaces `warmup_steps=50`. Keeps warmup proportional across `replay_ratio` values. |

Learning rate, optimizer, precision, gradient clipping, epochs, batch size, and the weight-decay value are unchanged from the basic fine-tune.

**Why `replay_ratio=1.0` as default:** A 50/50 mix is the most neutral starting point — neither dominated by Voynich (which would look like the basic fine-tune) nor by replay (which would barely move the model). Sweeps around this default should reveal the shape of the tradeoff.

## Data Pipeline Module

The fine-tuning entry point gains a `MixedDataLoader` (or equivalent) that wraps two underlying DataLoaders — one over `VoynichEntropyDataset`, one over `DCLMReplayDataset` — and yields batches according to the scheduling strategy above.

**Why a mixing wrapper rather than concatenation:** `torch.utils.data.ConcatDataset` with a weighted sampler could approximate this but doesn't give fine control over per-epoch ratio realization or over keeping the source of each batch known at training-step time. An explicit wrapper makes the logic auditable and makes per-source metric logging straightforward — each yielded batch arrives with a source tag.

**Batch shape:** Every batch (from either source) is a single 8192-token sequence right-padded with PAD_ID (2) if shorter. Identical to the basic fine-tune.

**Shuffling:** Voynich training chunks are shuffled each epoch (same as basic fine-tune). Replay chunks are shuffled each time the replay iterator is exhausted and restarted.

## Lightning Module Changes

The `VoynichEntropyFineTune` Lightning module from the basic fine-tune is extended (not replaced) to become `ReplayEntropyFineTune`:

1. **`__init__`:** Same as basic — load pre-trained entropy model, enable all gradients. Accept additional config: `replay_ratio`, `replay_pool_size`, `replay_seed`, `replay_schedule_seed`.
2. **`training_step`:** Receive a batch along with a `source` tag (`"voynich"` or `"replay"`). Compute cross-entropy loss. Log under the appropriate `train/loss/<source>` key in addition to the combined metric.
3. **`validation_step`:** Run two validation passes — one on Voynich val folios, one on the replay validation pool. Log `val/loss/voynich` and `val/loss/replay` separately.
4. **`configure_optimizers`:** AdamW with cosine LR schedule, warmup as `warmup_fraction * total_steps` where `total_steps = epochs * (num_voynich_chunks * (1 + replay_ratio))`.

**Why separate validation passes rather than a mixed val loader:** Validation metrics must be directly comparable across runs with different `replay_ratio`. A mixed val loader's loss would depend on the ratio itself, which is not what we want to measure. Two explicit passes keep each metric clean.

## Experiment Tracking

### Run Naming

Timestamp-based IDs with a `rft` prefix (replay fine tune): `rft-YYYYMMDD-HHMM-r{ratio}`.

Example: `rft-20260420-1430-r1.0` for a balanced run; `rft-20260420-1430-r4.0` for replay-dominant.

**Why include the ratio in the ID:** `replay_ratio` is the primary knob being swept. Putting it in the run ID makes it visible in ClearML's task list and in checkpoint paths without having to open config files.

### Logged Configuration

In addition to everything logged by the basic fine-tune, record:
- `replay_ratio`
- `replay_source`
- `replay_pool_size`
- `replay_seed`
- `replay_schedule_seed`
- Hash or count of replay documents actually fetched (to detect HF dataset version drift)
- `realized_replay_ratio` at end of run (average over all epochs)

## Evaluation

### Primary Metrics

Two perplexities, tracked in parallel:

1. **Voynich val perplexity** — on held-out folios. Measures how much Voynich structure was learned.
2. **Replay val perplexity** — on the held-out DCLM pool. Measures how much pre-training knowledge was preserved.

The central output of a replay run is the **pair** of final perplexities, not either alone. A successful configuration is one where Voynich perplexity drops meaningfully (vs. the pre-trained model) while replay perplexity stays close to the pre-trained baseline.

### Baseline for Comparison

Three baselines to contextualize results:

1. **Pre-trained model** — epoch-0 perplexity on both Voynich val and replay val pools.
2. **Basic fine-tune** — perplexities from the spec in `basic_fine_tune.md`. Sets the upper bound on Voynich-specific adaptation and (expected) lower bound on preserved replay performance.
3. **Replay-only (R = ∞, i.e. training only on DCLM)** — would be a sanity check that replay alone drives replay val loss back down, confirming the replay data path itself is correct. Optional — not required for the primary sweep.

### Expected Behavior

1. **Epoch 0:** Identical perplexities to the basic fine-tune at epoch 0 (same pre-trained model).
2. **Low `replay_ratio` (0.0–0.25):** Behavior approaches the basic fine-tune. Voynich perplexity drops quickly; replay perplexity degrades.
3. **Moderate `replay_ratio` (0.5–1.0):** Voynich perplexity still drops but more slowly. Replay perplexity remains close to baseline.
4. **High `replay_ratio` (2.0–4.0):** Voynich perplexity drops slowly or plateaus above the basic fine-tune's final value. Replay perplexity essentially unchanged from baseline.
5. **`replay_ratio = 0.0`:** Must produce results bit-identical (up to scheduler noise) to the basic fine-tune — this is a regression check on the mixing machinery.

**What would indicate a problem:**
- If replay perplexity rises at `replay_ratio >= 1.0`, something is wrong with the replay data path (encoding mismatch, wrong dataset, padding contamination).
- If Voynich perplexity fails to drop even at `replay_ratio = 0.25`, the replay schedule may be drowning out Voynich gradients before they can take effect — investigate warmup and LR.
- If `realized_replay_ratio` differs meaningfully from configured `replay_ratio`, the scheduler is broken.

## Checkpoints and Artifacts

Same structure as basic fine-tune, under `data/experiments/rft-.../`:

```
data/experiments/rft-20260420-1430-r1.0/
├── config.yaml
├── checkpoints/
│   ├── epoch=010.ckpt
│   ├── ...
│   └── last.ckpt
├── logs/
└── eval/
    ├── val_voynich_metrics.json
    └── val_replay_metrics.json
```

**Why two eval files:** The two validation passes produce independent metric streams. Writing them to separate files keeps each file simple and makes it obvious at the filesystem level which metric is which.

**Replay pool caching:** The first run fetches the replay pool from HF and caches it as `data/experiments/_replay_cache/dclm-seed{replay_seed}-n{pool_size}.jsonl`. Subsequent runs with the same `(replay_seed, pool_size)` read from this cache rather than re-fetching.

**Why cache:** HF streaming is reproducible in principle but depends on dataset version pinning and network availability. A local cache keyed on the seed and pool size guarantees that all runs in a sweep see exactly the same replay data, even if HF is slow or the dataset is updated upstream.

## Integration with Existing Pipeline

The output checkpoint format is identical to the basic fine-tune's, so the same loading path into `entropy_proc` / `blt_example.py` works unchanged. The interesting comparison is three-way:

1. Pre-trained model → reference entropy values.
2. Basic fine-tune model → Voynich-adapted but forgetful.
3. Replay fine-tune model(s) at various `replay_ratio` → Voynich-adapted with preserved general knowledge.

Rendering these side-by-side in `voy_entropy_display` will reveal which bytes move in entropy as a function of ratio, which is exactly the diagnostic needed to choose a production ratio.

## Assumptions and Constraints

- Everything from [basic_fine_tune.md](basic_fine_tune.md) — single GPU, HF access, `sdpa` attention, bf16, no data augmentation — applies unchanged.
- **Network access at run start:** The first run of a given `(replay_seed, pool_size)` configuration requires network access to HF Hub. Subsequent runs may be offline via the replay cache.
- **Replay source breadth:** This spec uses only DCLM. If BLT is re-trained in the future on a different corpus, the replay source should be revisited to match.
- **No replay for validation-time inference:** At inference time (downstream entropy analysis), the fine-tuned model is used as-is. Replay is a training-time construct only.
- **Compute cost scales with ratio:** Wall-clock time per epoch grows linearly in `(1 + replay_ratio)`. At `replay_ratio=4.0`, a run takes ~5× as long as the basic fine-tune. This is a bounded increase on a 100M model and is acceptable for a sweep.
