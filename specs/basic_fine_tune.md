# Basic Fine Tune Spec

## Purpose

This document specifies the first fine-tuning pass of the BLT entropy model on the Voynich Unicode BMP transcription dataset. The goal is to adapt the pre-trained entropy model so that its per-byte entropy values better reflect Voynich-specific patterns, producing a more meaningful signal for downstream cryptanalysis and visualization.

**Why fine-tune at all:** The pre-trained entropy model was trained on DCLM (~3.8 TB of English web text). When it encounters Voynich text — encoded as 3-byte UTF-8 sequences in the Unicode Private Use Area — every byte is maximally surprising because PUA codepoints never appeared in pre-training. The resulting entropy values measure "distance from English web text" rather than "predictability within Voynich." A fine-tuned model that has learned Voynich byte patterns will assign lower entropy to predictable glyphs and higher entropy to genuinely surprising ones, producing an entropy signal that is useful for structural and cryptanalytic analysis.

## Scope

This spec covers the "basic" fine-tune: a deliberate overfit of the entropy model on the full Voynich corpus. **Why start with overfitting:** The Voynich dataset is tiny (~30 KB of glyph text, ~4,043 manuscript lines across ~234 folios). Sophisticated regularization techniques (LoRA, mixed replay, curriculum learning) add complexity that obscures whether the model can learn Voynich patterns at all. The basic fine-tune establishes a lower bound on achievable loss and confirms the training pipeline works end-to-end. More precise methods (adapter-based fine-tuning, DCLM replay mixing) are deferred to a follow-up spec.

## Pipeline Context

The fine-tuning module sits between the existing data preparation pipeline and the entropy analysis pipeline:

```
voynpy.corpora.vms_unicode (DataFrame)
        ↓
vms_uprep.prepare()  →  list[str] (text with structural markers)
        ↓
vms_uprep.stack_lines(max_bytes=8192)  →  list[str] (byte-budget chunks)
        ↓
UTF-8 encoding  →  raw byte sequences
        ↓
Token IDs: [b + 4 for b in raw_bytes]  →  BLT vocab (260 tokens)
        ↓
┌─────────────────────────────────────────────┐
│  THIS SPEC: Fine-tune entropy model          │
│                                              │
│  PyTorch Lightning training loop             │
│  Loss: cross-entropy on next-byte prediction │
│  Checkpoints → data/experiments/             │
│  Tracking → ClearML                          │
└─────────────────────────────────────────────┘
        ↓
Fine-tuned model checkpoint
        ↓
(Future) Load into entropy analysis pipeline for pre/post comparison
```

**Why reuse the existing data pipeline:** The `vms_uprep` module already handles comma removal, structural marker insertion (pilcrow, line/paragraph separators), and byte-budget chunking to 8192 bytes — exactly what the entropy model expects. Building a separate data path would duplicate logic and risk subtle encoding differences.

## Target Model

The fine-tuning target is the **BLT entropy model only** — the ~100M parameter `LMTransformer` loaded from `facebook/blt-entropy` on HuggingFace.

| Property | Value |
|----------|-------|
| Architecture | `LMTransformer` (causal byte-level transformer) |
| Parameters | ~100M |
| Layers | 14 |
| Dimension | 768 |
| Attention heads | 12 |
| Vocabulary | 260 (256 bytes + 4 control tokens) |
| Context window | 8192 bytes |
| Attention | Sliding window, 512 positions |
| Pre-training data | DCLM Baseline 1.0 |
| Precision | bfloat16 |

**Why the entropy model and not the full BLT:** The entropy model is what drives the per-byte entropy values used in the analysis pipeline. It is small enough to fine-tune on a single GPU without distributed training. The full BLT (encoder + global transformer + decoder) is orders of magnitude larger and requires a fundamentally different fine-tuning strategy — that is out of scope for this basic pass.

## Training Data

### Source

All prepared Voynich Unicode text from `voynpy.corpora.vms_unicode`, processed through `vms_uprep.prepare()` and `vms_uprep.stack_lines()`.

### Dataset Size

The Voynich corpus is extremely small by language model standards:

- ~4,043 manuscript lines across ~234 folios
- Raw CSV: ~548 KB (mostly structural columns, not text)
- Prepared text (after marker insertion): estimated ~90–120 KB of UTF-8 bytes
- After chunking into 8192-byte windows: estimated 12–15 chunks

**Why this matters:** With so little data, the model will see every byte many times per epoch. This is acceptable for the basic fine-tune (which intentionally overfits) but must be addressed in follow-up work.

### Data Split

Random folio-level 80/20 split for training and validation.

**Why folio-level rather than line-level:** Consecutive lines within a folio share structural patterns (paragraph boundaries, glyph frequency distributions). Splitting at the line level would leak these patterns into the validation set, overstating generalization. Folio-level splitting ensures the validation set contains pages the model has never seen.

**Implementation:** Shuffle the list of unique folio identifiers with a fixed seed, assign the first 80% to training and the remainder to validation. All lines from a given folio stay together. The random seed must be recorded in the experiment config for reproducibility.

### Byte Encoding

Text is encoded to BLT token IDs using the same scheme as the existing inference pipeline:

```
token_ids = [byte_value + 4 for byte_value in text.encode("utf-8")]
```

Where the offset of 4 accounts for the control tokens: BOE (0), BOS (1), PAD (2), EOS (3).

**Why no BOS/EOS wrapping for chunks:** Each 8192-byte chunk is a contiguous slice of manuscript text, not a standalone document. Adding BOS/EOS would teach the model artificial boundaries that don't exist in the data. The structural markers (pilcrow, line separator, paragraph separator) already encode real manuscript boundaries.

## Training Configuration

### Framework

**PyTorch Lightning** for the training loop. **Why:** Lightning provides checkpointing, logging, gradient clipping, and mixed-precision support out of the box, and is portable across single-GPU, multi-GPU, and cluster environments without rewriting the training loop. The existing BLT training script (`blt/bytelatent/train.py`) uses raw PyTorch with FSDP — appropriate for pre-training at scale but excessive for fine-tuning a 100M-parameter model on 15 chunks of data.

### Loss Function

Standard cross-entropy loss on next-byte prediction, matching the pre-training objective:

```
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size),
                       target_ids[:, 1:].reshape(-1))
```

**Why cross-entropy and not a custom objective:** The entropy model's purpose is to produce calibrated next-byte probability distributions. Cross-entropy directly optimizes for this. Alternative objectives (contrastive loss, entropy regularization) would change what the model learns and complicate interpretation of its outputs.

### Hyperparameters

Starting point, derived from the pre-training config with adjustments for the tiny dataset:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | 4x lower than pre-training (4e-4). Smaller steps to avoid immediately overwriting pre-trained features. |
| LR schedule | Cosine with warmup | Matches pre-training. Cosine decay is well-suited to a known epoch count. |
| Warmup steps | 50 | Short warmup — dataset is tiny, so training is measured in hundreds of steps. |
| Batch size | 1 | Each chunk is 8192 bytes. With ~12 training chunks, batch size 1 gives ~12 steps per epoch. Larger batches would reduce steps per epoch below useful granularity. |
| Epochs | 100 | Intentionally high — this is a deliberate overfit. The model should reach near-zero training loss. |
| Gradient clipping | 10.0 | Matches pre-training config. Prevents instability from outlier gradients on unfamiliar byte patterns. |
| Precision | bfloat16 | Matches pre-training. Sufficient precision for this model size. |
| Optimizer | AdamW | Standard for transformer fine-tuning. Weight decay 0.1. |
| Weight decay | 0.1 | Minimal regularization — not trying to prevent overfitting in this pass. |

**Why these values:** The pre-training config (`entropy_model.yaml`) uses lr=4e-4 with 500 warmup steps over 100K steps of DCLM data. Fine-tuning on a tiny, very different distribution calls for a lower learning rate (the model should adjust, not reinitialize) and much shorter warmup (there isn't enough data for a long warmup to matter). The high epoch count is deliberate: we want to confirm the model can drive training loss close to zero on Voynich bytes.

### Sliding Window Attention

The entropy model uses a sliding window of 512 positions. This is preserved unchanged during fine-tuning.

**Why not expand the window:** The 8192-byte context window already covers the full manuscript in ~15 chunks. The 512-position sliding window means each byte attends to its 512 nearest predecessors — sufficient for learning local glyph patterns. Expanding the window would require architectural changes and is not warranted for this basic pass.

## Experiment Tracking

### ClearML

All training runs are logged to a self-hosted ClearML server at `https://clearml.acet.network/`. The training entry point calls `Task.set_credentials(...)` followed by `Task.init(...)` to register the run. A small custom `ClearMLLogger` (in `fine_tune/clearml_logger.py`) implements the PyTorch Lightning `Logger` interface and forwards scalar logs to `Task.get_logger().report_scalar(...)`, splitting keys of the form `"train/loss"` into a ClearML chart title and series so related metrics are grouped on the same plot.

**Logged metrics:**
- `train/loss` — per-step cross-entropy loss
- `train/perplexity` — exp(loss), for interpretability
- `train/lr` — learning rate (to verify schedule)
- `val/loss` — validation cross-entropy loss (computed every epoch)
- `val/perplexity` — exp(val_loss)
- `epoch` — current epoch

**Logged config:** All hyperparameters, data split seed, folio lists for train/val, model checkpoint source, git commit hash — connected to the ClearML task under the `run` section via `task.connect(...)`.

**Credentials:** Access key and secret key for the ClearML server are wired up in `train.py`. Any of `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`, `CLEARML_API_ACCESS_KEY`, or `CLEARML_API_SECRET_KEY` environment variables override the hardcoded defaults, so a shared checkout can be pointed at a different ClearML server without code edits.

**Why ClearML:** The project uses a self-hosted ClearML instance that supports task tracking, metric visualization, comparison across runs, and artifact management without relying on an external SaaS. Lightning integration is handled by the small custom logger rather than ClearML's framework auto-capture, which keeps the emitted scalar namespace predictable and avoids unrelated auto-capture side-effects.

### Run Naming

Timestamp-based run IDs: `bft-YYYYMMDD-HHMM` (e.g., `bft-20260416-1430`).

**Why timestamp-based:** Avoids the need for a counter or registry. Each run is inherently unique. The prefix `bft` (basic fine tune) distinguishes these runs from future experiment types. The same run ID is used as the ClearML task name.

## Artifacts and Storage

All artifacts are stored under `data/experiments/<run-id>/`:

```
data/experiments/bft-20260416-1430/
├── config.yaml              # Full experiment configuration (frozen at run start)
├── checkpoints/
│   ├── epoch=010.ckpt       # Lightning checkpoint every 10 epochs
│   ├── epoch=020.ckpt
│   ├── ...
│   └── last.ckpt            # Most recent checkpoint (overwritten each epoch)
├── logs/                    # Lightning default log dir (ClearML streams scalars to the server)
└── eval/
    └── val_metrics.json     # Final validation metrics
```

**Checkpoint strategy:**
- Save a checkpoint every 10 epochs (configurable).
- Always keep `last.ckpt` for resumption.
- Keep all periodic checkpoints (no pruning). **Why:** With ~100 epochs total, this produces ~10 checkpoints at ~400 MB each (100M params in bf16 + optimizer state). Total storage is under 5 GB — negligible, and having the full trajectory allows post-hoc analysis of when Voynich patterns were learned.

**Config snapshot:** The full experiment configuration (hyperparameters, data split, model source) is serialized to `config.yaml` at the start of each run. **Why:** Reproducibility. The config file is the single source of truth for recreating a run, independent of ClearML server availability.

## Evaluation

### Primary Metric

**Held-out perplexity** (exp of cross-entropy loss) on the 20% validation folios, computed at the end of each epoch.

**Why perplexity:** It directly measures how well the model predicts Voynich bytes. A perplexity of 256 means the model is guessing uniformly over all byte values (no knowledge). A perplexity of 1 means perfect prediction. The trajectory from the initial (high) perplexity toward a lower value quantifies how much Voynich structure the model has captured.

### Expected Behavior

1. **Epoch 0 (pre-trained model):** High perplexity on both train and val sets. The model has never seen PUA bytes and will assign near-uniform probability to them.
2. **Early epochs (1–10):** Rapid perplexity drop as the model learns common Voynich byte patterns (the 3-byte PUA prefix `0xEF 0xBx 0x8x` for U+FDxx–U+FExx range).
3. **Mid epochs (10–50):** Slower descent as the model learns glyph-level and token-level patterns.
4. **Late epochs (50–100):** Training perplexity approaches 1 (overfit). Validation perplexity plateaus or rises — the gap quantifies generalization. The plateau value is the interesting result: it tells us how much structure transfers across folios.

**What to watch for:**
- If validation perplexity never drops meaningfully below the epoch-0 value, the model is memorizing byte positions rather than learning transferable patterns. This would suggest the Voynich data lacks the kind of local byte-sequence regularity that a 512-window transformer can exploit.
- If validation perplexity drops and stays low, Voynich has learnable structure that generalizes across folios — a positive signal for cryptanalysis.

## Data Pipeline Module

### PyTorch Dataset

A `VoynichEntropyDataset` class wrapping the existing `vms_uprep` pipeline:

1. At initialization: load `vms_unicode.df`, apply the folio-level train/val split, call `vms_uprep.prepare()` and `vms_uprep.stack_lines()` on the assigned folios.
2. `__len__`: returns the number of 8192-byte chunks.
3. `__getitem__`: returns a tensor of BLT token IDs (byte values + 4 offset) for the chunk at the given index.

**Why a custom Dataset rather than BLT's data pipeline:** BLT's data pipeline (`bytelatent/data/`) is designed for streaming terabytes of pre-shuffled data from disk via `iterators`. It expects pre-processed arrow files, shard manifests, and distributed data loading. For 15 chunks of Voynich data that fit entirely in memory, this machinery is unnecessary overhead. A simple map-style Dataset is sufficient and avoids coupling the fine-tuning code to BLT's data infrastructure.

### Sequence Handling

Each chunk is at most 8192 UTF-8 bytes, producing at most 8192 token IDs. Chunks shorter than 8192 tokens are right-padded with PAD_ID (2).

**Why pad rather than pack:** Packing multiple short sequences into one 8192-token window would create artificial adjacencies between unrelated text segments. The structural markers in the prepared text encode real manuscript boundaries; padding preserves them.

### DataLoader

Standard PyTorch DataLoader with:
- `batch_size=1` (one chunk per step)
- `shuffle=True` for training, `shuffle=False` for validation
- `num_workers=0` — data is in memory, no I/O bottleneck

## Lightning Module

A `VoynichEntropyFineTune` Lightning module wrapping the pre-trained `LMTransformer`:

1. **`__init__`:** Load the pre-trained entropy model using the same loading logic as `blt_example.py` (`hf_hub_download` → `LMTransformerArgs` → `LMTransformer` → `load_file`). Enable `requires_grad` on all parameters.
2. **`training_step`:** Forward pass through the model, compute cross-entropy loss on next-byte prediction, log `train/loss` and `train/perplexity`.
3. **`validation_step`:** Same forward pass and loss, log `val/loss` and `val/perplexity`.
4. **`configure_optimizers`:** AdamW with cosine LR schedule and warmup.

**Why enable all gradients (no freezing):** This is the basic fine-tune — full parameter updates with intentional overfitting. Freezing layers would be a regularization technique, deferred to the follow-up spec.

## Integration with Existing Pipeline

The fine-tuned model checkpoint can be loaded into the existing entropy analysis pipeline (`blt_example.py`, `entropy_proc`) by replacing the HuggingFace model load with a Lightning checkpoint load. This enables side-by-side comparison of per-byte entropy values before and after fine-tuning.

**Details deferred:** The mechanics of loading a Lightning checkpoint into the inference pipeline, and the visualization of pre/post entropy comparisons, will be specified separately. This spec focuses on the training process.

## Assumptions and Constraints

- **Single GPU:** The entropy model (100M parameters in bf16) fits comfortably in a single GPU's memory. No distributed training is needed.
- **HuggingFace access:** The `facebook/blt-entropy` model is gated. The user must have accepted the license and authenticated via `huggingface-cli login`.
- **Attention backend:** The fine-tuning code should use `sdpa` (PyTorch native scaled dot-product attention) rather than `xformers`, matching the inference pipeline in `blt_example.py`. **Why:** `xformers` is an optional dependency with build complexity. SDPA is built into PyTorch and sufficient for single-GPU fine-tuning.
- **No data augmentation:** The basic fine-tune uses the Voynich data as-is. Augmentation strategies (byte-level noise, manuscript section resampling) are deferred.
- **Git-tracked config, not checkpoints:** The `config.yaml` for each experiment should be committed. Checkpoint files (~400 MB each) should be in `.gitignore`.
