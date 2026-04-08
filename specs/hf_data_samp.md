# HF Data Sampling Utility

## Purpose

This document specifies the `hf_data_samp` module, a lightweight utility for pulling small text samples from Hugging Face datasets for comparative entropy analysis and fine tuning. It enables researchers to run reference texts (e.g., English web text from DCLM, Wikipedia, or any HF dataset) through the same BLT entropy model pipeline used for Voynich manuscript analysis, without downloading or preprocessing full datasets.

## Pipeline Context

The `hf_data_samp` module provides an alternative entry point into the existing entropy analysis pipeline, running parallel to the Voynich-specific `vms_uprep` path:

```
                      ┌──────────────────────────────┐
                      │ voynpy.corpora (DataFrame)    │
                      │         ↓                     │
                      │ vms_uprep.prepare()            │
                      └────────┬─────────────────────┘
                               │
 ┌──────────────────────────┐  │
 │ hf_data_samp.sample()    │  │
 │         ↓                │  │
 │ list[str] (text chunks)  │  │
 └────────┬─────────────────┘  │
          │                    │
          ▼                    ▼
   vms_uprep.stack_lines()  /  direct byte encoding
          │
          ▼
   BLT entropy model inference  →  per-byte entropy floats
          │
          ▼
   entropy_plot.plot_entropy()  →  visualization
```

**Why this module exists:** The project already analyzes Voynich manuscript entropy patterns. Comparing those patterns against a baseline of "normal" text (e.g., English web text from DCLM, the dataset used to train BLT) requires pulling small reference samples. The existing BLT data pipeline (`blt/setup/download_prepare_hf_data.py`) downloads entire datasets and preprocesses them with terashuf — overkill for grabbing 10–100 documents. This module provides a streaming, sample-oriented alternative.

## New Dependency

The module requires the `datasets` library from Hugging Face:

```
datasets>=3.0
```

This should be added to the `gpu` dependency group in `/workspace/pyproject.toml`, alongside the existing `huggingface-hub==0.30.*`.

**Why `datasets` and not `huggingface_hub` alone:** The `huggingface_hub` package (already a dependency) provides `snapshot_download()` which downloads entire dataset repositories. The `datasets` library provides `load_dataset(..., streaming=True)` which yields individual examples on demand via HTTP range requests against parquet files, without downloading the full dataset. For pulling 10–100 documents from a multi-terabyte dataset like DCLM, streaming is the only practical approach.

## Data Structures

### `DatasetSpec` (Dataclass)

Specifies which HF dataset to sample from:

| Field | Type | Default | Description |
|---|---|---|---|
| `repo_id` | `str` | *(required)* | HF dataset repository ID (e.g., `"mlfoundations/dclm-baseline-1.0"`) |
| `split` | `str` | `"train"` | Dataset split name |
| `text_column` | `str \| None` | `None` | Column containing text; auto-detected if `None` |
| `subset` | `str \| None` | `None` | Dataset config/subset name (for multi-config datasets) |

**Why `text_column` is optional with auto-detection:** Different HF datasets use different column names for the text field. DCLM uses `"text"`, some datasets use `"content"`, others use `"document"`. The auto-detection logic checks for `"text"`, then `"content"` (matching the fallback order in BLT's `get_text()` at `blt/bytelatent/preprocess/preprocess_entropies.py`), then raises a clear error listing available columns. Explicit specification overrides auto-detection.

**Why `subset` exists:** Some HF datasets have multiple configurations (e.g., Wikipedia has per-language configs like `"20220301.en"`). The `subset` parameter maps to the `datasets.load_dataset()` `name` argument.

### `HFSample` (Dataclass)

A sampled document with provenance metadata:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text, truncated to `max_bytes` |
| `doc_index` | `int` | 0-based index in the (possibly shuffled) iteration stream |
| `dataset_id` | `str` | Copy of `spec.repo_id` for provenance |
| `byte_length` | `int` | UTF-8 byte length of `text` |
| `truncated` | `bool` | Whether the original document exceeded `max_bytes` |

### `DCLM` (Module-Level Constant)

A pre-configured `DatasetSpec` for the DCLM baseline:

```python
DCLM = DatasetSpec(
    repo_id="mlfoundations/dclm-baseline-1.0",
    split="train",
    text_column="text",
)
```

**Why a preset:** DCLM is the primary comparison dataset for this project — it was used to train BLT. A named constant prevents typos and documents the correct schema.

## Specified Capabilities

### Sampling

The primary function `sample()` pulls text documents from a HF dataset via streaming:

```python
def sample(
    spec: DatasetSpec,
    n: int = 10,
    *,
    offset: int = 0,
    seed: int | None = None,
    max_bytes: int = 8192,
) -> list[str]:
```

Returns a list of plain text strings, one per sampled document, each truncated to fit within `max_bytes` of UTF-8 encoding.

**Why these parameters:**

- **`n`**: Number of documents to retrieve. Default 10 is small enough for quick iteration but large enough for meaningful comparison.
- **`offset`**: Skip the first N documents in the stream before sampling. Enables deterministic access to different regions of the dataset without randomization. `offset=0, n=10` always returns the same 10 documents for a given dataset version.
- **`seed`**: When provided, enables shuffled sampling. The `datasets` library's streaming mode supports `dataset.shuffle(seed=seed)` which provides a pseudo-random iteration order using a shuffle buffer. When `None`, documents are returned in dataset order (sequential from `offset`). **Why:** Shuffled sampling gives a more representative cross-section of the dataset, but sequential access is simpler and fully reproducible.
- **`max_bytes`**: Per-document byte budget, defaulting to 8192 to match the BLT entropy model's input window. Documents longer than this are truncated at a UTF-8 character boundary. **Why:** The BLT model has a fixed context window of 8192 bytes. Text beyond this limit would be silently dropped during inference. Truncating at sample time makes the budget explicit.

### Sampling with Metadata

An extended function preserves dataset provenance:

```python
def sample_with_metadata(
    spec: DatasetSpec,
    n: int = 10,
    *,
    offset: int = 0,
    seed: int | None = None,
    max_bytes: int = 8192,
) -> list[HFSample]:
```

Same sampling behavior as `sample()`, but returns `HFSample` objects that track the source dataset, stream position, byte length, and whether truncation occurred.

**Why:** When comparing entropy across multiple sources, researchers need to track which sample came from where and whether it was truncated (truncation could affect entropy patterns at the document boundary).

### Text Column Auto-Detection

When `spec.text_column` is `None`, the module inspects the dataset's column names (available from the streamed dataset's `.features` or first yielded example) and selects the text column using this priority order:

1. `"text"`
2. `"content"`

If neither is found, raise `ValueError` listing the available columns with a message suggesting the user set `text_column` explicitly.

**Why this order:** Matches the convention in BLT's `get_text()` function (`blt/bytelatent/preprocess/preprocess_entropies.py`), ensuring consistency with how BLT itself reads these datasets.

### UTF-8 Safe Truncation

Documents exceeding `max_bytes` are truncated to the largest whole UTF-8 character that fits within the budget. The truncation must not split multi-byte characters.

Implementation: `text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")`. The `errors="ignore"` drops the final incomplete character if the cut falls mid-codepoint.

**Why:** The BLT model encodes text as raw UTF-8 bytes. Truncating mid-character would produce an invalid byte sequence, which could corrupt entropy analysis near the truncation point.

### Integration with Existing Pipeline

The output of `sample()` is a `list[str]` — plain text strings. These integrate with the existing pipeline in two ways:

1. **Via `stack_lines()`:** A list of sampled documents can be passed to `vms_uprep.stack_lines()` to pack them into BLT-sized chunks, reusing the existing byte-budgeting logic.

2. **Direct to inference:** Each string can be individually encoded to bytes and fed to the BLT entropy model, as done in `src/basic_run/blt_example.py`: `token_ids = [b + BLT_BYTE_OFFSET for b in sample_text.encode("utf-8")]`.

The module deliberately outputs plain strings rather than `AnnotatedChunk` objects. **Why:** HF dataset samples do not have Voynich manuscript provenance metadata (folio, paragraph, line). The annotation system is Voynich-specific. For comparative analysis, plain text flowing through `entropy_plot.plot_entropy()` (which is intentionally Voynich-agnostic) is the correct integration path.

## Error Handling

### Network Errors

Streaming from HF Hub requires network access. The module lets `datasets` library exceptions propagate naturally (e.g., `ConnectionError`, `HfHubHTTPError`) with no retry logic of its own. **Why:** Retry policy is better handled by the `datasets` library's built-in retry behavior. Adding custom retry logic would duplicate what the library already does.

### Insufficient Documents

If the dataset yields fewer than `n` documents (e.g., due to a large `offset` or small dataset), return whatever was collected without error. The caller can check `len(result) < n`. **Why:** Raising an exception for "not enough data" would be annoying for exploratory use.

### Authentication

Gated datasets require authentication. The module does not handle authentication directly — it relies on the user having run `huggingface-cli login` or set `HF_TOKEN`. **Why:** Authentication is an environment concern, not a sampling concern.

## Implementation Notes

- The module lives at `/workspace/src/hf_data_samp.py`, alongside the existing pipeline modules.
- Streaming is implemented via `datasets.load_dataset(repo_id, name=subset, split=split, streaming=True)`, which returns an `IterableDataset`. This makes only HTTP range requests for the specific parquet row groups needed, avoiding full dataset download.
- When `seed` is provided, call `.shuffle(seed=seed, buffer_size=1000)` on the iterable dataset before taking samples. The buffer size of 1000 provides reasonable randomization for small sample counts without excessive memory use.
- When `offset > 0` and `seed is None`, use `.skip(offset)` on the iterable dataset. When `seed` is provided, `offset` skips into the shuffled stream (i.e., shuffle then skip).
- Use `.take(n)` after skip/shuffle to limit the stream to exactly `n` documents.
- The module should be importable without `torch` or any GPU dependencies. It only depends on `datasets` (and transitively `huggingface_hub`, `fsspec`). This keeps it usable in CPU-only environments for data exploration.
- DCLM-specific note: `mlfoundations/dclm-baseline-1.0` is approximately 3.8 TB / 3B documents. Streaming mode avoids downloading any of this. The dataset stores parquet files, and the `datasets` library handles parquet streaming transparently.
- `sample()` is implemented in terms of `sample_with_metadata()`, extracting just the `.text` field from each `HFSample`. This avoids duplicating the streaming logic.

## Example Usage

```python
from hf_data_samp import sample, sample_with_metadata, DCLM, DatasetSpec

# Pull 10 DCLM documents, sequential from start
texts = sample(DCLM, n=10)

# Pull 20 shuffled DCLM documents
texts = sample(DCLM, n=20, seed=42)

# Pull from an arbitrary dataset
wiki = DatasetSpec(repo_id="wikipedia", subset="20220301.en", split="train")
texts = sample(wiki, n=5, seed=42)

# With metadata for tracking
samples = sample_with_metadata(DCLM, n=10, seed=42)
for s in samples:
    print(f"[{s.dataset_id}] doc {s.doc_index}: {s.byte_length} bytes, truncated={s.truncated}")

# Feed into existing entropy pipeline
from vms_uprep import stack_lines
chunks = stack_lines(texts, max_bytes=8192)
# Each chunk is now ready for BLT inference, same as Voynich text
```
