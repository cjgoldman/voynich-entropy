"""
Test basic functionality of the BLT entropy model.

Loads the pretrained ~100M parameter entropy model from HuggingFace,
runs inference on sample text, and displays per-byte entropy values
with rich terminal visualization and a matplotlib line plot.

Prerequisites:
  - Accept the model license at https://huggingface.co/facebook/blt-entropy
  - Authenticate with: huggingface-cli login

Run with:
uv run python basic_run/blt_example.py
"""

import json
import os

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"

import plotext as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bytelatent.transformer import LMTransformer, LMTransformerArgs

HF_REPO = "facebook/blt-entropy"

console = Console()


def entropy(scores):
    """Compute per-token Shannon entropy from logits. Uses natural log."""
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    return -p_log_p.sum(dim=-1)


def entropy_color(value, min_val, max_val):
    """Map an entropy value to a rich color: green (low) -> yellow -> red (high)."""
    if max_val == min_val:
        return "white"
    t = (value - min_val) / (max_val - min_val)
    if t < 0.5:
        return "green" if t < 0.25 else "yellow"
    return "dark_orange" if t < 0.75 else "red"


def spark_bar(value, max_val, width=20):
    """Create a colored inline bar string."""
    if max_val == 0:
        return ""
    filled = int(round(value / max_val * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def load_entropy_model():
    """Load the BLT entropy model from HuggingFace Hub."""
    with console.status("[bold cyan]Loading entropy model from HuggingFace..."):
        try:
            config_path = hf_hub_download(HF_REPO, "config.json")
            weights_path = hf_hub_download(HF_REPO, "model.safetensors")
        except Exception as e:
            if "401" in str(e) or "Gated" in str(e) or "restricted" in str(e):
                console.print(
                    f"\n[bold red]Error:[/] {HF_REPO} is a gated model. To use it:\n"
                    f"  1. Accept the license at https://huggingface.co/{HF_REPO}\n"
                    "  2. Run: huggingface-cli login\n",
                )
            raise

        with open(config_path) as f:
            config = json.load(f)

        args_dict = config.get("args", config)
        torch.set_default_dtype(torch.bfloat16)
        model_args = LMTransformerArgs(**args_dict)
        model = LMTransformer(model_args)

        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        model.attn_impl = "sdpa"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"[bold green]Loaded entropy model:[/] {param_count:.0f}M parameters ({device})")
    return model, model_args


def display_results(sample_text, token_ids, ent_values):
    """Display a rich table with per-byte entropy and an inline spark bar."""
    min_e, max_e = min(ent_values), max(ent_values)
    mean_e = sum(ent_values) / len(ent_values)

    # Build a mapping from byte position to the source unicode character.
    # For multi-byte chars, only the first byte gets the character label.
    byte_to_char = {}
    byte_offset = 0
    for ch in sample_text:
        encoded = ch.encode("utf-8")
        byte_to_char[byte_offset] = ch
        byte_offset += len(encoded)

    # Header panel
    console.print()
    console.print(Panel(f"[bold]{sample_text}[/]", title="Input Text", border_style="cyan"))

    # Per-byte table
    table = Table(title="Per-Byte Entropy", show_lines=False, header_style="bold magenta")
    table.add_column("Pos", justify="right", style="dim", width=4)
    table.add_column("Byte", justify="right", width=5)
    table.add_column("Char", justify="center", width=5)
    table.add_column("Entropy", justify="right", width=8)
    table.add_column("Distribution", width=24)

    for i, (tok, ent) in enumerate(zip(token_ids, ent_values)):
        # Show the source character on the first byte of each unicode codepoint
        if i in byte_to_char:
            char = byte_to_char[i]
        else:
            char = f"\\x{tok:02x}"
        color = entropy_color(ent, min_e, max_e)
        bar = spark_bar(ent, max_e)
        table.add_row(
            str(i),
            str(tok),
            char,
            Text(f"{ent:.4f}", style=f"bold {color}"),
            Text(bar, style=color),
        )

    console.print(table)

    # Summary stats
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("Mean entropy:", f"{mean_e:.4f}")
    summary.add_row("Max entropy:", f"[red]{max_e:.4f}[/]")
    summary.add_row("Min entropy:", f"[green]{min_e:.4f}[/]")
    console.print(Panel(summary, title="Summary", border_style="cyan"))


def build_sliding_window_causal_mask(seq_len, window_size, device="cpu"):
    """Build a sliding window causal attention mask for sdpa."""
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    # Causal: attend only to past positions; sliding window: at most window_size back
    mask = (rows >= cols) & (rows - cols < window_size)
    # sdpa expects float mask with -inf for masked positions
    attn_mask = torch.where(mask, 0.0, float("-inf"))
    return attn_mask


def main():
    entropy_model, model_args = load_entropy_model()
    device = next(entropy_model.parameters()).device
    sample_text = "Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R R Martin"

    # BLT vocab: [BOE=0, BOS=1, EOS=2, BPE=3, byte_0=4, ..., byte_255=259]
    BLT_BYTE_OFFSET = 4
    token_ids = [b + BLT_BYTE_OFFSET for b in sample_text.encode("utf-8")]
    tokens = torch.tensor([token_ids], dtype=torch.long).to(device)

    sliding_window = model_args.sliding_window or 512
    attn_mask = build_sliding_window_causal_mask(len(token_ids), sliding_window, device=device)

    with console.status("[bold cyan]Running entropy model inference..."):
        with torch.no_grad():
            logits = entropy_model(tokens, mask=attn_mask)
            ent_tensor = entropy(logits)

    ent_values = [e.item() for e in ent_tensor[0]]

    raw_byte_ids = [b - BLT_BYTE_OFFSET for b in token_ids]
    display_results(sample_text, raw_byte_ids, ent_values)

    # Terminal line plot of per-byte entropy using plotext
    positions = list(range(len(ent_values)))
    plt.clear_figure()
    plt.plot(positions, ent_values, marker="braille")
    plt.title("Per-Byte Entropy")
    plt.xlabel("Byte Position")
    plt.ylabel("Entropy")
    plt.show()


if __name__ == "__main__":
    main()
