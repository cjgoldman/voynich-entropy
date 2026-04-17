"""PyTorch Lightning module for Voynich entropy model fine-tuning."""

from __future__ import annotations

import json
import os

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

import lightning as L
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from bytelatent.transformer import LMTransformer, LMTransformerArgs
from fine_tune.config import FineTuneConfig, PAD_ID
from fine_tune.dataset import build_sliding_window_causal_mask


class VoynichEntropyFineTune(L.LightningModule):
    """Lightning module wrapping the BLT entropy model for fine-tuning."""

    def __init__(self, config: FineTuneConfig, total_steps: int) -> None:
        super().__init__()
        self.config = config
        self.total_steps = total_steps
        self.model, self.model_args = self._load_pretrained_model()

    def _load_pretrained_model(self) -> tuple[LMTransformer, LMTransformerArgs]:
        """Load the pre-trained BLT entropy model from HuggingFace Hub."""
        config_path = hf_hub_download(self.config.hf_repo, "config.json")
        weights_path = hf_hub_download(self.config.hf_repo, "model.safetensors")

        with open(config_path) as f:
            hf_config = json.load(f)

        args_dict = hf_config.get("args", hf_config)

        # Model must be constructed in bfloat16 to match pre-trained weights
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model_args = LMTransformerArgs(**args_dict)
            model = LMTransformer(model_args)
        finally:
            torch.set_default_dtype(prev_dtype)

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        # Use PyTorch native SDPA instead of xformers
        model.attn_impl = "sdpa"

        # Enable gradients for all parameters (full fine-tune)
        model.requires_grad_(True)

        return model, model_args

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entropy model.

        Args:
            token_ids: (batch, seq_len) integer token IDs.

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        seq_len = token_ids.shape[1]
        window = self.model_args.sliding_window or self.config.sliding_window
        mask = build_sliding_window_causal_mask(seq_len, window, device=token_ids.device)
        return self.model(token_ids, mask=mask)

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss on next-byte prediction.

        Args:
            batch: (batch_size, seq_len) token ID tensor.

        Returns:
            Scalar loss tensor.
        """
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        logits = self.forward(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/perplexity", torch.exp(loss), on_step=True, on_epoch=False)
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            lr = scheduler.get_last_lr()[0]
            self.log("train/lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        warmup = self.config.warmup_steps
        remaining = max(self.total_steps - warmup, 1)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8 / self.config.learning_rate if self.config.learning_rate > 0 else 1e-8,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
