"""Lightning module for replay-based Voynich entropy fine-tuning."""

from __future__ import annotations

import os

os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from fine_tune.module import VoynichEntropyFineTune
from fine_tune.replay_config import ReplayFineTuneConfig
from fine_tune.replay_loader import REPLAY, VOYNICH

_DATALOADER_SOURCES = (VOYNICH, REPLAY)


class ReplayEntropyFineTune(VoynichEntropyFineTune):
    """Extends :class:`VoynichEntropyFineTune` with per-source logging.

    Training batches are dicts ``{"tokens": LongTensor, "source": str}`` from
    :class:`MixedDataLoader`. Validation is run with two dataloaders (Voynich
    then replay), routed by ``dataloader_idx``.
    """

    def __init__(self, config: ReplayFineTuneConfig, total_steps: int) -> None:
        super().__init__(config, total_steps)
        self.config: ReplayFineTuneConfig = config
        self._voynich_batch_count = 0
        self._replay_batch_count = 0

    def on_train_epoch_start(self) -> None:
        self._voynich_batch_count = 0
        self._replay_batch_count = 0

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        source = batch["source"]
        loss = self._compute_loss(tokens)
        ppl = torch.exp(loss)

        self.log(
            "train/loss/combined", loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train/perplexity/combined", ppl, on_step=True, on_epoch=False)
        self.log(f"train/loss/{source}", loss, on_step=True, on_epoch=False)
        self.log(f"train/perplexity/{source}", ppl, on_step=True, on_epoch=False)

        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log(
                "train/lr", scheduler.get_last_lr()[0], on_step=True, on_epoch=False
            )

        if source == VOYNICH:
            self._voynich_batch_count += 1
        else:
            self._replay_batch_count += 1
        return loss

    def on_train_epoch_end(self) -> None:
        total = self._voynich_batch_count + self._replay_batch_count
        if total == 0:
            return
        realized = self._replay_batch_count / total
        self.log("replay_ratio_realized", realized, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        source = _DATALOADER_SOURCES[dataloader_idx]
        loss = self._compute_loss(batch)
        self.log(
            f"val/loss/{source}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(source == VOYNICH),
            add_dataloader_idx=False,
        )
        self.log(
            f"val/perplexity/{source}",
            torch.exp(loss),
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        warmup = max(1, int(self.config.warmup_fraction * self.total_steps))
        remaining = max(self.total_steps - warmup, 1)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=(
                1e-8 / self.config.learning_rate
                if self.config.learning_rate > 0
                else 1e-8
            ),
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
