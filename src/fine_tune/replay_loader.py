"""Batch scheduler and mixed dataloader for replay fine-tuning."""

from __future__ import annotations

import math
from typing import Iterator

import lightning as L
import torch
from torch.utils.data import Dataset

VOYNICH = "voynich"
REPLAY = "replay"


def _is_integer_ratio(ratio: float) -> bool:
    """True when ``ratio`` is a non-negative integer value."""
    if ratio < 0:
        return False
    return abs(ratio - round(ratio)) < 1e-9


def replay_batch_count(num_voynich: int, ratio: float) -> int:
    """Number of replay batches per epoch for a given Voynich count and ratio."""
    if _is_integer_ratio(ratio):
        return int(round(ratio)) * num_voynich
    return math.ceil(num_voynich * ratio)


class BatchScheduler:
    """Produces a per-step source schedule (``voynich`` vs ``replay``).

    For integer ``ratio`` the schedule is deterministic: one Voynich batch
    followed by ``R`` replay batches, repeated ``num_voynich`` times. For
    non-integer ``ratio`` the exact counts are fixed
    (``num_voynich`` and ``ceil(num_voynich * ratio)``) and the order is
    shuffled with a ``torch.Generator`` seeded from ``(seed, epoch)``.
    """

    def __init__(self, num_voynich: int, ratio: float, seed: int) -> None:
        if num_voynich <= 0:
            raise ValueError("num_voynich must be positive")
        if ratio < 0:
            raise ValueError("replay_ratio must be non-negative")
        self.num_voynich = num_voynich
        self.ratio = ratio
        self.seed = seed

    @property
    def is_deterministic(self) -> bool:
        return _is_integer_ratio(self.ratio)

    @property
    def num_replay(self) -> int:
        return replay_batch_count(self.num_voynich, self.ratio)

    @property
    def total_steps(self) -> int:
        return self.num_voynich + self.num_replay

    def schedule(self, epoch: int) -> list[str]:
        if self.is_deterministic:
            r_int = int(round(self.ratio))
            if r_int == 0:
                return [VOYNICH] * self.num_voynich
            block = [VOYNICH] + [REPLAY] * r_int
            return block * self.num_voynich

        tokens = [VOYNICH] * self.num_voynich + [REPLAY] * self.num_replay
        g = torch.Generator()
        g.manual_seed(self.seed + int(epoch))
        perm = torch.randperm(len(tokens), generator=g).tolist()
        return [tokens[i] for i in perm]


class MixedDataLoader:
    """Iterable that yields Voynich and replay batches per the scheduler.

    Each yielded batch is a ``{"tokens": LongTensor, "source": str}`` dict.
    Tokens carry a leading batch dimension of size 1, matching the basic
    fine-tune's ``batch_size=1`` convention. Voynich indices are reshuffled
    each epoch; replay indices are reshuffled each time the replay iterator
    is exhausted and restarted.
    """

    def __init__(
        self,
        voynich_ds: Dataset,
        replay_ds: Dataset,
        scheduler: BatchScheduler,
        voynich_shuffle_seed: int,
        replay_shuffle_seed: int,
    ) -> None:
        if len(voynich_ds) != scheduler.num_voynich:
            raise ValueError(
                f"Scheduler expects {scheduler.num_voynich} voynich chunks but "
                f"dataset has {len(voynich_ds)}"
            )
        if len(replay_ds) == 0 and scheduler.num_replay > 0:
            raise ValueError(
                "replay dataset is empty but scheduler requires replay batches"
            )
        self.voynich_ds = voynich_ds
        self.replay_ds = replay_ds
        self.scheduler = scheduler
        self.voynich_shuffle_seed = voynich_shuffle_seed
        self.replay_shuffle_seed = replay_shuffle_seed
        self._epoch = 0
        self._last_realized_ratio: float | None = None

    def __len__(self) -> int:
        return self.scheduler.total_steps

    def set_epoch(self, epoch: int) -> None:
        """Externally override the epoch counter used for seeding."""
        self._epoch = int(epoch)

    @property
    def last_realized_ratio(self) -> float | None:
        """Realized replay fraction from the most recently completed epoch."""
        return self._last_realized_ratio

    def _voynich_order(self, epoch: int) -> list[int]:
        g = torch.Generator()
        g.manual_seed(self.voynich_shuffle_seed + epoch)
        return torch.randperm(len(self.voynich_ds), generator=g).tolist()

    def _replay_cycle(self, epoch: int) -> Iterator[int]:
        cycle = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self.replay_shuffle_seed + epoch * 100003 + cycle)
            order = torch.randperm(len(self.replay_ds), generator=g).tolist()
            for i in order:
                yield i
            cycle += 1

    def __iter__(self):
        epoch = self._epoch
        schedule = self.scheduler.schedule(epoch)
        v_iter = iter(self._voynich_order(epoch))
        r_iter = self._replay_cycle(epoch)

        seen_voynich = 0
        seen_replay = 0
        for source in schedule:
            if source == VOYNICH:
                idx = next(v_iter)
                tokens = self.voynich_ds[idx].unsqueeze(0)
                seen_voynich += 1
            else:
                idx = next(r_iter)
                tokens = self.replay_ds[idx].unsqueeze(0)
                seen_replay += 1
            yield {"tokens": tokens, "source": source}

        total = seen_voynich + seen_replay
        self._last_realized_ratio = (seen_replay / total) if total else 0.0
        self._epoch = epoch + 1


class MixedLoaderEpochCallback(L.Callback):
    """Syncs ``MixedDataLoader._epoch`` with Lightning's ``trainer.current_epoch``.

    The loader's internal epoch counter is used only as a fallback; this
    callback overrides it at the start of every training epoch so the
    schedule and shuffle seeds are tied to Lightning's notion of the epoch
    even if ``__iter__`` is called more than once per epoch.
    """

    def __init__(self, loader: "MixedDataLoader") -> None:
        self._loader = loader

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        self._loader.set_epoch(trainer.current_epoch)
