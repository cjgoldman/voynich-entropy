"""Batch scheduler and mixed dataloader for replay fine-tuning."""

from __future__ import annotations

import math

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
        self._deterministic_schedule: list[str] | None = None

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
        # The deterministic-branch return is cached; callers must treat it as
        # read-only since the same list is reused across epochs.
        if self.is_deterministic:
            if self._deterministic_schedule is None:
                r_int = int(round(self.ratio))
                if r_int == 0:
                    self._deterministic_schedule = [VOYNICH] * self.num_voynich
                else:
                    block = [VOYNICH] + [REPLAY] * r_int
                    self._deterministic_schedule = block * self.num_voynich
            return self._deterministic_schedule

        tokens = [VOYNICH] * self.num_voynich + [REPLAY] * self.num_replay
        g = torch.Generator()
        g.manual_seed(self.seed + int(epoch))
        perm = torch.randperm(len(tokens), generator=g).tolist()
        return [tokens[i] for i in perm]


class MixedDataLoader:
    """Iterable that yields Voynich and replay batches per the scheduler.

    Each yielded batch is a ``{"tokens": LongTensor, "source": str}`` dict.
    Tokens carry a leading batch dimension of size 1, matching the basic
    fine-tune's ``batch_size=1`` convention. Replay indices are drawn from a
    single global permutation so every replay batch across the full run is
    unique; Voynich indices are still reshuffled each epoch.
    """

    def __init__(
        self,
        voynich_ds: Dataset,
        replay_ds: Dataset,
        scheduler: BatchScheduler,
        voynich_shuffle_seed: int,
        replay_shuffle_seed: int,
        total_epochs: int,
        pin_memory: bool = False,
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
        self._total_replay_steps = scheduler.num_replay * total_epochs
        if len(replay_ds) < self._total_replay_steps:
            raise ValueError(
                "replay dataset is too small for a globally-unique run: "
                f"have {len(replay_ds)} chunks but need "
                f"{self._total_replay_steps} "
                f"(num_voynich={scheduler.num_voynich}, "
                f"ratio={scheduler.ratio}, epochs={total_epochs})"
            )
        self.voynich_ds = voynich_ds
        self.replay_ds = replay_ds
        self.scheduler = scheduler
        self.voynich_shuffle_seed = voynich_shuffle_seed
        self.replay_shuffle_seed = replay_shuffle_seed
        self.total_epochs = total_epochs
        self._epoch = 0
        self._last_realized_ratio: float | None = None
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self._voynich_order_cache: dict[int, list[int]] = {}

        g = torch.Generator()
        g.manual_seed(replay_shuffle_seed)
        order = torch.randperm(len(replay_ds), generator=g).tolist()
        self._global_replay_order = order[: self._total_replay_steps]
        self._replay_cursor = 0

    def __len__(self) -> int:
        return self.scheduler.total_steps

    def set_epoch(self, epoch: int) -> None:
        """Externally override the epoch counter used for seeding.

        Only affects Voynich shuffling and non-integer schedule seeding; the
        replay cursor is intentionally not reset so replay indices stay
        globally unique across the whole run.
        """
        self._epoch = int(epoch)

    @property
    def last_realized_ratio(self) -> float | None:
        """Realized replay fraction from the most recently completed epoch."""
        return self._last_realized_ratio

    def _voynich_order(self, epoch: int) -> list[int]:
        cached = self._voynich_order_cache.get(epoch)
        if cached is not None:
            return cached
        g = torch.Generator()
        g.manual_seed(self.voynich_shuffle_seed + epoch)
        order = torch.randperm(len(self.voynich_ds), generator=g).tolist()
        self._voynich_order_cache[epoch] = order
        return order

    def __iter__(self):
        epoch = self._epoch
        schedule = self.scheduler.schedule(epoch)
        v_iter = iter(self._voynich_order(epoch))

        seen_voynich = 0
        seen_replay = 0
        for source in schedule:
            if source == VOYNICH:
                idx = next(v_iter)
                tokens = self.voynich_ds[idx].unsqueeze(0)
                seen_voynich += 1
            else:
                if self._replay_cursor >= len(self._global_replay_order):
                    raise RuntimeError(
                        "replay cursor exhausted the global permutation; "
                        "this should have been caught at __init__ time "
                        f"(cursor={self._replay_cursor}, "
                        f"total={len(self._global_replay_order)})"
                    )
                idx = self._global_replay_order[self._replay_cursor]
                self._replay_cursor += 1
                tokens = self.replay_ds[idx].unsqueeze(0)
                seen_replay += 1
            if self.pin_memory:
                tokens = tokens.pin_memory()
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
