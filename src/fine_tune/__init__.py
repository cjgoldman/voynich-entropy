"""Basic and replay-based Voynich entropy model fine-tuning package.

GPU-dependent imports (VoynichEntropyFineTune, ReplayEntropyFineTune) are
deferred to avoid import errors in CPU-only environments.
"""

from fine_tune.config import FineTuneConfig
from fine_tune.dataset import VoynichEntropyDataset
from fine_tune.replay_config import ReplayFineTuneConfig
from fine_tune.replay_dataset import (DCLMReplayDataset,
                                      load_or_fetch_replay_pool)
from fine_tune.replay_loader import (BatchScheduler, MixedDataLoader,
                                     MixedLoaderEpochCallback)

__all__ = [
    "FineTuneConfig",
    "VoynichEntropyDataset",
    "VoynichEntropyFineTune",
    "ReplayFineTuneConfig",
    "DCLMReplayDataset",
    "load_or_fetch_replay_pool",
    "BatchScheduler",
    "MixedDataLoader",
    "MixedLoaderEpochCallback",
    "ReplayEntropyFineTune",
]


def __getattr__(name):
    if name == "VoynichEntropyFineTune":
        from fine_tune.module import VoynichEntropyFineTune

        return VoynichEntropyFineTune
    if name == "ReplayEntropyFineTune":
        from fine_tune.replay_module import ReplayEntropyFineTune

        return ReplayEntropyFineTune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
