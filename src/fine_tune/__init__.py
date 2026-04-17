"""Basic Voynich entropy model fine-tuning package.

GPU-dependent imports (VoynichEntropyFineTune) are deferred to avoid
import errors in CPU-only environments.
"""

from fine_tune.config import FineTuneConfig
from fine_tune.dataset import VoynichEntropyDataset

__all__ = ["FineTuneConfig", "VoynichEntropyDataset", "VoynichEntropyFineTune"]


def __getattr__(name):
    if name == "VoynichEntropyFineTune":
        from fine_tune.module import VoynichEntropyFineTune

        return VoynichEntropyFineTune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
