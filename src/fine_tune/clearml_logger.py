"""PyTorch Lightning logger that routes scalars to a ClearML Task."""

from __future__ import annotations

from typing import Mapping, Optional

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class ClearMLLogger(Logger):
    """Minimal Lightning logger backed by a pre-initialized ClearML Task.

    Scalar keys of the form ``"train/loss"`` are split on ``"/"`` into
    ``title`` (chart) and ``series`` (line) so ClearML groups related metrics
    on a single plot.
    """

    def __init__(self, task) -> None:
        super().__init__()
        self._task = task

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def version(self) -> str:
        return self._task.id

    @property
    def experiment(self):
        return self._task.get_logger()

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs) -> None:
        if hasattr(params, "items"):
            params_dict = dict(params)
        else:
            params_dict = vars(params)
        self._task.connect(params_dict, name="lightning_hyperparams")

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        logger = self._task.get_logger()
        iteration = int(step) if step is not None else 0
        for key, value in metrics.items():
            if value is None:
                continue
            if "/" in key:
                title, series = key.split("/", 1)
            else:
                title, series = key, key
            logger.report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=iteration,
            )

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._task.get_logger().flush()
