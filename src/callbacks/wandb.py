from typing import List

from classy.pl_callbacks.prediction import PredictionCallback
from classy.pl_modules.base import ClassyPLModule
from classy.utils.log import get_project_logger

from src.data.data_drivers.base import ExmodSample
from pytorch_lightning.loggers import WandbLogger as PLWandbLogger

import pytorch_lightning as pl

logger = get_project_logger(__name__)


class ExmodWANDBLoggerPredictionCallback(PredictionCallback):
    def __call__(
        self,
        name: str,
        path: str,
        predicted_samples: List[ExmodSample],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):
        if trainer.logger is None:
            logger.warning(
                "WANDBLoggerPredictionCallback has been included as a PredictionCallback, however it seems wandb is not being used (did you pass `--wandb [...]`?)"
            )
            return

        if not isinstance(trainer.logger, PLWandbLogger):
            logger.warning(
                "WANDBLoggerPredictionCallback has been included as a PredictionCallback, however trainer.logger does not seem to be a WandbLogger"
            )
            return

        columns = ["type", "language-pair", "input", "prediction"]
        data = []

        for predicted_sample in predicted_samples:
            data.append(
                [
                    predicted_sample.source,
                    f"{predicted_sample.source_language} => {predicted_sample.target_language}",
                    str(predicted_sample.D),
                    str(predicted_sample.predicted_annotation),
                ]
            )

        trainer.logger.log_text(
            key=f"{name}-predictions",
            columns=columns,
            data=data,
            step=trainer.global_step,
        )
