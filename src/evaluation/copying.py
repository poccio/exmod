import json
from typing import List, Dict, Tuple, Union

import faiss
import numpy as np
import pytorch_lightning as pl
from classy.evaluation.base import Evaluation
from classy.pl_callbacks.prediction import PredictionCallback
from classy.pl_modules.base import ClassyPLModule
from classy.utils.commons import chunks
from classy.utils.log import get_project_logger
from pytorch_lightning.loggers import WandbLogger as PLWandbLogger
from sentence_transformers import SentenceTransformer

from src.data.data_drivers.base import ExmodSample

logger = get_project_logger(__name__)


class CopyingEvaluation(Evaluation):
    def __init__(
        self, sentence_transformer: str, index_path: str, batch_size: int = 16
    ):
        # load sentences
        index_path = f"{index_path}/copying.index"
        with open(f"{index_path}/sentences.json") as f:
            self.sentences = json.load(f)
        # load faiss
        self.faiss_index = faiss.read_index(f"{index_path}/index.faiss")
        self.model = SentenceTransformer(sentence_transformer)
        self.batch_size = batch_size

    def __call__(
        self,
        path: str,
        predicted_samples: List[ExmodSample],
        return_retrieved_sentences: bool = False,
    ) -> Union[Dict, Tuple[Dict, List[str]]]:
        distances, retrieved_sentences = [], []
        for predicted_samples_group in chunks(predicted_samples, self.batch_size):
            _sentences = [
                example[0]
                for sample in predicted_samples_group
                for example in sample.predicted_annotation
            ]
            if len(_sentences) == 0:
                continue
            _queries = self.model.encode(_sentences)
            _queries = _queries / np.expand_dims(np.linalg.norm(_queries, axis=1), 1)
            faiss_search = self.faiss_index.search(_queries, k=1)
            _distances, _retrieved_sentences = faiss_search[0][:, 0], [
                self.sentences[_rsi[0]] for _rsi in faiss_search[1]
            ]
            distances = np.append(distances, _distances)
            retrieved_sentences += _retrieved_sentences
        return (
            ({"distance": np.mean(distances)}, retrieved_sentences)
            if return_retrieved_sentences
            else {"distance": np.mean(distances)}
        )


class CopyingPredictionCallback(PredictionCallback):
    def __init__(
        self, evaluation: CopyingEvaluation, log_sentences_on_wandb: bool = False
    ):
        self.evaluation = evaluation
        self.log_sentences_on_wandb = log_sentences_on_wandb

    def __call__(
        self,
        name: str,
        path: str,
        predicted_samples: List[ExmodSample],
        model: ClassyPLModule,
        trainer: pl.Trainer,
    ):

        # evaluate and log evaluation
        logger.info(f"Starting evaluation {self.__class__.__name__} with name {name}")
        copying_results, copied_sentences = self.evaluation(
            path, predicted_samples, return_retrieved_sentences=True
        )
        for k, v in copying_results.items():
            model.log(f"{name}_{k}", v, prog_bar=True, on_step=False, on_epoch=True)
        str_results = ", ".join([f"{k}={v}" for k, v in copying_results.items()])
        logger.info(
            f"Evaluation {self.__class__.__name__} with name {name} completed with results: ({str_results})"
        )

        # log data on wandb
        if self.log_sentences_on_wandb:
            if trainer.logger is None or not isinstance(trainer.logger, PLWandbLogger):
                logger.warning(
                    "CopyingPredictionCallback has been included as a PredictionCallback with log_sentences_on_wandb=True, however it seems wandb is not being used (did you pass `--wandb [...]`?)"
                )
                return

            columns = ["input", "generated-example", "closest-copy"]
            data = []

            for predicted_sample, closest_copy in zip(
                predicted_samples, copied_sentences
            ):
                data.append(
                    [
                        str(predicted_sample.input),
                        str(predicted_sample.predicted_annotation),
                        str(closest_copy),
                    ]
                )

            trainer.logger.log_text(
                key=f"{name}-copying",
                columns=columns,
                data=data,
                step=trainer.global_step,
            )
