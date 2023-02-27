from typing import Optional

import torch
from classy.pl_modules.base import ClassificationOutput
from classy.pl_modules.hf.generation import (
    BartGenerativeModule,
    MBartGenerativeModule,
    T5GenerativeModule,
)
from classy.utils.log import get_project_logger
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM, AutoConfig

from src.model.prediction_mixin import ExModSeq2SeqPredictionMixin
from src.model.serve_mixin import ExModServeMixin

logger = get_project_logger(__name__)


class ExModSeq2SeqBartGenerativeModule(
    ExModServeMixin, ExModSeq2SeqPredictionMixin, BartGenerativeModule
):
    pass
