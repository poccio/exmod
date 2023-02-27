import collections
import itertools
import json
from typing import Iterator, Tuple, List, Callable, Iterable

import numpy as np
import torch
from classy.utils.log import get_project_logger

from src.utils.pmi import PMI
from src.utils.raganato import WSDInstance


logger = get_project_logger(__name__)


class Sampler:
    def __init__(
        self,
        samples_iterator: Callable[[], Iterable[Tuple[str, str, List[WSDInstance]]]],
    ):
        self.samples_iterator = samples_iterator
        self.fit()

    def fit(self):
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[List[Tuple[int, str, str]], List[str]]]:
        raise NotImplementedError

    @staticmethod
    def wsd_sentence_to_tokens(wsd_sentence) -> List[str]:
        return [wi.annotated_token.text for wi in wsd_sentence]


class IterativeSampler(Sampler):
    def fit(self):
        pass

    def __iter__(self) -> Iterator[Tuple[List[Tuple[int, str, str]], List[str]]]:
        pos_chain_skips_n, pos_chain_skips_d = 0, 0
        for sentence_idx, (_, _, sentence) in enumerate(self.samples_iterator()):
            # skip if POS structure is not ok
            pos_s = set([wsdi.annotated_token.pos for wsdi in sentence])
            pos_chain_skips_d += 1
            if ("NOUN" not in pos_s or "PRON" not in pos_s) or (
                "AUX" not in pos_s and "VERB" not in pos_s
            ):
                pos_chain_skips_n += 1
                continue
            # get labels
            labels = [
                (idx, np.random.choice(wsdi.labels), wsdi.annotated_token.lemma)
                for idx, wsdi in enumerate(sentence)
                if wsdi.labels is not None
            ]
            # yield
            for idx, label, lemma in labels:
                yield [(idx, label, lemma)], self.wsd_sentence_to_tokens(sentence)
        logger.warning(
            f"{pos_chain_skips_n} / {pos_chain_skips_d} sentences have been skipped due to POS chain"
        )
