import collections
import os
from builtins import enumerate
from pathlib import Path
from typing import Iterator, List

import hydra
import numpy as np
from classy.data.data_drivers import (
    DataDriver,
    GenerationSample,
)
from classy.utils.hydra import fix_paths
from classy.utils.log import get_project_logger
from omegaconf import ListConfig, OmegaConf

from src.data.data_drivers.base import ExmodSample, ExmodDataDriver
from src.utils.raganato import read_from_raganato, expand_raganato_path

logger = get_project_logger(__name__)


@DataDriver.register("generation", "raganato")
class RaganatoDataDriver(ExmodDataDriver):
    def get_unique_sentences(self, path: str) -> List[str]:
        seen_sentences = set()
        config = self._read(path)
        for _c in config:
            for _, _, wsds in read_from_raganato(
                *expand_raganato_path(_c.raganato_path)
            ):
                sentence = " ".join(wsdi.annotated_token.text for wsdi in wsds)
                if sentence not in seen_sentences:
                    yield sentence
                    seen_sentences.add(sentence)

    def dataset_exists_at_path(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        try:
            self._read(path)
            return True
        except Exception:
            return False

    def _read(self, path: str) -> ListConfig:
        config = fix_paths(
            OmegaConf.load(path),
            check_fn=lambda path: os.path.exists(
                hydra.utils.to_absolute_path(path[: path.rindex("/")])
            ),
            fix_fn=lambda path: hydra.utils.to_absolute_path(path),
        )
        assert all(
            all(Path(p).exists() for p in expand_raganato_path(_config.raganato_path))
            for _config in config
        )
        return config

    def read_from_path(self, path: str) -> Iterator[GenerationSample]:

        config = self._read(path)

        # build iterating structures

        dataset_p = []
        inventory_mappers, examples_languages = [], []
        samplers, samplers_p = [], []
        its, done = [], []

        def fn_samples_iterator(c):
            return lambda: read_from_raganato(*expand_raganato_path(c.raganato_path))

        for _c in config:
            dataset_p.append(_c.pop("p"))
            inventory_mappers.append(
                hydra.utils.instantiate(_c.pop("inventory_mapper"))
            )
            examples_languages.append(_c.examples_language)
            samplers_p.append([_s.pop("p") for _s in _c.samplers])
            samplers.append(
                [
                    hydra.utils.instantiate(
                        _s, samples_iterator=fn_samples_iterator(_c)
                    )
                    for _s in _c.samplers
                ]
            )
            its.append([iter(sampler) for sampler in samplers[-1]])
            done.append([False for _ in samplers[-1]])

        # normalize probabilities
        dataset_p = [p / sum(dataset_p) for p in dataset_p]
        samplers_p = [[p / sum(ps) for p in ps] for ps in samplers_p]

        n_skipped_samples = 0
        n_samples = 0
        unique_sentences_yielded = collections.defaultdict(
            lambda: collections.Counter()
        )

        while True:

            if all(all(_d) for _d in done):
                break

            i = np.random.choice(len(samplers), p=dataset_p)
            j = np.random.choice(len(samplers[i]), p=samplers_p[i])

            try:

                located_labels, tokenized_s = next(its[i][j])
                located_labels = [
                    (idx, label, lemma) for idx, label, lemma in located_labels
                ]
                should_skip = False

                # compute source language
                source_language = inventory_mappers[i].get_language()

                # if language "changed", update lemma
                if source_language != examples_languages[i]:
                    _located_labels = []
                    for idx, label, lemma in located_labels:
                        try:
                            _located_labels.append(
                                (
                                    idx,
                                    label,
                                    inventory_mappers[i].get_lemma(
                                        language=source_language, label=label
                                    ),
                                )
                            )
                        except KeyError:
                            should_skip = True
                            continue
                    located_labels = _located_labels

                # compute definitions
                located_definitions = []
                for idx, label, lemma in located_labels:
                    try:
                        definition = inventory_mappers[i].get_definition(
                            language=source_language, label=label
                        )
                    except KeyError:
                        should_skip = True
                        continue
                    located_definitions.append((idx, definition, lemma))

                # skip if empty
                n_samples += 1
                if should_skip:
                    n_skipped_samples += 1
                    continue
                else:
                    unique_sentences_yielded[examples_languages[i]][
                        tuple(tokenized_s)
                    ] += 1

                # yield
                try:
                    yield ExmodSample.from_tokenized_s(
                        source_language=source_language,
                        target_language=examples_languages[i],
                        D=[
                            (lemma, definition)
                            for _, definition, lemma in located_definitions
                        ],
                        tokenized_s=tokenized_s,
                        idxs=[idx for idx, _, _ in located_definitions],
                    )
                except ValueError:
                    logger.warning(
                        f"Token-char mapping failed on tokenized string: {tokenized_s}"
                    )
                    continue

            except StopIteration:
                logger.info(f"Sampler [{i}][{j}] completed. Resetting it.")
                done[i][j] = True
                its[i][j] = iter(samplers[i][j])

        logger.warning(f"{n_skipped_samples} / {n_samples} samples have been skipped")

        for k, v in unique_sentences_yielded.items():
            percentiles = [10, 25, 50, 75, 90, 95, 98, 99, 99.9, 99.99, 100]
            percentiles_v = np.percentile(list(v.values()), q=percentiles)
            logger.info(
                f"[{k}] {len(v)} unique sentences yielded; "
                + ", ".join(
                    [f"{_p}: {_pv:.2f}" for _p, _pv in zip(percentiles, percentiles_v)]
                )
            )

    def save(
        self,
        samples: Iterator[ExmodSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError
