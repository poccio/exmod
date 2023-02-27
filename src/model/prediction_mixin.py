import collections
import copy
import json
from typing import Iterator, Dict, Union, List, Tuple, Optional

from classy.data.data_drivers import ClassySample, get_data_driver
from classy.utils.log import get_project_logger
from omegaconf import DictConfig

from src.data.data_drivers.base import ExmodSample
from src.utils.stanza_tagging import get_stanza_pipeline, tag_sentences

logger = get_project_logger(__name__)


class ExModSeq2SeqPredictionMixin:

    __data_driver = get_data_driver("generation", "exmj")

    def __init__(
        self, additional_special_tokens: Optional[List[str]] = None, *args, **kwargs
    ):
        super().__init__(
            additional_special_tokens=additional_special_tokens, *args, **kwargs
        )
        self.additional_special_tokens = (
            additional_special_tokens  # todo move in classy
        )
        self.exemplification_params = {}
        self.stanza_cache = {}

    def read_input_from_bash(self) -> ExmodSample:

        source_language = input("Enter source_language: ").strip()
        target_language = input("Enter target_language: ").strip()
        D = []

        while True:
            lemma = input("Enter lemma: ").strip()
            if lemma == "":
                break
            definition = input("Enter definition: ").strip()
            D.append((lemma, definition))

        sample = json.dumps(
            dict(source_language=source_language, D=D, target_language=target_language)
        )
        return next(self.__data_driver.read([sample]))

    def load_prediction_params(self, prediction_params: Dict):
        self.generation_params = prediction_params["generation_params"]
        self.exemplification_params = prediction_params["exemplification_params"]

    def postprocess(
        self,
        D: List[Tuple[str, str]],
        generated_text: str,
    ):

        # strip surrounding special tokens
        if self.model.name_or_path == "facebook/bart-large":
            generated_text = generated_text[generated_text.rindex("<s>") + 3 :].lstrip()
        elif self.model.name_or_path == "facebook/mbart-large-cc25":
            generated_text = generated_text[5:].lstrip()
        elif self.model.name_or_path.startswith("google/mt5"):
            generated_text = generated_text[5:].lstrip()
        else:
            raise ValueError

        if generated_text.endswith("<pad>"):
            generated_text = generated_text[: generated_text.index("<pad>")]
        if generated_text.endswith("</s>"):
            generated_text = generated_text[:-4]

        # cleanup tokenization space
        generated_text = self.tokenizer.clean_up_tokenization(generated_text)

        # compute special tokens boundaries
        boundaries = []
        for i in range(len(D)):
            try:
                start = generated_text.index(self.additional_special_tokens[i * 2])
                end = generated_text.index(
                    self.additional_special_tokens[i * 2 + 1]
                ) + len(self.additional_special_tokens[i * 2 + 1])
                assert start + len(
                    self.additional_special_tokens[i * 2]
                ) + 1 < end - len(self.additional_special_tokens[i * 2 + 1])
                assert all((start >= _e or end <= _s) for _, _s, _e in boundaries)
                boundaries.append((i, start, end))
            except Exception as e:
                logger.warning(
                    f'Failed markup parsing of {i}-th lemma-def {D[i]} in generated text "{generated_text}" due to {e}'
                )

        # sort boundaries
        boundaries = sorted(boundaries, key=lambda x: x[1])

        # build s and phi
        s, phi = "", {}
        lower_boundary = 0
        for _idx, _s, _e in boundaries:
            # extract left and materialization
            _left = generated_text[lower_boundary:_s].rstrip(" ") + " "
            _materialization = generated_text[
                _s
                + len(self.additional_special_tokens[i * 2])
                + 1 : _e
                - len(self.additional_special_tokens[i * 2 + 1])
            ]
            # update s and phi
            phi[_idx] = (
                len(s) + len(_left),
                len(s) + len(_left) + len(_materialization),
            )
            s += _left + _materialization
            # update lower_boundary
            lower_boundary = _e

        # discard possible starting space
        if s.startswith(" "):
            s = s[1:]
            phi = {i: (s - 1, e - 1) for i, (s, e) in phi.items()}

        # add remaining sequence
        if lower_boundary != len(generated_text):
            s += generated_text[lower_boundary:]

        # build actual phi
        phi = [
            list(range(*phi.get(idx))) if idx in phi else None for idx in range(len(D))
        ]

        # return
        return s, phi

    def batch_predict(self, *args, **kwargs) -> Iterator[ExmodSample]:
        # run underlying seq2seq
        sample_it = super().batch_predict(*args, **kwargs)
        # postprocess and yield
        for sample in sample_it:
            exmod_sample = sample.exmod_sample
            exmod_sample.raw_predicted_annotation = (
                sample.predicted_annotation_group or [sample.predicted_annotation]
            )
            exmod_sample.predicted_annotation = [
                self.postprocess(exmod_sample.D, p)
                for p in exmod_sample.raw_predicted_annotation
            ]
            yield exmod_sample

    def predict(
        self,
        samples: Iterator[ClassySample],
        dataset_conf: Union[Dict, DictConfig],
        token_batch_size: int = 1024,
        progress_bar: bool = False,
        **kwargs,
    ) -> Iterator[ExmodSample]:

        # set prebatching to false for simplicity
        dataset_conf["prebatch"] = False

        # expand exmod samples
        # this method allows for future implementantions to change batch_predict (hierarchial, retrieval-based, ...)
        # but leave predict unchanged
        def samples_it():
            # todo (perhaps k1 -> 3 with k2)
            for sample in samples:
                _sample = copy.deepcopy(sample)
                _sample._original_exmod_sample = sample
                yield _sample

        # do predict
        predicted_samples = super().predict(
            samples_it(),
            dataset_conf,
            token_batch_size,
            progress_bar=progress_bar,
            **kwargs,
        )

        # filter examples with None phi

        def filter_it():
            for sample in predicted_samples:
                sample.predicted_annotation = [
                    (s, phi)
                    for s, phi in sample.predicted_annotation
                    if all(_phi is not None for _phi in phi)
                ]
                yield sample

        # pos tag

        def pos_tag_it():
            language_samples = collections.defaultdict(list)
            for sample in filter_it():
                language_samples[sample.target_language].append(sample)
                if len(language_samples[sample.target_language]) == 1_000:
                    # check stanza model is available
                    if sample.target_language not in self.stanza_cache:
                        self.stanza_cache[sample.target_language] = get_stanza_pipeline(
                            sample.target_language,
                            use_gpu=self.exemplification_params["tagging_device"] >= 0,
                            sentence_split=False,
                            tokenize=True,
                        )
                    # tag sentences
                    _stanzas = tag_sentences(
                        sentences=[
                            _e[0]
                            for _s in language_samples[sample.target_language]
                            for _e in _s.predicted_annotation
                        ],
                        stanza_pipeline=self.stanza_cache[sample.target_language],
                    )
                    # assign stanza structures
                    i = 0
                    for _sample in language_samples[sample.target_language]:
                        _sample.stanza = _stanzas[
                            i : i + len(_sample.predicted_annotation)
                        ]
                        i += len(_sample.predicted_annotation)
                        yield _sample
                    # reset
                    language_samples[sample.target_language] = []

            # checkout any remaining ones
            for language, _samples in language_samples.items():
                if len(_samples) > 0:
                    # check stanza model is available
                    if language not in self.stanza_cache:
                        self.stanza_cache[language] = get_stanza_pipeline(
                            language,
                            use_gpu=self.exemplification_params["tagging_device"] >= 0,
                            sentence_split=False,
                            tokenize=True,
                        )
                    # tag sentences
                    _stanzas = tag_sentences(
                        sentences=[
                            _e[0] for _s in _samples for _e in _s.predicted_annotation
                        ],
                        stanza_pipeline=self.stanza_cache[language],
                    )
                    # assign stanza structures
                    i = 0
                    for _sample in _samples:
                        _sample.stanza = _stanzas[
                            i : i + len(_sample.predicted_annotation)
                        ]
                        i += len(_sample.predicted_annotation)
                        yield _sample

        # backmap

        def grouped_predicted_samples_it():

            last_sample, grouped_samples = None, []
            it = pos_tag_it()

            while True:
                try:
                    predicted_sample = next(it)
                except StopIteration:
                    break

                if (
                    last_sample is None
                    or predicted_sample._original_exmod_sample != last_sample
                ):

                    if last_sample is not None:
                        last_sample.predicted_annotation = grouped_samples
                        yield last_sample

                    # reset
                    grouped_samples = []
                    last_sample = predicted_sample._original_exmod_sample

                grouped_samples.append(predicted_sample)

            if last_sample is not None:
                last_sample.predicted_annotation = grouped_samples
                yield last_sample

        # side-effect and yield

        for sample in grouped_predicted_samples_it():
            sample.grouped_predicted_annotation = sample.predicted_annotation
            sample.predicted_annotation = [
                _example
                for _sample in sample.predicted_annotation
                for _example in _sample.predicted_annotation
            ]

            # yield
            yield sample
