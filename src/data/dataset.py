import numpy as np
from classy.data.data_drivers import GenerationSample

from classy.data.dataset.hf.generation import (
    BartHFGenerationDataset,
)
from classy.utils.commons import flatten


language_map = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
}


class ExModSeq2SeqMixin:
    def __init__(
        self,
        add_language: bool,
        add_task_description: bool,
        shuffle_D: bool = True,
        lemma_masking: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shuffle_D = shuffle_D
        self.add_language = add_language
        self.add_task_description = add_task_description
        self.lemma_masking = lemma_masking

    @property
    def samples_iterator(self):
        def closure():
            def get_mask_token():
                i = 0
                while True:
                    if self.tokenizer.name_or_path.startswith("google/mt5"):
                        yield f"<extra_id_{i}>"
                    else:
                        yield self.tokenizer.mask_token
                    i += 1

            for exmod_sample in self._samples_iterator():

                # create encoded input
                D = exmod_sample.D
                shuffle_permutation = None
                if self.shuffle_D:
                    shuffle_permutation = list(range(len(D)))
                    np.random.shuffle(shuffle_permutation)
                    D = [D[idx] for idx in shuffle_permutation]

                mask_token_it = get_mask_token()

                source_sequence = [
                    (
                        self.additional_special_tokens[i * 2],
                        f" {l}"
                        if (
                            self.for_inference
                            or np.random.random() > self.lemma_masking
                        )
                        else next(mask_token_it),
                        self.additional_special_tokens[i * 2 + 1],
                        f" {d.rstrip('.')}.",
                    )
                    for i, (l, d) in enumerate(D)
                ]
                source_sequence = "".join(flatten(source_sequence))

                if self.add_task_description:
                    source_sequence = f"generate example from {language_map[exmod_sample.source_language]} to {language_map[exmod_sample.target_language]}: {source_sequence}"

                # create encoded output
                target_sequence = None

                if exmod_sample.reference_annotation is not None:

                    s, phi = exmod_sample.reference_annotation
                    if shuffle_permutation is not None:
                        phi = [phi[idx] for idx in shuffle_permutation]

                    phi = [(idx, _phi) for idx, _phi in enumerate(phi)]
                    phi = sorted(phi, key=lambda x: x[1][1], reverse=True)
                    assert len(phi) > 0

                    upper_boundary = len(s)
                    target_sequence = []

                    for idx, _phi in phi:
                        _s, _e = min(_phi), max(_phi)
                        target_sequence.append(s[_e:upper_boundary])
                        target_sequence.append(
                            self.additional_special_tokens[idx * 2 + 1]
                        )
                        target_sequence.append(f" {s[_s: _e]}")
                        target_sequence.append(self.additional_special_tokens[idx * 2])
                        upper_boundary = _s

                    assert upper_boundary >= 0
                    if upper_boundary > 0:
                        target_sequence.append(s[:_s])

                    target_sequence = "".join(reversed(target_sequence))

                # yield
                generation_sample = GenerationSample(
                    source_sequence=source_sequence,
                    target_sequence=target_sequence,
                    source_language=exmod_sample.source_language
                    if self.add_language
                    else None,
                    target_language=exmod_sample.target_language
                    if self.add_language
                    else None,
                )
                generation_sample.exmod_sample = exmod_sample
                yield generation_sample

        return closure


class ExModSeq2SeqBartHFGenerationDataset(ExModSeq2SeqMixin, BartHFGenerationDataset):
    pass
