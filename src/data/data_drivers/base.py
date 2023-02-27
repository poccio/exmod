import json
from typing import Iterator, Optional, Tuple, List

from classy.data.data_drivers import DataDriver, ClassySample
from classy.utils.log import get_project_logger

from src.utils.nlp import detokenize

logger = get_project_logger(__name__)


class ExmodSample(ClassySample):
    @classmethod
    def from_tokenized_s(
        cls,
        source_language: str,
        D: List[Tuple[str, str]],
        target_language: str,
        tokenized_s: List[str],
        idxs: List[int],
        **kwargs,
    ):

        # build s
        s = detokenize(" ".join(tokenized_s), language=source_language)

        # build token to char mapping
        try:
            token2char = {}
            offset, sliding_s = 0, s
            for _i, _t in enumerate(tokenized_s):
                try:
                    assert sliding_s.startswith(_t)
                except AssertionError:
                    _t = detokenize(_t, language=source_language)
                    assert sliding_s.startswith(_t)
                token2char[_i] = list(range(offset, offset + len(_t) + 1))
                new_sliding_s = sliding_s[len(_t) :].lstrip()
                offset += len(sliding_s) - len(new_sliding_s)
                sliding_s = new_sliding_s
        except Exception:
            raise ValueError(
                f"Token-char mapping failed on tokenized string: {tokenized_s}"
            )

        # build phi
        phi = [token2char[idx] for idx in idxs]

        return cls(
            source_language=source_language,
            D=D,
            target_language=target_language,
            s=s,
            phi=phi,
            **kwargs,
        )

    def __init__(
        self,
        source_language: str,
        D: List[Tuple[str, str]],
        target_language: str,
        s: Optional[str] = None,
        phi: Optional[List[List[int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_language = source_language
        self.D = D
        self.target_language = target_language
        self._reference_annotation = (s, phi) if s is not None else None
        self._predicted_annotation = None

    def _get_reference_annotation(self) -> Optional[Tuple[str, List[List[int]]]]:
        return self._reference_annotation

    def _update_reference_annotation(
        self, reference_annotation: Tuple[str, List[List[int]]]
    ):
        self._reference_annotation = reference_annotation

    def _get_predicted_annotation(
        self,
    ) -> Optional[Optional[List[Tuple[str, List[List[int]]]]]]:
        return self._predicted_annotation

    def _update_predicted_annotation(
        self, predicted_annotation: List[Tuple[str, List[List[int]]]]
    ):
        self._predicted_annotation = predicted_annotation

    def pretty_print(self) -> str:

        parts = [
            f"# source_language: {self.source_language}",
            f"# D: {self.D}",
        ]

        if self.reference_annotation is not None:
            parts += [
                f"\t# gold s: {self.reference_annotation[0]}",
                f"\t# gold phi: {self.reference_annotation[1]}",
            ]

        if self.predicted_annotation is not None:
            for _example in self.predicted_annotation:
                parts += [
                    f"\t- # predicted s: {_example[0]}",
                    f"\t  # predicted phi: {_example[1]}",
                ]

        return "\n".join(parts)

    @property
    def input(self) -> str:
        if self.source is not None:
            return f"[{self.source} {self.source_language} -> {self.target_language}] {str(self.D)}"
        else:
            return f"[{self.source_language} -> {self.target_language}] {str(self.D)}"


class ExmodDataDriver(DataDriver):
    def get_unique_sentences(self, path: str) -> List[str]:
        raise NotImplementedError


@DataDriver.register("generation", "exmj")
class ExmodJSONLDataDriver(ExmodDataDriver):
    def get_unique_sentences(self, path: str) -> List[str]:
        seen_sentences = set()
        for sample in self.read_from_path(path):
            assert sample.reference_annotation is not None
            if sample.reference_annotation[0] not in seen_sentences:
                yield sample.reference_annotation[0]
                seen_sentences.add(sample.reference_annotation[0])

    def read_from_path(self, path: str) -> Iterator[ExmodSample]:
        def r():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

        return self.read(r())

    def read(self, lines: Iterator[str]) -> Iterator[ExmodSample]:
        for line in lines:
            yield ExmodSample(**json.loads(line))

    def save(
        self,
        samples: Iterator[ExmodSample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        with open(path, "w") as f:
            for sample in samples:
                used_annotation = (
                    sample.predicted_annotation
                    if use_predicted_annotation
                    else [sample.reference_annotation]
                )
                d = {
                    "source_language": sample.source_language,
                    "D": sample.D,
                    "target_language": sample.target_language,
                    **sample.get_additional_attributes(),
                }
                for _used_annotation in used_annotation:
                    if use_predicted_annotation:
                        d["s"] = _used_annotation[0]
                        d["phi"] = _used_annotation[1]
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
