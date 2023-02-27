from typing import Type, List, Tuple

import pydantic
from classy.pl_modules.mixins.task_serve import MarshalInputSample, MarshalOutputSample

from src.data.data_drivers.base import ExmodSample


class ExmodMarshalInputSample(pydantic.BaseModel, MarshalInputSample):
    source_language: str
    D: List[Tuple[str, str]]
    target_language: str

    def unmarshal(self) -> ExmodSample:
        return ExmodSample(
            source_language=self.source_language,
            D=self.D,
            target_language=self.target_language,
        )


class ExmodMarshalOutputSample(ExmodMarshalInputSample, MarshalOutputSample):
    s: str
    phi: List[List[int]]

    @classmethod
    def marshal(cls, sample: ExmodSample):
        return cls(
            source_language=sample.source_language,
            D=sample.D,
            target_language=sample.target_language,
            s=sample.predicted_annotation[0][
                0
            ],  # todo first [0] is because exmod output can be a list
            phi=sample.predicted_annotation[0][1],
        )


class ExModServeMixin:
    @property
    def serve_input_class(self) -> Type[ExmodMarshalInputSample]:
        return ExmodMarshalInputSample

    @property
    def serve_output_class(self) -> Type[ExmodMarshalOutputSample]:
        return ExmodMarshalOutputSample
