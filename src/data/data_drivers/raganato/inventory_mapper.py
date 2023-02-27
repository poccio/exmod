import json
from typing import Dict, Optional

import numpy as np

from src.utils.wordnet import wn_offset_from_sense_key


class InventoryMapper:
    def __init__(
        self,
        language_probability: Optional[Dict[str, float]] = None,
    ):
        if language_probability is not None:
            self.languages, self.languages_p = [], []
            for k, v in language_probability.items():
                self.languages.append(k)
                self.languages_p.append(v)
            # normalize probability
            self.languages_p = [p / sum(self.languages_p) for p in self.languages_p]
        else:
            self.languages, self.languages_p = None, None

    def get_language(self):
        if self.languages is None:
            raise ValueError
        return np.random.choice(self.languages, p=self.languages_p)

    def get_lemma(self, language: str, label: str):
        raise NotImplementedError

    def get_lemmas(self, language: str, label: str):
        raise NotImplementedError

    def get_definition(self, language: str, label: str):
        raise NotImplementedError


class EnglishWordNetInventoryMapper(InventoryMapper):

    _cache = {
        "lemmas": {},
        "definitions": {},
    }

    def __init__(
        self,
        lemmas: str,
        definitions: str,
        language_probability: Optional[Dict[str, float]] = None,
        input_is_sense_key: bool = True,
    ):
        super().__init__(language_probability)
        self.input_is_sense_key = input_is_sense_key

        if lemmas not in self._cache["lemmas"]:
            with open(lemmas) as f:
                self._cache["lemmas"][lemmas] = json.load(f)
        self.lemmas = self._cache["lemmas"][lemmas]

        if definitions not in self._cache["definitions"]:
            with open(definitions) as f:
                self._cache["definitions"][definitions] = json.load(f)
        self.definitions = self._cache["definitions"][definitions]

    def get_language(self):
        if self.languages is None:
            raise ValueError
        return np.random.choice(self.languages, p=self.languages_p)

    def get_lemma(self, language: str, label: str):
        if self.input_is_sense_key:
            label = wn_offset_from_sense_key(label)
        return np.random.choice(self.lemmas[label][language])

    def get_lemmas(self, language: str, label: str):
        if self.input_is_sense_key:
            label = wn_offset_from_sense_key(label)
        return self.lemmas[label][language]

    def get_definition(self, language: str, label: str):
        if self.input_is_sense_key:
            label = wn_offset_from_sense_key(label)
        return self.definitions[label][language]
