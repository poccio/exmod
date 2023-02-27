import functools
from typing import Tuple

from sacremoses import MosesDetokenizer, MosesTokenizer


@functools.lru_cache(maxsize=1_000)
def get_tok_detok_pair(language: str) -> Tuple:
    return MosesTokenizer(lang=language), MosesDetokenizer(lang=language)


def detokenize(sequence: str, language: str = "en") -> str:
    sequence = sequence.split()
    sequence = get_tok_detok_pair(language)[1].detokenize(
        sequence, return_str=True, unescape=False
    )
    return sequence.strip()
