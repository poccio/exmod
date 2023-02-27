import functools
from typing import List, Tuple, Optional

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

patching_data = {
    "ddc%1:06:01::": "dideoxycytosine.n.01.DDC",
    "ddi%1:06:01::": "dideoxyinosine.n.01.DDI",
    "earth%1:15:01::": "earth.n.04.earth",
    "earth%1:17:02::": "earth.n.01.earth",
    "moon%1:17:03::": "moon.n.01.moon",
    "sun%1:17:02::": "sun.n.01.Sun",
    "kb%1:23:01::": "kilobyte.n.02.kB",
    "kb%1:23:03::": "kilobyte.n.01.kB",
}


@functools.lru_cache(maxsize=500_000)
def wn_sense_keys_from_lemmapos(lemma: str, pos: str) -> List[str]:
    return [l.key() for l in wn.lemmas(lemma, pos)]


@functools.lru_cache(maxsize=500_000)
def wn_gloss_from_sense_key(sense_key: str) -> str:
    if sense_key in patching_data:
        return wn.lemma(patching_data[sense_key]).synset().definition()
    else:
        return wn.lemma_from_key(sense_key).synset().definition()


@functools.lru_cache(maxsize=500_000)
def wn_gloss_from_offset(offset: str) -> str:
    offset, pos = int(offset[:-1]), offset[-1]
    return wn.synset_from_pos_and_offset(pos, offset).definition()


@functools.lru_cache(maxsize=500_000)
def wn_lemma_from_sense_key(sense_key: str) -> str:
    if sense_key in patching_data:
        return wn.lemma(patching_data[sense_key]).name()
    else:
        return wn.lemma_from_key(sense_key).name()


@functools.lru_cache(maxsize=500_000)
def wn_pos_from_sense_key(sense_key: str) -> str:
    if sense_key in patching_data:
        return wn.lemma(patching_data[sense_key]).synset().pos()
    else:
        return wn.lemma_from_key(sense_key).synset().pos()


def wn_offset_from_synset(synset: Synset) -> str:
    return str(synset.offset()).zfill(8) + synset.pos()


@functools.lru_cache(maxsize=500_000)
def wn_offset_from_sense_key(sense_key: str) -> str:
    if sense_key in patching_data:
        synset = wn.lemma(patching_data[sense_key]).synset()
    else:
        synset = wn.lemma_from_key(sense_key).synset()
    return wn_offset_from_synset(synset)


@functools.lru_cache(maxsize=500_000)
def wn_sense_key_from_offset(offset: str, lemma: Optional[str] = None) -> str:
    offset, pos = int(offset[:-1]), offset[-1]
    lemmas = wn.synset_from_pos_and_offset(pos, offset).lemmas()
    if lemma is not None:
        # define editing functions over target lemma and inventory lemmas
        editing_functions = [
            (lambda l: l, lambda l: l),
            (lambda l: l.lower(), lambda l: l),
            (lambda l: l.upper(), lambda l: l),
            (lambda l: l.lower(), lambda l: l.lower()),
        ]
        # try to find correct lemma
        search = None
        for f, g in editing_functions:
            _lemmas = [l for l in lemmas if f(lemma) == g(l.name())]
            if len(_lemmas) == 1:
                search = _lemmas
                break
        assert search is not None
    return lemmas[0].key()


@functools.lru_cache(maxsize=500_000)
def wn_offset_and_sensekey_from_lemma_definition(
    lemma: str, definition: str
) -> Tuple[str, str]:
    candidates = []
    for l in wn.lemmas(lemma):
        if l.synset().definition().startswith(definition):
            candidates.append((wn_offset_from_synset(l.synset()), l.key()))
    assert len(candidates) == 1, f"f({lemma}, {definition}) ==> {candidates}"
    return candidates[0]
