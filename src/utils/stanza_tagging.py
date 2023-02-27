from typing import List, Tuple

import stanza
from dataclasses import dataclass


_wnpos2upos = {"a": "ADJ", "s": "ADJ", "n": "NOUN", "r": "ADV", "v": "VERB"}


def wn_pos_to_upos(wn_pos: str):
    return _wnpos2upos[wn_pos]


_pos2score = {
    "ADJ": 1,
    "ADP": 0,
    "ADV": 1,
    "CCONJ": 0,
    "DET": 0,
    "INTJ": 0,
    "NOUN": 3,
    "NUM": 0,
    "PART": 0,
    "PRON": 0,
    "PUNCT": 0,
    "SCONJ": 0,
    "SYM": 0,
    "VERB": 2,
    "X": 1,
}


def multipos_to_pos(pos_s: List[str]) -> str:
    transformations = {"AUX": "VERB", "PROPN": "NOUN"}
    pos_s = [transformations.get(pos, pos) for pos in pos_s]
    if len(pos_s) == 1 or len(set(pos_s)) == 1:
        return pos_s[0]
    else:
        if "NOUN" in pos_s:
            return "NOUN"
        elif "VERB" in pos_s:
            return "VERB"
        else:
            return max(pos_s, key=lambda x: _pos2score[x])


@dataclass
class Token:
    text: str
    pos: str
    lemma: str
    start_char: int
    end_char: int


@dataclass
class Sentence:
    text: str
    tokens: List[Token]


def get_stanza_pipeline(
    language: str,
    use_gpu: bool = False,
    sentence_split: bool = True,
    tokenize: bool = True,
):
    try:
        return stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,lemma"
            if language in ["ja"]
            else "tokenize,mwt,pos,lemma",
            tokenize_no_ssplit=not sentence_split,
            tokenize_pretokenized=not tokenize,
            use_gpu=use_gpu,
        )
    except Exception:
        stanza.download(language)
        return stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,lemma"
            if language in ["ja"]
            else "tokenize,mwt,pos,lemma",
            tokenize_no_ssplit=not sentence_split,
            tokenize_pretokenized=not tokenize,
            use_gpu=use_gpu,
        )


def tag_sentences(sentences: List[str], stanza_pipeline) -> List[Sentence]:
    def merge_words(token, start_offset):
        if len(token.words) == 1:
            word = token.words[0]
            return Token(
                word.text,
                word.upos,
                word.lemma if word.lemma is not None else word.text,
                start_char=token.start_char - start_offset,
                end_char=token.end_char - start_offset,
            )
        else:
            return Token(
                token.text,
                multipos_to_pos([w.upos for w in token.words]),
                token.text if token.lemma is not None else token.text,
                start_char=token.start_char - start_offset,
                end_char=token.end_char - start_offset,
            )

    # tag
    doc = stanza_pipeline("\n\n".join(sentences))

    # extract
    sentences = []
    for sentence in doc.sentences:
        start_offset = sentence.tokens[0].start_char if len(sentence.tokens) > 0 else 0
        sentences.append(
            Sentence(
                text=sentence.text,
                tokens=[
                    merge_words(token, start_offset=start_offset)
                    for token in sentence.tokens
                ],
            )
        )

    # return
    return sentences
