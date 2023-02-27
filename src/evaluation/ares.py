from typing import Dict, List, Tuple, Iterable

import torch
from classy.evaluation.base import Evaluation
from classy.utils.commons import chunks, flatten
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from src.data.data_drivers.base import ExmodSample
from src.utils.babelnet import wn_id2bn_id
from src.utils.stanza_tagging import get_stanza_pipeline
from src.utils.wordnet import wn_offset_from_sense_key


class AresBertEncoder:
    def __init__(self, bert_transformer_model: str, device: int, batch_size: int = 16):
        self.device = torch.device(device if device >= 0 else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_transformer_model, fast=True
        )
        self.model = AutoModel.from_pretrained(
            bert_transformer_model, output_hidden_states=True
        )
        self.model.eval()
        self.model.to(self.device)
        self.batch_size = batch_size

    def __encode(self, input_ids: List[torch.Tensor]) -> Iterable[torch.Tensor]:
        for batch_input_ids in chunks(input_ids, self.batch_size):
            batch = pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            attention_mask = torch.ones_like(batch)
            attention_mask[batch == self.tokenizer.pad_token_id] = 0
            with torch.no_grad():
                embedded_sentences = self.model(batch, attention_mask=attention_mask)[
                    2
                ][-4:]
                embedded_sentences = torch.sum(
                    torch.stack(embedded_sentences), dim=0
                )  # sum?
            yield from [x.cpu() for x in embedded_sentences]

    # return embedded sentences and a mapping token -> bpes
    def encode(
        self, sentences: List[str], phis: List[List[List[int]]]
    ) -> List[List[torch.Tensor]]:

        # tokenize sentences

        input_ids = []
        token_phis = []

        for sentence, phi in zip(sentences, phis):

            # map phi to bpe
            tokenization_out = self.tokenizer(sentence, return_tensors="pt")
            token_phi = []
            for _phi in phi:
                _token_phi = []
                for _c_idx in _phi:
                    _t_idx = tokenization_out.char_to_token(_c_idx)
                    if _t_idx is not None and (
                        len(_token_phi) == 0 or _token_phi[-1] != _t_idx
                    ):
                        _token_phi.append(_t_idx)
                token_phi.append((min(_token_phi), max(_token_phi)))
            token_phis.append(token_phi)

            # add input ids
            input_ids.append(tokenization_out.input_ids.squeeze())

        # encode

        encoded_materializations = []

        for encoded_sentence, token_phi in zip(self.__encode(input_ids), token_phis):
            _encoded_materializations = []
            encoded_materializations.append(_encoded_materializations)
            for _token_phi in token_phi:
                start, end = _token_phi
                _encoded_materializations.append(
                    torch.mean(encoded_sentence[start : end + 1], dim=0)
                )

        # return
        return encoded_materializations


class AresMeasure:
    def __init__(
        self,
        bert_transformer_model: str,
        ares_embedding_path: str,
        key_type: str,
        device: int = -1,
        batch_size: int = 16,
    ):
        self.device = device
        self.ares_embedding: Dict[str, torch.Tensor] = dict()
        self.key_type = key_type
        assert self.key_type in ["sense_key", "babelnet"]
        self.stanza = {}
        self.bert = AresBertEncoder(
            bert_transformer_model, device, batch_size=batch_size
        )
        self._load_vectors(ares_embedding_path)

    def get_stanza(self, language: str):
        if language not in self.stanza:
            self.stanza[language] = get_stanza_pipeline(
                language, self.device >= 0, sentence_split=False, tokenize=False
            )
        return self.stanza[language]

    def _load_vectors(self, embedding_path: str) -> None:
        embedding_file = open(embedding_path)
        next(embedding_file)
        for line in embedding_file:
            word, *vector = line.strip().split(" ")
            vector = [float(v) for v in vector]
            vector = torch.tensor(vector)
            self.ares_embedding[word] = vector

    def embedding_from_senses(self, sense_keys: List[str]) -> torch.Tensor:
        return torch.stack([self.ares_embedding[sk] for sk in sense_keys])

    def matrix_cos_sim(
        self, matrix1: torch.Tensor, matrix2: torch.Tensor
    ) -> List[float]:
        matrix_bit_prod = (matrix1 / torch.norm(matrix1, dim=-1).unsqueeze(dim=-1)) * (
            matrix2 / torch.norm(matrix2, dim=-1).unsqueeze(dim=-1)
        )
        matrix_bit_prod = torch.sum(matrix_bit_prod, dim=-1)
        result = matrix_bit_prod.squeeze().tolist()
        if type(result) != list:
            # bug fix that can happen due to pytorch: torch.tensor(0).tolist() is a float
            result = [result]
        return result

    def ares_distance(self, exmod_samples: List[ExmodSample]) -> Tuple[float, float]:

        # prepare input
        sentences, phis = [], []
        senses = []
        for exmod_sample in exmod_samples:
            for _example in exmod_sample.predicted_annotation:
                if all(_phi is not None and len(_phi) > 0 for _phi in _example):
                    sentences.append(_example[0])
                    phis.append(_example[1])
                    for sense_key in exmod_sample.original_sense_keys:
                        if self.key_type == "sense_key":
                            senses.append(sense_key)
                        elif self.key_type == "babelnet":
                            senses.append(
                                wn_id2bn_id(f"wn:{wn_offset_from_sense_key(sense_key)}")
                            )

        # guard if
        if len(sentences) == 0:
            return 0.0, 0.0

        # compute embeddings of materializations
        encoded_materializations = self.bert.encode(sentences, phis)
        flattened_encoded_materializations = flatten(encoded_materializations)
        flattened_encoded_materializations = torch.stack(
            flattened_encoded_materializations
        )
        flattened_encoded_materializations = torch.cat(
            [flattened_encoded_materializations, flattened_encoded_materializations],
            dim=-1,
        )

        # compute sense embedding
        flattened_sense_embeddings = self.embedding_from_senses(senses)

        # compute cosine
        matrix_similarities = self.matrix_cos_sim(
            matrix1=flattened_encoded_materializations,
            matrix2=flattened_sense_embeddings,
        )

        # return
        numerator = sum(matrix_similarities)
        ares = numerator / sum(len(exmod_sample.D) for exmod_sample in exmod_samples)
        ares_wo0 = numerator / len(matrix_similarities)
        return ares, ares_wo0


class AresEvaluation(Evaluation):
    def __init__(self, ares_measure: AresMeasure):
        self.ares_measure = ares_measure

    def __call__(
        self,
        path: str,
        predicted_samples: List[ExmodSample],
    ) -> Dict:
        ares, ares_wo0 = self.ares_measure.ares_distance(predicted_samples)
        return {"ares": ares, "ares-wo0": ares_wo0}
