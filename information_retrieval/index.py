import re
import numpy as np


class CorpusIndex:
    """Stores all corpus structure: tokens, vocab, raw counts, inverted index.

    This class owns only structural logic — preprocessing, vocabulary building,
    raw term counts, and the inverted index. Scoring is left to retriever classes.
    """

    def __init__(self, corpus: list[str]):
        self.token_pattern = r'\b\w+\b'
        self.preprocessed = self.preprocess(corpus)
        self.tokens = self.build_vocab(self.preprocessed)
        self.count_matrix = self.build_count_matrix()
        self.inverted_index = self.build_inverted_index()
        self.avg_doc_length = self._compute_avg_doc_length()

    def preprocess(self, corpus: list[str]) -> list[list[str]]:
        preprocessed_corpus = []
        for document in corpus:
            preprocessed_corpus.append(
                re.findall(self.token_pattern, document.lower()))
        return preprocessed_corpus

    def build_vocab(self, preprocessed: list[list[str]]) -> dict[str, int]:
        vocab_set = set()
        for document in preprocessed:
            for word in document:
                vocab_set.add(word)

        return {word: i for i, word in enumerate(sorted(vocab_set))}

    def build_count_matrix(self) -> np.ndarray:
        """Builds an M×N matrix of raw integer term counts."""
        M = len(self.preprocessed)
        N = len(self.tokens)
        count_matrix = np.zeros([M, N], dtype=int)
        for doc_id, document in enumerate(self.preprocessed):
            for term in document:
                term_index = self.tokens[term]
                count_matrix[doc_id, term_index] += 1
        return count_matrix

    def build_inverted_index(self) -> dict[str, list[int]]:
        inverted_index = {}
        for doc_id, document in enumerate(self.preprocessed):
            for term in set(document):
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append(doc_id)
        return inverted_index

    def _compute_avg_doc_length(self) -> float:
        return float(np.mean([len(doc) for doc in self.preprocessed]))

    @property
    def n_docs(self) -> int:
        return len(self.preprocessed)

    @property
    def n_vocab(self) -> int:
        return len(self.tokens)
