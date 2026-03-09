import re
from collections import Counter


class CorpusIndex:
    """Stores all corpus structure: tokens, vocab, inverted index with TF, doc lengths.

    This class owns only structural logic — preprocessing, vocabulary building,
    and the inverted index. Scoring is left to retriever classes.

    The inverted index maps each term to a list of (doc_id, tf) pairs, enabling
    scalable retrieval without materializing a full document-term matrix.
    """

    def __init__(self, corpus: list[str]):
        self.token_pattern = r'\b\w+\b'
        self.preprocessed = self.preprocess(corpus)
        self.tokens = self.build_vocab(self.preprocessed)
        self.inverted_index = self.build_inverted_index()
        self.doc_lengths = [len(doc) for doc in self.preprocessed]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

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

    def build_inverted_index(self) -> dict[str, list[tuple[int, int]]]:
        """Builds an inverted index mapping each term to [(doc_id, tf), ...] pairs."""
        inverted_index = {}
        for doc_id, document in enumerate(self.preprocessed):
            for term, tf in Counter(document).items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, tf))
        return inverted_index

    @property
    def n_docs(self) -> int:
        return len(self.preprocessed)

    @property
    def n_vocab(self) -> int:
        return len(self.tokens)
