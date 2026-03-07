import math
import numpy as np
from base_retriever import BaseRetriever
from corpus import dataset_IR_v1


class BM25Retriever(BaseRetriever):
    """Retriever using Okapi BM25 scoring with cosine similarity.

    BM25 score per term:
        IDF(t) × f(t,d)×(k1+1) / (f(t,d) + k1×(1 - b + b×|d|/avgdl))

    BM25 IDF:
        log((N - df + 0.5) / (df + 0.5) + 1)

    Query vectors are binary (1 if term present).
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        # k1 and b must be set BEFORE super().__init__() because
        # super().__init__() calls build_score_matrix() which uses them.
        self.k1 = k1
        self.b = b
        super().__init__(corpus)

    def build_score_matrix(self) -> np.ndarray:
        """Build an M×N BM25 score matrix. Sets self.idf as a side-effect."""
        self.idf = self._compute_idf_bm25()
        M = self.index.n_docs
        N = self.index.n_vocab
        score_matrix = np.zeros([M, N])
        avgdl = self.index.avg_doc_length

        for doc_id in range(M):
            doc_len = len(self.index.preprocessed[doc_id])
            for term in set(self.index.preprocessed[doc_id]):
                term_index = self.index.tokens[term]
                f = float(self.index.count_matrix[doc_id, term_index])
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score_matrix[doc_id, term_index] = self.idf[term] * (numerator / denominator)

        return score_matrix

    def vectorize_query(self, tokens: list[str]) -> np.ndarray:
        """Binary query vector: 1.0 for each term present in the vocabulary."""
        vector = np.zeros([1, self.index.n_vocab])
        for term in tokens:
            if term in self.index.tokens:
                vector[0, self.index.tokens[term]] = 1.0
        return vector

    def _compute_idf_bm25(self) -> dict[str, float]:
        """BM25 IDF = log((N - df + 0.5) / (df + 0.5) + 1)."""
        N = self.index.n_docs
        idf = {}
        for term in self.index.tokens:
            df = len(self.index.inverted_index.get(term, []))
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf


if __name__ == "__main__":
    corpus = dataset_IR_v1

    retriever = BM25Retriever(corpus)

    print("Vocab:", retriever.index.tokens)
    print("BM25 row 0:", retriever.score_matrix[0])

    query = "Do que gatos e cachorros gostam ?"
    answers = retriever.retrieve(query, top_k=2)
    print("Query:", query)
    print("Results:", answers)
