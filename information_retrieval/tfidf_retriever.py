import math
import numpy as np
from collections import Counter
from base_retriever import BaseRetriever
from corpus import dataset_IR_v1


class TFIDFRetriever(BaseRetriever):
    """Retriever using log-normalized TF × IDF scoring with cosine similarity."""

    def build_score_matrix(self) -> np.ndarray:
        """Build an M×N TF-IDF matrix. Sets self.idf as a side-effect."""
        self.idf = self._compute_idf()
        M = self.index.n_docs
        N = self.index.n_vocab
        score_matrix = np.zeros([M, N])

        for doc_id in range(M):
            for term in set(self.index.preprocessed[doc_id]):
                term_index = self.index.tokens[term]
                count = self.index.count_matrix[doc_id, term_index]
                score_matrix[doc_id, term_index] = self._compute_tf(count) * self.idf[term]

        return score_matrix

    def vectorize_query(self, tokens: list[str]) -> np.ndarray:
        counts = Counter(tokens)
        vector = np.zeros([1, self.index.n_vocab])
        for term, count in counts.items():
            if term in self.index.tokens:
                tfidf_term = self._compute_tf(count) * self.idf[term]
                vector[0, self.index.tokens[term]] = tfidf_term
        return vector

    def _compute_tf(self, count: int) -> float:
        """Log-normalized TF: 1 + log10(count), or 0 if count is 0."""
        if count == 0:
            return 0.0
        return 1 + math.log10(count)

    def _compute_idf(self) -> dict[str, float]:
        """IDF = log10(N / df) for each term in the vocabulary."""
        N = self.index.n_docs
        idf = {}
        for term in self.index.tokens:
            df = len(self.index.inverted_index.get(term, []))
            idf[term] = math.log10(N / df) if df > 0 else 0.0
        return idf


if __name__ == "__main__":
    corpus = dataset_IR_v1

    retriever = TFIDFRetriever(corpus)

    print("Vocab:", retriever.index.tokens)
    print("TF-IDF row 0:", retriever.score_matrix[0])

    query = "Do que gatos e cachorros gostam ?"
    answers = retriever.retrieve(query, top_k=2)
    print("Query:", query)
    print("Results:", answers)
