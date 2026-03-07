from abc import ABC, abstractmethod
import numpy as np
from index import CorpusIndex


class BaseRetriever(ABC):
    """Abstract base class for corpus retrievers.

    Implements the retrieve() template method. Subclasses must provide
    build_score_matrix() and vectorize_query() to define their scoring model.
    """

    def __init__(self, corpus: list[str]):
        self.index = CorpusIndex(corpus)
        self.score_matrix = self.build_score_matrix()

    @abstractmethod
    def build_score_matrix(self) -> np.ndarray:
        """Build an M×N weighted document-term matrix."""
        pass

    @abstractmethod
    def vectorize_query(self, tokens: list[str]) -> np.ndarray:
        """Convert preprocessed query tokens into a (1×N) query vector."""
        pass

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple]:
        """Retrieve top-k documents for a query using cosine similarity.

        Pipeline: preprocess → candidate filtering via inverted index →
        vectorize_query → cosine similarity → top-k ranking.
        """
        preproc_query = self.index.preprocess([query])[0]

        candidate_doc_ids = set()
        for term in preproc_query:
            if term in self.index.tokens:
                candidate_doc_ids.update(
                    self.index.inverted_index.get(term, []))

        if not candidate_doc_ids:
            return []

        candidate_ids = list(candidate_doc_ids)
        query_vector = self.vectorize_query(preproc_query)
        candidate_matrix = self.score_matrix[candidate_ids]
        similarities = self._cosine_similarity(query_vector, candidate_matrix.T)[0]

        idx = np.argsort(similarities)

        answers = []
        for i in range(min(top_k, len(candidate_ids))):
            doc_id = candidate_ids[idx[-(i + 1)]]
            answers.append(
                (" ".join(self.index.preprocessed[doc_id]),
                 similarities[idx[-(i + 1)]]))

        return answers

    def _cosine_similarity(self, vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a (1×N) vector and a (N×K) matrix.

        Returns a (1×K) array of similarity scores, one per column.
        """
        dot_product = np.dot(vector, matrix)
        magnitude_vec = np.linalg.norm(vector)
        magnitude_cols = np.linalg.norm(matrix, axis=0)

        if magnitude_vec == 0:
            return np.zeros_like(dot_product)

        denominator = magnitude_vec * magnitude_cols
        denominator = np.where(denominator == 0, 1e-10, denominator)
        return dot_product / denominator
