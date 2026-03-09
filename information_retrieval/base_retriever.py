from abc import ABC, abstractmethod
from index import CorpusIndex


class BaseRetriever(ABC):
    """Abstract base class for scalable corpus retrievers.

    Uses term-at-a-time (TAAT) scoring: for each query term, iterates over its
    posting list and accumulates scores per document. Only documents sharing at
    least one term with the query are scored, avoiding full matrix operations.

    Subclasses must implement:
      - _precompute(): compute term statistics (e.g., IDF) after index is built.
      - _score_posting(): score contribution of one (term, document) posting.
    """

    def __init__(self, corpus: list[str]):
        self.index = CorpusIndex(corpus)
        self._precompute()

    @abstractmethod
    def _precompute(self):
        """Precompute term statistics needed for scoring (e.g., IDF)."""
        pass

    @abstractmethod
    def _score_posting(self, term: str, query_tokens: list[str], doc_id: int, tf: int) -> float:
        """Score contribution of term appearing in doc_id with frequency tf."""
        pass

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple]:
        """Retrieve top-k documents for a query using TAAT score accumulation.

        Pipeline: preprocess → iterate posting lists → accumulate scores → top-k.
        """
        tokens = self.index.preprocess([query])[0]

        scores: dict[int, float] = {}
        for term in set(tokens):
            for doc_id, tf in self.index.inverted_index.get(term, []):
                scores[doc_id] = scores.get(doc_id, 0.0) + self._score_posting(term, tokens, doc_id, tf)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(" ".join(self.index.preprocessed[doc_id]), score) for doc_id, score in ranked]
