import math
from base_retriever import BaseRetriever
from corpus import dataset_IR_v1


class TFIDFRetriever(BaseRetriever):
    """Retriever using log-normalized TF × IDF scoring with dot-product ranking.

    Score contribution of term t in document d for query q:
        tf_idf(t, q) × tf_idf(t, d)

    where tf_idf(t, x) = (1 + log10(tf(t, x))) × log10(N / df(t))

    Scoring is done term-at-a-time over posting lists; no full matrix is built.
    """

    def _precompute(self):
        """Precompute IDF for all vocabulary terms."""
        N = self.index.n_docs
        self.idf: dict[str, float] = {}
        for term, postings in self.index.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log10(N / df) if df > 0 else 0.0

    def _score_posting(self, term: str, query_tokens: list[str], doc_id: int, tf: int) -> float:
        """TF-IDF dot-product score contribution of term t in document doc_id."""
        query_tf = query_tokens.count(term)
        doc_score = self._compute_tf(tf) * self.idf[term]
        query_score = self._compute_tf(query_tf) * self.idf[term]
        return query_score * doc_score

    def _compute_tf(self, count: int) -> float:
        """Log-normalized TF: 1 + log10(count), or 0 if count is 0."""
        return 0.0 if count == 0 else 1 + math.log10(count)


if __name__ == "__main__":
    corpus = dataset_IR_v1

    retriever = TFIDFRetriever(corpus)

    query = "Do que gatos e cachorros gostam ?"
    answers = retriever.retrieve(query, top_k=2)
    print("Query:", query)
    print("Results:", answers)
