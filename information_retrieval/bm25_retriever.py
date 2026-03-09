import math
from base_retriever import BaseRetriever
from corpus import dataset_IR_v1


class BM25Retriever(BaseRetriever):
    """Retriever using Okapi BM25 scoring.

    BM25 score contribution of term t in document d:
        IDF(t) × f(t,d)×(k1+1) / (f(t,d) + k1×(1 - b + b×|d|/avgdl))

    BM25 IDF:
        log((N - df + 0.5) / (df + 0.5) + 1)

    Scoring is done term-at-a-time over posting lists; no full matrix is built.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        super().__init__(corpus)

    def _precompute(self):
        """Precompute BM25 IDF for all vocabulary terms."""
        N = self.index.n_docs
        self.idf: dict[str, float] = {}
        for term, postings in self.index.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _score_posting(self, term: str, query_tokens: list[str], doc_id: int, tf: int) -> float:
        """BM25 score contribution of term t in document doc_id."""
        doc_len = self.index.doc_lengths[doc_id]
        avgdl = self.index.avg_doc_length
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        return self.idf[term] * (numerator / denominator)


if __name__ == "__main__":
    corpus = dataset_IR_v1

    retriever = BM25Retriever(corpus)

    query = "Do que gatos e cachorros gostam ?"
    answers = retriever.retrieve(query, top_k=2)
    print("Query:", query)
    print("Results:", answers)
