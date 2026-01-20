
import math
from collections import Counter
import numpy as np


class TFIDFRetriever:

    def __init__(self, corpus):
        self.tfidf = TFIDF(corpus)
        self.bow = self.tfidf.compute_tfidf_with_bow()

    def retrieve(self, query, top_k=2):
        preproc_query = self.tfidf.preprocess([query])[0]
        print(preproc_query)

        vector = np.zeros([1, len(self.tfidf.vocab)])
        for term in preproc_query:
            print(f"Checking if {term} in vocab...")
            if term in self.tfidf.vocab:
                print(f"Term  {term} is in vocab, doing the math")
                tfidf_term = self.tfidf.compute_tf(
                    term, preproc_query) * self.tfidf.idf[term]
                term_index = self.tfidf.tokenize[term]
                vector[0, term_index] = tfidf_term

        similarities = self.cosine_similarity(vector, self.tfidf.bow.T)[0]

        idx = np.argsort(similarities)

        answers = []
        for i in range(top_k):
            answers.append(
                (" ".join(self.tfidf.corpus[idx[-(i+1)]]),
                 similarities[idx[-(i+1)]]))

        return answers

    def cosine_similarity(self, vector, bow):
        """
        Calculates the cosine similarity between two 1D NumPy vectors.
        """
        dot_product = np.dot(vector, bow)
        magnitude_vec1 = np.linalg.norm(vector)
        magnitude_vec2 = np.linalg.norm(bow)

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0

        cosine_similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
        return cosine_similarity


class TFIDF:

    def __init__(self, corpus):
        self.corpus = self.preprocess(corpus)
        self.vocab, self.tokenize = self.build_vocab(self.corpus)
        self.bow = self.build_bow()

    def build_bow(self):
        M = len(self.corpus)
        N = len(self.vocab)

        bow = np.zeros([M, N])

        return bow

    def preprocess(self, corpus):
        preprocessed_corpus = []
        for document in corpus:
            document = document.lower().split()
            preprocessed_corpus.append(document)
        return preprocessed_corpus

    def build_vocab(self, preprocessed_corpus):
        vocab = set()
        for document in preprocessed_corpus:
            for word in document:
                vocab.add(word)

        tokenize = {}
        for i, word in enumerate(vocab):
            tokenize[word] = i
        return list(vocab), tokenize

    def compute_tf(self, term, document):
        count_term = 0
        total_words = 0

        for word in document:
            if word == term:
                count_term += 1
            total_words += 1

        return count_term/total_words

    def compute_df(self):
        df = Counter()

        for doc in self.corpus:
            set_terms = set(doc)
            for word in set_terms:
                df[word] += 1

        return df

    def compute_idf(self):
        idf = {}
        df = self.compute_df()
        N = len(self.corpus)

        for term, freq in df.items():
            idf[term] = math.log(N/(freq+1))

        return idf

    def compute_tfidf_with_bow(self):

        self.idf = self.compute_idf()

        for index, document in enumerate(self.corpus):
            for term in document:
                tfidf_term = self.compute_tf(term, document) * self.idf[term]
                term_index = self.tokenize[term]
                self.bow[index, term_index] = tfidf_term

        return self.bow


if __name__ == "__main__":
    corpus = ["gatos gostam de leite",
              "gatos gostam de peixe",
              "cachorros gostam de ossos",
              "cachorros gostam de brincar",
              "gatos e cachorros podem ser amigos",
              "leite é bom para gatos",
              "peixe é comida de gato",
              "ossos são comida de cachorro",
              "animais gostam de comida",
              "gatos e cachorros gostam de comida"]

    tfidf = TFIDF(corpus)

    print(tfidf.vocab)

    print(tfidf.compute_tf("leite", tfidf.corpus[0]))

    print(tfidf.compute_df())

    print(tfidf.tokenize)

    print(tfidf.compute_tfidf_with_bow()[0])

    retriever = TFIDFRetriever(corpus)

    query = "Do que gatos e cachorros gostam ?"

    answers = retriever.retrieve(query, top_k=2)

    print(answers)
