import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf

        ## VSM statistics
        self.term_doc_freq = np.sum(tf != 0, axis=0)
        self.term_coll_freq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis=0) / self.term_doc_freq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        ## LMD statistics

        self.docLenVert = np.reshape(self.docLen, (-1, 1))

        ## LMJM statistics

        self.lc = np.sum(self.tf[-1, :])

        ## BM25 statistics
        self.avgDocLen = np.average(self.docLen)

    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))
        a = np.dot(query_vector, self.tfidf.T)
        doc_scores = a / (0.0001 + self.docNorms * query_norm)
        #       doc_scores = np.nan_to_num(doc_scores, 0)

        return doc_scores

    def score_lmd(self, query, mu):
        query_vector = self.vectorizer.transform([query]).toarray()
        mct = np.multiply(query_vector, self.term_coll_freq)
        ftd = np.multiply(self.tf, (query_vector != 0))
        const = np.add(self.docLenVert, mu)

        som = np.add(ftd, np.multiply(mct, mu))
        div = np.divide(som, const)
        doc_scores = np.power(div, query_vector)
        doc_scores = np.prod(doc_scores, axis=1)
        return doc_scores

    def score_lmjm(self, query, lmbd):
        query_vector = self.vectorizer.transform([query]).toarray()

        ftd = np.multiply(self.tf, (query_vector != 0))
        pmd = np.divide(ftd, self.docLenVert)

        lt = np.multiply((query_vector != 0), self.term_coll_freq)
        pmc = np.divide(lt, self.lc)

        x = np.multiply(lmbd, pmd)
        y = np.multiply((1 - lmbd), pmc)

        doc_scores = np.add(x, y)
        doc_scores = np.power(doc_scores, query_vector)
        doc_scores = np.prod(doc_scores, axis=1)

        doc_scores = np.where(doc_scores == np.inf, 0, doc_scores)
        doc_scores = np.nan_to_num(doc_scores)
        return doc_scores

    def score_bm25(self, query, k1, b):
        query_vector = self.vectorizer.transform([query]).toarray()
        ftd = np.multiply(self.tf, (query_vector != 0))

        superior = np.multiply(ftd, (k1 + 1))
        a = np.multiply(b, np.divide(self.docLen, self.avgDocLen))
        b = np.add(1 - b, a)
        c = np.multiply(k1, b)
        inferior = np.add(ftd, np.reshape(np.multiply(k1, c), (-1, 1)))
        division = np.divide(superior, inferior)

        idft = np.multiply(self.tfidf, (query_vector != 0))
        doc_scores = np.multiply(np.multiply(query_vector, division), idft)

        doc_scores = np.power(doc_scores, query_vector)
        doc_scores = np.sum(doc_scores, axis=1)
        doc_scores = np.where(doc_scores == np.inf, 0, doc_scores)
        doc_scores = np.nan_to_num(doc_scores)
        return doc_scores

    def scoreRM3(self, query):
        return 0
