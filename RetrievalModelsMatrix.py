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

        ## LMJM statistics

        self.features = self.vectorizer.get_feature_names()
        self.lc = len(self.features)
        self.d= np.shape(self.tf)[0]
        print("---->lc:" + str(self.lc))
        print("---->d:" + str(self.d))


        ## BM25 statistics

    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)
        #       doc_scores = np.nan_to_num(doc_scores, 0)

        return doc_scores

    def score_lmd(self, query):
        return doc_scores

    def score_lmjm(self, query, lmbdq):
        queryvector = self.vectorizer.build_tokenizer()
        for term in queryvector(query):
            try:
                index = self.features.index(term)
                lt = np.sum(self.tf[:, index], axis=0)
                np.shape(self.tf)
                for i in range(0, len(self.tf)):
                    ftd = self.tf[i, index]
            except:
                return 0
            first = float(lmbdq)*(lt/self.lc)
            second = (1-float(lmbdq))*(ftd/self.d)

        return first + second

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0
