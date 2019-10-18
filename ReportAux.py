import matplotlib.pyplot as plt
import numpy as np

import RetrievalModelsMatrix
import simpleparser as parser


def lmjm(vectorizer, cl, verbose, lmbd):
    plt.clf()
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfiled = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfiled, vectorizer)
    scores_array = []
    p10aux = 0;
    map_lmjm = 0
    j = 1
    for query in cl.queries_cranfield['query']:
        score = models.score_lmjm(parser.stemSentence(query), lmbd)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(score, j)
        map_lmjm = map_lmjm + average_precision
        p10aux = p10aux + p10
        scores_array.append(average_precision)
        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', j, 'LMJM     AP=', average_precision)
        j = j + 1
    map_lmjm = map_lmjm / cl.num_queries
    p10aux = p10aux / cl.num_queries

    plt.plot(recall_11point, precision_11point, color='b', alpha=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.fill_between(recall_11point,
                     np.mean(scores_array, axis=0) - np.std(scores_array, axis=0),
                     np.mean(scores_array, axis=0) + np.std(scores_array, axis=0), facecolor='b', alpha=0.1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_lmjm))
    plt.savefig('results/LMJMResult.png', dpi=100)

    finalres = [map_lmjm, p10aux]
    return finalres


def bm25(vectorizer, cl, verbose, k1, b):
    plt.clf()
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfield = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)
    i = 1
    map_bm25 = 0
    p10aux = 0;
    precision_bm25 = []
    for query in cl.queries_cranfield['query']:
        scores = models.score_bm25(parser.stemSentence(query), k1, b)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(scores, i)
        map_bm25 = map_bm25 + average_precision
        p10aux = p10aux + p10
        precision_bm25.append(average_precision)

        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', i, 'BM25    AP=', average_precision)
        i = i + 1
    map_bm25 = map_bm25 / cl.num_queries
    p10aux = p10aux / cl.num_queries

    plt.plot(recall_11point, precision_11point, color='b', alpha=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.fill_between(recall_11point,
                     np.mean(precision_bm25, axis=0) - np.std(precision_bm25, axis=0),
                     np.mean(precision_bm25, axis=0) + np.std(precision_bm25, axis=0), facecolor='b', alpha=0.1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_bm25))
    plt.savefig('results/BM25Result.png', dpi=100)

    finalres = [map_bm25, p10aux]
    return finalres


def vsm(vectorizer, cl, verbose):
    plt.clf()
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfield = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)
    i = 1
    map_vsm = 0
    p10aux = 0;
    precision_vsm = []
    recallarr = []
    for query in cl.queries_cranfield['query']:
        scores = models.score_vsm(parser.stemSentence(query))
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(scores, i)
        map_vsm = map_vsm + average_precision
        p10aux = p10aux + p10
        precision_vsm.append(average_precision)
        recallarr.append(recall_11point)

        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', i, 'VSM     AP=', average_precision)
        i = i + 1

    map_vsm = map_vsm / cl.num_queries
    p10aux = p10aux / cl.num_queries

    plt.plot(recall_11point, precision_11point, color='b', alpha=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.fill_between(recall_11point,
                     np.mean(precision_vsm, axis=0) - np.std(precision_vsm, axis=0),
                     np.mean(precision_vsm, axis=0) + np.std(precision_vsm, axis=0), facecolor='b', alpha=0.1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_vsm))
    plt.savefig('results/VSMResult.png', dpi=100)

    finalres = [map_vsm, p10aux]
    return finalres


def lmd(vectorizer, cl, verbose, mu):
    plt.clf()
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfiled = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfiled, vectorizer)
    scores_array = []
    map_lmd = 0
    p10aux = 0;
    j = 1
    for query in cl.queries_cranfield['query']:
        score = models.score_lmd(parser.stemSentence(query), mu)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(score, j)
        map_lmd = map_lmd + average_precision
        p10aux = p10aux + p10
        scores_array.append(average_precision)
        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', j, 'LMD     AP=', average_precision)
        j = j + 1

    map_lmd = map_lmd / cl.num_queries
    p10aux = p10aux / cl.num_queries
    plt.plot(recall_11point, precision_11point, color='b', alpha=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.fill_between(recall_11point,
                     np.mean(scores_array, axis=0) - np.std(scores_array, axis=0),
                     np.mean(scores_array, axis=0) + np.std(scores_array, axis=0), facecolor='b', alpha=0.1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_lmd))
    plt.savefig('results/LMDResult.png', dpi=100)
    finalres = [map_lmd, p10aux]
    return finalres
