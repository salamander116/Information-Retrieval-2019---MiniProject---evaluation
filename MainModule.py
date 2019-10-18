import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import RetrievalModelsMatrix
import collectionloaders
import simpleparser as parser


def model_inverted_index():
    pass


def helpz():
    print("Commands:")
    print("q ---> quit")
    print("mm ---> retrieval model matrix")
    print("mii ---> retrieval model inverted index")
    pass


def drawLMJMGraphic(lmjarray):
    plt.figure()
    i = 0.1
    for scorelm in lmjarray:
        jet = plt.get_cmap('jet')
        colors = iter(jet(np.linspace(0, 1, 10)))
        color = next(colors)

        plt.plot(i, scorelm, color='b', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        i = i + 0.1

    plt.xlabel('Lambda')
    plt.ylabel('Precision@10')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('LMJM (P@10={0:0.2f})')
    plt.show()
    plt.close()
    pass


def graphic(precision_vsmArr, recall, map_vsm):
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0, 1, 10)))
    color = next(colors)
    try:
        for precision_vsm in precision_vsmArr:
            plt.plot(recall, np.mean(precision_vsm, axis=0), color=color, alpha=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.fill_between(recall,
                             np.mean(precision_vsm, axis=0) - np.std(precision_vsm, axis=0),
                             np.mean(precision_vsm, axis=0) + np.std(precision_vsm, axis=0), facecolor=color,
                             alpha=0.1)
            color = next(colors)
    except:
        plt.plot(recall, np.mean(precision_vsmArr, axis=0), color='b', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill_between(recall,
                         np.mean(precision_vsmArr, axis=0) - np.std(precision_vsmArr, axis=0),
                         np.mean(precision_vsmArr, axis=0) + np.std(precision_vsmArr, axis=0), facecolor=color,
                         alpha=0.1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall (MAP={0:0.2f})'.format(map_vsm))
    plt.savefig('results/maplmd.png', dpi=100)
    pass


def inputParser(line):
    if line == "none":
        return 0

    ngrams = []
    for arg in line.split(" "):
        ngrams.append(arg)

    return ngrams


def lmd(vectorizer, cl, verbose, mu):
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfiled = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfiled, vectorizer)
    scores_array = []
    map_lmd = 0
    j = 1
    for query in cl.queries_cranfield['query']:
        score = models.score_lmd(parser.stemSentence(query), mu)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(score, j)
        map_lmd = map_lmd + average_precision
        scores_array.append(average_precision)
        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', j, 'LMD     AP=', average_precision)
        j = j + 1

    map_lmd = map_lmd / cl.num_queries
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
    plt.savefig('results/lmjmtest.png', dpi=100)
    finalres = [scores_array, recall_11point, map_lmd]
    return finalres


def lmjm(vectorizer, cl, verbose, lmbd):
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfiled = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfiled, vectorizer)
    scores_array = []
    map_lmjm = 0
    j = 1
    for query in cl.queries_cranfield['query']:
        score = models.score_lmjm(parser.stemSentence(query), lmbd)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(score, j)
        map_lmjm = map_lmjm + average_precision
        scores_array.append(average_precision)
        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', j, 'LMJM     AP=', average_precision)
        j = j + 1
    map_lmjm = map_lmjm / cl.num_queries

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
    plt.savefig('results/lmjmtest.png', dpi=100)

    finalres = [scores_array, precision_11point, map_lmjm]
    return finalres


def bm25(vectorizer, cl, verbose, k1, b):
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfield = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)
    i = 1
    map_bm25 = 0
    precision_bm25 = []
    for query in cl.queries_cranfield['query']:
        scores = models.score_bm25(parser.stemSentence(query), k1, b)
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(scores, i)
        map_bm25 = map_bm25 + average_precision
        precision_bm25.append(average_precision)

        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', i, 'BM25    AP=', average_precision)
        i = i + 1
    map_bm25 = map_bm25 / cl.num_queries

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
    plt.savefig('results/bm25test.png', dpi=100)

    finalres = [precision_bm25, recall_11point, map_bm25]
    return finalres


def vsm(vectorizer, cl, verbose):
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfield = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)
    i = 1
    map_vsm = 0
    precision_vsm = []
    recallarr = []
    for query in cl.queries_cranfield['query']:
        scores = models.score_vsm(parser.stemSentence(query))
        [average_precision, precision_11point, recall_11point, p10] = cl.eval(scores, i)
        map_vsm = map_vsm + average_precision
        precision_vsm.append(average_precision)
        recallarr.append(recall_11point)

        if verbose:
            plt.plot(recall_11point, precision_11point, color='silver', alpha=0.1)
            print('qid =', i, 'VSM     AP=', average_precision)
        i = i + 1

    map_vsm = map_vsm / cl.num_queries

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
    plt.savefig('results/vsmtest.png', dpi=100)

    finalres = [precision_vsm, precision_11point, map_vsm]
    return finalres


######################################################    Main Code    #############################################################################################################
cl = collectionloaders.CranfieldTestBed()
user_input = input("Command?")
verbose = True

while user_input.lower() != "q":
    if user_input.lower() == "mm":
        user_model_option = input("model : ex. vsm/lmjm/lmd/bm25")
        user_input = input("Number of N-grams? \t ex: 1 2 3")
        result = inputParser(user_input)
        if result == 0:
            vectorizer = CountVectorizer(ngram_range=(1, int(num) + 1), token_pattern=r'\b\w+\b',
                                         min_df=1, stop_words='english')
        else:
            vsmArray = []
            for num in result:
                vectorizer = CountVectorizer(ngram_range=(1, int(num)), token_pattern=r'\b\w+\b',
                                             min_df=1, stop_words='english')
        if user_model_option.lower() == "vsm":
            if result == 0:
                [precision_vsm, recall_11point, map_vsm] = vsm(vectorizer, cl, verbose)
            # graphic(precision_vsm, recall_11point, map_vsm)
            else:
                [precision_vsm, recall_11point, map_vsm] = vsm(vectorizer, cl, verbose)
                # vsmArray.append(recall_11point)
                # graphic(vsmArray, recall_11point, map_vsm)
        elif user_model_option.lower() == "lmjm":
            lmjm(vectorizer, cl, verbose, lmbd=0.75)
        elif user_model_option.lower() == "lmd":
            lmd(vectorizer, cl, verbose, mu=0.3)
        else:
            bm25(vectorizer, cl, verbose, k1=1.75, b=0.75)
    elif user_input.lower() == "mii":
        model_inverted_index()
    else:
        helpz()
    user_input = input("Command?")
#######################################################################################################################################################################################
