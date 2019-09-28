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
    plt.savefig('results/prec-try2.png', dpi=100)
    pass


def inputParser(line):
    if line == "none":
        return 0

    ngrams = []
    for arg in line.split(" "):
        ngrams.append(arg)

    return ngrams


def vsm(vectorizer, cl, verbose, ):
    corpus = parser.stemCorpus(cl.corpus_cranfield['abstract'])
    tf_cranfield = vectorizer.fit_transform(corpus).toarray()
    models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)
    i = 1
    map_vsm = 0
    precision_vsm = []
    for query in cl.queries_cranfield['query']:
        scores = models.score_vsm(parser.stemSentence(query))
        [average_precision, precision, recall, thresholds] = cl.eval(scores, i)
        map_vsm = map_vsm + average_precision
        precision_vsm.append(precision)

        if verbose:
            plt.plot(recall, precision, color='silver', alpha=0.1)
            print('qid =', i, 'VSM     AP=', average_precision)
        i = i + 1

    map_vsm = map_vsm / cl.num_queries
    finalres = [precision_vsm, recall, map_vsm]
    return finalres


######################################################    Main Code    #############################################################################################################
cl = collectionloaders.CranfieldTestBed()
user_input = input("Command?")
verbose = True
while user_input.lower() != "q":
    if user_input.lower() == "mm":
        user_input = input("Number of N-grams? \t ex: 1 2 3")
        result = inputParser(user_input)
        if result == 0:
            vectorizer = CountVectorizer()
            [precision_vsm, recall, map_vsm] = vsm(vectorizer, cl, verbose)
            graphic(precision_vsm, recall, map_vsm)
        else:
            vsmArray = []
            for num in result:
                vectorizer = CountVectorizer(ngram_range=(1, int(num)+1), token_pattern='(?=([a-zA-Z]{' + num + '}))',
                                             min_df=0, stop_words='english')
                [precision_vsm, recall, map_vsm] = vsm(vectorizer, cl, verbose)
                vsmArray.append(precision_vsm)
                graphic(vsmArray, recall, map_vsm)

    elif user_input.lower() == "mii":
        model_inverted_index()
    else:
        helpz()
    user_input = input("Command?")
#######################################################################################################################################################################################
