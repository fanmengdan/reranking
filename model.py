import json
from xml.dom import minidom

# numpy
import numpy as np

# gensim modules
from gensim.models import Doc2Vec

# pre-processing utilities
from myutils import preprocessor, consine

# MultiLayerPerceptron
from sklearn.neural_network import MLPClassifier

DB = '<< DEBUG >>'
config = json.load(open('config.json', 'r'))

def loadDoc2Vec(mode):
    modelPath = config['DOC2VEC'][mode]['path']
    modelName = config['DOC2VEC'][mode]['name']
    doc2vec = Doc2Vec.load(modelPath + modelName)
    return doc2vec

def transformLabel(label):
    if label == 'Good':
        return [1, 0]
    return [0, 1]

def trainNN(doc2vec, data):
    """ Train MLP """
    mlp = MLPClassifier( solver = 'adam', \
        hidden_layer_sizes = (100,), \
        early_stopping = False, \
        activation = 'relu', \
        random_state = 1, \
        max_iter = 1000, \
        verbose = True )
    X = []
    Y = []
    for q, cl in data:
        q_w = preprocessor(q[1])
        q_v = doc2vec.infer_vector(q_w)
        for c in cl:
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            X.append(np.append(q_v, c_v))
            Y.append(transformLabel(c[2]))
    mlp.fit(X, Y)
    return mlp

def predictAux(q_v, c_v, mlp):
    if mlp is None:
        """ cosine similarity """
        score = ( 1.0 + consine(q_v, c_v) ) / 2.0
        if score >= 0.5:
            return (score, 'true')
        return (score, 'false')
    """ mlp prediction """
    pred = mlp.predict_proba([ np.append(q_v, c_v) ])[0]
    if pred[0] > pred[1]:
        return pred[0], 'true'
    return pred[1], 'false'

def predict(doc2vec, data, output, mlp = None):
    """ Answer Reranking with rank ~ cosine(q_i, a_i)^(-1) """
    # data : zip(questions, commentsL) ... see 'constructData'
    out = open(output, 'w')
    for q, cl in data:
        scores = []
        q_w = preprocessor(q[1])
        q_v = doc2vec.infer_vector(q_w)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            score, pred = predictAux(q_v, c_v, mlp)
            scores.append( [ score, j, 0, pred ] )
        scores = sorted(scores, key=lambda score: score[0], reverse=True)
        for i in range(len(scores)):
            scores[i][2] = i + 1
        scores = sorted(scores, key=lambda score: score[1])
        for score in scores:
            out.write('\t'.join([q[0], cl[score[1]][0], str(score[2]), str(score[0]), score[3]]))
            out.write('\n')
    out.close()

def constructData(dataPath, fileList):
    """ constructs question and related comments data """
    # returns data = zip(questions, commentsL)
    # questions : list of (questionId, question) pairs
    #   ( [(qid1, q1) (qid2, q2) ... (qidN, qN)] )
    # commentsL : list of list of (commentId, comment, label) pairs
    #   ( [ [(cid1, c1, l1) (cid2, c2, l2) ... (cidK1, cK1, lK1)] ... [(cid1, c1, l1) (cid2, c2, l2) ... (cidKN, cKN, lKN)] ] )
    print DB, 'DATA IMPORT STARTED'
    labels = []
    questions = []
    commentsL = []
    for xmlFile in fileList:
        print DB, dataPath + xmlFile
        doc = minidom.parse(dataPath + xmlFile)
        threads = doc.getElementsByTagName("Thread")
        for tid, thread in enumerate(threads):
            # constructing the question string
            relQ = thread.getElementsByTagName('RelQuestion')[0]
            Qid = relQ.getAttribute('RELQ_ID')
            bodyQ = relQ.getElementsByTagName('RelQBody')[0]
            body = bodyQ._get_firstChild().data if bodyQ._get_firstChild() is not None else ''
            subjQ = relQ.getElementsByTagName('RelQSubject')[0]
            subj = subjQ._get_firstChild().data if subjQ._get_firstChild() is not None else ''
            questions.append( (Qid, subj + ' ' + body) )
            # constructing the list of comments
            comments = []
            for relC in thread.getElementsByTagName('RelComment'):
                Cid = relC.getAttribute('RELC_ID')
                label = relC.getAttribute('RELC_RELEVANCE2RELQ')
                comment = relC.getElementsByTagName('RelCText')[0]._get_firstChild().data
                comments.append( (Cid, comment, label) )
            commentsL.append(comments)
    print DB, 'DATA IMPORT FINISHED'
    return zip(questions, commentsL)

if __name__ == '__main__':
    print '== IMPORT DOC2VEC MODEL =='
    doc2vec = loadDoc2Vec('full')
    """ TRAIN MODE """
    print '======= TRAIN MODE ======='
    dataPath = config['TRAIN_NN']['path']
    fileList = config['TRAIN_NN']['files']
    data = constructData(dataPath, fileList)
    mlp = trainNN(doc2vec, data)
    """ VALIDATION MODE """
    print '======= VALIDATION ======='
    dataPath = config['VALIDATION']['path']
    fileList = config['VALIDATION']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['VALIDATION']['predictions']
    predict(doc2vec, data, output, mlp)
    """ TEST MODE """
    print '======= TEST MODE ======='
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['TEST_NN']['2016']['predictions']
    predict(doc2vec, data, output, mlp)
    print '======== FINISHED ========'
