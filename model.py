import sys
import json
from xml.dom import minidom

# numpy
import numpy as np
from numpy.linalg import norm

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

param = {}
def populateParam():
    hns = [500, 250, 100, 50]
    solver = sys.argv[1]
    activation = sys.argv[2]
    hlt = []
    for i in range(3):
        x = int(sys.argv[i + 3])
        if x < 4:
            hlt.append(hns[x])
    param['solver'] = solver
    param['activation'] = activation
    param['hidden'] = tuple(hlt)
    # print solver, activation, hlt

def getAverageCV(doc2vec, cl):
    """ get (norm-ed) average comment vector """
    ac_v = None
    for c in cl:
        c_w = preprocessor(c[1])
        c_v = doc2vec.infer_vector(c_w)
        c_v /= norm(c_v)
        for i in range( len(c_v) ):
            if ac_v is None:
                ac_v = [0] * len(c_v)
            ac_v[i] += c_v[i]
    ac_v = [ float(x)/len(cl) for x in ac_v ]
    ac_v /= norm(ac_v)
    return ac_v

def trainNN(doc2vec, data):
    """ Train MLP """
    mlp = MLPClassifier( solver = param['solver'], \
        hidden_layer_sizes = param['hidden'], \
        activation = param['activation'], \
        learning_rate = 'adaptive', \
        early_stopping = False, \
        random_state = 1, \
        max_iter = 1000, \
        verbose = True )
    X = []
    Y = []
    for q, cl in data:
        q_w = preprocessor(q[1])
        q_v = doc2vec.infer_vector(q_w)
        q_v /= norm(q_v)
        ac_v = getAverageCV(cl)
        for c in cl:
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            c_v /= norm(c_v)
            X.append(np.append(q_v, c_v, ac_v))
            Y.append(transformLabel(c[2]))
    mlp.fit(X, Y)
    return mlp

def predictAux(q_v, c_v, ac_v, mlp):
    if mlp is None:
        """ cosine similarity """
        score = ( 1.0 + consine(q_v, c_v) ) / 2.0
        if score >= 0.5:
            return (score, 'true')
        return (score, 'false')
    """ mlp prediction """
    q_v /= norm(q_v)
    c_v /= norm(c_v)
    pred = mlp.predict_proba([ np.append(q_v, c_v, ac_v) ])[0]
    if pred[0] > pred[1]:
        return (0.5 + 0.5 * pred[0]), 'true'
    return (0.5 - 0.5 * pred[1]), 'false'

def predict(doc2vec, data, output, mlp = None):
    """ Answer Reranking with rank ~ cosine(q_i, a_i)^(-1) """
    # data : zip(questions, commentsL) ... see 'constructData'
    out = open(output, 'w')
    for q, cl in data:
        scores = []
        q_w = preprocessor(q[1])
        q_v = doc2vec.infer_vector(q_w)
        ac_v = getAverageCV(cl)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            score, pred = predictAux(q_v, c_v, ac_v, mlp)
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
    sys.stdout.flush()
    labels = []
    questions = []
    commentsL = []
    for xmlFile in fileList:
        print DB, dataPath + xmlFile
        sys.stdout.flush()
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
    sys.stdout.flush()
    return zip(questions, commentsL)

if __name__ == '__main__':
    populateParam()
    print '== IMPORT DOC2VEC MODEL =='
    sys.stdout.flush()
    doc2vec = loadDoc2Vec('full')
    """ TRAIN MODE """
    print '======= TRAIN MODE ======='
    sys.stdout.flush()
    dataPath = config['TRAIN_NN']['path']
    fileList = config['TRAIN_NN']['files']
    data = constructData(dataPath, fileList)
    mlp = trainNN(doc2vec, data)
    """ VALIDATION MODE """
    print '======= VALIDATION ======='
    sys.stdout.flush()
    dataPath = config['VALIDATION']['path']
    fileList = config['VALIDATION']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['VALIDATION']['predictions']
    predict(doc2vec, data, output, mlp)
    """ TEST MODE """
    print '======= TEST MODE ======='
    sys.stdout.flush()
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['TEST_NN']['2016']['predictions']
    predict(doc2vec, data, output, mlp)
    print '======== FINISHED ========'
    sys.stdout.flush()
