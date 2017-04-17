import sys
import json

# numpy
import numpy as np
from numpy.linalg import norm

# gensim modules
from gensim.models import Doc2Vec

# pre-processing utilities
from myutils import preprocessor, cosine, constructData, debug

# MultiLayerPerceptron
from sklearn.neural_network import MLPClassifier

# for semantic and metadata features
from features import getFeatures

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
        if ac_v is None:
            ac_v = [0] * len(c_v)
        for i in range( len(c_v) ):
            ac_v[i] += c_v[i]
    if ac_v is not None:
        ac_v = [ float(x)/len(cl) for x in ac_v ]
        ac_v /= norm(ac_v)
    return ac_v if ac_v is not None else []

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
        ac_v = getAverageCV(doc2vec, cl)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            c_v /= norm(c_v)
            f_v = getFeatures(doc2vec, q_w, c_w, \
                { 'qid' : q[0], 'cid' : c[0], 'rank' : j }, config)
            X.append(np.append( np.append(q_v, c_v), np.append(ac_v, f_v) ))
            Y.append(transformLabel(c[2]))
    mlp.fit(X, Y)
    return mlp

def predictAux(q_v, c_v, ac_v, f_v, mlp):
    if mlp is None:
        """ cosine similarity """
        score = ( 1.0 + cosine(q_v, c_v) ) / 2.0
        if score >= 0.5:
            return (score, 'true')
        return (score, 'false')
    """ mlp prediction """
    q_v /= norm(q_v)
    c_v /= norm(c_v)
    pred = mlp.predict_proba([ np.append( np.append(q_v, c_v), np.append(ac_v, f_v) ) ])[0]
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
        ac_v = getAverageCV(doc2vec, cl)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            f_v = getFeatures(doc2vec, q_w, c_w, \
                { 'qid' : q[0], 'cid' : c[0], 'rank' : j }, config)
            score, pred = predictAux(q_v, c_v, ac_v, f_v, mlp)
            scores.append( [ score, j, 0, pred ] )
        scores = sorted(scores, key=lambda score: score[0], reverse=True)
        for i in range(len(scores)):
            scores[i][2] = i + 1
        scores = sorted(scores, key=lambda score: score[1])
        for score in scores:
            out.write('\t'.join([q[0], cl[score[1]][0], str(score[2]), str(score[0]), score[3]]))
            out.write('\n')
    out.close()

if __name__ == '__main__':
    populateParam()
    debug('== IMPORT DOC2VEC MODEL ==')
    doc2vec = loadDoc2Vec('full')
    """ TRAIN MODE """
    debug('======= TRAIN MODE =======')
    dataPath = config['TRAIN_NN']['path']
    fileList = config['TRAIN_NN']['files']
    data = constructData(dataPath, fileList)
    mlp = trainNN(doc2vec, data)
    """ VALIDATION MODE """
    debug('======= VALIDATION =======')
    dataPath = config['VALIDATION']['path']
    fileList = config['VALIDATION']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['VALIDATION']['predictions']
    predict(doc2vec, data, output, mlp)
    """ TEST MODE """
    debug('======= TEST MODE =======')
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['TEST_NN']['2016']['predictions']
    predict(doc2vec, data, output, mlp)
    debug('======== FINISHED ========')
