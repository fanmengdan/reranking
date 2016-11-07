import json
from xml.dom import minidom

# pre-processing utilities
from myutils import preprocessor, consine

config = json.load(open('config.json', 'r'))

def loadDoc2Vec(mode):
    modelPath = config['DOC2VEC'][mode]['path']
    modelName = config['DOC2VEC'][mode]['name']
    doc2vec = Doc2Vec.load(modelPath + modelName)
    return doc2vec

def predictOneAux(score):
    if score >= 0.5:
        return 'true'
    return 'false'

def predictOne(doc2vec, data, output):
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
            scores.append( ( consine(q_v, c_v), j ) )
        scores = sorted(scores, reverse=True)
        for rank, score in enumerate(scores):
            pred = predictOneAux(score[0])
            out.write('\t'.join([q[0], cl[score[1]], str(rank+1), str(score[0]), pred]))
            out.write('\n')
    out.close()

def constructData(dataPath, fileList):
    """ constructs question and related comments data """
    # returns data = zip(questions, commentsL)
    # questions : list of (questionId, question) pairs
    #   ( [(qid1, q1) (qid2, q2) ... (qidN, qN)] )
    # commentsL : list of list of (commentId, comment, label) pairs
    #   ( [ [(cid1, c1, l1) (cid2, c2, l2) ... (cidK1, cK1, lK1)] ... [(cid1, c1, l1) (cid2, c2, l2) ... (cidKN, cKN, lKN)] ] )
    labels = []
    questions = []
    commentsL = []
    for xmlFile in fileList:
        doc = minidom.parse(dataPath + xmlFile)
        threads = doc.getElementsByTagName("Thread")
        for thread in threads:
            # constructing the question string
            relQ = thread.getElementsByTagName('RelQuestion')[0]
            Qid = relQ.getAttribute('RELQ_ID')
            body = relQ.getElementsByTagName('RelQBody')[0]._get_firstChild().data
            subj = relQ.getElementsByTagName('RelQSubject')[0]._get_firstChild().data
            questions.append( (Qid, subj + ' ' + body) )
            # constructing the list of comments
            comments = []
            for relC in thread.getElementsByTagName('RelComment'):
                Cid = relC.getAttribute('RELC_ID')
                label = relC.getAttribute('RELC_RELEVANCE2RELQ')
                comment = relC.getElementsByTagName('RelCText')[0]._get_firstChild().data
                comments.append( (Cid, comment, label) )
            commentsL.append(comments)
    return zip(questions, commentsL)

if __name__ == '__main__':
    doc2vec = loadDoc2Vec('small')
    # """ TRAIN MODE """
    # print '======= TRAIN MODE ======='
    # dataPath = config['TRAIN_NN']['path']
    # fileList = config['TRAIN_NN']['files']
    # data = constructData(dataPath, fileList)
    # # TODO : TRAIN CODE
    """ VALIDATION MODE """
    print '======= VALIDATION ======='
    dataPath = config['VALIDATION']['path']
    fileList = config['VALIDATION']['files']
    data = constructData(dataPath, fileList)
    output = datapath + config['VALIDATION']['predictions']
    predictOne(doc2vec, data, output)
    """ TEST MODE """
    print '======= TEST MODE ======='
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['TEST_NN']['2016']['predicitons']
    predictOne(doc2vec, data, output)
    print '======== FINISHED ========'
