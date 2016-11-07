import json
from xml.dom import minidom

# gensim modules
from gensim.models import Doc2Vec

# pre-processing utilities
from myutils import preprocessor, consine

DB = '<< DEBUG >>'
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
            score = ( 1.0 + consine(q_v, c_v) ) / 2.0;
            scores.append( [ score, j, 0 ] )
        scores = sorted(scores, key=lambda score: score[0], reverse=True)
        for i in range(len(scores)):
            scores[i][2] = i + 1
        scores = sorted(scores, key=lambda score: score[1])
        for score in scores:
            pred = predictOneAux(score[0])
            out.write('\t'.join([q[0], cl[score[1]][0], str(score[2]), str(score[0]), pred]))
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
        print DB, dataPath + xmlFile
        doc = minidom.parse(dataPath + xmlFile)
        threads = doc.getElementsByTagName("Thread")
        for tid, thread in enumerate(threads):
            # constructing the question string
            relQ = thread.getElementsByTagName('RelQuestion')[0]
            Qid = relQ.getAttribute('RELQ_ID')
            bodyQ = relQ.getElementsByTagName('RelQBody')[0]
            body = bodyQ._get_firstChild().data if bodyQ._get_firstChild() is not None else ''
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
    print '== IMPORT DOC2VEC MODEL =='
    doc2vec = loadDoc2Vec('full')
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
    output = dataPath + config['VALIDATION']['predictions']
    predictOne(doc2vec, data, output)
    """ TEST MODE """
    print '======= TEST MODE ======='
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    output = dataPath + config['TEST_NN']['2016']['predictions']
    predictOne(doc2vec, data, output)
    print '======== FINISHED ========'
