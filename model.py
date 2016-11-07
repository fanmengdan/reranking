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

def predictOne(doc2vec, data):
    """ Answer Reranking with rank ~ cosine(q_i, a_i)^(-1) """
    # data : zip(questions, commentsL) ... see 'constructData'
    for q, cl in data:
        q_w = preprocessor(q)
        q_v = doc2vec.infer_vector(q_w)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[1])
            c_v = doc2vec.infer_vector(c_w)
            score.append( ( consine(q_v, c_v), j ) )
        score = sorted(score)
        score.reverse()
        # TODO : PRINT OUT RESULTS

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
            body = relQ.getElementsByTagName('RelQBody')[0]._get_firstChild().data
            subj = relQ.getElementsByTagName('RelQSubject')[0]._get_firstChild().data
            questions.append(subj + ' ' + body)
            # constructing the list of comments
            comments = []
            for relC in thread.getElementsByTagName('RelComment'):
                label = relC.getAttribute('RELC_RELEVANCE2RELQ')
                comment = relC.getElementsByTagName('RelCText')[0]._get_firstChild().data
                comments.append( (comment, label) )
            commentsL.append(comments)
    return zip(questions, commentsL)

if __name__ == '__main__':
    doc2vec = loadDoc2Vec('small')
    # TRAIN MODE
    dataPath = config['TRAIN_NN']['path']
    fileList = config['TRAIN_NN']['files']
    data = constructData(dataPath, fileList)
    predictOne(doc2vec, data)
    # TEST MODE
    dataPath = config['TEST_NN']['path']
    fileList = config['TEST_NN']['2016']['files']
    data = constructData(dataPath, fileList)
    predictOne(doc2vec, data)
