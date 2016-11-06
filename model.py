import json
from xml.dom import minidom

# pre-processing utilities
from myutils import preprocessor, consine

config = json.load(open('config.json', 'r'))
dataPath = config['TRAIN_NN']['dataPath']
fileList = config['TRAIN_NN']['fileList']

def loadDoc2Vec():
    modelPath = config['DOC2VEC']['modelPath']
    modelName = config['DOC2VEC']['modelName']
    doc2vec = Doc2Vec.load(modelPath + modelName)
    return doc2vec

def predictOne(doc2vec, data):
    """ Answer Reranking with rank ~ cosine(q_i, a_i)^(-1) """
    # data : zip(questions, commentsL) ... see 'constructData'
    for q, cl in data:
        q_w = preprocessor(q)
        q_v = doc2vec.infer_vector(q_w)
        for j, c in enumerate(cl):
            c_w = preprocessor(c[0])
            c_v = doc2vec.infer_vector(c_w)
            score.append( ( consine(q_v, c_v), j ) )
        score = sorted(score)
        score.reverse()

def constructData():
    """ constructs question and related comments data """
    # returns data = zip(questions, commentsL)
    # questions : list of questions
    #   ( [q1 q2 ... qN])
    # commentsL : list of list of (comment, label) pairs 
    #   ( [ [(c1, l1) (c2, l2) ... (cK1, lK1)] ... [(c1, l1) (c2, l2) ... (cKN, lKN)] ])
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
    doc2vec = loadDoc2Vec()
    data = constructData()
    predictOne(doc2vec, data)
