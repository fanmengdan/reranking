# regex
import re

# sys (flush)
import sys

# gensim
from gensim import utils

# xml parsing
from xml.dom import minidom

# numpy
from numpy import dot
from numpy.linalg import norm

# nltk-corpus
from nltk.corpus import stopwords
cached_stopwords = set( stopwords.words("english") )

""" process 'doc' string into 'word_list' """
""" 'word_list' intended as an input to 'infer_vector' """
def preprocessor(sentence):
    # convert 'sentence' to unicode
    sentence = utils.to_unicode(sentence)
    # remove numerics, special characters (except '?')
    sentence = re.sub('[$&!"#%\'()*+,-./:;<=>@\[\]^_{|}~1234567890`\\\]', ' ', sentence)
    # separate word from '?'
    sentence = re.sub('[?]', ' ? ', sentence)
    sentence = sentence.lower().split()
    # generating 'word_list' excluding all 'stopwords'
    word_list = [ w for w in sentence if w not in cached_stopwords ]
    return word_list

""" implements cosine similarity metric for two vectors """
""" vectors are normalized before computing dot product """
def cosine(vec_a, vec_b):
    vec_a /= norm(vec_a)
    vec_b /= norm(vec_b)
    sim = dot(vec_a, vec_b)
    return sim

""" converts a list of tuples of form [ (word, pos_tag) ... ] """
""" into string of the form 'word+pos_tag|word+pos_tag|...' """
def tagsToString(tags):
    stags = [ t[0] + '+' + t[1] for t in tags ]
    return '|'.join(stags)

""" inverse function to tagsToString """
def stringToTags(tagStr):
    stags = tagStr.split('|')
    tags = [ tuple(s.split('+')) for s in stags ]
    return tags

DB = '<< DEBUG >>'
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
