# gensim
from gensim import utils

# nltk-corpus
from nltk.corpus import stopwords
cached_stopwords = set( stopwords.words("english") )

""" process 'doc' string into 'word_list' """
""" 'word_list' intended as an input to 'infer_vector' """
def preprocessor(sentence):
    # convert 'sentence' to unicode
    sentence = utils.to_unicode(sentence)
    sentence = sentence.lower().split()
    # generating 'word_list' excluding all 'stopwords'
    word_list = [ w for w in sentence if w not in cached_stopwords ]
    return word_list

""" implements consine similarity metric for two vectors """
""" vectors are normalized before computing dot product """
def consine(vec_a, vec_b):
    vec_a /= norm(vec_a)
    vec_b /= norm(vec_b)
    sim = dot(vec_a, vec_b)
    return sim