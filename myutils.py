# regex
import re

# gensim
from gensim import utils

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
    sentence = re.sub('[!"#%\'()*+,-./:;<=>@\[\]^_{|}~1234567890`\\\]', ' ', sentence)
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