# gensim
from gensim import utils

# nltk-corpus
from nltk.corpus import stopwords
cached_stopwords = set( stopwords.words("english") )

def preprocessor(sentence):
    sentence = utils.to_unicode(sentence)
    sentence = sentence.lower().split()
    word_list = [ w for w in sentence if w not in cached_stopwords ]
    return word_list