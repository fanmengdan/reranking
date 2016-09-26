# nltk-corpus
from nltk.corpus import stopwords
cached_stopwords = set( stopwords.words("english") )

def preprocessor(sentence):
    sentence = sentence.lower().split()
    word_list = [ w for w in sentence if w not in cached_stopwords ]
    return word_list