""" LDA TOPIC SIMILARITY """
import json

# gensim corpus lib
from gensim import corpora

# LDA topic modeling lib
from gensim.models.ldamodel import LdaModel

# pre-processing utilities
from myutils import preprocessor, constructData, debug

config = json.load(open('config.json', 'r'))

dataPath = config['TRAIN_NN']['path']
fileList = config['TRAIN_NN']['files']
data = constructData(dataPath, fileList)

debug('====== CONSTRUCTING DOCS AND TEXTS ======')
docs = []
for q, cl in data:
    docs.append(q[1])
    for c in cl:
        docs.append(c[1])
texts = [ preprocessor(d) for d in docs ]

debug('====== CONSTRUCTING DICTIONARY ======')
dictionary = corpora.Dictionary(texts)
dictionary.save('models/lda/semeval.dict')

debug('====== CONSTRUCTING CORPUS ======')
corpus = [ dictionary.doc2bow(text) for text in texts ]
corpora.MmCorpus.serialize('models/lda/semeval.mm', corpus)

debug('====== CONSTRUCTING LDA MODEL ======')
lda = LdaModel(corpus, num_topics=100)
lda.save('models/lda/semeval.lda')

debug('====== FINISHED ======')