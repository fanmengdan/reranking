""" WORD CLUSTERS (K-MEANS) """
import json

import numpy as np

from myutils import debug

# gensim modules
from gensim.models import Doc2Vec

# KMeans clustering
from sklearn.cluster import KMeans

# Model persistence
from sklearn.externals import joblib

config = json.load(open('config.json', 'r'))

cluster_cache = {}

debug('====== IMPORTING DOC2VEC MODEL ======')
modelPath = config['DOC2VEC']['full']['path']
modelName = config['DOC2VEC']['full']['name']
doc2vec = Doc2Vec.load(modelPath + modelName)

debug('====== CONSTRUCTING DATA POINTS ======')
vocab = doc2vec.vocab.keys()
X = np.array([ doc2vec[w] for w in vocab ])
X.dtype = np.float64

debug('====== RUNNING KMEANS ======')
kmeans = KMeans(n_clusters=1000).fit(X)
joblib.dump(kmeans, 'models/cluster/kmeans.pkl')

debug('====== SAVING RESULTS ======')
for i, w in enumerate(vocab):
    cluster_cache[w] = int(kmeans.labels_[i])
json.dump(cluster_cache, open('cluster_cache.json', 'w'))

debug('====== FINISHED ======')