""" Helper script for POS tagging of train-dev-test data """

import json

# pre-processing utilities
from myutils import preprocessor, tagsToString, constructData

# Standford POS Tagger
# CLASSPATH env-var contains path to JAR
from nltk.tag import StanfordPOSTagger
from nltk.internals import config_java

config = json.load(open('config.json', 'r'))

postagpath = config['POS_TAG']['path']
classpath = postagpath + config['POS_TAG']['jar']
modelpath = postagpath + config['POS_TAG']['model']
tagger = StanfordPOSTagger(modelpath, classpath)
config_java(options='-Xms4096M -Xmx4096M', verbose=False)

tagger_cache = {}
unique_tags = []

def findUniqueTags(tags):
    global unique_tags
    for t in tags:
        if t[1] not in unique_tags:
            unique_tags.append(t[1])

def addToCache(id, wl):
    global tagger, tagger_cache
    if tagger_cache.get(id) is not None:
        return
    tags = tagger.tag(wl)
    findUniqueTags(tags)
    tagger_cache[id] = tagsToString(tags)

def POSTag(data):
    for q, cl in data:
        q_w = preprocessor(q[1])
        addToCache(q[0], q_w)
        for c in cl:
            c_w = preprocessor(c[1])
            addToCache(c[0], c_w)

print '======= TRAIN DATA ======='
dataPath = config['TRAIN_NN']['path']
fileList = config['TRAIN_NN']['files']
data = constructData(dataPath, fileList)
POSTag(data)

print '======= TEST DATA \'16 ======='
dataPath = config['TEST_NN']['path']
fileList = config['TEST_NN']['2016']['files']
data = constructData(dataPath, fileList)
POSTag(data)

print '======= TEST DATA \'17 ======='
dataPath = config['TEST_NN']['path']
fileList = config['TEST_NN']['2017']['files']
data = constructData(dataPath, fileList)
POSTag(data)

json.dump(tagger_cache, open('tagger_cache.json', 'w'))

print '====== UNIQUE TAGS ======\n', unique_tags
print '====== FINISHED ======'
