""" METADATA CONSTRUCTION """
import json

from myutils import debug

# xml parsing
from xml.dom import minidom

config = json.load(open('config.json', 'r'))

meta_cache = {}
unique_cats = []

def constructMetaData(dataPath, fileList):
    for xmlFile in fileList:
        debug(dataPath + xmlFile)
        doc = minidom.parse(dataPath + xmlFile)
        threads = doc.getElementsByTagName("Thread")
        for thread in threads:
            relQ = thread.getElementsByTagName('RelQuestion')[0]
            Qid = relQ.getAttribute('RELQ_ID')
            meta_cache[Qid] = { 'author' : relQ.getAttribute('RELQ_USERID'),
                'category' : relQ.getAttribute('RELQ_CATEGORY'),
                'time' : relQ.getAttribute('RELQ_DATE') }
            if meta_cache[Qid]['category'] not in unique_cats:
                unique_cats.append(meta_cache[Qid]['category'])
            user_tracker = {}
            for relC in thread.getElementsByTagName('RelComment'):
                Cid = relC.getAttribute('RELC_ID')
                meta_cache[Cid] = { 'author' : relC.getAttribute('RELC_USERID'),
                    'time' : relC.getAttribute('RELC_DATE') }
                if meta_cache[Cid]['author'] not in user_tracker:
                    user_tracker[ meta_cache[Cid]['author'] ] = 0
                user_tracker[ meta_cache[Cid]['author'] ] += 1
                meta_cache[Cid]['comment#'] = user_tracker[ meta_cache[Cid]['author'] ]
            for relC in thread.getElementsByTagName('RelComment'):
                Cid = relC.getAttribute('RELC_ID')
                meta_cache[Cid]['#comment'] = user_tracker[ meta_cache[Cid]['author'] ]

debug('======= TRAIN DATA =======')
dataPath = config['TRAIN_NN']['path']
fileList = config['TRAIN_NN']['files']
constructMetaData(dataPath, fileList)

debug('======= TEST DATA \'16 =======')
dataPath = config['TEST_NN']['path']
fileList = config['TEST_NN']['2016']['files']
constructMetaData(dataPath, fileList)

debug('======= TEST DATA \'17 =======')
dataPath = config['TEST_NN']['path']
fileList = config['TEST_NN']['2017']['files']
constructMetaData(dataPath, fileList)

json.dump(meta_cache, open('meta_cache.json', 'w'))

debug('====== UNIQUE CATEGORIES ======\n' + str(unique_cats))
debug('====== FINISHED ======')