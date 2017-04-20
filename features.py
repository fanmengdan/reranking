import math

import json

# min-heap library
import heapq as hq

# parsing time/date
from datetime import datetime as dt

# pre-processing utilities
from myutils import cosine, stringToTags

# gensim corpus lib
from gensim import corpora

# LDA topic modeling lib
from gensim.models.ldamodel import LdaModel

POS_TAGS = ['RB', 'NN', 'UH', 'FW', 'VBG', '.', 'VBZ', 'NNS', 'PRP', 'VB', 'VBN', 'VBP', \
    'IN', 'JJS', 'JJ', 'CD', 'VBD', 'CC', 'RBR', 'MD', 'DT', 'NNP', 'JJR', 'WP', 'SYM', \
    'TO', 'LS', 'RP', 'WP$', 'WRB', 'WDT', 'RBS', 'PRP$', 'NNPS', 'PDT', 'POS']

ALL_CATS = {u'Life in Qatar': 29, u'Computers and Internet': 26, u'Investment and Finance': 28, \
    u'Opportunities': 7, u'Environment': 11, u'Family Life in Qatar': 3, u'Sports in Qatar': 19, \
    u'Welcome to Qatar': 4, u'Beauty and Style': 22, u'Pets and Animals': 9, u'Electronics': 27, \
    u'Cars': 30, u'Salary and Allowances': 6, u'Visas and Permits': 8, u'Socialising': 16, \
    u'Health and Fitness': 25, u'Qatar Musicians': 33, u'Cars and driving': 20, \
    u'Qatar Living Lounge ': 32, u'Qatar 2022': 31, u'Funnies': 17, \
    u'Sightseeing and Tourist attractions': 2, u'Language': 24, u'Qatar Living Lounge': 5, \
    u'Doha Shopping': 13, u'Qatar Living Tigers....': 23, u'Qatari Culture': 14, \
    u'Missing home!': 18, u'Working in Qatar': 15, u'Advice and Help': 10, u'Moving to Qatar': 1, \
    u'Politics': 12, u'Education': 21}

def auxAdd(x, y):
    if x is None:
        return y
    x += y
    return x

meta_cache = json.load(open('meta_cache.json', 'r'))
tagger_cache = json.load(open('tagger_cache.json', 'r'))
cluster_cache = json.load(open('cluster_cache.json', 'r'))

lda = LdaModel.load('models/lda/semeval.lda')
dictionary = corpora.Dictionary.load('models/lda/semeval.dict')

""" filter out and track non-vocabulary words """
def vfilter(vocab, meta, x_w, tag_xw):
    # vocab : corpus vocabulary
    # meta  : contains rank, Qid, Cid
    # x_w   : question/comment word list
    # tag_xw: postags corresp. to x_w
    x_wp = []
    tag_xwp = []

    i = 0
    while i < len(x_w):
        if i < len(tag_xw) and x_w[i] == tag_xw[i][0]:
            i += 1
            continue
        # print '<< features.py >>', '|',
        # print 'non-tagged word', ( x_w[i], ), '|',
        # print 'qid', meta['qid'], '|',
        # print 'cid', meta['cid'], '|'
        del x_w[i]

    for i in range( len(x_w) ):
        w = x_w[i]
        t = tag_xw[i]
        if w in vocab:
            x_wp.append(w)
            tag_xwp.append(t)
        # else:
            # print '<< features.py >>', '|',
            # print 'non-vocabulary word', ( w, ), '|',
            # print 'qid', meta['qid'], '|',
            # print 'cid', meta['cid'], '|'
    return x_wp, tag_xwp

""" Get semantic and metadata features """
def getFeatures(model, q_w, c_w, meta):
    # model : doc2vec model trained on corpus
    # q_w   : words of question text
    # c_w   : words of comment text
    # meta  : contains rank, Qid, Cid
    feature_vector = []

    global tagger_cache
    tag_qw = stringToTags( tagger_cache[meta['qid']] ) if len(q_w) else []
    tag_cw = stringToTags( tagger_cache[meta['cid']] ) if len(c_w) else []

    q_w, tag_qw = vfilter(model.vocab, meta, q_w, tag_qw)
    c_w, tag_cw = vfilter(model.vocab, meta, c_w, tag_cw)

    ## Semantic features (x45)

    # Question to Comment similarity (x1)
    q_cv = None
    for w in q_w:
        if q_cv is None:
            q_cv = model[w]
        else:
            q_cv += model[w]
    c_cv = None
    for w in c_w:
        if c_cv is None:
            c_cv = model[w]
        else:
            c_cv += model[w]
    if q_cv is not None and c_cv is not None:
        q_cv = [ float(x)/len(q_w) for x in q_cv ]
        c_cv = [ float(x)/len(c_w) for x in c_cv ]
        feature_vector.append(cosine(q_cv, c_cv))
    else:
        feature_vector.append(0)

    # Maximized similarity (x5)
    maxsims = [ 0.0 ] * 5
    for w in c_w:
        hq.heappush(maxsims, cosine(q_cv, model[w]))
        hq.heappop(maxsims)
    maxsims = sorted(maxsims, reverse=True)

    cntsim = 0
    sumsim = 0.0
    for sim in maxsims:
        cntsim += 1
        sumsim += sim
        avgsim = sumsim/cntsim
        feature_vector.append(avgsim)

    # Aligned similarity (x1)
    alisims = []
    for w_q in q_w:
        bestsim = 0.0
        for w_c in c_w:
            bestsim = max(bestsim, cosine(model[w_q], model[w_c]))
        alisims.append(bestsim)
    feature_vector.append(sum(alisims)/len(alisims))

    # Part of speech (POS) based word vector similarities (x36)
    dict_t = { 'q' : {}, 'c' : {} }
    for tag in POS_TAGS:
        dict_t['q'][tag] = [ None, 0 ]
        dict_t['c'][tag] = [ None, 0 ]
    for w_tag in tag_qw:
        dict_t['q'][w_tag[1]][1] += 1
        dict_t['q'][w_tag[1]][0] = auxAdd(dict_t['q'][w_tag[1]][0], model[w_tag[0]])
    for w_tag in tag_cw:
        dict_t['c'][w_tag[1]][1] += 1
        dict_t['c'][w_tag[1]][0] = auxAdd(dict_t['c'][w_tag[1]][0], model[w_tag[0]])
    for tag in POS_TAGS:
        if dict_t['q'][tag][1] == 0:
            feature_vector.append(0)
            continue
        if dict_t['c'][tag][1] == 0:
            feature_vector.append(0)
            continue
        avg_tq = [ float(x) / dict_t['q'][tag][1] for x in dict_t['q'][tag][0] ]
        avg_tc = [ float(x) / dict_t['c'][tag][1] for x in dict_t['c'][tag][0] ]
        feature_vector.append(cosine(avg_tq, avg_tc))

    # Word clusters (WC) similarity (x1)
    q_w_clus = {}
    for w in q_w:
        value = 1
        key = cluster_cache[w]
        if key in q_w_clus:
            value += q_w_clus[key]
        q_w_clus[key] = value
    q_w_clus = q_w_clus.items()
    q_w_clus = sorted(q_w_clus, key = lambda tup : tup[0])
    q_w_norm = ( sum([ tup[1]**2 for tup in q_w_clus ]) )**0.5

    c_w_clus = {}
    for w in c_w:
        value = 1
        key = cluster_cache[w]
        if key in c_w_clus:
            value += c_w_clus[key]
        c_w_clus[key] = value
    c_w_clus = c_w_clus.items()
    c_w_clus = sorted(c_w_clus, key = lambda tup : tup[0])
    c_w_norm = ( sum([ tup[1]**2 for tup in c_w_clus ]) )**0.5

    i = 0
    j = 0
    wc_sim = 0.0
    while i < len(q_w_clus) and j < len(c_w_clus):
        if q_w_clus[i][0] < c_w_clus[j][0]:
            i += 1
            continue
        if q_w_clus[i][0] > c_w_clus[j][0]:
            j += 1
            continue
        wc_sim += (q_w_clus[i][1] / q_w_norm) * (c_w_clus[j][1] / c_w_norm)
        i += 1
        j += 1
    feature_vector.append(wc_sim)

    # LDA topic similarity (x1)
    q_w_bow = dictionary.doc2bow(q_w)
    q_w_lda = lda.get_document_topics(q_w_bow, minimum_probability=0)
    q_w_lda = [ tup[1] for tup in q_w_lda ]

    c_w_bow = dictionary.doc2bow(c_w)
    c_w_lda = lda.get_document_topics(c_w_bow, minimum_probability=0)
    c_w_lda = [ tup[1] for tup in c_w_lda ]

    feature_vector.append(cosine(q_w_lda, c_w_lda))

    ## Metadata features (x43)

    # Comment contains a question mark (x1)
    feature_vector.append(True in [ '?' in w for w in c_w ])

    # Comment length (x1)
    feature_vector.append(len(c_w))

    # Question length (x1)
    feature_vector.append(len(q_w))

    # Question to comment length (x1)
    feature_vector.append(float(len(c_w))/len(q_w) if len(q_w) else 0)

    # Answer rank in the thread (x1)
    feature_vector.append(meta['rank'] + 1)

    # Question and comment author same ? (x1)
    feature_vector.append(meta_cache[meta['qid']]['author'] \
        == meta_cache[meta['cid']]['author'])

    # Question category (x33)
    q_cat = [ 0 ] * len(ALL_CATS)
    q_cat[ALL_CATS[ meta_cache[meta['qid']]['category'] ] - 1] = 1
    feature_vector.extend(q_cat)

    # Number of comments by the same user on question thread (x1)
    feature_vector.append(int( meta_cache[meta['cid']]['#comment'] ))

    # Order of the comment by the same user (x1)
    feature_vector.append(int( meta_cache[meta['cid']]['comment#'] ))

    # Hyperlink in the comment text (x1)
    feature_vector.append( any(w in ' '.join(c_w) for w in ['http', 'www']) )

    # Time difference between Question and Comment posting (x1)
    Qtime = dt.strptime(meta_cache[meta['qid']]['time'], '%Y-%m-%d %H:%M:%S')
    Ctime = dt.strptime(meta_cache[meta['cid']]['time'], '%Y-%m-%d %H:%M:%S')
    delta = math.fabs( ( Ctime - Qtime ).total_seconds() )
    feature_vector.append( math.log10(delta + 1) )

    return [ float(f) for f in feature_vector ]