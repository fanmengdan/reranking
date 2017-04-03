# min-heap library
import heapq as hq

# pre-processing utilities
from myutils import consine

# Standford POS Tagger
# CLASSPATH env-var contains path to JAR
from nltk.tag import StanfordPOSTagger

POS_TAGS = [ "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", \
    "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", \
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", \
    "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", "''", "(", ")", ",", ".", ":", "``" ]

def auxAdd(x, y):
    if x in None:
        return y
    x += y
    return x

""" Get semantic and metadata features """
def getFeatures(model, q_w, c_w, config):
    # model : doc2vec model trained on corpus
    # q_w   : words of question text
    # c_w   : words of comment text
    feature_vector = []
    
    ## Semantic features (x52)

    # Question to Comment similarity (x1)
    q_cv = None
    for w in q_w:
        if q_cv is None:
            q_cv = model[w]
        else:
            q_cv += model[w]
    q_cv = [ float(x)/len(q_w) for x in q_cv ]
    c_cv = None
    for w in c_w:
        if c_cv is None:
            c_cv = model[w]
        else:
            c_cv += model[w]
    c_cv = [ float(x)/len(c_w) for x in c_cv ]
    feature_vector.append(cosine(q_cv, c_cv))

    # Maximized similarity (x5)
    maxsims = []
    for w in c_w:
        hq.heappush(maxsims, cosine(q_cv, model[w]))
        if len(maxsims) > 5:
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

    # Part of speech (POS) based word vector similarities (x45)
    classpath = config['POS_TAG']['classpath']
    modelpath = classpath + config['POS_TAG']['modelpath']
    tagger = StanfordPOSTagger(modelpath, classpath)
    tag_qw = tagger.tag(q_w)
    tag_cw = tagger.tag(c_w)
    
    dict_t = {}
    for tag in POS_TAGS:
        dict_t['q'][tag] = ( None, 0 )
        dict_t['c'][tag] = ( None, 0 )
    for w_tag in tag_qw:
        dict_t['q'][w_tag[1]][1] += 1
        dict_t['q'][w_tag[1]][0] = auxAdd(dict_t['q'][w_tag[1]][0], model[w_tag[0]])
    for w_tag in tag_cw:
        dict_t['c'][w_tag[1]][1] += 1
        dict_t['c'][w_tag[1]][0] = auxAdd(dict_t['c'][w_tag[1]][0], model[w_tag[0]])
    for tag in POS_TAGS:
        if dict_t['q'][tag] == ( None, 0 ):
            feature_vector.append(0)
            continue
        if dict_t['c'][tag] == ( None, 0 ):
            feature_vector.append(0)
            continue
        avg_tq = [ float(x) / dict_t['q'][tag][1] for x in dict_t['q'][tag][0] ]
        avg_tc = [ float(x) / dict_t['c'][tag][1] for x in dict_t['c'][tag][0] ]
        feature_vector.append(cosine(avg_tq, avg_tc))

    # Word clusters (WC) similarity (SKIPPED)
    # LDA topic similarity (SKIPPED)

    ## Metadata features (x4)

    # Comment contains a question mark (x1)
    feature_vector.append(True in [ '?' in w for w in c_w ])

    # Comment length (x1)
    feature_vector.append(len(c_w))

    # Question length (x1)
    feature_vector.append(len(q_w))

    # Question to comment length (x1)
    feature_vector.append(float(len(q_w))/len(c_w))

    # Question and comment author same ? (SKIPPED)

    return feature_vector