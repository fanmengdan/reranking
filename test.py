import sys

# gensim modules
from gensim import utils
from gensim.models import Doc2Vec

# pre-processing utilities
from myutils import preprocessor

# numpy
from numpy import dot
from numpy.linalg import norm

# random
from random import shuffle

# """ Finding similar words """
# print '\n\nWords similar to "bank"'
# for x in model.most_similar('bank'):
#     print '\t', x
# print '\n\nWords similar to "qatar"'
# for x in model.most_similar('qatar'):
#     print '\t', x

""" Computing squared-error on infered Para2Vec representations """

""" path to FULLDATA.txt, an aggregation of all data (questions and comments) of SEMEVAL'16 dataset (TASK 3 - SUBTASK A) """

home = '/media/sandeshc/Windows_OS/Sandesh/'
# data_path = home + 'MTP/Data/semeval2016-task3/useful/train-DR/FULLDATA.txt'
data_path = home + 'MTP/Data/semeval2016-task3/useful/train-DR/SMALLDATA.txt'

data_prefix = data_path.split('/')[-1].split('.')[0]

""" Loading trained Doc2Vec model """
windowsize = int(sys.argv[1])
dimension = int(sys.argv[4])
nepoch = int(sys.argv[2])
mode = sys.argv[3]
name_tuple = ( data_prefix.strip('DATA').lower(), windowsize, nepoch )
model = Doc2Vec.load('./models/' + mode + '/' + str(dimension) + 'd' + '/semeval-%s-lc-ns-%dw-%de.d2v' % name_tuple)

nsamp = 0
sqerr = 0.0
nsqerr = 0.0
sentences = []
with utils.smart_open(data_path) as fin:
    for item_no, line in enumerate(fin):
        sentences.append(line)
        words = preprocessor(line)
        model_v = model.docvecs[ data_prefix + '_%s' % item_no ]
        infer_v = model.infer_vector(words)
        sim = dot(model_v, infer_v)
        sqerr += ( ( 1 - sim ) * ( 1 - sim ) )
        model_v /= norm(model_v)
        infer_v /= norm(infer_v)
        sim = dot(model_v, infer_v)
        nsqerr += ( ( 1 - sim ) * ( 1 - sim ) )
        nsamp += 1

rsqerr = 0.0
rnsqerr = 0.0
shuffle(sentences)
for item_no in range(nsamp):
    words = preprocessor(sentences[item_no])
    infer_v = model.infer_vector(words)
    model_v = model.docvecs[ data_prefix + '_%s' % item_no ]
    sim = dot(model_v, infer_v)
    rsqerr += ( ( 1 - sim ) * ( 1 - sim ) )
    model_v /= norm(model_v)
    infer_v /= norm(infer_v)
    sim = dot(model_v, infer_v)
    rnsqerr += ( ( 1 - sim ) * ( 1 - sim ) )

print 'Results over %d samples :' % nsamp
print 'Squared Error is %lf' % (sqerr/nsamp)**0.5
print 'Normalized Squared Error is %lf' % (nsqerr/nsamp)**0.5
print 'Random Inferered Vector - Squared Error is %lf' % (rsqerr/nsamp)**0.5
print 'Random Inferered Vector - Normalized Squared Error is %lf' % (rnsqerr/nsamp)**0.5

record = [mode, dimension, windowsize, nepoch, nsamp, (sqerr/nsamp)**0.5, (nsqerr/nsamp)**0.5, (rsqerr/nsamp)**0.5, (rnsqerr/nsamp)**0.5]

out = open('out/test-' + data_prefix.strip('DATA').lower() + '.log', 'a')
out.write(','.join([str(x) for x in record]) + '\n')