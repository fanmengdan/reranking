# gensim modules
from gensim import utils
from gensim.models import Doc2Vec

# pre-processing utilities
from myutils import preprocessor

# numpy
from numpy import dot

""" Loading trained Doc2Vec model """
model = Doc2Vec.load('./models/semeval-small-lowercase-nostopwords.d2v')
print '#1 Loading trained Doc2Vec model Done'

# """ Finding similar words """
# print '\n\nWords similar to "bank"'
# for x in model.most_similar('bank'):
#     print '\t', x
# print '\n\nWords similar to "qatar"'
# for x in model.most_similar('qatar'):
#     print '\t', x

""" Computing squared-error on infered Para2Vec representations """

""" path to FULLDATA.txt, an aggregation of all data (questions and comments) of SEMEVAL'16 dataset (TASK 3 - SUBTASK A) """
# data_path = '/media/sandeshc/Windows_OS/Sandesh/MTP/Data/semeval2016-task3-cqa-ql-traindev-v3.2/useful/train-DR/FULLDATA.txt'
data_path = '/media/sandeshc/Windows_OS/Sandesh/MTP/Data/semeval2016-task3-cqa-ql-traindev-v3.2/useful/train-DR/SMALLDATA.txt'
data_prefix = data_path.split('/')[-1].split('.')[0]

nsamp = 0
sqerr = 0.0
with utils.smart_open(data_path) as fin:
    for item_no, line in enumerate(fin):
        words = preprocessor(line)
        model_v = model.docvecs[ data_prefix + '_%s' % item_no ]
        infer_v = model.infer_vector(words)
        sim = dot(model_v, infer_v)
        sqerr += ( ( 1 - sim ) * ( 1 - sim ) )
        nsamp += 1

print 'Squared Error over %d samples is %lf' % (nsamp, sqerr**0.5)