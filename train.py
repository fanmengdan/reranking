# gensim modules
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

# random
from random import shuffle

# pre-processing utilities
from myutils import preprocessor

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(preprocessor(utils.to_unicode(line)), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(preprocessor(utils.to_unicode(line)), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

""" path to FULLDATA.txt, an aggregation of all data (questions and comments) of SEMEVAL'16 dataset (TASK 3 - SUBTASK A) """
data_file = 'FULLDATA' # 'SMALLDATA'
data_path = '/media/sandeshc/Windows_OS/Sandesh/MTP/Data/semeval2016-task3-cqa-ql-traindev-v3.2/useful/train-DR/' + data_file + '.txt'
data_prefix = data_path.split('/')[-1].split('.')[0]

sources = { data_path : data_prefix }
sentences = LabeledLineSentence(sources)
print '#1 Constructed "LabeledLineSentences"'

""" Constructing Doc2Vec Model """
# negative  : negative-sampling [https://www.quora.com/What-is-negative-sampling]
# window--  : window size of skip-gram model [http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf]
# dbow_words: if set to 1 trains word-vectors (in skip-gram fashion) 
#   simultaneous with DBOW doc-vector training;
#   default is 0 (faster training of doc-vectors only).
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
print '#2 Constructed "Doc2Vec" Model'

""" Building the Model Vocabulary """
model.build_vocab(sentences.to_array())
print '#3 Building Vocabulary Done'

""" Training the Model """
nepoch = 10
for epoch in range(nepoch):
    model.train(sentences.sentences_perm())
    print '#4 Training Model on %s/%s epoch Done' % ( (epoch+1), nepoch )

""" Save the model for future usage """
model.save('./models/semeval-%s-lowercase-nostopwords.d2v' % data_prefix.strip('DATA').lower())