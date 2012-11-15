"""
# Turkish
dev_text = '/home/vchahune/data/ldmt/lm-exp/tur/corpus/dev.txt'
model_pickle = '/home/vchahune/data/ldmt/lm-exp/tur/models/unigram-mp.1000.pickle'
backend = 'xfst'
analyzer = '/home/vchahune/data/ldmt/analyzers/tur/rev-generator.fst'
add_null = False
"""

# Kinyarwanda
dev_text = '/home/vchahune/data/ldmt/lm-exp/kin-bbc/2012-11-13/corpus/dev.txt'
model_pickle = '/home/vchahune/data/ldmt/lm-exp/kin-bbc/2012-11-13/models/unigram-mp2.1000.pickle'
backend = 'foma'
analyzer = '/home/vchahune/data/ldmt/analyzers/kin/2012-11-13.fst'
add_null = False
annotation_database = '/home/vchahune/data/ldmt/annotation/kin/test.db'

"""
# Russian
dev_text = '/home/vchahune/data/ldmt/lm-exp/rus/corpus/dev.txt'
model_pickle = '/home/vchahune/data/ldmt/lm-exp/rus/models_guesser/unigram-mp2.1000.pickle'
backend = 'foma'
analyzer = '/home/vchahune/data/ldmt/analyzers/rus/rusmorph.fst'
add_null = True
annotation_database = '/home/vchahune/data/ldmt/annotation/rus/test.db'
"""

users = {'chahuneau': 'victor',
         'dyer': 'chris',
         'smith': 'noah',
         'baldridge': 'jason',
         'mielens': 'jason',
         'jerro': 'kyle'}

n_sentences = 100

# Load training sentences
with open(dev_text) as dev:
    sentences = [s.decode('utf8') for s in dev][:n_sentences]

import logging
logging.basicConfig(level=logging.INFO)

# Load analyzer
from morpholm.analyzers import all_analyzers
from morpholm.analyzers.analyzer import parse_analysis, Analysis
analyzer = all_analyzers[backend](analyzer)
logging.info('Using %s for analysis', analyzer)

# Load model
import cPickle
with open(model_pickle) as m:
    model = cPickle.load(m)
morpho_process = model.backoff

# XXX
from morpholm.ngram.eval import fix_model
fix_model(model)
# XXX

# __STEM__ -> stem
S = model.morpheme_vocabulary['__STEM__']
model.morpheme_vocabulary.id2word[S] = 'stem'

from vpyp.corpus import Corpus

def analyze_sentence(sentence):
    corpus = Corpus([sentence.encode('utf8')], model.vocabulary)
    corpus.stem_vocabulary = model.stem_vocabulary
    corpus.morpheme_vocabulary = model.morpheme_vocabulary
    corpus.pattern_vocabulary = model.pattern_vocabulary
    corpus.analyses = model.analyses
    analyzer.analyze_corpus(corpus, add_null)
    return corpus.segments[0]

# Color palette
import colorsys
n_colors = float(len(model.morpheme_vocabulary))
morpheme_colors = dict((morpheme, colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.8)) 
        for (i, morpheme) in enumerate(sorted(model.morpheme_vocabulary.id2word,
            key=lambda w: hash(w))))


logging.info('ready')
