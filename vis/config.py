dev_text = '/home/vchahune/data/ldmt/lm-exp/tur/corpus/dev.txt'
model_pickle = '/home/vchahune/data/ldmt/lm-exp/tur/models/unigram-mp.1000.pickle'
fst = '/home/vchahune/data/ldmt/analyzers/tur/rev-generator.fst'
backend = 'xfst'

"""
dev_text = '/home/vchahune/data/ldmt/lm-exp/kin-small/corpus/dev.txt'
model_pickle = '/home/vchahune/data/ldmt/lm-exp/kin-small/models/unigram-mp.1000.pickle'
fst = '/home/vchahune/data/ldmt/analyzers/kin/2012-10-24.fst'
backend = 'foma'
"""

# Load training sentences
with open(dev_text) as dev:
    sentences = [s.decode('utf8') for s in dev]

# Load MorphoLM
import sys
sys.path.append('../morpholm')

# Load analyzer
import analysis
analyzer = analysis.Analyzer(fst, backend=backend)

# Load model
import cPickle
with open(model_pickle) as m:
    data = cPickle.load(m)
model = data['model']
morpho_process = model.base
vocabularies = data['vocabularies']
word_analyses = data['analyses']

import logging

def analyze_sentence(sentence):
    for word in sentence.split():
        w = vocabularies['word'][word]
        try:
            if not w in word_analyses:
                analyses = analyzer.get_analyses(word)
                yield [analysis.Analysis(ana, vocabularies) for ana in analyses]
            else:
                yield word_analyses[w]
        except analysis.AnalysisError as ana:
            logging.error(u'Analysis error for {0}: {1}'.format(word, ana))
            yield [analysis.Analysis(word, vocabularies)]

# Color palette
import colorsys
import random
n_colors = float(len(vocabularies['morpheme']))
morpheme_colors = dict((morpheme, colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.8)) 
        for (i, morpheme) in enumerate(sorted(vocabularies['morpheme'].id2word,
            key=lambda w: hash(w))))


logging.info('ready')
