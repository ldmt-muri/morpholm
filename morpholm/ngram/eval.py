import argparse
import logging
import cPickle
from vpyp.corpus import Corpus
from vpyp.ngram.eval import print_ppl
from ..analyze import load_analyses

""""""""""""""""""""""""""""""""""""
from vpyp.charlm import CharLM
from vpyp.pyp import PYP
from vpyp.ngram.model import PYPLM
from ..models import MorphoProcess
def fix_model(model):
    def get_charlm(m):
        if isinstance(m, CharLM):
            return m
        elif isinstance(m, MorphoProcess):
            return get_charlm(m.stem_model)
        elif isinstance(m, PYPLM):
            return get_charlm(m.backoff)
        elif isinstance(m, PYP):
            return get_charlm(m.base)
    model.stem_vocabulary = get_charlm(model).vocabulary
""""""""""""""""""""""""""""""""""""

def get_match_level(model, seq, vocabulary):
    if isinstance(model, PYPLM):
        m = model.models.get(seq[:-1], None)
        if m and seq[-1] in m.tables:
            return ' | '+' '.join(vocabulary[w] for w in seq[:-1])+') ~ '+str(model.order)+'-gram'
        elif isinstance(model.backoff, MorphoProcess):
            return ') ~ MorphoProcess'
        elif isinstance(model.backoff, CharLM):
            return ') ~ char LM'
        else:
            return get_match_level(model.backoff, seq[1:], vocabulary)

import math
from vpyp.corpus import ngrams, STOP
def print_debug(model, corpus):
    n_words = 0
    n_sentences = 0
    total_lp = 0
    for sentence in corpus:
        n_sentences += 1
        for seq in ngrams(sentence, model.order):
            if seq[-1] != STOP:
                n_words += 1
            word = model.vocabulary[seq[-1]]
            lp = math.log10(model.prob(seq[:-1], seq[-1]))
            total_lp += lp
            level = get_match_level(model, seq, model.vocabulary)
            print(u'p({0}{1}) = {2}'.format(word, level, lp).encode('utf8'))
    print('logprob = {0} ppl = {1}'.format(total_lp, 10**(-total_lp/(n_words+n_sentences))))

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Evaluate n-gram model')
    parser.add_argument('--backend', help='analyzer backend (foma/xfst/pymorphy/xerox)',
            default='foma')
    parser.add_argument('--analyzer', help='analyzer option', required=True)
    parser.add_argument('--addnull', help='add null analysis', action='store_true')
    parser.add_argument('--rmnull', help='remove null analysis', action='store_true')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--debug', help='debug ppl computation', action='store_true')

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)
    # fix_model(model)

    logging.info('Reading evaluation corpus')
    with open(args.test) as test:
        test_corpus = Corpus(test, model.vocabulary)
    load_analyses(test_corpus, model)

    from ..analyzers import all_analyzers
    analyzer = all_analyzers[args.backend](args.analyzer)
    logging.info('Analyzing evaluation corpus using %s', analyzer)
    analyzer.analyze_corpus(test_corpus, args.addnull, args.rmnull)

    logging.info('Computing perplexity')
    if args.debug:
        print_debug(model, test_corpus)
    else:
        print_ppl(model, test_corpus)

if __name__ == '__main__':
    main()
