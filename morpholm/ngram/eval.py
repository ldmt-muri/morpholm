import argparse
import logging
import cPickle
from vpyp.corpus import Corpus
from vpyp.ngram.eval import print_ppl

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

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Evaluate n-gram model')
    parser.add_argument('--backend', help='analyzer backend (foma/xfst/pymorphy/xerox)',
            default='foma')
    parser.add_argument('--analyzer', help='analyzer option', required=True)
    parser.add_argument('--addnull', help='add null analysis', action='store_true')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)
    fix_model(model)

    logging.info('Reading evaluation corpus')
    with open(args.test) as test:
        test_corpus = Corpus(test, model.vocabulary)
    test_corpus.stem_vocabulary = model.stem_vocabulary
    test_corpus.morpheme_vocabulary = model.morpheme_vocabulary
    test_corpus.morpheme_vocabulary.frozen = True
    test_corpus.pattern_vocabulary = model.pattern_vocabulary
    test_corpus.analyses = model.analyses

    from ..analyzers import all_analyzers
    analyzer = all_analyzers[args.backend](args.analyzer)
    logging.info('Analyzing evaluation corpus using %s', analyzer)
    analyzer.analyze_corpus(test_corpus, args.addnull)

    logging.info('Computing perplexity')
    print_ppl(model, test_corpus)

if __name__ == '__main__':
    main()
