import argparse
import logging
import cPickle
from vpyp.ngram.arpa import print_arpa
from vpyp.corpus import Corpus
from ..analyze import load_analyses
from .eval import fix_model

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Export n-gram model as ARPA file')
    parser.add_argument('--backend', help='analyzer backend (foma/xfst/pymorphy/xerox)',
            default='foma')
    parser.add_argument('--analyzer', help='analyzer option')
    parser.add_argument('--addnull', help='add null analysis', action='store_true')
    parser.add_argument('--vocab', help='test corpus vocabulary (default: training vocabulary)')
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)

    fix_model(model)

    if args.vocab:
        assert args.analyzer
        logging.info('Reading vocabulary')
        with open(args.vocab) as vocab:
            vocab_corpus = Corpus(vocab, model.vocabulary)
        load_analyses(vocab_corpus, model)

        from ..analyzers import all_analyzers
        analyzer = all_analyzers[args.backend](args.analyzer)
        logging.info('Analyzing vocabulary words using %s', analyzer)
        analyzer.analyze_corpus(vocab_corpus, args.addnull)

        vocabulary = set(seg[0] for seg in vocab_corpus.segments)
    else:
        logging.info('Using training corpus vocabulary')
        vocabulary = set(xrange(2, len(model.vocabulary)))

    logging.info('Creating ARPA file')
    print_arpa(model, vocabulary)

if __name__ == '__main__':
    main()
