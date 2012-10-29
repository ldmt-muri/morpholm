import logging
import argparse
import math
import cPickle
import sys
from corpus import ngrams, encode_corpus
from analysis import Analyzer, analyze_corpus
from model import MorphoProcess

def print_ppl(model, corpus):
    n_words = 0
    loglik = 0
    for sentence in corpus.sentences:
        for seq in ngrams(sentence, model.order):
            n_words += 1
            loglik += math.log(model.prob(seq))
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)

get_base = lambda model: model if isinstance(model, MorphoProcess) else get_base(model.backoff)

def decode_corpus(model, corpus, vocabularies):
    for word in corpus:
        p, dec = model.decode(word)
        print(dec.decode(vocabularies).encode('utf8'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate n-gram MorphoLM')
    parser.add_argument('--fst', help='compiled morphoanalyzer')
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--notrust', help='keep non-analyzed word', action='store_true')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--ppl', help='compute perplexity', action='store_true')
    parser.add_argument('--decode', help='analyze input', action='store_true')
    args = parser.parse_args()

    if args.fst:
        logging.info('Loading trained model')
        with open(args.model) as f:
            data = cPickle.load(f)
        vocabularies = data['vocabularies']
        word_analyses = data['analyses']
        model = data['model']

        logging.info('Using model: %s', model)

        logging.info('Analyzing test corpus')
        analyzer = Analyzer(args.fst, args.backend, not args.notrust)
        test_corpus = analyze_corpus(sys.stdin, analyzer, vocabularies, word_analyses)

        if args.ppl:
            logging.info('Computing test corpus perplexity')
            print_ppl(model, test_corpus)
        elif args.decode:
            logging.info('Decoding')
            decode_corpus(get_base(model), test_corpus, vocabularies)

    else:
        logging.info('Loading trained model')
        with open(args.model) as f:
            data = cPickle.load(f)
        vocabulary = data['vocabulary']
        model = data['model']

        test_corpus = encode_corpus(sys.stdin, vocabulary)

        logging.info('Computing test corpus perplexity')
        print_ppl(model, test_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
