import sys
import logging
import argparse
import cPickle
import math
from analysis import Analyzer, analyze_corpus

def print_ppl(model, corpus):
    n_words = 0
    loglik = 0
    for word in corpus:
        n_words += 1
        loglik += math.log(model.prob(word))
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)

def decode_corpus(model, corpus, vocabularies):
    for word in corpus:
        p, dec = model.decode(word)
        print(dec.decode(vocabularies).encode('utf8'))

def main():
    parser = argparse.ArgumentParser(description='Evaluate 1-gram MorphoLM')
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--notrust', help='keep non-analyzed word', action='store_true')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--ppl', help='compute perplexity', action='store_true')
    parser.add_argument('--decode', help='analyze input', action='store_true')
    args = parser.parse_args()

    if not (args.ppl or args.decode):
        logging.error('--ppl or --decode is required')
        sys.exit(1)

    logging.info('Loading trained model')
    with open(args.model) as f:
        data = cPickle.load(f)
    vocabularies = data['vocabularies']
    word_analyses = data['analyses']
    model = data['model']

    logging.info('Analyzing test corpus')
    analyzer = Analyzer(args.fst, args.backend, not args.notrust)
    test_corpus = analyze_corpus(sys.stdin, analyzer, vocabularies, word_analyses)

    if args.ppl:
        logging.info('Computing test corpus perplexity')
        print_ppl(model, test_corpus)
    elif args.decode:
        logging.info('Decoding test corpus')
        decode_corpus(model, test_corpus, vocabularies)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
