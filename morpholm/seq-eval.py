import sys
import logging
import argparse
import cPickle
import math
from analysis import Analyzer, analyze_corpus

def print_ppl(model, corpus):
    n_words = len(corpus) + len(corpus.sentences) # + </s>
    loglik = sum(math.log(model.prob(sentence)) for sentence in corpus.sentences)
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)

def decode_corpus(model, corpus, vocabularies):
    for sentence in corpus.sentences:
        _, decoded = model.viterbi(sentence)
        print('\n'.join(analysis.decode(vocabularies).encode('utf8') for analysis in decoded))

def main():
    parser = argparse.ArgumentParser(description='Use MorphoHMM')
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--notrust', help='do not trust analyzer and also keep non-analyzed word', action='store_true')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--ppl', help='compute perplexity', action='store_true')
    parser.add_argument('--decode', help='analyze input', action='store_true')
    args = parser.parse_args()

    if not (args.ppl or args.decode):
        logging.error('--ppl or --decode is required')
        sys.exit(1)

    analyzer = Analyzer(args.fst, args.backend, not args.notrust)

    logging.info('Loading trained model')

    with open(args.model) as f:
        data = cPickle.load(f)
    vocabularies = data['vocabularies']
    word_analyses = data['analyses']
    model = data['model']

    logging.info('Using model: %s', model)

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
