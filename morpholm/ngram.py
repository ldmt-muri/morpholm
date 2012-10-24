import logging
import argparse
import math
from corpus import Vocabulary, ngrams, encode_corpus
from analysis import Analyzer, analyze_corpus, init_vocabularies, Analysis
from prob import CharLM
from pyp import PYP, PYPLM
from model import MorphoProcess, BigramPattern, PoissonUnigramPattern

# Main PYP
theta, d = 1.0, 0.8
# Stem PYP
alpha, p = 1.0, 0.8
# Pattern PYP
nu, q = 1.0, 0.8
# Morpheme prior
beta = 1.0
# Length prior
gamma, delta = 1.0, 1.0

def run_sampler(model, corpus, n_iter):
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence in corpus.sentences:
            for seq in ngrams(sentence, model.order):
                if it > 0: model.decrement(seq)
                model.increment(seq)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)

def print_ppl(model, corpus):
    n_words = 0
    loglik = 0
    for sentence in corpus.sentences:
        for seq in ngrams(sentence, model.order):
            n_words += 1
            loglik += math.log(model.prob(seq))
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)

def main():
    parser = argparse.ArgumentParser(description='Train PYP-MorphoLM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer')
    parser.add_argument('--charlm', help='character language model (KenLM format)',
                        required=True)
    parser.add_argument('--model', help='type of model to train (1/2/3)', type=int, default=2)
    parser.add_argument('-n', '--order', help='language model order',
                        required=True, type=int)
    parser.add_argument('-i', '--iterations', help='number of iterations',
                        required=True, type=int)
    args = parser.parse_args()

    char_lm = CharLM(args.charlm)

    if args.fst:
        logging.info('Training Morpo-LM of order %d', args.order)
        vocabularies = init_vocabularies('<s>', '</s>')
        word_analyses = {vocabularies['word']['<s>']: [Analysis('<s>', vocabularies)],
                         vocabularies['word']['</s>']: [Analysis('</s>', vocabularies)]}
        fsm = Analyzer(args.fst)

        logging.info('Analyzing training corpus')
        with open(args.train) as train:
            training_corpus = analyze_corpus(train, fsm, vocabularies, word_analyses)

        n_words = len(vocabularies['word'])
        n_stems = len(vocabularies['stem'])
        n_morphemes = len(vocabularies['morpheme'])
        n_analyses = sum(len(analyses) for analyses in word_analyses.itervalues())
        n_patterns = len(vocabularies['pattern'])

        logging.info('Corpus size: %d tokens', len(training_corpus))
        logging.info('Voc size: %d words / %d morphemes / %d stems',
                n_words, n_morphemes, n_stems)
        logging.info('Analyses: %d total -> %d patterns', n_analyses, n_patterns)

        assert len(word_analyses) == n_words

        char_lm.vocabulary = vocabularies['stem']
        if args.model == 1:
            pattern_model = PoissonUnigramPattern(n_morphemes, beta, gamma, delta,
                    vocabularies['pattern'])
        elif args.model == 2:
            pattern_model = BigramPattern(n_morphemes, beta, vocabularies['pattern'])
        elif args.model == 3:
            pattern_model = PYP(nu, q, BigramPattern(n_morphemes, beta,
                vocabularies['pattern']))

        mp = MorphoProcess(PYP(alpha, p, char_lm), pattern_model, word_analyses)

        model = PYPLM(theta, d, args.order, mp)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        logging.info('Computing test corpus perplexity')
        with open(args.test) as test:
            test_corpus = analyze_corpus(test, fsm, vocabularies, word_analyses)

        print_ppl(model, test_corpus)

    else:
        logging.info('Training baseline LM of order %d', args.order)
        vocabulary = Vocabulary('<s>', '</s>')

        logging.info('Reading training corpus')
        with open(args.train) as train:
            training_corpus = encode_corpus(train, vocabulary)

        logging.info('Corpus size: %d tokens | Voc size: %d words', len(training_corpus), len(vocabulary))

        char_lm.vocabulary = vocabulary
        model = PYPLM(theta, d, args.order, char_lm)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        logging.info('Computing test corpus perplexity')
        with open(args.test) as test:
            test_corpus = encode_corpus(test, vocabulary)

        print_ppl(model, test_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
