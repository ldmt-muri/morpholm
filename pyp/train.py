import logging
import argparse
import math
import cPickle
from corpus import FSM, Vocabulary, analyze_corpus, encode_corpus, init_vocabularies
from prob import CharLM
from pyp import PYP
from model import MorphoProcess, PoissonUnigram, Bigram

# Main PYP
theta = 1.0
d = 0.8
# Stem PYP
alpha = 1.0
p = 0.8
# Pattern PYP
nu = 1.0
q = 0.8
# Morpheme prior
beta = 1.0
# Length prior
gamma = 1.0
delta = 1.0

def run_sampler(model, corpus, Niter):
    for it in range(Niter):
        logging.info('Iteration %d/%d', it+1, Niter)
        for word in corpus:
            if it > 0: model.decrement(word)
            model.increment(word)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)

def print_ppl(model, corpus):
    n_words = 0
    loglik = 0
    for word in corpus:
        n_words += 1
        loglik += model.prob(word)
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)

def main():
    parser = argparse.ArgumentParser(description='Train PYP-MorphoLM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer')
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    parser.add_argument('--pyp', help='top model is PYP(p0=MP) instead of MP', action='store_true') 
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--model', help='type of model to train (1/2/3)', type=int, default=3)
    parser.add_argument('--notrust', help='do not trust analyzer and also keep non-analyzed word', action='store_true')
    parser.add_argument('-i', '--iterations', help='number of iterations', required=True, type=int)
    parser.add_argument('--output', '-o', help='model output path')
    args = parser.parse_args()

    char_lm = CharLM(args.charlm)

    if args.fst:
        logging.info('Training full system')
        vocabularies = init_vocabularies()
        word_analyses = {}
        fsm = FSM(args.fst, args.backend, not args.notrust)

        logging.info('Analyzing training corpus')
        with open(args.train) as train:
            training_corpus = analyze_corpus(train, fsm, vocabularies, word_analyses)

        n_words = len(vocabularies['word'])
        n_stems = len(vocabularies['stem'])
        n_morphemes = len(vocabularies['morpheme'])
        n_analyses = sum(len(analyzes) for analyzes in word_analyses.itervalues())
        n_patterns = len(set(analysis.pattern for analyses in word_analyses.itervalues()
                             for analysis in analyses))

        logging.info('Corpus size: %d tokens', len(training_corpus))
        logging.info('Voc size: %d words / %d morphemes / %d stems',
                n_words, n_morphemes, n_stems)
        logging.info('Analyses: %d total -> %d patterns', n_analyses, n_patterns)

        assert len(word_analyses) == n_words

        char_lm.vocabulary = vocabularies['stem']

        if args.model == 1:
            mp = MorphoProcess(PYP(alpha, p, char_lm), PoissonUnigram(n_morphemes, beta, gamma, delta), word_analyses)
        elif args.model == 2:
            mp = MorphoProcess(PYP(alpha, p, char_lm), Bigram(n_morphemes, beta), word_analyses)
        elif args.model == 3:
            mp = MorphoProcess(PYP(alpha, p, char_lm), PYP(nu, q, Bigram(n_morphemes, beta)), word_analyses)

        if args.pyp:
            model = PYP(theta, d, mp)
        else:
            model = mp

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        if args.output:
            logging.info('Saving model')
            data = {'model': model, 'vocabularies': vocabularies, 'analyses': word_analyses}
            with open(args.output, 'w') as output:
                cPickle.dump(data, output, protocol=-1)

        logging.info('Computing test corpus perplexity')
        with open(args.test) as test:
            test_corpus = analyze_corpus(test, fsm, vocabularies, word_analyses)

        print_ppl(model, test_corpus)
    else:
        logging.info('Training baseline system')
        vocabulary = Vocabulary()

        logging.info('Reading training corpus')
        with open(args.train) as train:
            training_corpus = encode_corpus(train, vocabulary)

        logging.info('Corpus size: %d tokens | Voc size: %d words', len(training_corpus), len(vocabulary))

        char_lm.vocabulary = vocabulary
        model = PYP(theta, d, char_lm)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        logging.info('Computing test corpus perplexity')
        with open(args.test) as test:
            test_corpus = encode_corpus(test, vocabulary)

        print_ppl(model, test_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
