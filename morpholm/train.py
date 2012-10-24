import logging
import argparse
import math
import cPickle
from prob import CharLM
from pyp import PYP
from model import MorphoProcess, PoissonUnigramPattern, BigramPattern

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
        for word in corpus:
            if it > 0: model.decrement(word)
            model.increment(word)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)

def main():
    parser = argparse.ArgumentParser(description='Train MorphoLM')
    parser.add_argument('-i', '--iterations', help='number of iterations', required=True, type=int)
    parser.add_argument('--train', help='compiled training corpus', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    parser.add_argument('--pyp', help='top model is PYP(p0=MP) instead of MP', action='store_true') 
    parser.add_argument('--model', help='type of model to train (1/2/3)', type=int, default=3)
    parser.add_argument('--output', '-o', help='model output path')
    args = parser.parse_args()

    logging.info('Reading training corpus')

    with open(args.train) as f:
        data = cPickle.load(f)
    vocabularies = data['vocabularies']
    word_analyses = data['analyses']
    training_corpus = data['corpus']

    n_morphemes = len(vocabularies['morpheme'])

    char_lm = CharLM(args.charlm)
    char_lm.vocabulary = vocabularies['stem']

    if args.model == 1:
        pattern_model = PoissonUnigramPattern(n_morphemes, beta, gamma, delta, vocabularies['pattern'])
    elif args.model == 2:
        pattern_model = BigramPattern(n_morphemes, beta, vocabularies['pattern'])
    elif args.model == 3:
        pattern_model = PYP(nu, q, BigramPattern(n_morphemes, beta, vocabularies['pattern']))

    mp = MorphoProcess(PYP(alpha, p, char_lm), pattern_model, word_analyses)

    if args.pyp:
        logging.info('Top model is PYP')
        model = PYP(theta, d, mp)
    else:
        logging.info('Top model is MorphoProcess')
        model = mp

    logging.info('Training model')
    run_sampler(model, training_corpus, args.iterations)

    if args.output:
        logging.info('Saving model')
        data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'model': model}
        with open(args.output, 'w') as output:
            cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
