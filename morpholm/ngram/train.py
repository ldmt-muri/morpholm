import argparse
import logging
import cPickle
from vpyp.prob import Uniform
from vpyp.charlm import CharLM
from vpyp.prior import PYPPrior, GammaPrior
from vpyp.pyp import PYP
from vpyp.ngram.model import PYPLM
from vpyp.ngram.train import run_sampler
from ..models import BigramPattern, UniformUnigramPattern, MorphoProcess, SwitchingMorphoProcess

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train n-gram model')
    parser.add_argument('--train', help='analyzed training corpus', required=True)
    parser.add_argument('--order', help='order of the model', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--charlm', help='stem character language model')
    parser.add_argument('--model', help='type of model', type=int, required=True)
    parser.add_argument('--switch', help='use switching model', action='store_true')
    parser.add_argument('--output', help='model output path')

    args = parser.parse_args()

    with open(args.train) as train:
        training_corpus = cPickle.load(train)

    if args.charlm:
        logging.info('Preloading stem CharLM')
        char_lm = CharLM(args.charlm, training_corpus.stem_vocabulary)
    else:
        logging.info('Using base uniform distribution over stems')
        char_lm = Uniform(len(training_corpus.stem_vocabulary))
    stem_model = PYP(char_lm, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0)) # G_s

    pvoc = training_corpus.pattern_vocabulary
    n_morphemes = len(training_corpus.morpheme_vocabulary)
    if args.model == 0:
        pattern_base = Uniform(len(pvoc))
    elif args.model == 10:
        pattern_base = UniformUnigramPattern(n_morphemes, 1.0, 1.0, pvoc)
    elif args.model == 2:
        alpha_prior = GammaPrior(1.0, 1.0, 1.0)
        pattern_base = BigramPattern(n_morphemes, alpha_prior, pvoc)
    pattern_model = PYP(pattern_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.)) # G_p

    if args.switch:
        if args.charlm:
            logging.info('Preloading word CharLM')
            word_base = CharLM(args.charlm, training_corpus.vocabulary)
        else:
            word_base = Uniform(len(training_corpus.vocabulary))
        word_model = PYP(word_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0))
        mp = SwitchingMorphoProcess(word_model, stem_model, pattern_model,
                training_corpus.analyses)
    else:
        mp = MorphoProcess(stem_model, pattern_model, training_corpus.analyses)

    model = PYPLM(args.order, mp)

    logging.info('Training model of order %d', args.order)
    run_sampler(model, training_corpus, args.iter)

    if args.output:
        model.vocabulary = training_corpus.vocabulary
        model.stem_vocabulary = training_corpus.stem_vocabulary
        model.morpheme_vocabulary = training_corpus.morpheme_vocabulary
        model.pattern_vocabulary = training_corpus.pattern_vocabulary
        model.analyses = training_corpus.analyses
        with open(args.output, 'w') as f:
            cPickle.dump(model, f, protocol=-1)

if __name__ == '__main__':
    main()
