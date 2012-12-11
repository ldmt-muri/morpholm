import argparse
import logging
import cPickle
from vpyp.prob import Uniform
from vpyp.charlm import CharLM
from vpyp.prior import PYPPrior, GammaPrior
from vpyp.pyp import PYP
from vpyp.lda.train import run_sampler
from ..models import BigramPattern
from model import MorphoLDA

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train MP topic model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--charlm', help='character language model')
    parser.add_argument('--pyp', help='G_w^0 is PYP(CharLM)', action='store_true')
    parser.add_argument('--model', help='model type', type=int, default=0)
    parser.add_argument('--output', help='model output prefix')

    args = parser.parse_args()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = cPickle.load(train)

    if args.charlm:
        logging.info('Preloading stem character language model')
        char_lm = CharLM(args.charlm, training_corpus.stem_vocabulary)
    else:
        logging.info('Uniform distribution over stems')
        char_lm = Uniform(len(training_corpus.stem_vocabulary))

    if args.pyp:
        stem_base = PYP(char_lm, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0))
    else:
        stem_base = char_lm

    if args.model == 0:
        pattern_base = Uniform(len(training_corpus.pattern_vocabulary))
    elif args.model == 2:
        alpha_prior = GammaPrior(1.0, 1.0, 1.0)
        n_morphemes = len(training_corpus.morpheme_vocabulary)
        pattern_base = BigramPattern(n_morphemes, alpha_prior, training_corpus.pattern_vocabulary)


    model = MorphoLDA(args.topics, len(training_corpus), 
            pattern_base, stem_base, training_corpus.analyses)

    def save_callback(it):
        if args.output and it % 100 == 99:
            logging.info('Saving model...')
            with open('{0}.{1}.pickle'.format(args.output, it+1), 'w') as f:
                model.vocabulary = training_corpus.vocabulary
                model.stem_vocabulary = training_corpus.stem_vocabulary
                model.morpheme_vocabulary = training_corpus.morpheme_vocabulary
                model.pattern_vocabulary = training_corpus.pattern_vocabulary
                model.analyses = training_corpus.analyses
                cPickle.dump(model, f, protocol=-1)

    logging.info('Training model with %d topics', args.topics)
    run_sampler(model, training_corpus, args.iter, cb=save_callback)

if __name__ == '__main__':
    main()
