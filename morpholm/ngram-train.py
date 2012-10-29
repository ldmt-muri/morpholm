import logging
import argparse
import math
import cPickle
from corpus import Vocabulary, ngrams, encode_corpus
from prob import CharLM
from pyp import PYPLM
from train import make_mp

# Main PYP
theta, d = 1.0, 0.8

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

def main():
    parser = argparse.ArgumentParser(description='Train n-gram MorphoLM')
    parser.add_argument('-i', '--iterations', help='number of iterations',
                        required=True, type=int)
    parser.add_argument('-n', '--order', help='language model order',
                        required=True, type=int)
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)',
                        required=True)
    parser.add_argument('--model', help='type of model to train (1/2/3)', type=int, default=3)
    parser.add_argument('--output', '-o', help='model output path')
    args = parser.parse_args()

    if args.train.endswith('.pickle'):
        logging.info('Training MorpoLM of order %d', args.order)

        logging.info('Reading training corpus')
        with open(args.train) as f:
            data = cPickle.load(f)
        vocabularies = data['vocabularies']
        word_analyses = data['analyses']
        training_corpus = data['corpus']

        logging.info('Training model')
        mp = make_mp(args, vocabularies, word_analyses)
        model = PYPLM(theta, d, args.order, mp)
        run_sampler(model, training_corpus, args.iterations)

        if args.output:
            logging.info('Saving model')
            data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'model': model}
            with open(args.output, 'w') as output:
                cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

    else:
        logging.info('Training baseline LM of order %d', args.order)
        vocabulary = Vocabulary('<s>', '</s>')

        logging.info('Reading training corpus')
        with open(args.train) as train:
            training_corpus = encode_corpus(train, vocabulary)

        logging.info('Corpus size: %d tokens | Voc size: %d words',
                len(training_corpus), len(vocabulary))

        logging.info('Pre-loading base CharLM')
        char_lm = CharLM(args.charlm, vocabulary)
        model = PYPLM(theta, d, args.order, char_lm)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        if args.output:
            logging.info('Saving model')
            data = {'vocabulary': vocabulary, 'model': model}
            with open(args.output, 'w') as output:
                cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
