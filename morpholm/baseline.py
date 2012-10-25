import logging
import argparse
from corpus import Vocabulary, encode_corpus
from prob import CharLM
from pyp import PYP
from train import run_sampler
from eval import print_ppl

def main():
    parser = argparse.ArgumentParser(description='Train PYP 1-gram LM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    parser.add_argument('--strength', help='PYP strength parameter', default=1.0)
    parser.add_argument('--discount', help='PYP discount parameter', default=0.8)
    parser.add_argument('-i', '--iterations', help='number of iterations', required=True, type=int)
    args = parser.parse_args()

    logging.info('Training baseline system')

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = encode_corpus(train, vocabulary)

    logging.info('Corpus size: %d tokens | Voc size: %d words', len(training_corpus), len(vocabulary))

    logging.info('Pre-loading base CharLM')
    char_lm = CharLM(args.charlm, vocabulary)

    model = PYP(args.strength, args.discount, char_lm)

    logging.info('Training model')
    run_sampler(model, training_corpus, args.iterations)

    logging.info('Computing test corpus perplexity')
    with open(args.test) as test:
        test_corpus = encode_corpus(test, vocabulary)

    print_ppl(model, test_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
