import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, encode_corpus
from prob import CharLM, Uniform
from model import TopicModel

def run_sampler(model, corpus, n_iter):
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for d, sentence in enumerate(corpus.sentences):
            for word in sentence:
                if it > 0: model.decrement(d, word)
                model.increment(d, word)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Train PYP-LDA')
    parser.add_argument('-i', '--iterations', help='number of iterations',
                        required=True, type=int)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--output', help='output directory', default=None)
    parser.add_argument('--charlm', help='character language model (KenLM format)')
    args = parser.parse_args()

    logging.info('Training model with %d topics', args.topics)

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = encode_corpus(train, vocabulary)

    logging.info('Corpus size: %d sentences | %d words | Voc size: %d words',
            len(training_corpus.sentences), len(training_corpus), len(vocabulary))

    logging.info('Training model')
    doc_base = Uniform(args.topics)
    if args.charlm:
        topic_base = CharLM(args.charlm, vocabulary)
    else:
        topic_base = Uniform(len(vocabulary))
    model = TopicModel(args.topics, len(training_corpus.sentences), doc_base, topic_base) 

    run_sampler(model, training_corpus, args.iterations)

    if args.output:
        data = {'vocabulary': vocabulary, 'model': model}
        with open(args.output, 'w') as f:
            cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
