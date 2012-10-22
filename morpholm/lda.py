import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, encode_corpus
from prob import Uniform
from pyp import PYP
from model import TopicModel

theta_doc = 1.0
d_doc = 0.8
theta_topic = 1.0
d_topic = 0.8

n_iter = 500

def run_sampler(model, corpus, output=None):
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for d, sentence in enumerate(corpus.sentences):
            for word in sentence:
                if it > 0: model.decrement(d, word)
                model.increment(d, word)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)
        if output:
            logging.info('Saving model...')
            with open(output+'/model.{0}'.format(it), 'w') as f:
                cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Train PYP-MorphoLM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--output', help='output directory', default=None)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    args = parser.parse_args()

    logging.info('Training model with %d topics', args.topics)

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = encode_corpus(train, vocabulary)

    logging.info('Corpus size: %d sentences | %d words | Voc size: %d words',
            len(training_corpus.sentences), len(training_corpus), len(vocabulary))

    if args.output:
        with open(args.output+'/voc', 'w') as f:
            cPickle.dump(vocabulary, f, cPickle.HIGHEST_PROTOCOL)

    doc_process = lambda: PYP(theta_doc, d_doc, Uniform(args.topics))
    topic_process = lambda: PYP(theta_topic, d_topic, Uniform(len(vocabulary)))
    model = TopicModel(args.topics, len(training_corpus.sentences), doc_process, topic_process) 

    logging.info('Training model')
    run_sampler(model, training_corpus, args.output)

if __name__ == '__main__':
    main()
