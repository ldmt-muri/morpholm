import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, encode_corpus
from prob import CharLM, Uniform
from pyp import PYP
from model import TopicModel, BaseTopicModel, MorphoProcess

# Document-topic
theta_doc, d_doc = 1.0, 0.1
# Topic-word
theta_topic, d_topic = 1.0, 0.8
# Stem PYP
alpha, p = 1.0, 0.8
# Pattern PYP
nu, q = 1.0, 0.8

def run_sampler(model, corpus, n_iter, callback=None):
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for d, sentence in enumerate(corpus.sentences):
            for word in sentence:
                if it > 0: model.decrement(d, word)
                model.increment(d, word)
        if it % 10 == 0:
            ll = model.log_likelihood()
            ppl = math.exp(-ll / len(corpus))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
            logging.info('Model: %s', model)
        if it % 100 == 99 and callback is not None:
            callback(it)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Train PYP-LDA')
    parser.add_argument('-i', '--iterations', help='number of iterations',
                        required=True, type=int)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--output', help='output prefix', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)')
    args = parser.parse_args()

    logging.info('Training model with %d topics', args.topics)

    if args.train.endswith('.pickle'):
        logging.info('Reading training corpus')
        with open(args.train) as f:
            data = cPickle.load(f)
        vocabularies = data['vocabularies']
        word_analyses = data['analyses']
        training_corpus = data['corpus']

        logging.info('Pre-loading stem CharLM')
        char_lm = CharLM(args.charlm, vocabularies['stem'])

        doc_base = Uniform(args.topics)
        make_document = lambda: PYP(theta_doc, d_doc, doc_base)
        ll_document = lambda dts: (sum(dt.log_likelihood(base=False) for dt in dts) 
                + doc_base.log_likelihood())

        pattern_model = PYP(nu, q, Uniform(len(vocabularies['pattern'])))
        stem_base = PYP(alpha, p, char_lm)
        # PYP-MP0
        make_topic = lambda: PYP(theta_topic, d_topic, 
                MorphoProcess(PYP(alpha, p, stem_base), pattern_model, word_analyses))
        ll_topic = lambda tws: (sum(tw.log_likelihood(base=False)
            + tw.base.stem_model.log_likelihood(base=False) for tw in tws)
            + stem_base.log_likelihood() + pattern_model.log_likelihood())

        model = BaseTopicModel(len(training_corpus.sentences), args.topics,
                make_document, make_topic, ll_document, ll_topic)

        def write_output(it):
            logging.info('Saving model')
            data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'model': model}
            with open('{0}.{1}.pickle'.format(args.output, it), 'w') as output:
                cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations, write_output)
    else:
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
        model = TopicModel(len(training_corpus.sentences), args.topics, doc_base, topic_base) 

        run_sampler(model, training_corpus, args.iterations)

        if args.output:
            data = {'vocabulary': vocabulary, 'model': model}
            with open(args.output, 'w') as f:
                cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
