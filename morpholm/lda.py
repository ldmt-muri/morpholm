import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, encode_corpus
from prob import CharLM, Uniform
from pyp import PYP
from model import PYPTopicModel, MorphoTopicModel, BigramPattern, PoissonUnigramPattern

# Pattern PYP
nu, q = 1.0, 0.8
# CharLM PYP
alpha, p = 1.0, 0.8

def run_sampler(model, corpus, n_iter, callback=None):
    assignments = [[None]*len(doc) for doc in corpus.sentences]
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for d, sentence in enumerate(corpus.sentences):
            doc_assignments = assignments[d]
            for i, word in enumerate(sentence):
                if it > 0: model.decrement(d, word, doc_assignments[i])
                doc_assignments[i] = model.increment(d, word)
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
    parser.add_argument('--model', help='pattern model', type=int, default=0)
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

        logging.info('Using pattern model %d', args.model)
        n_morphemes = len(vocabularies['morpheme'])
        pvoc = vocabularies['pattern']
        if args.model == 0:
            pattern_base = Uniform(len(pvoc))
        if args.model == 1:
            pattern_base = PoissonUnigramPattern(n_morphemes, 1.0, 1.0, 1.0, pvoc)
        elif args.model == 2:
            pattern_base = BigramPattern(n_morphemes, 1.0, pvoc)
        pattern_model = PYP(nu, q, pattern_base)

        model = MorphoTopicModel(len(training_corpus.sentences), args.topics,
                doc_base, char_lm, pattern_model, word_analyses)

        def write_output(it):
            logging.info('Saving model')
            data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'model': model}
            with open('{0}.{1}.pickle'.format(args.output, it+1), 'w') as output:
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

        doc_base = Uniform(args.topics)
        if args.charlm:
            logging.info('Pre-loading word CharLM')
            char_lm = CharLM(args.charlm, vocabulary)
            topic_base = PYP(alpha, p, char_lm)
        else:
            topic_base = Uniform(len(vocabulary))
        model = PYPTopicModel(len(training_corpus.sentences), args.topics, doc_base, topic_base)

        logging.info('Training model')
        run_sampler(model, training_corpus, args.iterations)

        if args.output:
            data = {'vocabulary': vocabulary, 'model': model}
            with open(args.output, 'w') as f:
                cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
