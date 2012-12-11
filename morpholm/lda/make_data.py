import logging
import argparse
import cPickle
import math
import sys
from itertools import izip
from collections import Counter
from vpyp.corpus import Corpus
from vpyp.prob import DirichletMultinomial, mult_sample

def read_all_labels(fn):
    with open(fn) as f:
        for line in f:
            yield line.split(' ||| ')[1].lower().split(',')

def read_labels(fn, threshold):
    label_count = Counter()
    for labels in read_all_labels(fn):
        for label in labels:
            label_count[label] += 1
    top = ', '.join(label for label, _ in label_count.most_common(10))
    kept = sum(c>=threshold for label, c in label_count.iteritems())
    logging.info('Most common labels: %s (kept %d total)', top, kept)
    for labels in read_all_labels(fn):
        yield [label for label in labels if label_count[label] >= threshold]

def topic_vector(dt, model):
    return dict((str(k), math.log(dt.prob(k))) for k in xrange(model.n_topics))

def sample_topics(doc, model, n_iter):
    assignments = [None] * len(doc)
    doc_topic = DirichletMultinomial(model.n_topics, model.alpha)
    for it in xrange(n_iter):
        for i, word in enumerate(doc):
            if it > 0: doc_topic.decrement(assignments[i])
            assignments[i] = mult_sample((k, (doc_topic.prob(k)
                * model.topic_word[k].prob(word))) for k in xrange(model.n_topics))
            doc_topic.increment(assignments[i])
    return topic_vector(doc_topic, model)

def format_topics(topics):
    return ' '.join('{0}:{1}'.format(*t) for t in topics.items())

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='?')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--labels', help='topic labels', required=True)
    parser.add_argument('--min', help='minimal label count', type=int, default=10)
    parser.add_argument('--iter', help='number of sampling iterations', type=int, default=1000)
    parser.add_argument('corpus', nargs='*')
    args = parser.parse_args()
    
    logging.info('Loading model')
    with open(args.model) as f:
        model = cPickle.load(f)

    logging.info('Computing topic vectors')

    all_labels = read_labels(args.labels, args.min)

    logging.info('Printing training corpus topics')
    for dt, labels in izip(model.document_topic, all_labels):
        topics = topic_vector(dt, model)
        print('{0} ||| {1}'.format(format_topics(topics), ','.join(labels)))

    logging.info('Inferring other topics')
    for corpus in args.corpus:
        with open(corpus) as corpus_file:
            if corpus.endswith('.pickle'):
                new_corpus = cPickle.load(corpus_file)
                model.stem_vocabulary.update(new_corpus.stem_vocabulary)
                model.morpheme_vocabulary.update(new_corpus.morpheme_vocabulary)
                model.pattern_vocabulary.update(new_corpus.pattern_vocabulary)
                model.analyses.update(new_corpus.analyses)
            else:
                new_corpus = Corpus(corpus_file, model.vocabulary)
        for doc, labels in izip(new_corpus.segments, all_labels):
            topics = sample_topics(doc, model, args.iter)
            print('{0} ||| {1}'.format(format_topics(topics), ','.join(labels)))
            sys.stderr.write('.')
            sys.stderr.flush()
    sys.stderr.write('\n')

if __name__ == '__main__':
    main()
