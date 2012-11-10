import argparse
import logging
import cPickle
import math
from itertools import izip
from collections import Counter
import creg

def read_all_labels(fn):
    with open(fn) as f:
        for line in f:
            yield line.split(' ||| ')[1].lower().split(',')

def read_labels(fn, threshold):
    label_count = Counter()
    for labels in read_all_labels(fn):
        for label in labels:
            label_count[label] += 1
    logging.info('Most common labels: %s', ', '.join(label for label, _ in label_count.most_common(10)))
    for labels in read_all_labels(fn):
        yield [label for label in labels if label_count[label] >= threshold]

def compute_map(dataset):
    total_ap = 0
    n_docs = 0
    for labels, ranked_list in dataset:
        relevant = set(labels)
        total_p = 0
        tp = 0
        for n,(_, label) in enumerate(ranked_list):
            if label in relevant:
                tp += 1
                total_p += tp/float(n+1)
        total_ap += total_p/len(relevant)
        n_docs += 1
    return total_ap/n_docs

def compute_at5(dataset):
    total_p = 0
    total_r = 0
    total_f1 = 0
    n_docs = 0
    for labels, ranked_list in dataset:
        relevant = set(labels)
        predicted = set(label for _, label in ranked_list[:5])
        p = len(predicted & relevant)/float(len(predicted))
        r = len(predicted & relevant)/float(len(relevant))
        total_p += p
        total_r += r
        total_f1 += 2/(1/(p+1e-6)+1/(r+1e-6))
        n_docs += 1
    return (total_p/n_docs, total_r/n_docs, total_f1/n_docs)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Use PYP-LDA to make predictions')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--labels', help='topic labels', required=True)
    parser.add_argument('--min', help='minimal label count', type=int, default=10)
    args = parser.parse_args()
    
    logging.info('Loading model')
    with open(args.model) as f:
        data = cPickle.load(f)
    #vocabulary = data['vocabulary']
    model = data['model']

    logging.info('Computing topic vectors')
    def topic_vector(dt):
        return dict((str(k), math.log(dt.prob(k))) for k in xrange(model.n_topics))

    all_instances = [(topic_vector(dt), labels) for dt, labels in 
            izip(model.document_topic, read_labels(args.labels, args.min))]
    n_train = int(len(all_instances)*0.8)
    n_dev = int(len(all_instances)*0.1)
    train_instances = all_instances[:n_train]
    dev_instances = all_instances[n_train:n_train+n_dev]
    test_instances = all_instances[n_train+n_dev:]
    logging.info('%d instances -> %d training + %d development + %d evaluation', 
            len(all_instances), len(train_instances), len(dev_instances), len(test_instances))

    training_dataset = creg.CategoricalDataset((fv, label) for fv, labels in train_instances for label in labels)
    logging.info('Training set: %s', training_dataset)
    logging.info('Label vocabulary size: %d', len(training_dataset.labels))

    def predict(classifier, instances):
        for topic_vector, gold_labels in test_instances:
            yield gold_labels, sorted(izip(classifier.predict_proba(topic_vector), training_dataset.labels), reverse=True)

    def tune_l1():
        for l1 in (1e-2, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
            logging.info('L1 strength: %s', l1)
            classifier = creg.LogisticRegression()
            classifier.fit(training_dataset, l1=l1, delta=1e-6)
            yield compute_map(predict(classifier, dev_instances)), l1

    logging.info('Tuning label classifier')
    best_map, best_l1 = max(tune_l1())
    logging.info('Best MAP: {0:.2%} for L1 strength: {1}'.format(best_map, best_l1))

    logging.info('Training final classifier with L1 strength: %s', best_l1)
    classifier = creg.LogisticRegression()
    classifier.fit(training_dataset, l1=best_l1, delta=1e-6)
    
    test_predictions = list(predict(classifier, test_instances))

    logging.info('MAP: {0:.2%}'.format(compute_map(test_predictions)))
    logging.info('P@5: {0:.2%} / R@5: {1:.2%} / F1@5: {2:.2%}'.format(*compute_at5(test_predictions)))

if __name__ == '__main__':
    main()
