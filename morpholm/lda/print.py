import argparse
import logging
import cPickle
import heapq

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print LDA model')
    parser.add_argument('model', help='trained model')
    args = parser.parse_args()

    with open(args.model) as m:
        model = cPickle.load(m)

    pm = model.pattern_model
    def dec(p):
        pattern = model.pattern_vocabulary[p]
        return '+'.join(model.morpheme_vocabulary[m] for m in pattern)
    patt_prob = ((pm.prob(p), p) for p in xrange(len(model.pattern_vocabulary)))
    for prob, p in heapq.nlargest(100, patt_prob):
        print(u'{0} {1}'.format(dec(p), prob).encode('utf8'))
    print('---------')

    
    for i, topic in enumerate(model.topic_word):
        print('Topic {0}'.format(i))
        stem_topic = topic.base.stem_model
        word_prob = ((stem_topic.prob(w), w) for w in xrange(len(model.stem_vocabulary)))
        for prob, w in heapq.nlargest(10, word_prob):
            print(u'{0} {1}'.format(model.stem_vocabulary[w], prob).encode('utf8'))
        print('---------')

if __name__ == '__main__':
    main()
