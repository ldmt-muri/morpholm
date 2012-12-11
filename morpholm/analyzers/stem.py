import logging
import argparse
import cPickle

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Stem analyzed corpus')
    parser.add_argument('corpus', help='analyzed corpus')
    args = parser.parse_args()

    logging.info('Reading analyzed corpus')
    with open(args.corpus) as corpus_file:
        corpus = cPickle.load(corpus_file)

    def stem(w):
        a = corpus.analyses[w][0]
        return corpus.stem_vocabulary[a.stem]

    for sentence in corpus.segments:
        print(' '.join(stem(w) for w in sentence).encode('utf8'))

if __name__ == '__main__':
    main()
