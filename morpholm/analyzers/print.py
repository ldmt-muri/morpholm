import logging
import argparse
import cPickle
import analyzer

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print analyzed corpus')
    parser.add_argument('--corpus', help='analyzed corpus', required=True)

    args = parser.parse_args()

    with open(args.corpus) as f:
        corpus = cPickle.load(f)

    S = corpus.morpheme_vocabulary[analyzer.STEM]
    def get_analysis(analysis):
        stem = corpus.stem_vocabulary[analysis.stem]
        pattern = corpus.pattern_vocabulary[analysis.pattern]
        return '+'.join((stem if m == S else corpus.morpheme_vocabulary[m]) for m in pattern)

    for sentence in corpus:
        for w in sentence:
            word = corpus.vocabulary[w]
            analyses = map(get_analysis, corpus.analyses[w])
            print((u'{0}\t{1}'.format(word, '\t'.join(analyses))).encode('utf8'))

if __name__ == '__main__':
    main()
