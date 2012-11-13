import logging
import argparse
import cPickle

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print analyzed corpus statistics')
    parser.add_argument('--corpus', help='analyzed corpus', required=True)

    args = parser.parse_args()

    with open(args.corpus) as f:
        corpus = cPickle.load(f)

    n_words = len(corpus.analyses)
    n_stems = sum(len(set(analysis.stem for analysis in analyses)) 
            for analyses in corpus.analyses.itervalues())
    n_analyses = sum(len(analyses) 
            for analyses in corpus.analyses.itervalues())
    n_non_analyzed = sum((len(analyses) == 1)
            for analyses in corpus.analyses.itervalues())

    print('avg # stems/word: {0:.2f}'.format(n_stems/float(n_words)))
    print('avg # analyses/word: {0:.2f}'.format(n_analyses/float(n_words)))
    print('non analyzed words: {0:.2%}'.format(n_non_analyzed/float(n_words)))

if __name__ == '__main__':
    main()
