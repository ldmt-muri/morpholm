import sys
import argparse
import logging
import cPickle
from analysis import Analyzer, init_vocabularies, analyze_corpus

def main():
    parser = argparse.ArgumentParser(description='Analyze corpus')
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--notrust', help='do not trust analyzer and also keep non-analyzed word', action='store_true')
    parser.add_argument('--output', '-o', help='analyzed corpus output path', required=True)
    args = parser.parse_args()

    word_analyses, vocabularies = init_vocabularies()
    analyzer = Analyzer(args.fst, args.backend, not args.notrust)

    logging.info('Analyzing corpus')
    corpus = analyze_corpus(sys.stdin, analyzer, vocabularies, word_analyses)

    n_words = len(vocabularies['word'])
    n_stems = len(vocabularies['stem'])
    n_morphemes = len(vocabularies['morpheme'])
    n_analyses = sum(len(analyses) for analyses in word_analyses.itervalues())
    n_patterns = len(vocabularies['pattern'])

    logging.info('Corpus size: %d tokens', len(corpus))
    logging.info('Voc size: %d words / %d morphemes / %d stems',
            n_words, n_morphemes, n_stems)
    logging.info('Analyses: %d total -> %d patterns', n_analyses, n_patterns)

    assert len(word_analyses) == n_words

    with open(args.output, 'w') as output:
        data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'corpus': corpus}
        cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
