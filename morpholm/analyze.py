import sys
import logging
import argparse
import cPickle
from vpyp.corpus import Corpus

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Analyze corpus')
    parser.add_argument('--backend', help='analyzer backend (foma/xfst/pymorphy/xerox)',
            default='foma')
    parser.add_argument('--analyzer', help='analyzer option', required=True)
    parser.add_argument('--addnull', help='add null analysis', action='store_true')
    parser.add_argument('--output', help='analyzed output path', required=True)

    args = parser.parse_args()

    logging.info('Reading corpus')
    corpus = Corpus(sys.stdin)
    n_tokens = sum(map(len, corpus))
    n_types = len(corpus.vocabulary)
    logging.info('Number of tokens: %d / types: %d', n_tokens, n_types)

    from analyzers import all_analyzers
    analyzer = all_analyzers[args.backend](args.analyzer)
    logging.info('Analyzing corpus using %s', analyzer)
    analyzer.analyze_corpus(corpus, args.addnull)
    logging.info('Found %d morphemes / %d stems / %d analyses -> %d patterns',
            len(corpus.morpheme_vocabulary), len(corpus.stem_vocabulary),
            sum(len(analyses) for analyses in corpus.analyses.itervalues()),
            len(corpus.pattern_vocabulary))

    logging.info('Saving analyzed corpus')
    with open(args.output, 'w') as out:
        cPickle.dump(corpus, out, protocol=-1)

if __name__ == '__main__':
    main()
