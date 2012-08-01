import sys
import cPickle
from corpus import FSM, analyze_corpus

def main(fst, vocab_out, corpus_out):
    fst = FSM(fst)
    corpus, vocabulary = analyze_corpus(fst, sys.stdin)
    print('Found {0} morphemes'.format(len(vocabulary['morpheme'])))
    print('Found {0} stems'.format(len(vocabulary['stem'])))
    sys.stderr.write('Saving vocabulary...\n')
    with open(vocab_out, 'w') as fp:
        cPickle.dump(vocabulary, fp, protocol=2)
    sys.stderr.write('Saving corpus...\n')
    with open(corpus_out, 'w') as fp:
        cPickle.dump(corpus, fp, protocol=2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: {argv[0]} fst vocab corpus\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
