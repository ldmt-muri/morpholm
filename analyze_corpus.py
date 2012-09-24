import sys
import cPickle
from corpus import FSM, analyze_corpus, init_vocabularies

def main(fst, vocab_out, corpus_out):
    vocabularies = init_vocabularies()
    word_analyses = {}
    fsm = FSM(fst)
    corpus = analyze_corpus(sys.stdin, fsm, vocabularies, word_analyses)
    corpus.analyses = word_analyses
    print('Found {0} morphemes'.format(len(vocabularies['morpheme'])))
    print('Found {0} stems'.format(len(vocabularies['stem'])))
    sys.stderr.write('Saving vocabularies...\n')
    with open(vocab_out, 'w') as fp:
        cPickle.dump(vocabularies, fp, protocol=2)
    sys.stderr.write('Saving corpus...\n')
    with open(corpus_out, 'w') as fp:
        cPickle.dump(corpus, fp, protocol=2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: {argv[0]} fst vocab corpus\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
