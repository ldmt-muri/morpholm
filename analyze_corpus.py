import sys
import cPickle
from corpus import FSM, analyze_corpus

if __name__ == '__main__':
    fst = FSM('malmorph.fst')
    corpus, vocabulary = analyze_corpus(fst, sys.stdin)
    print(len(vocabulary['morpheme']))
    print(len(vocabulary['stem']))
    sys.stderr.write('Saving vocabulary...\n')
    with open('vocab.pickle', 'w') as fp:
        cPickle.dump(vocabulary, fp, protocol=2)
    sys.stderr.write('Saving corpus...\n')
    with open('corpus.pickle', 'w') as fp:
        cPickle.dump(corpus, fp, protocol=2)
