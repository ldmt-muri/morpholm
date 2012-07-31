import cPickle
from corpus import FSM, analyze_corpus

if __name__ == '__main__':
    fst = FSM('malmorph.fst')
    text = 'train.txt'
    #text = 'mini.txt'
    corpus, vocabulary = analyze_corpus(fst, text)
    print(len(vocabulary['morpheme']))
    print(len(vocabulary['stem']))
    with open('vocab.pickle', 'w') as fp:
        cPickle.dump(vocabulary, fp, protocol=2)
    with open('corpus.pickle', 'w') as fp:
        cPickle.dump(corpus, fp, protocol=2)
