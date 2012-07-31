import sys
import cPickle
from itertools import izip
import kenlm
from corpus import Analysis, FSM

if __name__ == '__main__':
    with open('vocab.pickle') as fp:
        vocabulary = cPickle.load(fp)
    with open('model1.pickle') as fp:
        model = cPickle.load(fp)
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel('charlm.klm')
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True
    fsm = FSM('malmorph.fst')

    def decode(word):
        analyses = fsm.get_analyses(word)
        if not analyses: return word
        coded = [Analysis(analysis, vocabulary) for analysis in analyses]
        probs = map(model.prob, coded)
        return max(izip(probs, analyses))[1]

    for line in sys.stdin:
        words = line.decode('utf8').split()
        print(' '.join(decode(word).encode('utf8') for word in words))
