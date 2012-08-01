import sys
import cPickle
from itertools import izip
import kenlm
from corpus import Analysis, FSM


def main(vocab_file, model_file, charlm, fst):
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    with open(model_file) as fp:
        model = cPickle.load(fp)
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel(charlm)
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True
    fsm = FSM(fst)

    def decode(word):
        analyses = fsm.get_analyses(word)
        if not analyses: return word
        coded = [Analysis(analysis, vocabulary) for analysis in analyses]
        probs = map(model.prob, coded)
        return max(izip(probs, analyses))[1]

    for line in sys.stdin:
        words = line.decode('utf8').split()
        print(' '.join(decode(word).encode('utf8') for word in words))

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
