import sys
import cPickle
import numpy as np
import kenlm
from corpus import Analysis, FSM

def analyze(vocabulary, model, char_lm, fsm, test_corpus, nomorph=False):
    model.vocabulary = vocabulary
    model.char_lm = char_lm
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True

    def decode(word):
        analyses = fsm.get_analyses(word)
        if not analyses: return word
        analyses = list(analyses)
        coded = [Analysis(analysis, vocabulary) for analysis in analyses]
        probs = map(model.prob, coded)
        best = np.argmax(probs)
        if nomorph: # stem only
            return coded[best].decode_stem(vocabulary['stem'])
        return analyses[best]

    for line in test_corpus:
        words = line.decode('utf8').split()
        yield ' '.join(decode(word).encode('utf8') for word in words)

def main(vocab_file, model_file, charlm, fst, nomorph=False):
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    with open(model_file) as fp:
        model = cPickle.load(fp)
    char_lm = kenlm.LanguageModel(charlm)
    for line in analyze(vocabulary, model, char_lm, FSM(fst), sys.stdin, nomorph):
        print(line)

if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst [nomorph]\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
