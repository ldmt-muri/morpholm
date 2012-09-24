import sys
import cPickle
import numpy as np
from model import CharLM
from corpus import Analysis, FSM

def analyze(vocabularies, model, char_lm, fsm, test_corpus, nomorph=False):
    model.vocabularies = vocabularies
    model.char_lm = char_lm
    vocabularies['morpheme'].frozen = True
    vocabularies['stem'].frozen = True

    def decode(word):
        analyses = list(fsm.get_analyses(word))
        coded = [Analysis(analysis, vocabularies) for analysis in analyses]
        probs = map(model.analysis_prob, coded)
        best = np.argmax(probs)
        if nomorph: # stem only
            return coded[best].decode_stem(vocabularies['stem'])
        return analyses[best]

    for line in test_corpus:
        words = line.decode('utf8').split()
        yield ' '.join(decode(word).encode('utf8') for word in words)

def main(vocab_file, model_file, charlm, fst, nomorph=False):
    with open(vocab_file) as fp:
        vocabularies = cPickle.load(fp)
    with open(model_file) as fp:
        model = cPickle.load(fp)
    char_lm = CharLM(charlm)
    for line in analyze(vocabularies, model, char_lm, FSM(fst), sys.stdin, nomorph):
        print(line)

if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst [nomorph]\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
