import sys
import cPickle
import numpy as np
import kenlm
from corpus import Analysis, FSM, OOV

def main(vocab_file, model_file, charlm, fst):
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    with open(model_file) as fp:
        model = cPickle.load(fp)
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel(charlm)
    model.model_char = 0.5 # Instead of 0
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True
    fsm = FSM(fst)

    print('Analyzing corpus...')
    corpus_probs = []
    for line in sys.stdin:
        words = line.decode('utf8').split()
        for word in words:
            analyses = fsm.get_analyses(word)
            if not analyses: continue
            # keep small words as-is (i.e. use char LM)
            #    analyses = [Analysis(word, vocabulary, small=True)]
            try:
                analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
            except OOV, oov:
                print('Ignored analysis with OOV morpheme: {0}'.format(oov))
                continue
            stem_probs = map(model.stem_prob, analyses)
            char_probs = map(model.char_prob, analyses)
            sp = np.logaddexp.reduce(stem_probs)
            cp = np.logaddexp.reduce(char_probs)
            corpus_probs.append((sp, cp))

    print('Optimizing...')
    Niter = 10
    for it in range(Niter):
        count = -np.inf
        for sp, cp in corpus_probs:
            char_prob = cp + np.log(model.model_char)
            stem_prob = sp + np.log(1-model.model_char)
            prob = np.logaddexp(stem_prob, char_prob)
            count = np.logaddexp(char_prob - prob, count)
        model.model_char = np.exp(count) / len(corpus_probs)

    print('Best p_char: {}'.format(model.model_char))
    print('Updating model...')
    del model.char_lm
    del model.vocabulary
    with open(model_file, 'w') as fp:
        cPickle.dump(model, fp, protocol=2)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
