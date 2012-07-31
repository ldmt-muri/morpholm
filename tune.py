import sys
import cPickle
import numpy as np
import kenlm
from corpus import Analysis, FSM

if __name__ == '__main__':
    with open('vocab.pickle') as fp:
        vocabulary = cPickle.load(fp)
    with open('model1.pickle') as fp:
        model = cPickle.load(fp)
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel('charlm.klm')
    #model.model_char = 0.5
    CRM, SRM = np.log(model.model_char), np.log(1-model.model_char)
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True
    fsm = FSM('malmorph.fst')

    print('Analyzing corpus...')
    corpus_probs = []
    for line in sys.stdin:
        words = line.decode('utf8').split()
        for word in words:
            analyses = fsm.get_analyses(word)
            if not analyses: continue
            analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
            stem_probs, char_probs = zip(*map(model.stem_char_prob, analyses))
            sp = np.logaddexp.reduce(stem_probs) - SRM
            cp = np.logaddexp.reduce(char_probs) - CRM
            corpus_probs.append((sp, cp))

    print('Optimizing...')
    Niter = 10
    alpha = model.model_char
    for it in range(Niter):
        count = -np.inf
        for sp, cp in corpus_probs:
            char_prob = cp + np.log(alpha)
            stem_prob = sp + np.log(1-alpha)
            prob = np.logaddexp(stem_prob, char_prob)
            count = np.logaddexp(char_prob - prob, count)
        alpha = np.exp(count) / len(corpus_probs)
        print alpha
    print('Best alpha: {}'.format(alpha))
