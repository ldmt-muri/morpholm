import sys
import cPickle
import numpy as np
import kenlm
from corpus import Analysis, FSM

LOG10 = np.log(10)
class CharLM:
    def prob(self, word):
        return self.char_lm.score(' '.join(word))*LOG10

def main(vocab_file, model_file, charlm, fst):
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    with open(model_file) as fp:
        model = cPickle.load(fp)
    #model = CharLM()
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel(charlm)
    vocabulary['morpheme'].frozen = True
    vocabulary['stem'].frozen = True
    fsm = FSM(fst)

    total_prob = 0
    n_words = 0

    print('Reading corpus...')
    for line in sys.stdin:
        words = line.decode('utf8').split()
        for word in words:
            analyses = fsm.get_analyses(word)
            if not analyses: continue
            n_words += 1
            #total_prob += model.prob(word)
            analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
            probs = map(model.prob, analyses)
            total_prob += np.logaddexp.reduce(probs)

    ppl = np.exp(-total_prob/n_words)

    print('Log Likelihood: {0:.3f}'.format(total_prob))
    print('       # words: {0}'.format(n_words))
    print('    Perplexity: {0:.3f}'.format(ppl))

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
