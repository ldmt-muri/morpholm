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

    total_prob, total_viterbi_stem, total_viterbi_morph = 0, 0, 0
    n_words = 0

    print('Reading corpus...')
    for line in sys.stdin:
        words = line.decode('utf8').split()
        for word in words:
            analyses = fsm.get_analyses(word)
            if not analyses: continue
            n_words += 1
            #total_prob += model.prob(word) # for baseline char lm
            analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
            all_probs = np.array(map(model.probs, analyses))
            probs = all_probs.sum(axis=1)
            viterbi = all_probs[probs.argmax()]
            total_prob += np.logaddexp.reduce(probs)
            total_viterbi_morph += viterbi[0]
            total_viterbi_stem += viterbi[1]

    total_viterbi = total_viterbi_morph + total_viterbi_stem

    ppl = np.exp(-total_prob/n_words)
    ppl_viterbi = np.exp(-total_viterbi/n_words)
    ppl_viterbi_morph = np.exp(-total_viterbi_morph/n_words)
    ppl_viterbi_stem = np.exp(-total_viterbi_stem/n_words)

    print('Log Likelihood: {0:.3f}'.format(total_prob))
    print('    Viterbi LL: {0:.3f}'.format(total_viterbi))
    print('       # words: {0}'.format(n_words))
    print('    Perplexity: {0:.3f}'.format(ppl))
    print('Vit Perplexity: {0:.3f} = (morph) {1:.3f} * (stem) {2:.3f}'.format(ppl_viterbi,
        ppl_viterbi_morph, ppl_viterbi_stem))

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: {argv[0]} vocab model charlm fst\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
