from itertools import izip
import numpy as np

marginalize = np.logaddexp.reduce

def normalize(model):
    """ Make a probability distribution from counts """
    return model - marginalize(model)

def smooth(model): # add-1 smoothing
    return marginalize((model, np.zeros(len(model))))

LOG10 = np.log(10)

class Model:
    def char_lm_prob(self, analysis):
        stem = analysis.decode_stem(self.vocabulary['stem'])
        chars = ' '.join(stem)
        # KenLM score is logarithm in base 10
        return self.char_lm.score(chars)*LOG10

    # log(p_stem(analysis))
    def stem_prob(self, analysis):
        if analysis.oov: return -np.inf
        return self.common_prob(analysis) + self.model_stems[analysis.stem]
    
    # log(p_char(analysis)))
    def char_prob(self, analysis):
        return self.common_prob(analysis) + self.char_lm_prob(analysis)

    # log(p(analysis)),
    def prob(self, analysis):
        sp = self.stem_prob(analysis) + np.log(1-self.model_char)
        cp = self.char_prob(analysis) + np.log(self.model_char)
        if sp == cp == -np.inf: return -np.inf
        return np.logaddexp(sp, cp)

    # log(p_morph), log(p_stem)
    def probs(self, analysis):
        sp = (-np.inf if analysis.oov else
                self.model_stems[analysis.stem] + np.log(1-self.model_char))
        cp = self.char_lm_prob(analysis) + np.log(self.model_char)
        p_stem = np.logaddexp(sp, cp)
        return (self.common_prob(analysis), p_stem)

    def run_em(self, n_iterations, corpus):
        for it in range(n_iterations):
            print('Iteration {0}:'.format(it+1))
            self.init_counts()
            loglik = 0
            # Expectation
            for analyses in corpus:
                probs = map(self.stem_prob, analyses)
                #probs = marginalize((stem_probs, char_probs)) # p(char)+p(stem)
                norm = marginalize(probs)
                loglik += norm
                for analysis, lp in izip(analyses, probs):
                    self.count(analysis, lp-norm)
                #self.count_words += 1
            print(' Log likelihood: {0}'.format(loglik))
            # Maximization
            self.maximization()
        self.cleanup()
        return loglik
