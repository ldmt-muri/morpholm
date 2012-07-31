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
    def char_lm_prob(self, stem):
        # KenLM score is logarithm in base 10
        chars = ' '.join(self.vocabulary['stem'][stem])
        return self.char_lm.score(chars)*LOG10

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
            print(' Log likelihood: {0}\n'.format(loglik))
            # Maximization
            self.maximization()
        self.cleanup()
        return loglik
