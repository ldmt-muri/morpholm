from itertools import izip
import math
import numpy as np
import kenlm
from corpus import Analysis

marginalize = np.logaddexp.reduce

def normalize(model):
    """ Make a probability distribution from counts """
    return model - marginalize(model)

def smooth(model): # add-1 smoothing
    return marginalize((model, np.zeros(len(model))))

LOG10 = math.log(10)

class CharLM(kenlm.LanguageModel):
    def prob(self, word):
        # KenLM score is logarithm in base 10
        return self.score(' '.join(word))*LOG10

class Model:
    def char_lm_prob(self, analysis):
        stem = analysis.decode_stem(self.vocabulary['stem'])
        return self.char_lm.prob(stem)

    # log(p_stem(analysis))
    def stem_prob(self, analysis):
        if analysis.oov: return -np.inf
        return self.common_prob(analysis) + self.model_stems[analysis.stem]
    
    # log(p_char(analysis)))
    def char_prob(self, analysis):
        return self.common_prob(analysis) + self.char_lm_prob(analysis)

    # log(p(analysis))
    def analysis_prob(self, analysis):
        sp = self.stem_prob(analysis) + math.log(1-self.model_char)
        cp = self.char_prob(analysis) + math.log(self.model_char)
        if sp == cp == -np.inf: return -np.inf
        return np.logaddexp(sp, cp)

    # log(p_morph), log(p_stem+char)
    def probs(self, analysis):
        sp = (-np.inf if analysis.oov else
                self.model_stems[analysis.stem] + math.log(1-self.model_char))
        cp = self.char_lm_prob(analysis) + math.log(self.model_char)
        p_stem = np.logaddexp(sp, cp)
        return (self.common_prob(analysis), p_stem)

    # log(p(word)) = log(sum_analyses p(analysis))
    def prob(self, word):
        analyses = self.fsm.get_analyses(word)
        coded = [Analysis(analysis, self.vocabulary) for analysis in analyses]
        return np.logaddexp.reduce(map(self.analysis_prob, coded))

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

    def run_sampler(self, n_iterations, corpus):
        assignments = np.zeros(len(corpus), int)
        for it in range(n_iterations):
            print('Iteration {0}:'.format(it+1))
            for i, analyses in enumerate(corpus):
                if it > 0: self.increment(analyses[assignments[i]], -1)
                assignments[i] = self.sample(analyses)
                self.increment(analyses[assignments[i]], 1)
            self.map_estimate()
            loglik = sum(marginalize(map(self.stem_prob, analyses)) for analyses in corpus)
            print(' Log likelihood: {0}'.format(loglik))
        self.cleanup()

    def sample(self, analyses):
        if len(analyses) == 1: return 0
        weights = map(self.pred_weight, analyses)
        v = np.random.rand()*sum(weights)
        for z, w in enumerate(weights):
            if w > v:
                return z
            v -= w
