from itertools import izip, tee
import numpy as np
from model import Model, normalize, smooth
from corpus import OOV

START = STOP = 0

def bigrams(morphemes):
    ms = [START]
    ms.extend([m+1 for m in morphemes])
    ms.append(STOP)
    a, b = tee(ms)
    next(b)
    return izip(a, b)

def row_normalize(m):
    return m - np.logaddexp.reduce(m, axis=1).reshape((len(m), 1))

def square_noise(n):
    return np.random.randn(n**2).reshape((n, n))

class Model2(Model):
    def uniform_init(self, n_morphemes, n_stems):
        # Morphemes
        self.model_morphemes = np.zeros((n_morphemes+1, n_morphemes+1)) - np.log((n_morphemes+1))
        self.model_morphemes += square_noise(n_morphemes+1)
        self.model_morphemes = row_normalize(self.model_morphemes)
        # Stems
        self.model_stems = np.zeros(n_stems) - np.log(n_stems)
        self.model_stems += np.random.randn(n_stems)
        self.model_stems = normalize(self.model_stems)
        # Character model or stem model?
        self.model_char = 0

    def init_counts(self):
        self.count_morphemes = np.zeros(self.model_morphemes.shape) - np.inf
        self.count_stems = np.zeros(len(self.model_stems)) - np.inf

    def common_prob(self, analysis):
        # p(morphemes)
        return sum(self.model_morphemes[x, y] for x, y in bigrams(analysis.morphemes))

    # log(p_stem(analysis))
    def stem_prob(self, analysis):
        if analysis.stem == OOV: return -np.inf
        return self.common_prob(analysis) + self.model_stems[analysis.stem]
    
    # log(p_char(analysis)))
    def char_prob(self, analysis):
        return self.common_prob(analysis) + self.char_lm_prob(analysis.stem)

    # log(p(analysis)),
    def prob(self, analysis):
        sp = self.stem_prob(analysis) + np.log(1-self.model_char)
        cp = self.char_prob(analysis) + np.log(self.model_char)
        return np.logaddexp(sp, cp)

    # E step counts
    def count(self, analysis, lp):
        assert (analysis.stem != OOV)
        self.count_stems[analysis.stem] = np.logaddexp(self.count_stems[analysis.stem], lp)
        for x, y in bigrams(analysis.morphemes):
            self.count_morphemes[x, y] = np.logaddexp(self.count_morphemes[x, y], lp)

    # M step
    def maximization(self):
        self.model_morphemes = row_normalize(self.count_morphemes)
        self.model_stems = normalize(smooth(self.count_stems))
        n_total = len(self.model_morphemes.flat)
        n_params = sum(1 for v in self.model_morphemes.flat if v != -np.inf)
        print(' Active parameters: {0}/{1}'.format(n_params, n_total))

    def cleanup(self):
        del self.count_morphemes, self.count_stems
