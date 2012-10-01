from itertools import izip, tee
import math
import numpy as np
from model import Model, normalize, smooth

START = STOP = 0

def unigrams(morphemes):
    yield START
    for m in morphemes:
        yield m+1
    yield STOP

def bigrams(morphemes):
    a, b = tee(unigrams(morphemes))
    next(b)
    return izip(a, b)

def row_normalize(m):
    norm = np.logaddexp.reduce(m, axis=1)
    norm[norm == -np.inf] = 0
    return m - norm[:,np.newaxis]

def square_noise(n):
    return np.random.randn(n**2).reshape((n, n))

class Model2(Model):
    def uniform_init(self, n_morphemes, n_stems):
        # Morphemes
        self.model_morphemes = (np.zeros((n_morphemes+1, n_morphemes+1)) 
                - math.log((n_morphemes+1)))
        self.model_morphemes += square_noise(n_morphemes+1)
        self.model_morphemes = row_normalize(self.model_morphemes)
        # Stems
        self.model_stems = np.zeros(n_stems) - math.log(n_stems)
        self.model_stems += np.random.randn(n_stems)
        self.model_stems = normalize(self.model_stems)
        # Character model or stem model?
        self.model_char = 0

    def init_counts(self):
        self.count_morphemes = np.zeros(self.model_morphemes.shape) - np.inf
        self.count_stems = np.zeros(len(self.model_stems)) - np.inf

    def cleanup(self):
        del self.count_morphemes, self.count_stems

    def common_prob(self, analysis):
        # p(morphemes)
        return sum(self.model_morphemes[x, y] for x, y in bigrams(analysis.pattern.morphemes))

    # E step counts
    def count(self, analysis, lp):
        self.count_stems[analysis.stem] = np.logaddexp(self.count_stems[analysis.stem], lp)
        for x, y in bigrams(analysis.pattern.morphemes):
            self.count_morphemes[x, y] = np.logaddexp(self.count_morphemes[x, y], lp)

    # M step
    def maximization(self):
        self.model_morphemes = row_normalize(self.count_morphemes)
        self.model_stems = normalize(smooth(self.count_stems))
        n_total = len(self.model_morphemes.flat)
        n_params = sum(1 for v in self.model_morphemes.flat if v != -np.inf)
        print(' Active parameters: {0}/{1}'.format(n_params, n_total))
