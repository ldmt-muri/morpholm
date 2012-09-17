import numpy as np
import math
from model import Model, normalize, smooth
from model2 import square_noise, row_normalize, bigrams

class Model22(Model):
    def uniform_init(self, n_morphemes, n_stems):
        # Morphemes
        self.model_left = np.zeros((n_morphemes+1, n_morphemes+1)) - math.log((n_morphemes+1))
        self.model_left += square_noise(n_morphemes+1)
        self.model_left = row_normalize(self.model_left)
        self.model_right = np.zeros((n_morphemes+1, n_morphemes+1)) - math.log((n_morphemes+1))
        self.model_right += square_noise(n_morphemes+1)
        self.model_right = row_normalize(self.model_right)
        # Stems
        self.model_stems = np.zeros(n_stems) - math.log(n_stems)
        self.model_stems += np.random.randn(n_stems)
        self.model_stems = normalize(self.model_stems)
        # Character model or stem model?
        self.model_char = 0

    def init_counts(self):
        self.count_left = np.zeros(self.model_left.shape) - np.inf
        self.count_right = np.zeros(self.model_right.shape) - np.inf
        self.count_stems = np.zeros(len(self.model_stems)) - np.inf

    def cleanup(self):
        del self.count_right, self.count_left, self.count_stems

    def common_prob(self, analysis):
        # p(morphemes)
        p_left = sum(self.model_left[x, y] for x, y in bigrams(analysis.left_morphemes))
        p_right = sum(self.model_right[x, y] for x, y in bigrams(analysis.right_morphemes))
        return p_left + p_right

    # E step counts
    def count(self, analysis, lp):
        self.count_stems[analysis.stem] = np.logaddexp(self.count_stems[analysis.stem], lp)
        for x, y in bigrams(analysis.left_morphemes):
            self.count_left[x, y] = np.logaddexp(self.count_left[x, y], lp)
        for x, y in bigrams(analysis.right_morphemes):
            self.count_right[x, y] = np.logaddexp(self.count_right[x, y], lp)

    # M step
    def maximization(self):
        self.model_left = row_normalize(self.count_left)
        self.model_right = row_normalize(self.count_right)
        self.model_stems = normalize(smooth(self.count_stems))
        n_total = len(self.model_left.flat)
        n_params_l = sum(1 for v in self.model_left.flat if v != -np.inf)
        n_params_r = sum(1 for v in self.model_right.flat if v != -np.inf)
        print(' Active parameters: {0}/{2} + {1}/{2}'.format(n_params_l, n_params_r, n_total))
