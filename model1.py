import math
import numpy as np
from model import Model, normalize, smooth
from corpus import STEM

class Model1(Model):
    def uniform_init(self, n_morphemes, n_stems):
        # Morphemes
        self.model_morphemes = np.zeros(n_morphemes) - math.log(n_morphemes)
        self.model_morphemes += np.random.randn(n_morphemes)
        self.model_morphemes = normalize(self.model_morphemes)
        # Stems
        self.model_stems = np.zeros(n_stems) - math.log(n_stems)
        self.model_stems += np.random.randn(n_stems)
        self.model_stems = normalize(self.model_stems)
        # Length
        self.model_length = 1
        # Character model or stem model?
        self.model_char = 0

    def init_counts(self):
        self.count_morphemes = np.zeros(len(self.model_morphemes)) - np.inf
        self.count_stems = np.zeros(len(self.model_stems)) - np.inf
        self.count_length = [0.0, 0.0]
        #self.count_char = -np.inf
        #self.count_words = 0

    def cleanup(self):
        del self.count_morphemes, self.count_stems, self.count_length, \
                #self.count_char, self.count_words

    def length_prob(self, length):
        # ss.poisson.logpmf(length, self.model_length)
        return (- self.model_length +
                length * math.log(self.model_length)
                - math.lgamma(length + 1))

    def morpheme_prob(self, morpheme):
        if morpheme == STEM: return 0
        return self.model_morphemes[morpheme]

    def common_prob(self, analysis):
        # p(len)
        lp = self.length_prob(len(analysis))
        # p(morphemes)
        lp += sum(self.morpheme_prob(morpheme) for morpheme in analysis.morphemes)
        return lp

    # E step counts
    def count(self, analysis, lp):
        prob = np.exp(lp)
        self.count_length[0] += len(analysis) * prob
        self.count_length[1] += prob
        self.count_stems[analysis.stem] = np.logaddexp(self.count_stems[analysis.stem], lp)
        #self.count_char = np.logaddexp(self.count_char, lp_char)
        for morpheme in analysis.morphemes:
            if morpheme == STEM: continue
            self.count_morphemes[morpheme] = np.logaddexp(self.count_morphemes[morpheme], lp)

    # M step
    def maximization(self):
        self.model_morphemes = normalize(smooth(self.count_morphemes))
        self.model_stems = normalize(smooth(self.count_stems))
        self.model_length = self.count_length[0] / self.count_length[1]
        #self.model_char = np.exp(self.count_char)/self.count_words
        print(self.model_morphemes)
        print('Morph length: {0}'.format(self.model_length))
        #print('p(char): {}'.format(self.model_char))

    def init_sampler(self, n_morphemes, n_stems, alpha, beta, gamma, delta):
        self.count_morphemes = np.zeros(n_morphemes, int)
        self.count_stems = np.zeros(n_stems, int)
        self.count_length = [0, 0]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

    def increment(self, analysis, c):
        self.count_length[0] += len(analysis)*c
        self.count_length[1] += c
        self.count_stems[analysis.stem] += c
        for morpheme in analysis.morphemes:
            if morpheme == STEM: continue
            self.count_morphemes[morpheme] += c

    def pred_weight(self, analysis):
        L, N = self.count_length
        l = len(analysis)
        # length
        # ss.nbinom.pmf(l, L + self.gamma, (N + self.delta)/(N + self.delta + 1))
        # = math.exp(math.lgamma(l + L + self.gamma)
        #            - math.lgamma(L + self.gamma)
        #            - math.lgamma(l + 1))
        #            + (L + self.gamma) * math.log((N + self.delta)/(N + self.delta + 1)) # const
        #            - l * math.log(N + self.delta + 1)
        w = math.exp(math.lgamma(l + L + self.gamma) - math.lgamma(L + self.gamma)
                - math.lgamma(l + 1) - l * math.log(N + self.delta + 1))
        # stem
        S = len(self.count_morphemes)
        w *= (self.alpha + self.count_stems[analysis.stem])/(S * self.alpha + N - 1)
        # morphemes
        for morpheme in set(analysis.morphemes):
            if morpheme == STEM: continue
            w *= (self.beta + self.count_morphemes[morpheme])
        M = len(self.count_morphemes)
        w *= math.exp(math.lgamma(M * self.beta + L)
                - math.lgamma(M * self.beta + L + l))
        return w

    def map_estimate(self):
        L, N = self.count_length
        assert (self.count_morphemes.sum(), self.count_stems.sum()) == (L, N)
        self.model_morphemes = self.count_morphemes + self.beta
        self.model_morphemes /= self.model_morphemes.sum()
        self.model_morphemes = np.log(self.model_morphemes)
        self.model_stems = self.count_stems + self.alpha
        self.model_stems /= self.model_stems.sum()
        self.model_stems = np.log(self.model_stems)
        L, N = self.count_length
        self.model_length =  (L + self.gamma)/(N + self.delta)
        print(self.model_morphemes)
        print('Morph length: {0}'.format(self.model_length))
