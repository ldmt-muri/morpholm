import numpy as np
import scipy.stats as ss
from model import Model, normalize, smooth
from corpus import STEM, OOV

class Model1(Model):
    def uniform_init(self, n_morphemes, n_stems):
        # Morphemes
        self.model_morphemes = np.zeros(n_morphemes) - np.log(n_morphemes)
        self.model_morphemes += np.random.randn(n_morphemes)
        self.model_morphemes = normalize(self.model_morphemes)
        # Stems
        self.model_stems = np.zeros(n_stems) - np.log(n_stems)
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

    def length_prob(self, length):
        return ss.poisson.logpmf(length, self.model_length)

    def morpheme_prob(self, morpheme):
        if morpheme == STEM: return 0
        return self.model_morphemes[morpheme]

    def common_prob(self, analysis):
        # p(len)
        lp = self.length_prob(len(analysis))
        # p(morphemes)
        lp += sum(self.morpheme_prob(morpheme) for morpheme in analysis.morphemes)
        return lp

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
        prob = np.exp(lp)
        self.count_length[0] += len(analysis) * prob
        self.count_length[1] += prob
        assert (analysis.stem != OOV)
        self.count_stems[analysis.stem] = np.logaddexp(self.count_stems[analysis.stem], lp)
        #self.count_char = np.logaddexp(self.count_char, lp_char)
        for morpheme in analysis.morphemes:
            if morpheme == STEM: continue
            self.count_morphemes[morpheme] = np.logaddexp(self.count_morphemes[morpheme], lp)

    # M step
    def maximization(self):
        self.model_morphemes = normalize(self.count_morphemes)
        self.model_stems = normalize(smooth(self.count_stems))
        self.model_length = self.count_length[0] / self.count_length[1]
        #self.model_char = np.exp(self.count_char)/self.count_words
        print(self.model_morphemes)
        print('Morph length: {0}'.format(self.model_length))
        #print('p(char): {}'.format(self.model_char))

    def cleanup(self):
        del self.count_morphemes, self.count_stems, self.count_length, \
                #self.count_char, self.count_words
