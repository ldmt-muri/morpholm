import operator
import logging
import random
from collections import defaultdict
from itertools import tee, izip
from vpyp.prob import mult_sample, remove_random,\
        DirichletMultinomial, GammaPoisson, Uniform, BetaBernouilli
from vpyp.corpus import START, STOP

prod = lambda it: reduce(operator.mul, it, 1)

class UniformUnigramPattern:
    def __init__(self, K, gamma, delta, pattern_vocabulary):
        self.morpheme_model = Uniform(K-3) # -START, -STOP, -STEM
        self.length_model = GammaPoisson(gamma, delta)
        self.vocabulary = pattern_vocabulary

    def increment(self, pattern):
        n_morphemes = len(self.vocabulary[pattern])
        self.morpheme_model.count += n_morphemes-1
        self.length_model.increment(n_morphemes-1)

    def decrement(self, pattern):
        n_morphemes = len(self.vocabulary[pattern])
        self.morpheme_model.count -= n_morphemes-1
        self.length_model.decrement(n_morphemes-1)

    def prob(self, pattern):
        n_morphemes = len(self.vocabulary[pattern])
        morpheme_prob = 1./self.morpheme_model.K
        return (morpheme_prob**(n_morphemes-1) *
                self.length_model.prob(n_morphemes-1))

    def log_likelihood(self, full=False):
        return (self.morpheme_model.log_likelihood(full)
                + self.length_model.log_likelihood(full))

    def resample_hyperparemeters(self, n_iter):
        return self.morpheme_model.resample_hyperparemeters(n_iter)

    def __repr__(self):
        return ('UniformUnigram(length ~ {self.length_model},'
                ' morph ~ {self.morpheme_model})').format(self=self)

class PoissonUnigramPattern:
    def __init__(self, K, morpheme_prior, gamma, delta, pattern_vocabulary):
        self.morpheme_model = DirichletMultinomial(K-2, morpheme_prior) # -START, -STOP
        self.length_model = GammaPoisson(gamma, delta)
        self.vocabulary = pattern_vocabulary

    def increment(self, pattern):
        morphemes = self.vocabulary[pattern]
        for morpheme in morphemes:
            self.morpheme_model.increment(morpheme-2)
        self.length_model.increment(len(morphemes)-1)

    def decrement(self, pattern):
        morphemes = self.vocabulary[pattern]
        for morpheme in morphemes:
            self.morpheme_model.decrement(morpheme-2)
        self.length_model.decrement(len(morphemes)-1)

    def prob(self, pattern):
        morphemes = self.vocabulary[pattern]
        return (prod(self.morpheme_model.prob(m) for m in morphemes) *
                self.length_model.prob(len(morphemes)-1))

    def log_likelihood(self, full=False):
        return (self.morpheme_model.log_likelihood(full)
                + self.length_model.log_likelihood(full))

    def resample_hyperparemeters(self, n_iter):
        return self.morpheme_model.resample_hyperparemeters(n_iter)

    def __repr__(self):
        return ('PoissonUnigram(length ~ {self.length_model},'
                ' morph ~ {self.morpheme_model})').format(self=self)

class BigramPattern:
    def __init__(self, K, prior, pattern_vocabulary):
        self.prior = prior
        self.morpheme_models = [DirichletMultinomial(K, prior) for _ in range(K)]
        self.vocabulary = pattern_vocabulary

    def _unigrams(self, pattern):
        yield START
        for m in self.vocabulary[pattern]:
            yield m
        yield STOP

    def _bigrams(self, pattern):
        a, b = tee(self._unigrams(pattern))
        next(b)
        return izip(a, b)

    def increment(self, pattern):
        for x, y in self._bigrams(pattern):
            self.morpheme_models[x].increment(y)

    def decrement(self, pattern):
        for x, y in self._bigrams(pattern):
            self.morpheme_models[x].decrement(y)

    def prob(self, pattern):
        return prod(self.morpheme_models[x].prob(y) for x, y in self._bigrams(pattern))

    def log_likelihood(self, full=False):
        ll = sum(m.log_likelihood() for m in self.morpheme_models)
        if full:
            ll += self.prior.log_likelihood()
        return ll

    def resample_hyperparemeters(self, n_iter):
        return self.prior.resample(n_iter)

    def __repr__(self):
        return 'Bigram(stem\'|stem ~ Mult(K={K}) ~ Dir(alpha ~ {self.prior}))'.format(self=self, K = len(self.morpheme_models))

class MorphoProcess:
    def __init__(self, stem_model, pattern_model, analyses):
        self.stem_model = stem_model
        self.pattern_model = pattern_model
        self.analyses = analyses
        self.assignments = defaultdict(list)

    def increment(self, k):
        # Sample analysis & store assignment
        i = (0 if len(self.analyses[k]) == 1 else
                mult_sample((i, self.analysis_prob(analysis))
                    for i, analysis in enumerate(self.analyses[k])))
        self.assignments[k].append(i)
        analysis = self.analyses[k][i]
        # Increment models
        self.stem_model.increment(analysis.stem)
        self.pattern_model.increment(analysis.pattern)

    def decrement(self, k):
        # Select assigned analysis randomly and remove it
        analysis = self.analyses[k][remove_random(self.assignments[k])]
        # Decrement models
        self.stem_model.decrement(analysis.stem)
        self.pattern_model.decrement(analysis.pattern)

    def analysis_prob(self, analysis):
        return (self.stem_model.prob(analysis.stem) *
                self.pattern_model.prob(analysis.pattern))

    def prob(self, k):
        return sum(self.analysis_prob(analysis) for analysis in self.analyses[k])

    def decode(self, k):
        return max((self.analysis_prob(analysis), analysis)
            for analysis in self.analyses[k])

    def log_likelihood(self, full=False):
        return (self.stem_model.log_likelihood(full=full)
                + self.pattern_model.log_likelihood(full=full))

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling stem model hyperparameters')
        a1, r1 = self.stem_model.resample_hyperparemeters(n_iter)
        logging.info('Resampling pattern model hyperparameters')
        a2, r2 = self.pattern_model.resample_hyperparemeters(n_iter)
        return (a1+a2, r1+r2)

    def __repr__(self):
        return ('MorphoProcess(#words={N} | stem ~ {self.stem_model}; '
                'pattern ~ {self.pattern_model})'
                ).format(self=self, N=sum(map(len, self.assignments.itervalues())))

class SwitchingMorphoProcess:
    def __init__(self, word_model, stem_model, pattern_model, analyses):
        self.word_model = word_model
        self.mp = MorphoProcess(stem_model, pattern_model, analyses)
        self.switch_model = BetaBernouilli(1.0, 1e6)
        self.analyses = analyses
        self.switches = defaultdict(list)

    def increment(self, k):
        p_word, p_mp = self.probs(k)
        x = random.random() * (p_word + p_mp)
        if x < p_word:
            self.word_model.increment(k)
            switch = True
        else:
            self.mp.increment(k)
            switch = False
        self.switch_model.increment(switch)
        self.switches[k].append(switch)

    def decrement(self, k):
        switch = remove_random(self.switches[k])
        if switch:
            self.word_model.decrement(k)
        else:
            self.mp.decrement(k)
        self.switch_model.decrement(switch)

    def probs(self, k):
        p = self.switch_model.p
        p_word = p * self.word_model.prob(k)
        p_mp = 0 if len(self.analyses[k]) == 0 else (1 - p) * self.mp.prob(k)
        return (p_word, p_mp)

    def prob(self, k):
        return sum(self.probs(k))

    def log_likelihood(self, full=False):
        return (self.word_model.log_likelihood(full=full)
                + self.mp.log_likelihood(full=full)
                + self.switch_model.log_likelihood(full=full))

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling word model hyperparameters')
        a1, r1 = self.word_model.resample_hyperparemeters(n_iter)
        logging.info('Resampling mp hyperparameters')
        a2, r2 = self.mp.resample_hyperparemeters(n_iter)
        return (a1+a2, r1+r2)

    def __repr__(self):
        return ('Switching[{self.word_model}+{self.mp}'
                '|switch={self.switch_model}]').format(self=self)
