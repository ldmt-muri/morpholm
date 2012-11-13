import operator
import logging
from collections import defaultdict
from itertools import tee, izip
from vpyp.prob import mult_sample, remove_random, DirichletMultinomial
from vpyp.corpus import START, STOP

prod = lambda it: reduce(operator.mul, it, 1)

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
