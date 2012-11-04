import os
import math
import random
try:
    import numpypy as np
except ImportError:
    import numpy as np
import kenlm

def mult_sample(vals):
    vals = list(vals)
    if len(vals) == 1: return vals[0][0]
    x = random.random() * sum(v for _, v in vals)
    for k, v in vals:
        if x < v: return k
        x -= v
    return k

def remove_random(assignments):
    i = random.randrange(0, len(assignments))
    assignment = assignments[i]
    del assignments[i]
    return assignment

class CharLM:
    def __init__(self, path, vocabulary):
        self.lm = kenlm.LanguageModel(os.path.abspath(path))
        self.vocabulary = vocabulary
        self.K = len(vocabulary)
        self.count = np.zeros(self.K)
        self.probs = np.zeros(self.K)
        for k in xrange(self.K):
            self.probs[k] = self.get_prob(k)

    def increment(self, k):
        assert (0 <= k < self.K)
        self.count[k] += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.count[k] -= 1

    def get_prob(self, k):
        chars = ' '.join(self.vocabulary[k])
        return 10**self.lm.score(chars)

    def prob(self, k):
        assert (k >= 0)
        if k >= self.K: return self.get_prob(k)
        return self.probs[k]

    def log_likelihood(self):
        return np.log(self.probs).dot(self.count)

    def __getstate__(self):
        return (self.lm.path, self.vocabulary)

    def __setstate__(self, state):
        path, self.vocabulary = state
        self.lm = kenlm.LanguageModel(path)
        self.K = 0

    def __repr__(self):
        return 'CharLM(n={self.lm.order})'.format(self=self)

class DirichletMultinomial:
    def __init__(self, K, alpha):
        self.K = K
        self.alpha = alpha
        self.count = np.zeros(K)
        self.N = 0

    def increment(self, k):
        assert (0 <= k < self.K)
        self.count[k] += 1
        self.N += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.count[k] -= 1
        self.N -= 1

    def prob(self, k):
        # = self.weight(k) * self.gamma_factor(1)
        assert k >= 0
        if k >= self.K: return 0
        return (self.alpha + self.count[k])/(self.K * self.alpha + self.N)

    def weight(self, k):
        assert k >= 0
        if k >= self.K: return 0
        return (self.alpha + self.count[k])

    def gamma_factor(self, n):
        if n == 0: return 1
        if n == 1: return 1/(self.K * self.alpha + self.N)
        return math.exp(math.lgamma(self.K * self.alpha + self.N)
                - math.lgamma(self.K * self.alpha + self.N + n))

    def log_likelihood(self):
        return (math.lgamma(self.K * self.alpha) - math.lgamma(self.K * self.alpha + self.N)
                + sum(math.lgamma(self.alpha + self.count[k]) for k in xrange(self.K))
                - self.K * math.lgamma(self.alpha))

    def __getstate__(self):
        return (self.K, self.alpha, self.count.tolist(), self.N)

    def __setstate__(self, state):
        self.K, self.alpha, self.count, self.N = state
        #self.count = np.array(self.count)

    def __repr__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ Dir({self.alpha})'.format(self=self)

def log_binomial_coeff(k, n):
    if k == 0: return 0
    if k == 1: return math.log(n)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

class GammaPoisson:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.L, self.N = 0, 0

    def increment(self, l):
        self.L += l
        self.N += 1

    def decrement(self, l):
        self.L -= l
        self.N -= 1

    def prob(self, l):
        r = self.L + self.alpha
        p = 1 / (self.N + self.beta + 1)
        return math.exp(log_binomial_coeff(l, l + r - 1)
                + r * math.log(1 - p) + l * math.log(p))

    # TODO log_likelihood

    def __repr__(self):
        return 'Poisson(L={self.L}, N={self.N}) ~ Gamma({self.alpha}, {self.beta})'.format(self=self)

class Uniform:
    def __init__(self, N):
        self.N = N
        self.count = 0

    def increment(self, k):
        self.count += 1

    def decrement(self, k):
        self.count -= 1

    def prob(self, k):
        if k >= self.N: return 0
        return 1./self.N

    def log_likelihood(self):
        return - self.count * math.log(self.N)

    def __repr__(self):
        return 'Uniform(N={self.N})'.format(self=self)
