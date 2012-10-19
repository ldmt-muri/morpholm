import os
import math
import numpy as np
import random
from collections import Counter
import kenlm

def mult_sample(vals):
    vals = list(vals)
    if len(vals) == 1: return vals[0][0]
    x = random.random() * sum(v for _, v in vals)
    for k, v in vals:
        if x < v: return k
        x -= v

def remove_random(assignments):
    i = random.randrange(0, len(assignments))
    assignment = assignments[i]
    del assignments[i]
    return assignment

LOG10 = math.log(10)
SPECIAL = set(('<s>', '</s>'))

class CharLM(kenlm.LanguageModel):
    def __init__(self, path):
        super(CharLM, self).__init__(path)
        self.path = os.path.abspath(path)

    def increment(self, k): pass

    def decrement(self, k): pass

    def prob(self, k):
        word = self.vocabulary[k]
        if word in SPECIAL: return 0
        return self.score(' '.join(word))*LOG10

    def __repr__(self):
        return 'CharLM(n={self.order})'.format(self=self)

    def __reduce__(self):
        return (CharLM, (self.path,))

class FixedMixModel:
    def __init__(self, models, coeffs):
        assert sum(coeffs) == 1 and len(models) == len(coeffs)
        self.mix = zip(models, map(math.log, coeffs))

    def prob(self, k):
        return np.logaddexp.reduce([c+m.prob(k) for m, c in self.mix])

class DirichletMultinomial:
    def __init__(self, K, alpha):
        self.K = K
        self.alpha = alpha
        self.count = Counter()
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
        # = self.weight(k) + self.gamma_factor(1)
        assert k >= 0
        if k > self.K: return -np.inf
        return math.log((self.alpha + self.count[k])/(self.K * self.alpha + self.N))

    def weight(self, k):
        assert k >= 0
        if k > self.K: return -np.inf
        return math.log((self.alpha + self.count[k]))

    def gamma_factor(self, n):
        if n == 0: return 0
        if n == 1: return -math.log(self.K * self.alpha + self.N)
        return (math.lgamma(self.K * self.alpha + self.N)
                - math.lgamma(self.K * self.alpha + self.N + n))

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
        return log_binomial_coeff(l, l + r - 1) + r * math.log(1 - p) + l * math.log(p)

    def __repr__(self):
        return 'Poisson(L={self.L}, N={self.N}) ~ Gamma({self.alpha}, {self.beta})'.format(self=self)

class Uniform:
    def __init__(self, N):
        self.N = N
        self.logN = math.log(N)

    def increment(self, k): pass

    def decrement(self, k): pass

    def prob(self, k):
        if k > self.N: return -np.inf
        return -self.logN
