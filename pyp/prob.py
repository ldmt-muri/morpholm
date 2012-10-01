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

class CharLM(kenlm.LanguageModel):
    def increment(self, k): pass

    def decrement(self, k): pass

    def prob(self, k):
        return self.score(' '.join(self.vocabulary[k]))*LOG10

    def __str__(self):
        return 'CharLM(n={self.order})'.format(self=self)

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

    def pred_weight(self, k):
        assert (0 <= k < self.K)
        return math.log(self.alpha + self.count[k])

    def prob(self, k):
        if k > self.K: return -np.inf
        return math.log((self.alpha + self.count[k])/(self.K * self.alpha + self.N))

    def gamma_factor(self, n):
        if n == 0: return 0
        if n == 1: return -math.log(self.K * self.alpha + self.N)
        return (math.lgamma(self.K * self.alpha + self.N)
                - math.lgamma(self.K * self.alpha + self.N + n))

    def __str__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ Dir({self.alpha})'.format(self=self)

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

    def pred_weight(self, l):
        return (math.lgamma(l + self.L + self.alpha) - math.lgamma(self.L + self.alpha)
                - math.lgamma(l + 1) - l * math.log(self.N + self.beta + 1))

    def prob(self, l):
        _lambda = (self.L + self.alpha) / (self.N + self.beta)
        return (-_lambda + l * math.log(_lambda) - math.lgamma(l + 1))

    def __str__(self):
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
