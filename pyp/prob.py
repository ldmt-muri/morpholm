import math
import random
from collections import Counter
import kenlm

def mult_sample(vals):
    vals = list(vals)
    x = random.random() * sum(v for _, v in vals)
    for k, v in vals:
        if x < v:
            return k
        x -= v

LOG10 = math.log(10)

class CharLM(kenlm.LanguageModel):
    def increment(self, k): pass

    def decrement(self, k): pass

    def prob(self, k):
        return self.score(' '.join(self.vocabulary[k]))*LOG10

class Multinomial:
    def __init__(self, alpha):
        self.alpha = alpha
        self.count = Counter()

    def increment(self, k):
        self.count[k] += 1

    def decrement(self, k):
        self.count[k] -= 1

class Poisson:
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
        return (math.lalpha(l + self.L + self.alpha) - math.lalpha(self.L + self.alpha)
                - math.lalpha(l + 1) - l * math.log(self.N + self.beta + 1))
