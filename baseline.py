import math
import numpy as np
from collections import Counter
from corpus import Vocabulary, OOV

LOG10 = math.log(10)

class Model0:
    def __init__(self, counts):
        N = sum(counts.itervalues())
        V = len(counts)
        self.model = np.zeros(V)
        for i in range(V):
            self.model[i] = math.log((counts[i]+1.0)/(N+V))

    def token_prob(self, word):
        try:
            return self.model[self.vocabulary[word]]
        except OOV:
            return -np.inf

    def char_prob(self, word):
        chars = ' '.join(word)
        return self.char_lm.score(chars)*LOG10

    def prob(self, word):
        sp = self.token_prob(word) + math.log(1-self.model_char)
        cp = self.char_prob(word) + math.log(self.model_char)
        if sp == cp == -np.inf: return -np.inf
        return np.logaddexp(sp, cp)

def train_model(train_corpus):
    vocabulary = Vocabulary()
    counts = Counter()

    print('Counting training corpus words')
    for line in train_corpus:
        for word in line.decode('utf8').split():
            counts[vocabulary[word]] += 1

    print('Vocabulary size: {0}'.format(len(vocabulary)))
    print('MLE...')
    return Model0(counts), vocabulary

def tune_model(vocabulary, model, char_lm, dev_corpus):
    model.vocabulary = vocabulary
    model.char_lm = char_lm
    model.model_char = 0.5
    vocabulary.frozen = True

    print('Analyzing development corpus...')
    corpus_probs = []
    for line in dev_corpus:
        words = line.decode('utf8').split()
        for word in words:
            corpus_probs.append((model.token_prob(word), model.char_prob(word)))

    print('Optimizing...')
    Niter = 10
    for it in range(Niter):
        count = -np.inf
        for sp, cp in corpus_probs:
            char_prob = cp + math.log(model.model_char)
            token_prob = sp + math.log(1-model.model_char)
            prob = np.logaddexp(token_prob, char_prob)
            count = np.logaddexp(char_prob - prob, count)
        model.model_char = np.exp(count) / len(corpus_probs)

    print('Best p_char: {}'.format(model.model_char))
    print('Updating model...')
    del model.char_lm
    del model.vocabulary

def print_ppl(vocabulary, model, char_lm, test_corpus):
    model.vocabulary = vocabulary
    model.char_lm = char_lm
    vocabulary.frozen = True

    total_prob = 0
    n_words = 0

    print('Reading evaluation corpus...')
    for line in test_corpus:
        words = line.decode('utf8').split()
        for word in words:
            n_words += 1
            total_prob += model.prob(word)

    ppl = np.exp(-total_prob/n_words)

    print('Log Likelihood: {0:.3f}'.format(total_prob))
    print('       # words: {0}'.format(n_words))
    print('    Perplexity: {0:.3f}'.format(ppl))
