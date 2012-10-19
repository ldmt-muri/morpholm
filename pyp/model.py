import math
import numpy as np
from collections import defaultdict
from itertools import tee, izip
from prob import mult_sample, remove_random, DirichletMultinomial, GammaPoisson

class PoissonUnigram:
    def __init__(self, K, beta, gamma, delta):
        self.morpheme_model = DirichletMultinomial(K, beta)
        self.length_model = GammaPoisson(gamma, delta)

    def increment(self, pattern):
        for morpheme in pattern:
            self.morpheme_model.increment(morpheme)
        self.length_model.increment(len(pattern))

    def decrement(self, pattern):
        for morpheme in pattern:
            self.morpheme_model.decrement(morpheme)
        self.length_model.decrement(len(pattern))

    def prob(self, pattern):
        return (sum(map(self.morpheme_model.weight, pattern)) +
                self.morpheme_model.gamma_factor(len(pattern)) +
                self.length_model.prob(len(pattern)))

    def __repr__(self):
        return 'PoissonUnigram(length ~ {self.length_model}, morph ~ {self.morpheme_model})'.format(self=self)

class Bigram:
    def __init__(self, K, beta):
        self.morpheme_models = [DirichletMultinomial(K+1, beta) for _ in range(K+1)]
        self.START = K
        self.STOP = K

    def _unigrams(self, pattern):
        yield self.START
        for m in pattern.morphemes:
            yield m
        yield self.STOP

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
        return sum(self.morpheme_models[x].prob(y) for x, y in self._bigrams(pattern))

    def __repr__(self):
        return 'Bigram(stem\'|stem ~ Mult ~ Dir)'

class MorphoProcess:
    def __init__(self, stem_model, morpheme_model, analyses):
        self.stem_model = stem_model
        self.morpheme_model = morpheme_model
        self.assignments = defaultdict(list)
        self.analyses = analyses

    def increment(self, k):
        # Sample analysis & store assignment
        i = (0 if len(self.analyses[k]) == 1 else
                mult_sample((i, math.exp(self.analysis_prob(analysis)))
                    for i, analysis in enumerate(self.analyses[k])))
        self.assignments[k].append(i)
        analysis = self.analyses[k][i]
        # Increment models
        self.stem_model.increment(analysis.stem)
        self.morpheme_model.increment(analysis.pattern)

    def decrement(self, k):
        # Select assigned analysis randomly and remove it
        analysis = self.analyses[k][remove_random(self.assignments[k])]
        # Decrement models
        self.stem_model.decrement(analysis.stem)
        self.morpheme_model.decrement(analysis.pattern)

    def analysis_prob(self, analysis):
        return (self.stem_model.prob(analysis.stem) +
                self.morpheme_model.prob(analysis.pattern))

    def prob(self, k):
        return np.logaddexp.reduce([self.analysis_prob(analysis)
            for analysis in self.analyses[k]])

    def decode(self, k):
        return max((self.analysis_prob(analysis), analysis)
            for analysis in self.analyses[k])

    def log_likelihood(self):
        # XXX pseudo log-likelihood
        return sum(self.prob(k) * len(assignments)
                for k, assignments in self.assignments.iteritems())

    def __repr__(self):
        return 'MorphoProcess(#words={N} | stem ~ {self.stem_model}, morphemes ~ {self.morpheme_model})'.format(self=self, N=sum(map(len, self.assignments.itervalues())))

class TopicModel:
    def __init__(self, Ntopics, Ndocs, doc_process, topic_process):
        self.Ntopics = Ntopics
        self.document_topic = [doc_process() for _ in xrange(Ndocs)]
        self.topic_word = [topic_process() for _ in xrange(Ntopics)]
        self.assignments = defaultdict(list)

    def increment(self, doc, word):
        z = mult_sample((k, math.exp(self.topic_prob(doc, word, k)))
                for k in xrange(self.Ntopics))
        self.assignments[doc, word].append(z)
        self.document_topic[doc].increment(z)
        self.topic_word[z].increment(word)

    def decrement(self, doc, word):
        z = remove_random(self.assignments[doc, word])
        self.document_topic[doc].decrement(z)
        self.topic_word[z].decrement(word)

    def topic_prob(self, doc, word, k):
        return self.document_topic[doc].prob(k) + self.topic_word[k].prob(word)

    def prob(self, doc, word):
        return np.logaddexp.reduce([self.topic_prob(doc, word, k)
            for k in xrange(self.Ntopics)])

    def log_likelihood(self):
        # XXX pseudo log-likelihood
        return sum(self.prob(doc, word) * len(assignments)
                for (doc, word), assignments in self.assignments.iteritems())

    def __repr__(self):
        return 'TopicModel(#topics={self.Ntopics})'.format(self=self)
