import math
import numpy as np
from collections import defaultdict
from itertools import tee, izip
from analysis import STEM
from prob import mult_sample, remove_random, DirichletMultinomial, GammaPoisson

class PoissonUnigramPattern:
    def __init__(self, K, beta, gamma, delta, vocabulary):
        self.morpheme_model = DirichletMultinomial(K, beta)
        self.length_model = GammaPoisson(gamma, delta)
        self.vocabulary = vocabulary

    def _unigrams(self, pattern):
        return [m for m in self.vocabulary[pattern] if m != STEM]

    def increment(self, pattern):
        morphemes = self._unigrams(pattern)
        for morpheme in morphemes:
            self.morpheme_model.increment(morpheme)
        self.length_model.increment(len(morphemes))

    def decrement(self, pattern):
        morphemes = self._unigrams(pattern)
        for morpheme in morphemes:
            self.morpheme_model.decrement(morpheme)
        self.length_model.decrement(len(morphemes))

    def prob(self, pattern):
        morphemes = self._unigrams(pattern)
        return (sum(map(self.morpheme_model.weight, morphemes)) +
                self.morpheme_model.gamma_factor(len(morphemes)) +
                self.length_model.prob(len(morphemes)))

    def log_likelihood(self):
        return (self.morpheme_model.log_likelihood()
                + self.length_model.log_likelihood())

    def __repr__(self):
        return 'PoissonUnigram(length ~ {self.length_model}, morph ~ {self.morpheme_model})'.format(self=self)

class BigramPattern:
    def __init__(self, K, beta, vocabulary):
        self.morpheme_models = [DirichletMultinomial(K+1, beta) for _ in range(K+1)]
        self.START = K
        self.STOP = K
        self.vocabulary = vocabulary

    def _unigrams(self, pattern):
        yield self.START
        for m in self.vocabulary[pattern]:
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

    def log_likelihood(self):
        return sum(m.log_likelihood() for m in self.morpheme_models)

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

    def pseudo_log_likelihood(self):
        return sum(self.prob(k) * len(assignments)
                for k, assignments in self.assignments.iteritems())

    def log_likelihood(self):
        return (self.stem_model.log_likelihood()
                + self.morpheme_model.log_likelihood())

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

    def pseudo_log_likelihood(self):
        return sum(self.prob(doc, word) * len(assignments)
                for (doc, word), assignments in self.assignments.iteritems())

    def log_likelihood(self):
        return (sum(dt.log_likelihood() for dt in self.document_topic)
                + sum(tw.log_likelihood() for tw in self.topic_word))

    def __repr__(self):
        return 'TopicModel(#topics={self.Ntopics})'.format(self=self)

# TODO remove
class PoissonUnigram:
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__class__ = PoissonUnigramPattern

class Bigram:
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__class__ = BigramPattern
