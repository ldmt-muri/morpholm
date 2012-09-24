import math
import numpy as np
from collections import defaultdict
from prob import mult_sample, remove_random, DirichletMultinomial, GammaPoisson

class PoissonUnigram:
    def __init__(self, K, beta, gamma, delta):
        self.morpheme_model = DirichletMultinomial(K, beta)
        self.length_model = GammaPoisson(gamma, delta)

    def increment(self, analysis):
        for morpheme in analysis.non_stem_morphemes:
            self.morpheme_model.increment(morpheme)
        self.length_model.increment(len(analysis))

    def decrement(self, analysis):
        for morpheme in analysis.non_stem_morphemes:
            self.morpheme_model.decrement(morpheme)
        self.length_model.decrement(len(analysis))

    def pred_weight(self, analysis):
        return (sum(map(self.morpheme_model.pred_weight, analysis.non_stem_morphemes)) +
                self.morpheme_model.gamma_factor(len(analysis)) +
                self.length_model.pred_weight(len(analysis)))

    def prob(self, analysis):
        return (sum(map(self.morpheme_model.prob, analysis.non_stem_morphemes)) +
                self.length_model.prob(len(analysis)))

    def __str__(self):
        return 'PoissonUnigram(length ~ {self.length_model}, morph ~ {self.morpheme_model})'.format(self=self)

class MorphoProcess:
    def __init__(self, stem_model, morpheme_model):
        self.stem_model = stem_model
        self.morpheme_model = morpheme_model
        self.assignments = defaultdict(list)

    def increment(self, k):
        # Sample analysis & store assignment
        i = (0 if len(self.analyses[k]) == 1 else
                mult_sample((i, math.exp(self.pred_weight(analysis)))
                    for i, analysis in enumerate(self.analyses[k])))
        self.assignments[k].append(i)
        analysis = self.analyses[k][i]
        # Increment models
        self.stem_model.increment(analysis.stem)
        self.morpheme_model.increment(analysis)

    def decrement(self, k):
        # Select assigned analysis randomly and remove it
        analysis = self.analyses[k][remove_random(self.assignments[k])]
        # Decrement models
        self.stem_model.decrement(analysis.stem)
        self.morpheme_model.decrement(analysis)

    def pred_weight(self, analysis):
        return (self.stem_model.pred_weight(analysis.stem) +
                self.morpheme_model.pred_weight(analysis))

    def analysis_prob(self, analysis):
        return (self.stem_model.prob(analysis.stem) +
                self.morpheme_model.prob(analysis))

    def prob(self, k):
        return np.logaddexp.reduce([self.analysis_prob(analysis)
            for analysis in self.analyses[k]])

    def log_likelihood(self):
        return sum(self.prob(k) * len(assignments)
                for k, assignments in self.assignments.iteritems())

    def __str__(self):
        return 'MorphoProcess(#words={N} | stem ~ {self.stem_model}, morphemes ~ {self.morpheme_model})'.format(self=self, N=sum(map(len, self.assignments.itervalues())))

class TopicModel:
    def __init__(self, Ntopics, Ndocs, doc_process, topic_process):
        self.Ntopics = Ntopics
        self.document_topic = [doc_process() for _ in xrange(Ndocs)]
        self.topic_word = [topic_process() for _ in xrange(Ntopics)]
        self.assignments = defaultdict(list)

    def increment(self, doc, word):
        z = mult_sample((k, math.exp(self.pred_weight(doc, word, k)))
                for k in xrange(self.Ntopics))
        self.assignments[doc, word].append(z)
        self.document_topic[doc].increment(z)
        self.topic_word[z].increment(word)

    def decrement(self, doc, word):
        z = remove_random(self.assignments[doc, word])
        self.document_topic[doc].decrement(z)
        self.topic_word[z].decrement(word)

    def pred_weight(self, doc, word, k):
        return self.topic_prob(doc, word, k)

    def topic_prob(self, doc, word, k):
        return self.document_topic[doc].prob(k) + self.topic_word[k].prob(word)

    def prob(self, doc, word):
        return np.logaddexp.reduce([self.topic_prob(doc, word, k)
            for k in xrange(self.Ntopics)])

    def log_likelihood(self):
        return sum(self.prob(doc, word) * len(assignments)
                for (doc, word), assignments in self.assignments.iteritems())

    def __str__(self):
        return 'TopicModel(#topics={self.Ntopics})'.format(self=self)