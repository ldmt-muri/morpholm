import math
import random
import numpy as np
from collections import defaultdict
from prob import mult_sample

STEM = 0

class MorphoProcess:
    def __init__(self, stem_model, morpheme_model, length_model):
        self.stem_model = stem_model
        self.morpheme_model = morpheme_model
        self.length_model = length_model
        self.assignments = defaultdict(list)

    def increment(self, k):
        analysis = self.get_assignment(k)
        # Increment models
        self.stem_model.increment(analysis.stem)
        for morpheme in analysis.morphemes:
            if morpheme == STEM: continue
            self.morpheme_model.increment(morpheme)
        self.length_model.increment(len(analysis))

    def get_assignment(self, k):
        # Sample analysis & store assignment
        i = mult_sample((i, math.exp(self.pred_weight(analysis)))
                for i, analysis in enumerate(self.analyses[k]))
        self.assignments[k].append(i)
        return self.analyses[k][i]

    def decrement(self, k):
        analysis = self.remove_assignement(k)
        # Decrement models
        self.stem_model.decrement(analysis.stem)
        for morpheme in analysis.morphemes:
            if morpheme == STEM: continue
            self.morpheme_model.decrement(morpheme)
        self.length_model.decrement(len(analysis))

    def remove_assignement(self, k):
        # Select assigned analysis randomly and remove it
        assignments = self.assignments[k]
        a = random.randrange(0, len(assignments[k]))
        analysis = self.analyses[k][assignments[k][a]]
        del assignments[k][a]
        return analysis

    def pred_weight(self, analysis):
        return (self.stem_model.pred_weight(analysis.stem) +
                sum(self.morpheme_model.pred_weight(morph) for morph in analysis.morphemes) +
                self.length_model.pred_weight(len(analysis)))

    def analysis_prob(self, analysis):
        return (self.stem_model.prob(analysis.stem) +
                sum(self.morpheme_model.prob(morph) for morph in analysis.morphemes) +
                self.length_model.prob(len(analysis)))

    def prob(self, k):
        return np.logaddexp([self.analysis_prob(analysis) for analysis in self.analyses[k]])
