import math
import numpy
from prob import mult_sample, DirichletMultinomial
from corpus import START, STOP

logsumexp = numpy.logaddexp.reduce

class SimpleBigram:
    def __init__(self, alpha, n_ctx):
        self.alpha = alpha
        self.n_ctx = n_ctx
        self.models = {}

    def __getitem__(self, ctx):
        if ctx not in self.models:
            self.models[ctx] = self._get(ctx)
        return self.models[ctx]

    def _get(self, ctx):
        if ctx not in self.models:
            return DirichletMultinomial(self.n_ctx, self.alpha)
        return self.models[ctx]

    def increment(self, k):
        self[k[0]].increment(k[-1])

    def decrement(self, k):
        self.models[k[0]].decrement(k[-1])

    def prob(self, k):
        return self._get(k[0]).prob(k[-1])

    def log_likelihood(self):
        return sum(m.log_likelihood() for m in self.models.itervalues())

    def __repr__(self):
        return 'SimpleBigram(#ctx={0})'.format(len(self.models))

class SeqMorphoProcess:
    def __init__(self, stem_model, morpheme_model, analyses):
        self.stem_model = stem_model
        self.morpheme_model = morpheme_model
        self.analyses = analyses

    def increment(self, p_prev, p, p_next):
        k_prev, i_prev = p_prev
        k, _ = p
        k_next, i_next = p_next
        a_prev = self.analyses[k_prev][i_prev]
        # Sample assignment
        if i_next is None: # no future yet
            i = (0 if len(self.analyses[k]) == 1 else
                    mult_sample((i, math.exp(self.analysis_prob(a_prev, a)))
                        for i, a in enumerate(self.analyses[k])))
        else: # after 1st iteration
            a_next = self.analyses[k_next][i_next]
            i = (0 if len(self.analyses[k]) == 1 else
                    mult_sample((i, math.exp(self.analysis_prob(a_prev, a)
                                           + self.analysis_prob(a, a_next)))
                        for i, a in enumerate(self.analyses[k])))
        # Increment models
        a = self.analyses[k][i]
        self.stem_model.increment((a_prev.stem, a.stem))
        if i_next is not None:
            self.stem_model.increment((a.stem, a_next.stem))
        self.morpheme_model.increment((a_prev.pattern, a.pattern))
        if i_next is not None: self.morpheme_model.increment((a.pattern, a_next.pattern))
        return i

    def decrement(self, p_prev, p, p_next):
        k_prev, i_prev = p_prev
        k, i = p
        k_next, i_next = p_next
        # Decrement models
        a_prev = self.analyses[k_prev][i_prev]
        a = self.analyses[k][i]
        a_next = self.analyses[k_next][i_next]
        self.stem_model.decrement((a_prev.stem, a.stem))
        self.stem_model.decrement((a.stem, a_next.stem))
        self.morpheme_model.decrement((a_prev.pattern, a.pattern))
        self.morpheme_model.decrement((a.pattern, a_next.pattern))

    def analysis_prob(self, analysis1, analysis2):
        return (self.stem_model.prob((analysis1.stem, analysis2.stem)) +
                self.morpheme_model.prob((analysis1.pattern, analysis2.pattern)))

    def log_likelihood(self):
        return (self.stem_model.log_likelihood() 
                + self.morpheme_model.log_likelihood())

    def viterbi(self, sentence):
        probs = [[(0, None)]]
        seq = [START]+sentence+[STOP]
        for k in xrange(1, len(sentence)+2):
            probs.append([max((self.analysis_prob(a_prev, a) + probs[k-1][i_prev][0], i_prev)
                for i_prev, a_prev in enumerate(self.analyses[seq[k-1]]))
                for a in self.analyses[seq[k]]])
        _, best_prob = probs[-1][0]
        best = []
        i_next = 0
        for k in xrange(1, len(sentence)+1):
            _, i_next = probs[-k][i_next]
            best.append(self.analyses[sentence[-k]][i_next])
        return best_prob, best[::-1]

    def prob(self, sentence):
        probs = [[0]]
        seq = [START]+sentence+[STOP]
        for k in xrange(1, len(sentence)+2):
            probs.append([logsumexp([self.analysis_prob(a_prev, a) + probs[k-1][i_prev]
                for i_prev, a_prev in enumerate(self.analyses[seq[k-1]])])
                for a in self.analyses[seq[k]]])
        return probs[-1][0]

    def __repr__(self):
        return ('SeqMorphoProcess(stem ~ {self.stem_model}, '
                'morphemes ~ {self.morpheme_model})').format(self=self)
