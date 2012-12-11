import logging
from vpyp.prob import DirichletMultinomial
from vpyp.pyp import PYP
from vpyp.prior import GammaPrior, PYPPrior, stuple
from vpyp.lda.model import TopicModel
from ..models import MorphoProcess

class MorphoLDA(TopicModel):
    def __init__(self, n_topics, n_docs, pattern_base, stem_base, analyses):
        super(MorphoLDA, self).__init__(n_topics)
        self.alpha = GammaPrior(1.0, 1.0, 1.0)
        self.document_topic = [DirichletMultinomial(n_topics, self.alpha) for _ in xrange(n_docs)]
        self.stem_base = stem_base
        self.pattern_model = PYP(pattern_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)) # G_p
        def make_topic_word():
            stem_model = PYP(stem_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)) # G_s
            mp = MorphoProcess(stem_model, self.pattern_model, analyses)
            return PYP(mp, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)) # G_w
        self.topic_word = [make_topic_word() for _ in xrange(n_topics)]

    def log_likelihood(self):
        return (sum(d.log_likelihood() for d in self.document_topic)
                + self.alpha.log_likelihood()
                + sum(topic.log_likelihood() # G_w
                    + topic.prior.log_likelihood() # d_w, T_w
                    + topic.base.stem_model.log_likelihood() # G_s
                    + topic.base.stem_model.prior.log_likelihood() # d_s, T_s
                    for topic in self.topic_word)
                + self.pattern_model.log_likelihood(full=True) # G_p + ...
                + self.stem_base.log_likelihood(full=True)) # G_s^0 + ...

    def resample_hyperparemeters(self, n_iter):
        ar = stuple((0, 0))
        logging.info('Resampling doc-topic hyperparameters')
        ar += self.alpha.resample(n_iter)
        logging.info('Resampling stem base / pattern model hyperparameters')
        ar += self.pattern_model.resample_hyperparemeters(n_iter) # G_p
        ar += self.pattern_model.base.resample_hyperparemeters(n_iter) # G_p^0
        ar += self.stem_base.resample_hyperparemeters(n_iter) # G_s^0
        logging.info('Resampling all topic-word PYP hyperparameters')
        for topic in self.topic_word:
            ar += topic.resample_hyperparemeters(n_iter) # G_w
            ar += topic.base.stem_model.resample_hyperparemeters(n_iter) # G_s
        return ar

    def __repr__(self):
        return ('MorphoLDA(#topics={self.n_topics} '
                '| alpha={self.alpha}, beta=PYP(base=MP(stem ~ PYP(base={self.stem_base}); '
                'pattern ~ {self.pattern_model})))').format(self=self)
