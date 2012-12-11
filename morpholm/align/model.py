import logging
from vpyp.align.model import AlignmentModel, AlignmentDistribution
from vpyp.pyp import PYP
from vpyp.prob import BetaBernouilli
from vpyp.prior import GammaPrior, PYPPrior, stuple
from ..models import MorphoProcess

class MorphoAlignmentModel(AlignmentModel):
    def __init__(self, n_source, stem_base, pattern_model, analyses):
        """AlignmentModel(n_source, t_base) -> morpho-alignment model
        n_source: size of the source vocabulary
        stem_base: t-table MP stem base (G_s^0)
        pattern_model: t-table MP pattern model (G_p)
        analyses: t-table MP analyses"""
        self.null = BetaBernouilli(1.0, 1.0) # p(NULL) ~ Beta(1, .)
        self.a_table = AlignmentDistribution(GammaPrior(1.0, 1.0, 4.0))
        self.stem_base = stem_base # G_s^0
        self.pattern_model = pattern_model
        def make_t_word():
            stem_model = PYP(stem_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)) # G_s
            mp = MorphoProcess(stem_model, self.pattern_model, analyses)
            return PYP(mp, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0)) # G_w
        self.t_table = [make_t_word() for _ in xrange(n_source)]

    def log_likelihood(self):
        return (sum((t_word.log_likelihood() # G_w
                   + t_word.prior.log_likelihood() # d_w, T_w
                   + t_word.base.stem_model.log_likelihood() # G_s
                   + t_word.base.stem_model.prior.log_likelihood()) # d_s, T_s
                    for t_word in self.t_table)
                + self.pattern_model.log_likelihood(full=True) # G_p + ...
                + self.stem_base.log_likelihood(full=True) # G_s^0 + ...
                + self.null.log_likelihood()
                + self.a_table.log_likelihood() + self.a_table.scale_prior.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        ar = stuple((0, 0))
        logging.info('Resampling stem base / pattern model hyperparameters')
        ar += self.pattern_model.resample_hyperparemeters(n_iter) # G_p
        ar += self.pattern_model.base.resample_hyperparemeters(n_iter) # G_p^0
        ar += self.stem_base.resample_hyperparemeters(n_iter) # G_s^0
        logging.info('Resampling t-table word and stem hyperparameters')
        for t_word in self.t_table:
            ar += t_word.resample_hyperparemeters(n_iter) # G_w
            ar += t_word.base.stem_model.resample_hyperparemeters(n_iter) # G_s
        logging.info('Resampling alignment distribution scale parameter')
        ar += self.a_table.resample_hyperparemeters(n_iter)
        return ar

    def __repr__(self):
        return ('MorphoAlignmentModel(#source words={n_source} '
                '| t-table[f] ~ PYP(base=MP(stem ~ PYP(base={self.stem_base}); '
                'pattern ~ {self.pattern_model}))'
                '| a-table ~ {self.a_table} + p(NULL)={self.p_null} ~ {self.null})'
                ).format(self=self, n_source=len(self.t_table))
