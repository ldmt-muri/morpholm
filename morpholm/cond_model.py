# TODO remove

import seq_model

class SimpleBigram:
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__class__ = seq_model.SimpleBigram

class SeqMorphoProcess:
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__class__ = seq_model.SeqMorphoProcess
