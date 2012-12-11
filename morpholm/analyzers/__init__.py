import logging
from analyzer import Analyzer

class FomaAnalyzer(Analyzer):
    def __init__(self, fsm):
        self.fsm = foma.read_binary(fsm)

    def analyze_word(self, word):
        return set(self.fsm.apply_up(word))

    def __repr__(self):
        return 'FomaAnalyzer'

class XFSTAnalyzer(Analyzer):
    def __init__(self, fsm):
        self.fsm = xfst.read_binary(fsm)

    def analyze_word(self, word):
        return set(self.fsm.apply_up(word))

    def synthesize_word(self, analysis):
        return set(self.fsm.apply_down(analysis))

    def __repr__(self):
        return 'XFSTAnalyzer'

class PymorphyAnalyzer(Analyzer):
    pass

all_analyzers = {}

try:
    import foma
    all_analyzers['foma'] = FomaAnalyzer
except ImportError:
    logging.info('foma backend not available')
try:
    import xfst
    all_analyzers['xfst'] = XFSTAnalyzer
except ImportError:
    logging.info('xfst backend not available')
try:
    import pymorphy
    all_analyzers['pymorphy'] = PymorphyAnalyzer
except ImportError:
    logging.info('pymorphy backend not available')
try:
    from xerox import XeroxAnalyzer
    all_analyzers['xerox'] = XeroxAnalyzer
except ImportError:
    logging.info('xerox backend not available')
