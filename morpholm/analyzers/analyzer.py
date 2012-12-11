import re
import logging
from vpyp.corpus import Vocabulary, OOV

word_re = re.compile('^[^\W\d_]+$', re.UNICODE)
morph_re = re.compile('[A-Z]')

STEM = '__STEM__'

class AnalysisError(Exception):
    def __init__(self, cause, analysis):
        self.cause = cause
        self.analysis = analysis

    def __str__(self):
        return u'{self.cause} for {self.analysis}'.format(self=self)

def parse_analysis(analysis):
    stem = None
    morphemes = []
    for morph in analysis.split('+'):
        if not morph: continue
        if morph_re.search(morph):
            morphemes.append(morph)
        else:
            if stem is not None: raise AnalysisError('Multiple stems', analysis)
            stem = morph
            morphemes.append(STEM)
    if stem is None: raise AnalysisError('No stem', analysis)
    return stem, morphemes

class Analysis(object):
    def __init__(self, stem, pattern):
        self.stem = stem
        self.pattern = pattern

    def __eq__(self, other):
        return (isinstance(other, Analysis) and 
                (self.stem, self.pattern) == (other.stem, other.pattern))

    def __hash__(self):
        return hash((self.stem, self.pattern))

class Analyzer(object):
    def analyze_corpus(self, corpus, add_null=False, rm_null=False):
        if not hasattr(corpus, 'analyses'): # Pre-analyzed corpus
            corpus.analyses = {}
            corpus.stem_vocabulary = Vocabulary()
            corpus.morpheme_vocabulary = Vocabulary()
            corpus.pattern_vocabulary = Vocabulary()
        if not rm_null:
            null_pattern = corpus.pattern_vocabulary[(corpus.morpheme_vocabulary[STEM],)]
        for w, word in enumerate(corpus.vocabulary):
            if w in corpus.analyses: continue # Skip already analyzed words
            analyses = []
            if word_re.match(word):
                for analysis in self.analyze_word(word):
                    try:
                        stem, pattern = parse_analysis(analysis)
                    except AnalysisError as e:
                        logging.error('Analysis error for %s: %s', word, e)
                        continue
                    try:
                        morphemes = tuple(corpus.morpheme_vocabulary[m] for m in pattern)
                    except OOV as e:
                        logging.error('Unknown morpheme "%s" in %s', e, analysis)
                        continue
                    if rm_null and len(morphemes) < 2: continue
                    s = corpus.stem_vocabulary[stem]
                    p = corpus.pattern_vocabulary[morphemes]
                    analyses.append(Analysis(s, p))
            if (not analyses or add_null) and not rm_null:
                analyses.append(Analysis(corpus.stem_vocabulary[word], null_pattern))
            corpus.analyses[w] = tuple(set(analyses))
