import sys
import logging
import re
import corpus

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try loading analyzers
try:
    import foma
except ImportError:
    logger.info('foma backend not available')
try:
    import xfst
except ImportError:
    logger.info('xfst backend not available')
try:
    import pymorphy
except ImportError:
    logger.info('pymorphy backend not available')


wRE = re.compile('^[^\W\d_]+$', re.UNICODE)
morphRE = re.compile('[A-Z]')
STEM = 0
class AnalysisError(Exception): pass

info2morph = lambda info: '+'.join(m.title() for m in info.split(',') if m)
class_map = {
'-': 'UNK', # Errors
'S': 'N', # Noun
'A': 'J', # Adjective
'V': 'V', # Verb
'PR': 'PR', # Preposition
'CONJ': 'CONJ', # Conjunction
'ADV': 'ADV' # Adverb
}

class PyMorphy:
    def __init__(self, db):
        self.analyzer = pymorphy.get_morph(db)

    def apply_up(self, word):
        return [(analysis['norm'].lower()
            +'+'+class_map[analysis['class']]
            +'+'+info2morph(analysis['info']))
            for analysis in self.analyzer.get_graminfo(word.upper(), standard=True)]

class Analyzer:
    def __init__(self, fsm, backend='foma', trust=True):
        self.fsm = (PyMorphy(fsm) if backend == 'pymorphy' else
                (xfst if backend == 'xfst' else foma).read_binary(fsm))
        self.trust = trust

    def get_analyses(self, word):
        if '+' in word:
            word = word.replace('+', '#')
        if len(word) > 3 and wRE.match(word):
            analyses = set(self.fsm.apply_up(word))
            if self.trust: # always try to analyze
                return analyses or {word}
            else: # keep unanalyzed version even if analysis is possible
                return analyses | {word}
        return {word}

analysis_fix = re.compile('A\+(\d)')

class MorphemePattern:
    def __init__(self, morphemes):
        self.morphemes = morphemes
        self._hash = hash(tuple(self.morphemes))

    @property
    def right_morphemes(self):
        self._split()
        return self._right

    @property
    def left_morphemes(self):
        self._split()
        return self._left

    def _split(self):
        if not hasattr(self, '_right'):
            stem_index = self.morphemes.index(STEM)
            self._left = self.morphemes[stem_index-1::-1]
            self._right = self.morphemes[stem_index+1:]

    def __iter__(self):
        return (morpheme for morpheme in self.morphemes if morpheme != STEM)

    def __len__(self):
        return len(self.morphemes) - 1

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return self._hash

class Analysis:
    def __init__(self, analysis, vocabularies):
        """ Split the morphemes from the output of the analyzer """
        analysis = analysis_fix.sub(r'a+\1', analysis) # ??????????????
        morphs = analysis.split('+')
        morphs = (morph for morph in morphs if morph)
        self.stem = None
        morphemes = []
        for morph in morphs:
            if morphRE.search(morph): # non-stem
                morphemes.append(vocabularies['morpheme'][morph])
            else: # stem
                if self.stem is not None:
                    raise AnalysisError(analysis)
                self.stem = vocabularies['stem'][morph]
                morphemes.append(STEM)
        if self.stem is None:
            raise AnalysisError(analysis)
        self.pattern = MorphemePattern(morphemes)

    def decode(self, vocabularies):
        def m():
            for morph in self.pattern.morphemes:
                if morph == STEM:
                    yield self.decode_stem(vocabularies['stem'])
                else:
                    yield vocabularies['morpheme'][morph]
        return '+'.join(m())

def analyze_corpus(stream, fsm, vocabularies, word_analyses):
    sys.stderr.write('Reading corpus ')
    N = 0
    sentences = []
    for i, sentence in enumerate(stream):
        if i % 1000 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()
        words = []
        for word in sentence.decode('utf8').split():
            N += 1
            w = vocabularies['word'][word]
            try:
                if not w in word_analyses:
                    analyses = fsm.get_analyses(word)
                    analyses = [Analysis(analysis, vocabularies) for analysis in analyses]
                    word_analyses[w] = analyses
                words.append(w)
            except AnalysisError as analysis:
                print(u'Analysis error for {0}: {1}'.format(word, analysis))
                word_analyses[w] = [Analysis(word, vocabularies)]
        sentences.append(words)
    sys.stderr.write(' done\n')
    return corpus.Corpus(sentences)

def init_vocabularies(*init):
    vocabularies = {'word': corpus.Vocabulary(*init),
                    'morpheme': corpus.Vocabulary(),
                    'stem': corpus.Vocabulary()}
    assert (vocabularies['morpheme']['stem'] == STEM)
    return vocabularies
