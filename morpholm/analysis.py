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

# XXX fix for the Dalrymple analyzer
analysis_fix = re.compile('A\+(\d)')

class Analysis:
    def __init__(self, analysis, vocabularies):
        """ Split the morphemes from the output of the analyzer """
        analysis = analysis_fix.sub(r'a+\1', analysis) # XXX apply fix
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
        self.pattern = vocabularies['pattern'][tuple(morphemes)]

    def decode_pattern(self, morphemes, vocabularies):
        for morph in morphemes:
            if morph == STEM:
                yield vocabularies['stem'][self.stem]
            else:
                yield vocabularies['morpheme'][morph]

    def decode(self, vocabularies):
        return '+'.join(self.decode_pattern(vocabularies['pattern'][self.pattern], vocabularies))

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

def init_vocabularies():
    vocabularies = {'word': corpus.Vocabulary('<s>', '</s>'),
                    'stem': corpus.Vocabulary('<s>', '</s>'),
                    'pattern': corpus.Vocabulary(),
                    'morpheme': corpus.Vocabulary()}
    assert vocabularies['morpheme']['stem'] == STEM
    for name in ('word', 'stem'):
        for w, i in (('<s>', corpus.START), ('</s>', corpus.STOP)):
            assert vocabularies[w] == i
    word_analyses = {corpus.START: [Analysis('<s>', vocabularies)],
                     corpus.STOP: [Analysis('</s>', vocabularies)]}
    return word_analyses, vocabularies
