from collections import deque
from itertools import chain, repeat
import unicodedata
import sys
import logging
import re

# Try loading analyzers
try:
    import foma
except ImportError:
    logging.info('foma backend not available')
try:
    import xfst
except ImportError:
    logging.info('xfst backend not available')
try:
    import pymorphy
except ImportError:
    logging.info('pymorphy backend not available')

strip_accents = lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn')

#wRE = re.compile('^[a-z]+$')
wRE = re.compile('^[^\W\d_]+$', re.UNICODE)
morphRE = re.compile('[A-Z]')
STEM = 0

class OOV(Exception): pass
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

class FSM:
    def __init__(self, fsm, backend='foma', trust=True):
        self.fsm = (PyMorphy(fsm) if backend == 'pymorphy' else
                (xfst if backend == 'xfst' else foma).read_binary(fsm))
        self.trust = True

    def get_analyses(self, word):
        if '+' in word:
            word = word.replace('+', '#')
        #word = strip_accents(word)
        if len(word) > 3 and wRE.match(word):
            analyses = set(self.fsm.apply_up(word))
            if self.trust:
                return {word} if not analyses else analyses
            else:
                return analyses | {word}
        return {word}

class Vocabulary:
    def __init__(self, *init):
        self.word2id = {}
        self.id2word = []
        self.frozen = False
        for w in init:
            self.word2id[w] = len(self)
            self.id2word.append(w)

    def __getitem__(self, word):
        if isinstance(word, int):
            assert word >= 0
            return self.id2word[word]
        if word not in self.word2id:
            if self.frozen: raise OOV(word)
            self.word2id[word] = len(self)
            self.id2word.append(word)
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

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
        self.oov = False
        self.stem = None
        morphemes = []
        for morph in morphs:
            if morphRE.search(morph): # non-stem
                morphemes.append(vocabularies['morpheme'][morph])
            else: # stem
                try:
                    if self.stem is not None:
                        raise AnalysisError(analysis)
                    self.stem = vocabularies['stem'][morph]
                except OOV:
                    self.stem = morph # do not encode
                    self.oov = True
                morphemes.append(STEM)
        if self.stem is None:
            raise AnalysisError(analysis)
        self.pattern = MorphemePattern(morphemes)

    def decode_stem(self, vocabulary):
        if self.oov: return self.stem
        return vocabulary[self.stem]

    def decode(self, vocabularies):
        def m():
            for morph in self.pattern.morphemes:
                if morph == STEM:
                    yield self.decode_stem(vocabularies['stem'])
                else:
                    yield vocabularies['morpheme'][morph]
        return '+'.join(m())

class Corpus:
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            for word in sentence:
                yield word

    def __len__(self):
        return sum(len(sentence) for sentence in self.sentences)

def init_vocabularies(*init):
    vocabularies = {'word': Vocabulary(*init),
                    'morpheme': Vocabulary(),
                    'stem': Vocabulary()}
    assert (vocabularies['morpheme']['stem'] == STEM)
    return vocabularies

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
    return Corpus(sentences)

def encode_corpus(stream, vocabulary):
    return Corpus([[vocabulary[word] for word in sentence.decode('utf8').split()]
                        for sentence in stream])

START = 0
STOP = 1
def ngrams(sentence, order):
    if order == 1:
        for w in sentence:
            yield (w,)
        return
    s = chain(repeat(START, order-1), sentence, (STOP,))
    ngram = deque(maxlen=order)
    for w in s:
        ngram.append(w)
        if len(ngram) == order:
            yield tuple(ngram)
