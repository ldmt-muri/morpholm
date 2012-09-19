import unicodedata
import sys
import re
import foma

strip_accents = lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn')

wRE = re.compile('^[a-z]+$')
morphRE = re.compile('[A-Z]')
STEM = 0

class OOV(Exception): pass
class AnalysisError(Exception): pass

class FSM:
    def __init__(self, fsm):
        self.fsm = foma.read_binary(fsm)

    def get_analyses(self, word):
        if '+' in word:
            word = word.replace('+', '#')
        word = strip_accents(word)
        if len(word) > 3 and wRE.match(word):
            analyses = set(self.fsm.apply_up(word))
            if not analyses: return {word}
            return analyses
        return {word}

class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.frozen = False

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

# TODO: cache to reduce memory usage
class Analysis:
    def __init__(self, analysis, vocabulary):
        """ Split the morphemes from the output of the analyzer """
        analysis = analysis_fix.sub(r'a+\1', analysis) # ??????????????
        morphs = analysis.split('+')
        morphs = (morph for morph in morphs if morph)
        self.oov = False
        self.morphemes = []
        self.stem = None
        for morph in morphs:
            if morphRE.search(morph): # non-stem
                self.morphemes.append(vocabulary['morpheme'][morph])
            else: # stem
                try:
                    if self.stem is not None:
                        raise AnalysisError(analysis)
                    self.stem = vocabulary['stem'][morph]
                except OOV:
                    self.stem = morph # do not encode
                    self.oov = True
                self.morphemes.append(STEM)
        if self.stem is None:
            raise AnalysisError(analysis)

    def split(self):
        if not hasattr(self, '_right'):
            stem_index = self.morphemes.index(STEM)
            self._left = self.morphemes[stem_index-1::-1] # self.morphemes[:stem_index]
            self._right = self.morphemes[stem_index+1:]

    def decode_stem(self, vocabulary):
        if self.oov: return self.stem
        return vocabulary[self.stem]

    @property
    def right_morphemes(self):
        self.split()
        return self._right

    @property
    def left_morphemes(self):
        self.split()
        return self._left

    def __len__(self):
        return len(self.morphemes) - 1

class Corpus:
    def __init__(self, sentences=[]):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            for word in sentence:
                yield word

    def __len__(self):
        return sum(len(sentence) for sentence in self.sentences)

# TODO: store analyzes!!!
def analyze_corpus(fsm, stream):
    vocabulary = {'morpheme': Vocabulary(), 'stem': Vocabulary()}
    assert (vocabulary['morpheme']['***STEM***'] == STEM)
    corpus = Corpus()
    sys.stderr.write('Reading corpus ')
    for i, sentence in enumerate(stream):
        if i % 1000 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()
        analyzed_sentence = []
        for word in sentence.decode('utf8').split():
            analyses = fsm.get_analyses(word)
            try:
                analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
                analyzed_sentence.append(analyses)
            except AnalysisError as analysis:
                print('Analyzis error for {0}: {1}'.format(word, analysis))
        corpus.sentences.append(analyzed_sentence)
    sys.stderr.write(' done\n')
    return corpus, vocabulary
