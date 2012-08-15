import unicodedata
import sys
import re
import foma

strip_accents = lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn')

wRE = re.compile('^[a-z]+$')
MORPH = {'aPss', 'tafaPss', 'voaPss', 
         'PastTense', 'NonpastTense', 'PresentTense', 'PresentActive', 'FutureTense',
         'NominalizationMP', 'NominalizationF', 'NominalizationHA',
         'Ma', 'ActiveI', 'ActiveAN', 'ANACaus', 'AhaCaus', 'AhaAbil',
         'AmpCaus', 'AnkCaus', 'Recip',
         'Directive', 'Locative',  'NounRoot', 'AdjRoot', 'PassRoot', 'NonPassRoot', 
         'Verb', 'Oblique', 'Deictic',
         'Passive1','Passive2', 'VerbPassive2', 'PassImp', 'ActImp', 'Imp',
         '1SgGen', '2SgGen', '3Gen', '1PlIncGen', '1PlExcGen', '2PlGen',
         'OvertGenAgent', 'ActiveVoice'}
        #'Guess', 'Inch', 'Adj', 'Punct', 'GEN', 'Redup', 'Noun',  'Root'

STEM = 0

class FSM:
    def __init__(self, fsm):
        self.fsm = foma.read_binary(fsm)

    def get_analyses(self, word):
        if len(word) > 3 and wRE.match(word):
            analyses = set(self.fsm.apply_up(strip_accents(word)))
            if not analyses:
                analyses = {word}
            return analyses
        return None

class OOV(Exception): pass

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

class Analysis:
    def __init__(self, analysis, vocabulary):
        """ Split the morphemes from the output of the analyzer """
        morphs = analysis.split('+')
        morphs = (morph for morph in morphs if morph)
        self.oov = False
        self.morphemes = []
        for morph in morphs:
            if morph in MORPH:
                self.morphemes.append(vocabulary['morpheme'][morph])
            else:
                try:
                    self.stem = vocabulary['stem'][morph]
                except OOV:
                    self.stem = morph
                    self.oov = True
                self.morphemes.append(STEM)

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
        return len(self.morphemes)-1

def analyze_corpus(fsm, stream):
    vocabulary = {'morpheme': Vocabulary(), 'stem': Vocabulary()}
    assert (vocabulary['morpheme']['stem'] == STEM)
    corpus = []
    sys.stderr.write('Reading corpus ')
    for i, sentence in enumerate(stream):
        if i % 1000 == 0:
            sys.stderr.write('*')
            sys.stderr.flush()
        for word in sentence.decode('utf8').split():
            analyses = fsm.get_analyses(word)
            if not analyses: continue
            analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
            corpus.append(analyses)
    sys.stderr.write(' done\n')
    return corpus, vocabulary
