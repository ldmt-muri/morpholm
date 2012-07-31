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
OOV = -1

class FSM:
    def __init__(self, fsm):
        self.fsm = foma.read_binary(fsm)

    def get_analyses(self, word):
        if len(word) > 3 and wRE.match(word):
            return list(self.fsm.apply_up(strip_accents(word)))
        return None

class Vocabulary:
    def __init__(self):
        self.words = {}
        self.frozen = False

    def __getitem__(self, word):
        if word not in self.words:
            if self.frozen: return OOV
            self.words[word] = len(self.words)
        return self.words[word]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

class Analysis:
    def __init__(self, analysis, vocabulary):
        """ Split the morphemes from the output of the analyzer """
        morphs = analysis.split('+')
        morphs = (morph for morph in morphs if morph)
        self.morphemes = []
        for morph in morphs:
            if morph in MORPH:
                self.morphemes.append(vocabulary['morpheme'][morph])
            else:
                self.stem = vocabulary['stem'][morph]
                self.stem_str = morph
                self.morphemes.append(STEM)

    def __len__(self):
        return len(self.morphemes)-1

def analyze_corpus(fsm, text):
    vocabulary = {'morpheme': Vocabulary(), 'stem': Vocabulary()}
    assert (vocabulary['morpheme']['stem'] == STEM)
    corpus = []
    sys.stderr.write('Reading corpus ')
    with open(text) as fp:
        for i, sentence in enumerate(fp):
            if i % 1000 == 0:
                sys.stderr.write('*')
                sys.stderr.flush()
            if i % 100 == 0:
                sys.stderr.write('.')
                sys.stderr.flush()
            for word in sentence.decode('utf8').split():
                analyses = fsm.get_analyses(word)
                if not analyses: continue
                analyses = [Analysis(analysis, vocabulary) for analysis in analyses]
                corpus.append(analyses)
    sys.stderr.write(' done\n')
    return corpus, vocabulary
