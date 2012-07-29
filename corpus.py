from itertools import izip
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

should_analyze = lambda word: len(word) > 3 and wRE.match(word)

class OOV(Exception):
    pass

class Vocabulary:
    def __init__(self):
        self.words = {}
        self.frozen = False

    def __getitem__(self, word):
        if word not in self.words:
            if self.frozen: raise OOV(word)
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
        morphs = [morph for morph in morphs if morph]
        self.morphemes = []
        for morph in morphs:
            if morph in MORPH:
                self.morphemes.append(vocabulary['morpheme'][morph])
            else:
                self.lemma = vocabulary['lemma'][morph]

    def __len__(self):
        return len(self.morphemes)

class Corpus:
    def __init__(self, fsm, corpus):
        fsm = foma.read_binary(fsm)
        self.vocabulary = {'morpheme': Vocabulary(), 'lemma': Vocabulary()}
        self.corpus = []
        sys.stdout.write('Reading corpus ')
        with open(corpus) as fp:
            for i, sentence in enumerate(fp):
                if i % 1000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                for word in sentence.decode('utf8').split():
                    if should_analyze(word):
                        analyses = list(fsm.apply_up(strip_accents(word)))
                        if analyses:
                            self.corpus.append(map(self.make_analysis, analyses))
        self.n_morphemes = len(self.vocabulary['morpheme'])
        self.n_lemmas = len(self.vocabulary['lemma'])
        print(' found {0} lemmas'.format(self.n_lemmas))

    def make_analysis(self, analysis):
        return Analysis(analysis, self.vocabulary)

    def __iter__(self):
        return iter(self.corpus)

class Decoder:
    def __init__(self, fsm, model, vocabulary):
        self.fsm = foma.read_binary(fsm)
        self.model = model
        self.vocabulary = vocabulary
        vocabulary['morpheme'].frozen = True
        vocabulary['lemma'].frozen = True

    def make_analysis(self, analysis):
        return Analysis(analysis, self.vocabulary)

    def decode(self, word):
        if not should_analyze(word): return word
        analyses = list(self.fsm.apply_up(strip_accents(word)))
        if not analyses: return word
        try:
            probs = map(self.model.prob, map(self.make_analysis, analyses))
            return max(izip(probs, analyses))[1]
        except OOV:
            return word
