from collections import deque
from itertools import chain, repeat

class OOV(Exception): pass
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

class Corpus:
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            for word in sentence:
                yield word

    def __len__(self):
        return sum(len(sentence) for sentence in self.sentences)

def encode_corpus(stream, vocabulary):
    return Corpus([[vocabulary[word] for word in sentence.decode('utf8').split()]
                        for sentence in stream])

START = 0
STOP = 1
def ngrams(sentence, order):
    if order == 1:
        for w in sentence:
            yield (w,)
    else:
        ngram = deque(maxlen=order)
        for w in chain(repeat(START, order-1), sentence, (STOP,)):
            ngram.append(w)
            if len(ngram) == order:
                yield tuple(ngram)

def hmm_trigrams(sentence):
    trigram = deque(maxlen=3)
    for w in chain((START,), sentence, (STOP,),):
        trigram.append(w)
        if len(trigram) == 3:
            yield tuple(trigram)

# XXX remove:
Analysis = None
MorphemePattern = None
def reload_analysis():
    global Analysis, MorphemePattern
    import analysis
    Analysis = analysis.Analysis
    MorphemePattern = analysis.MorphemePattern
