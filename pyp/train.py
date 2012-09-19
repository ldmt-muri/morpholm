import sys
import logging
import argparse
import math
from corpus import Corpus, FSM, Vocabulary, Analysis, AnalysisError, STEM
from prob import Multinomial, Poisson, CharLM
from pyp import PYP
from model import MorphoProcess

def analyze_corpus(stream, fsm, vocabularies, word_analyses):
    corpus = Corpus()
    sys.stderr.write('Reading corpus ')
    for i, sentence in enumerate(stream):
        if i % 1000 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()
        words = []
        for word in sentence.decode('utf8').split():
            w = vocabularies['word'][word]
            try:
                if w in word_analyses:
                    analyses = word_analyses[w]
                else:
                    analyses = fsm.get_analyses(word)
                    analyses = [Analysis(analysis, vocabularies) for analysis in analyses]
                    word_analyses[w] = analyses
                words.append(w)
            except AnalysisError as analysis:
                print('Analyzis error for {0}: {1}'.format(word, analysis))
        corpus.sentences.append(words)
    sys.stderr.write(' done\n')
    return corpus

theta = 1
d = 0.8
alpha = 1
beta = 1
p = 0.8
gamma = 1
delta = 1

Niter = 3

def run_sampler(model, corpus):
    for it in range(Niter):
        logging.info('Iteration %d/%d', it+1, Niter)
        for word in corpus:
            if it > 0: model.decrement(word)
            model.increment(word)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)

def print_ppl(model, corpus):
    n_words = 0
    loglik = 0
    for word in corpus:
        n_words += 1
        loglik += model.prob(word)
    ppl = math.exp(-loglik / n_words)
    logging.info('Words: %d\tLL: %.0f\tppl: %.3f', n_words, loglik, ppl)


def main():
    parser = argparse.ArgumentParser(description='Train PYP-MorphoLM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    args = parser.parse_args()

    vocabularies = {'word': Vocabulary(),
                    'morpheme': Vocabulary(),
                    'stem': Vocabulary()}
    word_analyses = {}
    assert (vocabularies['morpheme']['stem'] == STEM)

    fsm = FSM(args.fst)
    char_lm = CharLM(args.charlm)

    logging.info('Analyzing training corpus')
    with open(args.train) as train:
        corpus = analyze_corpus(train, fsm, vocabularies, word_analyses)

    char_lm.vocabulary = vocabularies['stem']
    mp = MorphoProcess(PYP(alpha, p, char_lm), Multinomial(beta), Poisson(gamma, delta))
    mp.analyses = word_analyses
    model = PYP(theta, d, mp)

    logging.info('Training model')
    run_sampler(model, corpus)

    logging.info('Computing test corpus perplexity')
    with open(args.test) as test:
        test_corpus = analyze_corpus(test, fsm, vocabularies, word_analyses)

    print_ppl(model, test_corpus)

def encode_corpus(stream, vocabulary):
    return Corpus([[vocabulary[word] for word in sentence.decode('utf8').split()]
                        for sentence in stream])

def baseline_main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train PYPLM')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    args = parser.parse_args()

    vocabulary = Vocabulary()
    char_lm = CharLM(args.charlm)

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = encode_corpus(train, vocabulary)

    char_lm.vocabulary = vocabulary
    model = PYP(theta, d, char_lm)

    logging.info('Training model')
    run_sampler(model, training_corpus)

    logging.info('Computing test corpus perplexity')
    with open(args.test) as test:
        test_corpus = encode_corpus(test, vocabulary)

    print_ppl(model, test_corpus)

if __name__ == '__main__':
    baseline_main()
