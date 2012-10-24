import logging
import argparse
import math
import cPickle
from itertools import izip
from corpus import hmm_trigrams
from prob import CharLM
from pyp import PYPLM
from model import BigramPattern
from seq_model import SeqMorphoProcess, SimpleBigram

# Stem PYP
alpha, p = 1.0, 0.8
# Pattern PYP
nu, q = 1.0, 0.8
# Morpheme prior
beta = 1.0

def run_sampler(model, corpus, n_iter):
    assignments = [[None]*len(sentence)+[0] for sentence in corpus.sentences]
    for it in xrange(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence, sentence_assignments in izip(corpus.sentences, assignments):
            trigrams = hmm_trigrams(sentence)
            for k, seq in enumerate(trigrams):
                if it > 0 and len(model.analyses[seq[1]]) == 1: continue
                ass_seq = tuple(sentence_assignments[k+i] for i in (-1, 0, 1))
                if it > 0: model.decrement(*izip(seq, ass_seq))
                sentence_assignments[k] = model.increment(*izip(seq, ass_seq))
        ll = model.log_likelihood()
        ppl = math.exp(-ll / len(corpus))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        logging.info('Model: %s', model)

def main():
    parser = argparse.ArgumentParser(description='Train MorphoHMM')
    parser.add_argument('-i', '--iterations', help='number of iterations', required=True, type=int)
    parser.add_argument('--train', help='compiled training corpus', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)', required=True)
    parser.add_argument('--pattern-lm', help='use LM for pattern', action='store_true')
    parser.add_argument('--output', '-o', help='model output path')
    args = parser.parse_args()

    logging.info('Reading training corpus')

    with open(args.train) as f:
        data = cPickle.load(f)
    vocabularies = data['vocabularies']
    word_analyses = data['analyses']
    training_corpus = data['corpus']

    char_lm = CharLM(args.charlm)
    char_lm.vocabulary = vocabularies['stem']

    stem_model = PYPLM(alpha, p, 2, char_lm)
    if args.pattern_lm:
        logging.info('pattern ~ PYPLM')
        n_morphemes = len(vocabularies['morpheme'])
        pattern_model = PYPLM(nu, q, 2, BigramPattern(n_morphemes, beta, vocabularies['pattern']))
    else:
        logging.info('pattern ~ Simple bigram')
        pattern_model = SimpleBigram(nu, len(vocabularies['pattern']))
    model = SeqMorphoProcess(stem_model, pattern_model, word_analyses)

    logging.info('Training model')
    run_sampler(model, training_corpus, args.iterations)

    if args.output:
        logging.info('Saving model')
        data = {'vocabularies': vocabularies, 'analyses': word_analyses, 'model': model}
        with open(args.output, 'w') as output:
            cPickle.dump(data, output, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
