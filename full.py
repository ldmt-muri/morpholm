#!/usr/bin/env python
import argparse
from model import CharLM
import model1, model2, baseline
from corpus import FSM, analyze_corpus, init_vocabularies
import em, tune, ppl

MODELS = {1: model1.Model1,
          2: model2.Model2}

def main():
    parser = argparse.ArgumentParser(description='Run FST + prob model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--dev', help='development corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer')
    parser.add_argument('--charlm', help='character language model (KenLM format)',
            required=True)
    parser.add_argument('--model', help='type of model to train', default=1, type=int)
    parser.add_argument('--sampler', help='use Gibbs sampling instead of EM', action='store_true')
    args = parser.parse_args()

    fsm = FSM(args.fst)
    char_lm = CharLM(args.charlm)

    """
    import cPickle
    with open('models/model1.pickle') as m1:
        char_lm = cPickle.load(m1)
    with open('models/vocab.pickle') as voc:
        char_lm.vocabularies = cPickle.load(voc)
        char_lm.vocabularies['stem'].frozen = True
        char_lm.vocabularies['morpheme'].frozen = True
    char_lm.fsm = fsm
    char_lm.char_lm = CharLM(args.charlm)
    """

    if args.fst:
        print('Training Model: {0}'.format(MODELS[args.model]))
        # Train
        with open(args.train) as train:
            vocabularies = init_vocabularies()
            word_analyses = {}
            training_corpus = analyze_corpus(train, fsm, vocabularies, word_analyses)
            training_corpus.analyses = word_analyses
            model = em.train_model(MODELS[args.model], training_corpus, vocabularies, args.sampler)
        # Tune
        with open(args.dev) as dev:
            tune.tune_model(vocabularies, model, char_lm, fsm, dev)

        # Evaluate
        with open(args.test) as test:   
            ppl.print_ppl(vocabularies, model, char_lm, fsm, test)
    else:
        print('Training baseline')
        # Train
        with open(args.train) as train:
            model, vocabulary = baseline.train_model(train)

        # Tune
        with open(args.dev) as dev:
            baseline.tune_model(vocabulary, model, char_lm, dev)

        # Evaluate
        with open(args.test) as test:   
            baseline.print_ppl(vocabulary, model, char_lm, test)

if __name__ == '__main__':
    main()
