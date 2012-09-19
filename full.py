#!/usr/bin/env python
import argparse
from model import CharLM
import model1, model2, baseline
from corpus import FSM, analyze_corpus
import em, tune, ppl

MODELS = {1: model1.Model1,
          2: model2.Model2}

def main():
    parser = argparse.ArgumentParser(description='Run FST + prob model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--dev', help='development corpus', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--charlm', help='character language model (KenLM format)',
            required=True)
    parser.add_argument('--model', help='type of model to train')
    args = parser.parse_args()

    fsm = FSM(args.fst)
    char_lm = CharLM(args.charlm)

    """
    import cPickle
    with open('models/model1.pickle') as m1:
        char_lm = cPickle.load(m1)
    with open('models/vocab.pickle') as voc:
        char_lm.vocabulary = cPickle.load(voc)
        char_lm.vocabulary['stem'].frozen = True
        char_lm.vocabulary['morpheme'].frozen = True
    char_lm.fsm = fsm
    char_lm.char_lm = CharLM(args.charlm)
    """

    if args.model:
        print('Training Model: {0}'.format(MODELS[int(args.model)]))
        # Train
        with open(args.train) as train:
            training_corpus, vocabulary = analyze_corpus(fsm, train)
            model = em.train_model(MODELS[int(args.model)], training_corpus, vocabulary)
        # Tune
        with open(args.dev) as dev:
            tune.tune_model(vocabulary, model, char_lm, fsm, dev)

        # Evaluate
        with open(args.test) as test:   
            ppl.print_ppl(vocabulary, model, char_lm, fsm, test)
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
