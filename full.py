#!/usr/bin/env python
import argparse
import kenlm
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
    char_lm = kenlm.LanguageModel(args.charlm)
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
