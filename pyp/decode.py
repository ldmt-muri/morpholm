import logging
import argparse
import cPickle
from corpus import FSM, analyze_corpus
from pyp import PYP

def decode_corpus(model, corpus, vocabularies):
    for word in corpus:
        p, dec = model.decode(word)
        print(dec.decode(vocabularies).encode('utf8'))

def main():
    parser = argparse.ArgumentParser(description='Train PYP-MorphoLM')
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--fst', help='compiled morphoanalyzer', required=True)
    parser.add_argument('--backend', help='analyzer type (foma/xfst/pymorphy)', default='foma')
    parser.add_argument('--notrust', help='do not trust analyzer and also keep non-analyzed word', action='store_true')
    args = parser.parse_args()

    with open(args.model) as f:
        data = cPickle.load(f)
    
    vocabularies = data['vocabularies']
    word_analyses = data['analyses']
    model = data['model']

    mp = model.base if isinstance(model, PYP) else model
    mp.stem_model.base.vocabulary = vocabularies['stem']

    fsm = FSM(args.fst, args.backend, not args.notrust)

    logging.info('Reading test corpus')
    with open(args.test) as test:
        test_corpus = analyze_corpus(test, fsm, vocabularies, word_analyses)

    decode_corpus(model, test_corpus, vocabularies)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
