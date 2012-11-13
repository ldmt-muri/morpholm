import argparse
import logging
import cPickle

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Evaluate n-gram model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--model', help='trained model')
    args = parser.parse_args()

    with open(args.train) as train:
        training_corpus = cPickle.load(train)

    if not args.model:
        for s, stem in enumerate(training_corpus.stem_vocabulary):
            if s < 2: continue # skip START, STOP
            print(stem.encode('utf8'))
        return

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)
    mp = model.backoff

    for w, word in enumerate(training_corpus.vocabulary):
        if w < 2: continue # skip START, STOP
        _, best_stem = max((mp.analysis_prob(analysis), analysis.stem)
                for analysis in training_corpus.analyses[w])
        print(training_corpus.stem_vocabulary[best_stem].encode('utf8'))

if __name__ == '__main__':
    main()
