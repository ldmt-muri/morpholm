import argparse
import logging
import cPickle
from itertools import izip
from vpyp.corpus import Corpus
from vpyp.prob import Uniform
from vpyp.charlm import CharLM
from vpyp.prior import PYPPrior, GammaPrior
from vpyp.pyp import PYP
from vpyp.align.train import run_sampler
from ..models import BigramPattern
from model import MorphoAlignmentModel

NULL = '__NULL__'

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train MP alignment model')
    parser.add_argument('--source', '-f', help='source corpus', required=True)
    parser.add_argument('--target', '-e', help='target analyzed corpus', required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--charlm', help='character language model', required=True)
    parser.add_argument('--model', help='model type', type=int, default=0)
    parser.add_argument('--output', help='model output path')

    args = parser.parse_args()

    logging.info('Reading parallel training data')
    with open(args.source) as source:
        source_corpus = Corpus(source)
    with open(args.target) as target:
        target_corpus = cPickle.load(target)
    assert len(source_corpus.segments) == len(target_corpus.segments)
    N = source_corpus.vocabulary[NULL]
    training_corpus = [([N]+f, e) for f, e in izip(source_corpus, target_corpus)]
    logging.info('Read %d sentences', len(training_corpus))

    logging.info('Preloading stem character language model')
    char_lm = CharLM(args.charlm, target_corpus.stem_vocabulary)
    stem_base = PYP(char_lm, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0))

    pvoc = target_corpus.pattern_vocabulary
    n_morphemes = len(target_corpus.morpheme_vocabulary)
    if args.model == 0:
        pattern_base = Uniform(len(pvoc))
    elif args.model == 2:
        alpha_prior = GammaPrior(1.0, 1.0, 1.0)
        pattern_base = BigramPattern(n_morphemes, alpha_prior, pvoc)
    pattern_model = PYP(pattern_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0))

    n_source = len(source_corpus.vocabulary)
    model = MorphoAlignmentModel(n_source, stem_base, pattern_model, target_corpus.analyses)

    logging.info('Training alignment model')
    alignments = run_sampler(model, training_corpus, args.iter)

    if args.output:
        with open(args.output, 'w') as f:
            model.source_vocabulary = source_corpus.vocabulary
            model.target_vocabulary = target_corpus.vocabulary
            model.morpheme_vocabulary = target_corpus.morpheme_vocabulary
            model.pattern_vocabulary = target_corpus.pattern_vocabulary
            model.analyses = target_corpus.analyses
            cPickle.dump(model, f, protocol=-1)

    for a, (f, e) in izip(alignments, training_corpus):
        f_sentence = ' '.join(source_corpus.vocabulary[w] for w in f[1:])
        e_sentence = ' '.join(target_corpus.vocabulary[w] for w in e)
        al = ' '.join('{0}-{1}'.format(j-1, i) for i, j in enumerate(a) if j > 0)
        print(u'{0} ||| {1} ||| {2}'.format(f_sentence,  e_sentence, al).encode('utf8'))

if __name__ == '__main__':
    main()
