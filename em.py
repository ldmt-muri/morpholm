import sys
import cPickle
#from model1 import Model1
from model2 import Model2

def main(vocab_file, corpus_file, out):
    Niter = 10
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    n_morphemes = len(vocabulary['morpheme'])
    n_stems = len(vocabulary['stem'])
    with open(corpus_file) as fp:
        corpus = cPickle.load(fp)
    model = Model2()
    model.vocabulary = vocabulary
    #model.char_lm = kenlm.LanguageModel('charlm.klm')
    model.uniform_init(n_morphemes, n_stems)
    model.run_em(Niter, corpus)
    del model.vocabulary
    #del model.char_lm
    with open(out, 'w') as fp:
        cPickle.dump(model, fp, protocol=2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: {argv[0]} vocab corpus out\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
