import sys
import cPickle
from model1 import Model1
#from model2 import Model2
#from model22 import Model22

Niter = 10
def train_model(Model, corpus, vocabulary):
    n_morphemes = len(vocabulary['morpheme'])
    n_stems = len(vocabulary['stem'])
    model = Model()
    model.uniform_init(n_morphemes, n_stems)
    model.run_em(Niter, corpus)
    #model.init_sampler(n_morphemes, n_stems, 1, 1, 1, 1)
    #model.run_sampler(Niter, corpus)
    return model

def main(vocab_file, corpus_file, out):
    with open(vocab_file) as fp:
        vocabulary = cPickle.load(fp)
    with open(corpus_file) as fp:
        corpus = cPickle.load(fp)
    model = train_model(Model1, corpus, vocabulary)
    with open(out, 'w') as fp:
        cPickle.dump(model, fp, protocol=2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: {argv[0]} vocab corpus out\n'.format(argv=sys.argv))
        sys.exit(1)
    main(*sys.argv[1:])
