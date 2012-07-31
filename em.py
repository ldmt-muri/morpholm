import cPickle
import kenlm
from model1 import Model1

if __name__ == '__main__':
    Niter = 10
    with open('corpus.pickle') as fp:
        corpus = cPickle.load(fp)
    with open('vocab.pickle') as fp:
        vocabulary = cPickle.load(fp)
    with open('info.txt') as fp:
        n_morphemes = int(next(fp))
        n_stems = int(next(fp))
    model = Model1()
    model.vocabulary = vocabulary
    model.char_lm = kenlm.LanguageModel('charlm.klm')
    model.uniform_init(n_morphemes, n_stems)
    model.run_em(Niter, corpus)
    del model.char_lm
    with open('model1.pickle', 'w') as fp:
        cPickle.dump(model, fp, protocol=2)
