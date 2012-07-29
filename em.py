from itertools import izip
import pickle
import numpy as np
import scipy.stats as ss
from corpus import Corpus

 
def normalize(model):
    """ Make a probability distribution from counts """
    norm = np.logaddexp.reduce(model)
    return model - norm

def smooth(model): # add-1 smoothing
    return np.array([np.logaddexp(v, 0) for v in model])

class Model:
    def uniform_init(self, n_morphemes, n_lemmas):
        # Morphemes
        self.model_morphemes = np.zeros(n_morphemes) - np.log(n_morphemes)
        self.model_morphemes += np.random.randn(n_morphemes)
        self.model_morphemes = normalize(self.model_morphemes)
        # Lemmas
        self.model_lemmas = np.zeros(n_lemmas) - np.log(n_lemmas)
        self.model_lemmas += np.random.randn(n_lemmas)
        self.model_lemmas = normalize(self.model_lemmas)
        # Length
        self.model_length = 1

    def prob(self, analysis):
        lp = ss.poisson.logpmf(len(analysis), self.model_length)
        lp += self.model_lemmas[analysis.lemma]
        for morpheme in analysis.morphemes:
            lp += self.model_morphemes[morpheme]
        return lp
    
    def run_em(self, n_iterations, corpus):
        def count(analysis, lp):
            prob = np.exp(lp)
            count_length[0] += len(analysis) * prob
            count_length[1] += prob
            count_lemmas[analysis.lemma] = np.logaddexp(count_lemmas[analysis.lemma], lp)
            for morpheme in analysis.morphemes:
                count_morphemes[morpheme] = np.logaddexp(count_morphemes[morpheme], lp)
                    
        for it in range(n_iterations):
            print('Iteration {0}:'.format(it+1))
            count_morphemes = np.zeros(corpus.n_morphemes) - np.inf
            count_lemmas = np.zeros(corpus.n_lemmas) - np.inf
            count_length = [0.0, 0.0]
            
            loglik = 0
            # Expectation
            for analyses in corpus:
                probs = map(self.prob, analyses)
                norm = np.logaddexp.reduce(probs)
                loglik += norm
                for analysis, lp in izip(analyses, probs):
                    count(analysis, lp-norm)
            
            print(' Log likelihood: {0}\n'.format(loglik))
            # Maximization
            self.model_morphemes = normalize(count_morphemes)
            self.model_lemmas = normalize(smooth(count_lemmas))
            self.model_length = count_length[0] / count_length[1]

            print(self.model_morphemes)
            print('Morph length: {0}'.format(self.model_length))
            
        return loglik

if __name__ == '__main__':
    Niter = 10
    fst = 'malmorph.fst'
    corpus = '/cab1/vchahune/ldmt/global_voices/baseline-2/lm/leipzig.mg.tok.lc'
    corpus = Corpus(fst, corpus)
    with open('vocab.pickle', 'w') as fp:
        pickle.dump(corpus.vocabulary, fp)
    del corpus.vocabulary
    model = Model()
    model.uniform_init(corpus.n_morphemes, corpus.n_lemmas)
    model.run_em(Niter, corpus)
    with open('model.pickle', 'w') as fp:
        pickle.dump(model, fp)
