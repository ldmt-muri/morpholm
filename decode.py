import sys
import pickle
from em import Model
from corpus import Decoder

def load_model(fname):
    model = Model()
    with open(fname) as fp:
        data = pickle.load(fp)
    model.model_morphemes = data['morphemes']
    model.model_lemmas = data['lemmas']
    model.model_length = data['length']
    return model

if __name__ == '__main__':
    #model = load_model('model2.pickle')
    with open('model.pickle') as fp:
        model = pickle.load(fp)
    with open('vocab.pickle') as fp:
        vocabulary = pickle.load(fp)
    decoder = Decoder('malmorph.fst', model, vocabulary)

    for line in sys.stdin:
        words = line.decode('utf8').split()
        print(' '.join(decoder.decode(word).encode('utf8') for word in words))
