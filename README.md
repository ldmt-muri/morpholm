# Morphology-aware LMs

## Requirements

- Python 2.7 is required
- Install numpy
- Install the [foma module](https://github.com/vchahun/foma).
- Install the [KenLM module](https://github.com/vchahun/kenlm#python-module)

## Combined training + evaluation

To run Unigram Model 2 (~20m & 1G on the Global Voices data):

    python full.py --train training-corpus.txt --dev dev-corpus.txt --test test-corpus.txt\
        --fst analyzer.fst --charlm charlm.klm --model 2

## Training a model step by step

- Analyze the training corpus:

        python analyze_corpus.py analyzer.fst vocab.pickle corpus.pickle < training-corpus.txt

- Train a model:

        python em.py vocab.pickle corpus.pickle model1.pickle

- Tune the CharLM/MorphLM interpolation parameter:

        python tune.py vocab.pickle model1.pickle charlm.klm analyzer.fst < dev-corpus.txt

- Evaluate by computing the test set perplexity:

        python ppl.py vocab.pickle model1.pickle charlm.klm analyzer.fst < test-corpus.txt
