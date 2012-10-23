# Morphology-aware LMs

## Requirements

- Python 2.7
- numpy
- [foma module](https://github.com/vchahun/foma).
- [KenLM module](https://github.com/vchahun/kenlm#python-module)

## Training a model

- Analyze the training corpus:

        python morpholm/analyze.py --fst analyzer.fst --output train-corpus.pickle < train-corpus.txt

- Train the model:

        python morpholm/train.py -i 10 --train train-corpus.pickle --charlm charlm.klm --output model.pickle

- Evaluate the model by computing the test set perplexity:

        python morpholm/eval.py --fst analyzer.fst --model model.pickle --ppl < test-corpus.txt

## Using a model for decoding

- Analyze data:

    	python morpholm/eval.py --fst analyzer.fst --model model.pickle --decode < test-corpus.txt > analyzed-corpus.txt
