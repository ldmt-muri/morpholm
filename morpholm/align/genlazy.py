import argparse
import logging
import cPickle
import os
import math
from collections import defaultdict
from itertools import groupby

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Generate new translations')
    parser.add_argument('--corpus', help='large analyzed corpus', required=True)
    parser.add_argument('--model', help='trained model', required=True)
    parser.add_argument('in_grammar', help='input grammar path')
    parser.add_argument('out_grammar', help='output grammar path')

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)


    logging.info('Reading large analyzed corpus')
    with open(args.corpus) as corpus_file:
        corpus = cPickle.load(corpus_file)

    logging.info('Indexing analyzed words by stem')
    stem_patterns = defaultdict(dict)
    for i, word in enumerate(corpus.vocabulary):
        w = model.target_vocabulary[word]
        for analysis in corpus.analyses[i]:
            # Recode stem
            s = model.stem_vocabulary[corpus.stem_vocabulary[analysis.stem]]
            # Recode morphemes
            morphemes = (corpus.morpheme_vocabulary[m] for m in 
                    corpus.pattern_vocabulary[analysis.pattern])
            pattern = tuple(model.morpheme_vocabulary[m] for m in morphemes)
            # ... and pattern
            p = model.pattern_vocabulary[pattern]
            stem_patterns[s][p] = w

    logging.info('Found %d stems', len(stem_patterns))

    def dec_pattern(pattern):
        return '+'.join(model.morpheme_vocabulary[m] for m in model.pattern_vocabulary[pattern])

    def _generate_alternatives(f, e):
        t_word = model.t_table[model.source_vocabulary[f]]
        analyses = set(analysis.stem for analysis in model.analyses[model.target_vocabulary[e]])
        for stem in analyses:
            stem_p = t_word.base.stem_model.prob(stem)
            for pattern, new_e in stem_patterns[stem].iteritems():
                pattern_p = model.pattern_model.prob(pattern)
                if pattern_p == 0:
                    #logging.error('OOV pattern: %s', dec_pattern(pattern))
                    continue
                # PatternLength=len(pattern) ?
                yield model.target_vocabulary[new_e], stem, stem_p, pattern_p

    def generate_alternatives(f, e):
        # Aggregate rules with the same target word and stem (sum over patterns)
        groups = groupby(sorted(_generate_alternatives(f, e)), lambda t: t[:2])
        for (word, stem), group in groups:
            group = list(group)
            stem_p = max(stem_p for _, _, stem_p, _ in group)
            pattern_p = sum(pattern_p for _, _, _, pattern_p in group)
            yield stem_p, pattern_p, word

    def process_grammar(ing, outg):
        history = set()
        for line in ing:
            outg.write(line)
            nt, f, e, _, _ = line.split(' ||| ') # [X] ||| f ||| e ||| features ||| alignment
            f = f.split()
            e = e.split()
            if len(f) == len(e) == 1:
                f, e = f[0].decode('utf8'), e[0].decode('utf8')
                #logging.info('Expanding %s ||| %s', f, e)
                for stem_p, pattern_p, new_e in generate_alternatives(f, e):
                    if (f, new_e) in history: continue # Skip already generated alternatives
                    outg.write((
                        u'{0} ||| {1} ||| {2} ||| '
                        'StemProb={3} PatternProb={4} Alternative=1 ||| 0-0\n'
                            ).format(nt, f, new_e, 
                        math.log(stem_p), math.log(pattern_p)).encode('utf8'))
                    history.add((f, new_e))

    if not os.path.exists(args.out_grammar):
        os.mkdir(args.out_grammar)
    for grammar in os.listdir(args.in_grammar):
        with open(os.path.join(args.in_grammar, grammar)) as ing,\
                open(os.path.join(args.out_grammar, grammar), 'w') as outg:
            process_grammar(ing, outg)

if __name__ == '__main__':
    main()
