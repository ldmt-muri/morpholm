import argparse
import logging
import cPickle
import sys
import math
from collections import defaultdict

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Generate new translations')
    parser.add_argument('--backend', help='analyzer backend (foma/xfst/pymorphy/xerox)',
            default='foma')
    parser.add_argument('--analyzer', help='analyzer option', required=True)
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)

    from ..analyzers import all_analyzers
    analyzer = all_analyzers[args.backend](args.analyzer)

    def pattern_class(pattern):
        if len(pattern) < 2:
            return -1
        return pattern[1]

    logging.info('#Patterns: %d', len(model.pattern_vocabulary))
    pattern_classes = defaultdict(list)
    for p, pattern in enumerate(model.pattern_vocabulary):
        pattern_p = model.pattern_model.prob(p)
        pattern_classes[pattern_class(pattern)].append((pattern_p, p))
    logging.info('#Pattern classes: %d', len(pattern_classes))
    for cl in pattern_classes:
        pattern_classes[cl] = sorted(pattern_classes[cl], reverse=True)[:10]
    logging.info('After filtering: %d patterns', sum(map(len, pattern_classes.itervalues())))

    S = model.morpheme_vocabulary['__STEM__']
    def synthesize(s, p):
        stem = model.stem_vocabulary[s]
        pattern = model.pattern_vocabulary[p]
        morphemes = tuple(stem if m == S else model.morpheme_vocabulary[m] for m in pattern)
        words = set(map(unicode.lower, analyzer.synthesize_word('+'.join(morphemes))))
        return words

    def generate_alternatives(f, e):
        t_word = model.t_table[model.source_vocabulary[f]]
        analyses = set((analysis.stem, pattern_class(model.pattern_vocabulary[analysis.pattern]))
                for analysis in model.analyses[model.target_vocabulary[e]])
        for stem, cls in analyses:
            stem_p = t_word.base.stem_model.prob(stem)
            for pattern_p, pattern in pattern_classes[cls]:
                for new_e in synthesize(stem, pattern):
                    yield stem_p, pattern_p, new_e # PatternLength=len(pattern) ?

    for line in sys.stdin:
        sys.stdout.write(line)
        nt, f, e, _, _ = line.split(' ||| ') # [X] ||| f ||| e ||| features ||| alignment
        f = f.split()
        e = e.split()
        if len(f) == len(e) == 1:
            f, e = f[0].decode('utf8'), e[0].decode('utf8')
            logging.info('Expading %s ||| %s', f, e)
            for stem_p, pattern_p, new_e in generate_alternatives(f, e):
                print(u'{0} ||| {1} ||| {2} ||| StemProb={3} PatternProb={4} Alternative=1 ||| 0-0'.format(nt, f, new_e, 
                    math.log(stem_p), math.log(pattern_p)).encode('utf8'))

if __name__ == '__main__':
    main()
