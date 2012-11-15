from itertools import groupby
import math
from flask import Flask, render_template, request
import config

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', sentences=config.sentences)

@app.template_filter('morph_norm')
def morph_norm(morpheme):
        return morpheme.replace('^', '-').replace('.', '-')

astem = lambda a: a.stem

def decode_pattern(analysis):
    a_prob = math.log(config.morpho_process.analysis_prob(analysis))
    s_prob = math.log(config.morpho_process.stem_model.prob(analysis.stem))
    p_prob = math.log(config.morpho_process.pattern_model.prob(analysis.pattern))
    return (s_prob, p_prob, a_prob,
            [config.model.morpheme_vocabulary[m] for m in 
                config.model.pattern_vocabulary[analysis.pattern]])

def word_score(word):
    score = math.log(config.model.prob((), word))
    return score

def is_known_stem(stem):
    return stem in config.morpho_process.stem_model.tables

def is_known_word(word):
    return word in config.model[()].tables

def stem_score(anas):
    return math.log(sum(map(config.morpho_process.analysis_prob, anas)))

def get_patterns(word):
    analyses = config.model.analyses[word]
    return sorted([(stem_score(anas),
                    config.model.stem_vocabulary[stem], is_known_stem(stem),
                    sorted(map(decode_pattern, anas), reverse=True))
                   for stem, anas in ((stem, list(anas_group))
                       for stem, anas_group in groupby(sorted(analyses, key=astem), key=astem))],
                   reverse=True)

@app.route('/sentence')
def sentence():
    s = request.args.get('q')
    words = [(config.model.vocabulary[word], word_score(word), 
        is_known_word(word), get_patterns(word))
             for word in config.analyze_sentence(s)]
    return render_template('sentence.html', words=words, palette=config.morpheme_colors)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=34959)
