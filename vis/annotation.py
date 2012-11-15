from itertools import groupby, izip
from functools import wraps
import math
import logging
import sqlite3
from flask import Flask, render_template, request, Response, g
import config

def check_auth(username, password):
    return config.users.get(username.lower(), None) == password.lower()

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response('Invalid username/password (should be last name/first name)', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

app = Flask(__name__)

@app.before_request
def before_request():
    g.connection = sqlite3.connect(config.annotation_database)
    g.cursor = g.connection.cursor()

@app.teardown_request
def teardown_request(exception):
    g.cursor.close()
    g.connection.commit()
    g.connection.close()

def all_sentences(username):
    for i, sentence in enumerate(config.sentences):
        g.cursor.execute('select count(*) from annotations '
                'where sentence_id = ? and username = ?', (i, username))
        done, = g.cursor.fetchone()
        yield done/float(len(sentence.split())), sentence

@app.route('/')
@requires_auth
def index():
    username = request.authorization.username.lower()
    g.cursor.execute('select count(*) from annotations where username = ?', (username,))
    total_count, = g.cursor.fetchone()
    max_count = sum(len(sentence.split()) for sentence in config.sentences)
    return render_template('index_annotation.html', sentences=list(all_sentences(username)),
            name=config.users[username]+' '+username,
            total_count=total_count, max_count=max_count)

@app.template_filter('morph_norm')
def morph_norm(morpheme):
        return morpheme.replace('^', '-').replace('.', '-')

def pattern_score(pattern):
    return math.log(config.morpho_process.pattern_model.prob(pattern))

@app.template_filter('analysis_str')
def analysis_str(analysis):
    def morph():
        for m in config.model.pattern_vocabulary[analysis.pattern]:
            if m == config.S:
                yield config.model.stem_vocabulary[analysis.stem]
            else:
                yield config.model.morpheme_vocabulary[m]
    return '+'.join(morph())

def decode_pattern(analysis):
    return (pattern_score(analysis.pattern), analysis,
            [config.model.morpheme_vocabulary[m] for m in 
                config.model.pattern_vocabulary[analysis.pattern]])

def stem_score(anas):
    return math.log(sum(map(config.morpho_process.analysis_prob, anas)))

astem = lambda a: a.stem

def get_patterns(word):
    analyses = config.model.analyses[word]
    grouped_analyses = ((stem, list(anas_group))
            for stem, anas_group in groupby(sorted(analyses, key=astem), key=astem))
    return sorted(((stem_score(anas),
                    config.model.stem_vocabulary[stem],
                    sorted(map(decode_pattern, anas), reverse=True))
                   for stem, anas in grouped_analyses),
                   reverse=True)

def parse_analysis(analysis):
    if analysis is None: return None
    stem, pattern = config.parse_analysis(analysis)
    s = config.model.stem_vocabulary[stem]
    p = tuple(config.model.morpheme_vocabulary[m] for m in pattern)
    return config.Analysis(s, config.model.pattern_vocabulary[p])

def is_other(annotation, word):
    return (annotation is not None) and (annotation not in config.model.analyses[word])

@app.route('/sentence')
@requires_auth
def sentence():
    username = request.authorization.username.lower()
    sentence_id = int(request.args.get('id'))
    sentence = config.sentences[sentence_id]
    analyzed_sentence = config.analyze_sentence(sentence)
    g.cursor.execute('select word_id, analysis from annotations '
            'where sentence_id = ? and username = ?', (sentence_id, username))
    annotations = dict(g.cursor.fetchall())
    annotations = [parse_analysis(annotations.get(word_id, None))
        for word_id in range(len(sentence.split()))]
    words = [(config.model.vocabulary[word], get_patterns(word), 
                annotation, is_other(annotation, word))
             for annotation, word in izip(annotations, analyzed_sentence)]
    return render_template('annotation.html', sentence_id=sentence_id,
            words=words, palette=config.morpheme_colors, 
            prev=(None if sentence_id == 0 else sentence_id-1),
            nxt=(None if sentence_id == len(config.sentences) else sentence_id+1))

@app.route('/annotate', methods=('POST',))
@requires_auth
def annotate():
    username = request.authorization.username.lower()
    sentence_id = int(request.form['sentence_id'])
    word_id = int(request.form['word_id'])
    analysis = request.form['analysis'].replace(' ', '')
    logging.info("%s/%d/%d:%s", username, sentence_id, word_id, analysis)
    try:
        parse_analysis(analysis)
    except Exception as e:
        logging.error("-> invalid analysis: %s (%s)", analysis, e)
        return 'INVALID'
    try:
        g.cursor.execute('insert or replace into annotations'
                '(sentence_id, word_id, username, analysis) '
                'values (?, ?, ?, ?)', (sentence_id, word_id, username, analysis))
    except Exception as e:
        logging.error("-> update error (%s)", e)
        return 'ERROR'
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=34957)
