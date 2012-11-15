import random
import time
import requests
import json
import lxml.html
from itertools import izip
import logging
from analyzer import Analyzer, word_re

supported_languages = {'cs': 'Czech', 'en': 'English', 'fr': 'French', 
'de': 'German', 'el': 'Greek', 'hu': 'Hungarian', 'it': 'Italian', 
'pl': 'Polish', 'ru': 'Russian', 'tr': 'Turkish'}

FORM_URL = 'http://open.xerox.com/Services/fst-nlp-tools/Consume/176?serviceId=57'

def get_blocks(stream, block_size):
    while True:
        block = []
        for i, line in enumerate(stream):
            block.append(line)
            if i >= block_size - 1: break
        if block:
            yield block
        else:
            break

WAIT = 1 # wait on average 1s between each request

def fix_analysis(analysis):
    return '+'+'+'.join(m.title() for m in analysis.split('+') if m)

class XeroxAnalyzer(Analyzer):
    def __init__(self, language):
        if not language in supported_languages:
            raise NotImplemented('No analyzer available for {0}'.format(language))
        self.language = supported_languages[language]

    def analyze_corpus(self, corpus, add_null=False):
        self.analyses = {}
        # Get analyses for all words in the vocabulary not already analyzed
        exclude = ({} if not hasattr(corpus, 'analyses') 
                else set(corpus.vocabulary[w] for w in corpus.analyses))
        analyzable = sorted(word for word in corpus.vocabulary 
                if word_re.match(word) and word not in exclude)
        logging.info('Retrieving analyses for %d words', len(analyzable))
        for block in get_blocks(iter(analyzable), 1000):
            words = ' '.join(block)
            logging.info('New request (%d words)', len(block))
            response = requests.post(FORM_URL,
                    data={'form_action_url': '/Services/fst-nlp-tools/Consume/176?id=57',
                          'form_save_test_url': '/Services/fst-nlp-tools/SaveAsTest',
                          'serviceId': '57',
                          'formId': '176',
                          'operationId': '316',
                          'inputtext_TEXT': words.encode('utf8'),
                          'inputtext_type': 'TEXT',
                          'language': self.language,
                          'ajax': 'true'})
            logging.info('Response status code: %d', response.status_code)

            result = json.loads(response.content)
            doc = lxml.html.fromstring(result['Result'])
            ul_it = iter(doc.cssselect('body > ul')[0])
            for li, ul in izip(ul_it, ul_it):
                word = li[0].text
                analyses = [ali[0].text.lower()+fix_analysis(ali[1].text) for ali in ul]
                self.analyses[word] = analyses

            wait = random.random() * 2 * WAIT
            logging.info('Sleep during %.3f s', wait)
            time.sleep(wait)

        logging.info('Retrieved %d analyses', len(self.analyses))
        if len(self.analyses) != len(analyzable):
            logging.warn('|analyses| != |analyzable|: some analyses might be missing')
        super(XeroxAnalyzer, self).analyze_corpus(corpus, add_null)

    def analyze_word(self, word):
        return self.analyses.get(word, (word, ))

    def __repr__(self):
        return 'XeroxAnalyzer(lang={self.language})'.format(self=self)
