import calendar
import re

import nltk

from .stopwords import SIMPLE_STOPWORDS
from gensim.summarization import summarizer
from gensim.summarization.textcleaner import get_sentences

TOKEN_PATTERN = re.compile('^(20|19)\d{2}|(?=[A-Z])[\w\-\d]+$', re.IGNORECASE)


def replace_punctuation(text):
    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<|≥|≤|~|`', '', text)
    t = re.sub('/', ' ', t)
    t = t.replace("'", '')
    return t


def clean(text):
    t = text.lower()
    t = replace_punctuation(t)
    return t


def tokenize(text):
    words = nltk.word_tokenize(text)
    return [word for word in words
            if len(word) > 1
            and not word in SIMPLE_STOPWORDS
            and TOKEN_PATTERN.match(word)
            # and (word.isalpha() or (word.isnumeric() and len(word) ==4))
            # and not (word.isnumeric() and len(word) is not 4)
            # and (not word.isnumeric() or word.isalpha())
            ]


def preprocess(text):
    t = clean(text)
    tokens = tokenize(t)
    return tokens


months = list(calendar.month_abbr)
seasons = {'Winter': 'Dec', 'Autumn': 'Sep', 'Spring': 'April', 'Fall': 'Sep', 'Summer': 'June'}


def extract_publish_date(dates):
    year_month = dates.str.extract('(?P<year>\d{4}) ?(?P<month>\w+)?')
    has_month = ~year_month.month.isnull()
    year_month.loc[has_month, 'month'] = year_month.loc[has_month, 'month'] \
        .replace(seasons).apply(lambda m: str(months.index(m[:3])).zfill(2))
    year_month.month = year_month.month.fillna('01').apply(str)
    return year_month.year + '-' + year_month.month


def shorten(text, length=200):
    if text:
        _len = min(len(text), length)
        shortened_text = text[:_len]
        ellipses = '...' if len(shortened_text) < len(text) else ''
        return f'{shortened_text}{ellipses}'
    return ''


def num_sentences(text):
    if not text:
        return 0
    return len(list(get_sentences(text)))


def summarize(text, word_count=120):
    if num_sentences(text) > 1:
        try:
            word_count_summary = summarizer.summarize(text, word_count=word_count)
        except ValueError:
            return text
        if word_count_summary:
            return word_count_summary
        else:
            ratio_summary = summarizer.summarize(text, ratio=0.2)
            if ratio_summary:
                return ratio_summary
    return text