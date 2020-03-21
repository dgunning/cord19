import re
import nltk
from nltk.corpus import stopwords
import calendar
import numpy as np

english_stopwords = list(set(stopwords.words('english')))


def replace_punctuation(text):
    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)
    t = re.sub('/', ' ', t)
    t = t.replace("'", '')
    return t


def clean(text):
    t = text.lower()
    t = replace_punctuation(t)
    return t


def tokenize(text):
    words = nltk.word_tokenize(text)
    return list(set([word for word in words
                     if len(word) > 1
                     and not word in english_stopwords
                     and not (word.isnumeric() and len(word) is not 4)
                     and (not word.isnumeric() or word.isalpha())])
                )


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
    _len = min(len(text), length)
    if text:
        return f'{text[:_len]}...'
    return ''
