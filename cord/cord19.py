import gc
import json
from functools import reduce, lru_cache, partial
import time
import ipywidgets as widgets
import nltk
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from requests import HTTPError
import re
from cord.core import parallel, ifnone, add, render_html, show_common
from cord.text import preprocess, extract_publish_date, shorten
from cord.dates import fix_dates, add_date_diff
from cord.nlp import get_lda_model, get_top_topic, get_topic_vector

nltk.download("punkt")
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords

english_stopwords = list(set(stopwords.words('english')))
import pickle
from pathlib import Path, PurePath

CORD_CHALLENGE_PATH = 'CORD-19-research-challenge'
SARS_DATE = '2002-11-01'
SARS_COV_2_DATE = '2019-11-30'
_MINIMUM_SEARCH_SCORE = 2


class Author:

    def __init__(self, first=None, last=None, middle=None):
        self.first = ifnone(first, '')
        self.last = ifnone(last, '')

    def __repr__(self):
        return f'{self.first} {self.last}'


class JPaper:

    def __init__(self, paper):
        self.paper = paper
        self.sha = paper['paper_id']
        self.abstract = '\n'.join([a['text'] for a in paper['abstract']])
        self.title = paper['metadata']['title']
        self.authors = [Author(a.get('first'), a.get('last'), a.get('middle'))
                        for a in paper['metadata']['authors']]
        self.sections = [s['text'] for s in paper['body_text']]

    def get_text(self):
        return ' \n '.join([t[1] for t in self.sections])

    def _repr_html_(self):
        return render_html('JPaper', paper=self)

    def __repr__(self):
        return 'JPaper'


#@lru_cache(maxsize=2048)
def load_json(json_file):
    with open(json_file, 'r') as f:
        contents = json.load(f)
        jpaper = JPaper(contents)
    return jpaper


class JCatalog:

    def __init__(self, papers):
        self.papers = papers
        self.paper_index = {p.sha: p for p in self.papers}
        self.index = pd.Series(self.papers, index=[p.sha for p in papers])

    @classmethod
    def load(cls, json_catalog_path):
        print('Load JSON from', json_catalog_path)
        papers = parallel(load_json, list(json_catalog_path.glob('*.json')))
        return cls(papers)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.papers[item]
        return self.index.loc[item].values[0]

    def __len__(self):
        return len(self.papers)

    def __add__(self, o):
        return JCatalog(self.papers + o.papers)


def get(url, timeout=6):
    try:
        r = requests.get(url, timeout=timeout)
        return r.text
    except ConnectionError:
        print(f'Cannot connect to {url}')
        print(f'Remember to turn Internet ON in the Kaggle notebook settings')
    except HTTPError:
        print('Got http error', r.status, r.text)


_DISPLAY_COLS = ['sha', 'title', 'abstract', 'publish_time', 'authors', 'has_text']
_RESEARCH_PAPERS_SAVE_FILE = 'ResearchPapers.pickle'
_COVID = ['sars-cov-2', '2019-ncov', 'covid-19', 'covid-2019', 'wuhan', 'hubei', 'coronavirus']


# Convert the doi to a url
def doi_url(d):
    if not d:
        return '#'
    return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'


_abstract_terms_ = '(Publisher|Abstract|Summary|BACKGROUND|INTRODUCTION)'


# Some titles are is short and unrelated to viruses
# This regex keeps some short titles if they seem relevant
_relevant_re_ = '.*vir.*|.*sars.*|.*mers.*|.*corona.*|.*ncov.*|.*immun.*|.*nosocomial.*'
_relevant_re_ = _relevant_re_ +  '.*epidem.*|.*emerg.*|.*vacc.*|.*cytokine.*'


def remove_common_terms(abstract):
    return re.sub(_abstract_terms_, '', abstract)


def start(data):
    return data.copy()


def clean_title(data):
    # Set junk titles to NAN
    title_relevant = data.title.fillna('').str.match(_relevant_re_, case=False)
    title_short = data.title.fillna('').apply(len) < 30
    title_junk = title_short & ~title_relevant
    data.loc[title_junk, 'title'] = ''
    return data


def clean_abstract(data):
    # Set unknowns to NAN
    abstract_unknown = data.abstract == 'Unknown'
    data.loc[abstract_unknown, 'abstract'] = np.nan

    # Fill missing abstract with the title
    data.abstract = data.abstract.fillna(data.title)

    # Remove common terms like publisher
    data.abstract = data.abstract.fillna('').apply(remove_common_terms)

    # Remove the abstract if it is too common
    common_abstracts = show_common(data, 'abstract').query('abstract > 2') \
                            .reset_index().query('~(index =="")')['index'].tolist()
    data.loc[data.abstract.isin(common_abstracts),'abstract'] = ''

    return data


def drop_missing(data):
    missing = (data.published.isnull()) & \
              (data.sha.isnull()) & \
              (data.title == '') & \
              (data.abstract == '')
    return data[~missing].reset_index(drop=True)


def fill_nulls(data):
    data.authors = data.authors.fillna('')
    data.doi = data.doi.fillna('')
    return data


def clean_metadata(metadata):
    print('Cleaning metadata')
    return metadata.pipe(start) \
                   .pipe(clean_title) \
                   .pipe(clean_abstract) \
                   .pipe(fix_dates) \
                   .pipe(add_date_diff) \
                   .pipe(drop_missing) \
                   .pipe(fill_nulls)


def get_json_path(data_path, text_path, sha):
    return Path(data_path) / text_path / text_path / f'{sha}.json'


_COVID_KEYWORDS = {'covid-19': 100, '2019-ncov': 100, 'sars-cov-2': 100, 'sars-cov': 20, 'quarantine': 10,
                      'outbreak': 10, 'severe': 5, '2019': 10, 'coronavirus': 25, 'novel': 25, 'new': 10,
                      'china': 5, 'wuhan': 20, 'hubei': 30, 'ace2': 30, 'pneumonia': 10}


class ResearchPapers:

    def __init__(self, metadata, data_dir='data', index_tokens=None):
        self.data_path = Path(data_dir) / CORD_CHALLENGE_PATH

        self.num_results = 10
        if index_tokens is None:
            self.metadata = metadata
            print('Indexing research papers')
            tick = time.time()
            index_tokens = self._create_index_tokens()
            # Add antiviral column
            self.metadata['antivirals'] = index_tokens.apply(lambda t:
                                                             ','.join([token for token in t if token.endswith('vir')]))
            # Does it have any covid term?
            self.metadata['covid_related'] = index_tokens.apply(
                lambda tokens: sum([_COVID_KEYWORDS[token] for token in tokens if token in _COVID_KEYWORDS]) > 50)
            self.bm25 = BM25Okapi(index_tokens.tolist())
            self.index_tokens = index_tokens

            tock = time.time()
            print('Finished Indexing in', round(tock - tick, 0), 'seconds')
        else:
            self.metadata = metadata
            self.bm25 = BM25Okapi(index_tokens.tolist())
            self.index_tokens = index_tokens

    def nlp(self):
        # Topic model
        lda_model, dictionary, corpus = get_lda_model(self.index_tokens)
        print('Assigning LDA topics')
        topic_vector = self.index_tokens.apply(lambda tokens: get_topic_vector(lda_model, dictionary, tokens))
        self.metadata['topic_vector'] = topic_vector
        self.metadata['top_topic'] = topic_vector.apply(np.argmax)
        return self

    def create_document_index(self):
        print('Indexing research papers')
        tick = time.time()
        index_tokens = self._create_index_tokens()
        # Add antiviral column
        self.metadata['antivirals'] = index_tokens.apply(lambda t:
                                                         ','.join([token for token in t if token.endswith('vir')]))
        # Does it have any covid term?
        self.metadata['covid_related'] = index_tokens.apply(lambda t: any([covid_term in t for covid_term in _COVID]))
        self.bm25 = BM25Okapi(index_tokens.tolist())
        tock = time.time()
        print('Finished Indexing in', round(tock - tick, 0), 'seconds')

    def get_json_paths(self):
        return self.metadata.apply(lambda d:
                                   np.nan if not d.has_text else get_json_path(self.data_path, d.full_text_file, d.sha),
                                   axis=1)

    def __getitem__(self, item):
        if isinstance(item, int):
            paper = self.metadata.iloc[item]
        else:
            paper = self.metadata[self.metadata.sha == item]
        return Paper(paper, self.data_path)

    def covid_related(self):
        return self.query('covid_related')

    def not_covid_related(self):
        return self.query('~covid_related')

    def __len__(self):
        return len(self.metadata)

    def _make_copy(self, new_data):
        _index = new_data.index
        new_tokens = self.index_tokens.loc[_index]
        return ResearchPapers(metadata=new_data.copy(),
                              data_dir=self.data_path,
                              index_tokens=new_tokens)

    def query(self, query):
        data = self.metadata.query(query)
        return self._make_copy(data)

    def after(self, date, include_null_dates=False):
        cond = self.metadata.published >= date
        if include_null_dates:
            cond = cond | self.metadata.published.isnull()
        return self._make_copy(self.metadata[cond])

    def before(self, date, include_null_dates=False):
        cond = self.metadata.published < date
        if include_null_dates:
            cond = cond | self.metadata.published.isnull()
        return self._make_copy(self.metadata[cond])

    def since_sars(self, include_null_dates=False):
        return self.after(SARS_DATE, include_null_dates)

    def before_sars(self, include_null_dates=False):
        return self.before(SARS_DATE, include_null_dates)

    def since_sarscov2(self, include_null_dates=False):
        return self.after(SARS_COV_2_DATE, include_null_dates)

    def before_sarscov2(self, include_null_dates=False):
        return self.before(SARS_COV_2_DATE, include_null_dates)

    def with_text(self):
        return self.query('has_text')

    def head(self, n):
        return self._make_copy(self.metadata.head(n))

    def tail(self, n):
        return self._make_copy(self.metadata.tail(n).copy())

    def abstracts(self):
        return pd.Series([self.__getitem__(i).abstract() for i in range(len(self))])

    def titles(self):
        return pd.Series([self.__getitem__(i).title() for i in range(len(self))])

    def _repr_html_(self):
        display_cols = ['title', 'abstract', 'journal', 'source', 'authors',
                        'has_text', 'published', 'when']
        return self.metadata[display_cols]._repr_html_()

    @staticmethod
    def load_metadata(data_path=Path('data') / CORD_CHALLENGE_PATH):
        print('Loading metadata from', data_path)
        metadata_path = PurePath(data_path) / 'metadata.csv'
        dtypes = {'Microsoft Academic Paper ID': 'str', 'pubmed_id': str}
        renames = {'source_x': 'source', 'has_full_text': 'has_text'}
        metadata = pd.read_csv(metadata_path, dtype=dtypes).rename(columns=renames)
        # category_dict = {'license': 'category', 'source_x': 'category',
        #                 'journal': 'category', 'full_text_file': 'category'}
        metadata = clean_metadata(metadata)
        return metadata

    @classmethod
    def from_data_dir(cls, data_dir='data'):
        data_path = Path(data_dir) / 'CORD-19-research-challenge'
        metadata = cls.load_metadata(data_path)
        return cls(metadata, data_dir)

    @staticmethod
    def from_pickle(save_dir='data'):
        save_path = PurePath(save_dir) / _RESEARCH_PAPERS_SAVE_FILE
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    def save(self, save_dir='data'):
        save_path = PurePath(save_dir) / _RESEARCH_PAPERS_SAVE_FILE
        print('Saving to', save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def _create_index_tokens(self):
        abstract_tokens = self.metadata.abstract.apply(preprocess)
        return abstract_tokens

    def search(self, search_string,
               num_results=None,
               covid_related=False,
               start_date=None,
               end_date=None):
        if not self.bm25:
            self.create_document_index()

        n_results = num_results or self.num_results
        search_terms = preprocess(search_string)
        doc_scores = self.bm25.get_scores(search_terms)

        # Get the index from the doc scores
        ind = np.argsort(doc_scores)[::-1]
        results = self.metadata.iloc[ind].copy()
        results['Score'] = doc_scores[ind].round(1)

        # Filter covid related
        if covid_related:
            results = results[results.covid_related]

        # Filter by dates
        if start_date:
            results = results[results.published >= start_date]

        if end_date:
            results = results[results.published < end_date]

        # Only include results over a minimum threshold
        results = results[results.Score > _MINIMUM_SEARCH_SCORE]

        # Show only up to n_results
        results = results.head(n_results)

        # Create the final results
        results = results.reset_index(drop=True)

        # Return Search Results
        return SearchResults(results, self.data_path)

    def _search_papers(self, SearchTerms: str):
        search_results = self.search(SearchTerms)
        if len(search_results) > 0:
            display(search_results)
        return search_results

    def searchbar(self, search_terms='cruise ship', num_results=10):
        self.num_results = num_results
        return widgets.interactive(self._search_papers, SearchTerms=search_terms)


# Convert the doi to a url
def doi_url(d):
    if not d:
        return ''
    return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'


class Paper:
    '''
    A single research paper
    '''

    def __init__(self, item, data_path):
        self.sha = item.sha
        self.paper = item.T
        self.paper.columns = ['Value']
        self.data_path = data_path

    def doi(self):
        return self.paper.loc['doi'].values[0]

    def html(self):
        '''
        Load the paper from doi.org and display as HTML. Requires internet to be ON
        '''
        doi = self.doi()
        if doi:
            url = doi_url(doi)
            text = get(url)
            return widgets.HTML(text)

    def text(self):
        '''
        Load the paper from doi.org and display as text. Requires Internet to be ON
        '''
        if self.json_paper is not None:
            return self.json_paper.text()
        return get(self.doi())

    def abstract(self):
        _abstract = self.paper.loc['abstract'].values[0]
        if _abstract:
            return _abstract
        return ''

    def title(self):
        return self.paper.loc['title'].values[0]

    def has_text(self):
        return self.paper.loc['has_text']

    def full_text_path(self):
        if self.has_text():
            text_file = self.paper.loc['full_text_file']
            text_path = self.data_path / text_file / text_file / f'{self.sha}.json'
            return text_path

    def get_json_paper(self):
        text_path = self.full_text_path()
        if text_path:
            return load_json(str(text_path.resolve()))

    def authors(self, split=False):
        if self.json_paper:
            return self.json_paper.authors
        '''
        Get a list of authors
        '''
        authors = self.paper.loc['authors'].values[0]
        if not authors:
            return []
        if not split:
            return authors
        if authors.startswith('['):
            authors = authors.lstrip('[').rstrip(']')
            return [a.strip().replace("\'", "") for a in authors.split("\',")]

        # Todo: Handle cases where author names are separated by ","
        return [a.strip() for a in authors.split(';')]

    def _repr_html_(self):
        return self.paper.fillna('').to_frame()._repr_html_()
        #return render_html('Paper', paper=self)


class SearchResults:

    def __init__(self, data: pd.DataFrame, data_path):
        self.data_path = data_path
        self.results = data.dropna(subset=['title'])
        self.results.authors = self.results.authors.apply(str).replace("'", '').replace('[', '').replace(']', '')
        self.results['url'] = self.results.doi.apply(doi_url)
        self.columns = [col for col in ['sha', 'title', 'authors', 'when', 'Score'] if col in data]

    def __getitem__(self, item):
        return Paper(self.results.loc[item], self.data_path)

    def __len__(self):
        return len(self.results)

    def _results_view(self, search_results):
        return [{'title': rec['title'],
                 'authors': rec['authors'],
                 'abstract': shorten(rec['abstract'], 300),
                 'when': rec['when'],
                 'url' : rec['url']
                 }
                for rec in search_results.to_dict('records')]

    def _repr_html_(self):
        search_results=self._results_view(self.results)
        #display_cols = [col for col in self.columns if not col == 'sha']
        return render_html('SearchResultsHTML',
                           search_results=search_results)
