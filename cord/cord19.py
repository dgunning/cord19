import re
import re
import time

import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from requests import HTTPError

from cord.core import ifnone, render_html, show_common, describe_dataframe, is_kaggle, CORD_CHALLENGE_PATH, \
    JSON_CATALOGS, KAGGLE_INPUT, NON_KAGGLE_DATA_DIR
from cord.dates import fix_dates, add_date_diff
from cord.jsonpaper import load_json_paper, load_json_texts
from cord.nlp import get_lda_model, get_topic_vector
from cord.text import preprocess, shorten

english_stopwords = list(set(stopwords.words('english')))
import pickle
from pathlib import Path, PurePath

SARS_DATE = '2002-11-01'
SARS_COV_2_DATE = '2019-11-30'
_MINIMUM_SEARCH_SCORE = 2


class Author:

    def __init__(self, first=None, last=None, middle=None):
        self.first = ifnone(first, '')
        self.last = ifnone(last, '')

    def __repr__(self):
        return f'{self.first} {self.last}'


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
_relevant_re_ = _relevant_re_ + '.*epidem.*|.*emerg.*|.*vacc.*|.*cytokine.*'


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
    data.loc[data.abstract.isin(common_abstracts), 'abstract'] = ''

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
    data.journal = data.journal.fillna('')
    return data


def rename_publish_time(data):
    return data.rename(columns={'publish_time': 'published'})


def clean_metadata(metadata):
    print('Cleaning metadata')
    return metadata.pipe(start) \
        .pipe(clean_title) \
        .pipe(clean_abstract) \
        .pipe(rename_publish_time) \
        .pipe(add_date_diff) \
        .pipe(drop_missing) \
        .pipe(fill_nulls)


def get_json_path(data_path, text_path, sha):
    return Path(data_path) / text_path / text_path / f'{sha}.json'


_COVID_KEYWORDS = {'covid-19': 100, '2019-ncov': 100, 'sars-cov-2': 100, 'sars-cov': 20, 'quarantine': 10,
                   'outbreak': 10, 'severe': 5, '2019': 10, 'coronavirus': 25, 'novel': 25, 'new': 10,
                   'china': 5, 'wuhan': 20, 'hubei': 30, 'ace2': 30, 'pneumonia': 10}

covid19_synonyms = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2',  # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b']


def _get_bm25Okapi(index_tokens):
    has_tokens = index_tokens.apply(len).sum() > 0
    if not has_tokens:
        index_tokens.loc[0] = ['no', 'tokens']
    return BM25Okapi(index_tokens.tolist())


def lookup_tokens(shas, token_map):
    if not isinstance(shas, str): return []
    for sha in shas.split(';'):
        tokens = token_map.get(sha.strip())
        if tokens:
            return tokens


def _set_index_from_text(metadata, data_dir):
    print('Creating the BM25 index from the text contents of the papers')
    tick = time.time()
    for catalog in JSON_CATALOGS:
        catalog_idx = metadata.full_text_file == catalog
        metadata_papers = metadata.loc[catalog_idx, ['sha']].copy().reset_index()

        # Load the json catalog
        json_papers = load_json_texts(json_dirs=catalog, data_path=data_dir, tokenize=True)

        # Set the index tokens from the json_papers to the metadata
        sha_tokens = metadata_papers.merge(json_papers, how='left', on='sha').set_index('index')

        # Handle records with multiple shas
        has_multiple = (sha_tokens.sha.fillna('').str.contains(';'))
        token_map = json_papers[['sha', 'index_tokens']].set_index('sha').to_dict()['index_tokens']
        sha_tokens.loc[has_multiple, 'index_tokens'] \
            = sha_tokens.loc[has_multiple, 'sha'].apply(lambda sha: lookup_tokens(sha, token_map))

        metadata.loc[catalog_idx, 'index_tokens'] = sha_tokens.index_tokens
        null_tokens = metadata.index_tokens.isnull()
        # Fill null tokens with an empty list
        metadata.loc[null_tokens, 'index_tokens'] = \
            metadata.loc[null_tokens, 'index_tokens'].fillna('').apply(lambda d: d.split(' '))
    tock = time.time()
    print('Finished Indexing texts in', round(tock - tick, 0), 'seconds')
    return metadata


class ResearchPapers:

    def __init__(self, metadata, data_dir='data', index='abstract', display='html'):
        self.data_path = Path(data_dir) / CORD_CHALLENGE_PATH
        self.num_results = 10
        self.display = display
        self.metadata = metadata
        print('\nIndexing research papers')
        if 'index_tokens' not in metadata:
            if any([index == t for t in ['text', 'texts', 'content', 'contents']]):
                _set_index_from_text(self.metadata, data_dir)
            else:
                print('Creating the BM25 index from the abstracts of the papers')
                print('Use index="text" if you want to index the texts of the paper instead')
                tick = time.time()
                self.metadata['index_tokens'] = metadata.abstract.apply(preprocess)
                tock = time.time()
                print('Finished Indexing in', round(tock - tick, 0), 'seconds')

        self.bm25 = _get_bm25Okapi(self.metadata.index_tokens)

        if 'antivirals' not in self.metadata:
            # Add antiviral column
            self.metadata['antivirals'] = self.metadata.index_tokens \
                .apply(lambda t:
                       ','.join([token for token in t if token.endswith('vir')]))
        if not 'covid_related' in self.metadata:
            # Does it have any covid term?
            self.metadata['covid_related'] = self.metadata.index_tokens.apply(
                lambda tokens: sum([_COVID_KEYWORDS[token] for token in tokens
                                    if token in _COVID_KEYWORDS]) > 50)

    def nlp(self):
        # Topic model
        lda_model, dictionary, corpus = get_lda_model(self.index_tokens, num_topics=8)
        print('Assigning LDA topics')
        topic_vector = self.index_tokens.apply(lambda tokens: get_topic_vector(lda_model, dictionary, tokens))
        self.metadata['topic_vector'] = topic_vector
        self.metadata['top_topic'] = topic_vector.apply(np.argmax)

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

    def describe(self):
        cols = [col for col in self.metadata if not col in ['sha', 'index_tokens']]
        return describe_dataframe(self.metadata, cols)

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
        return ResearchPapers(metadata=new_data.copy(),
                              data_dir=self.data_path,
                              display=self.display)

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

    def get_papers(self, sub_catalog):
        return self.query(f'full_text_file =="{sub_catalog}"')

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
        display_cols = ['title', 'abstract', 'journal', 'authors', 'published', 'when']
        return self.metadata[display_cols]._repr_html_()

    def files(self):
        def full_text_path(self):
            if self.has_text():
                text_file = self.paper.loc['full_text_file']
                text_path = self.data_path / text_file / text_file / f'{self.sha}.json'
                return text_path

    @staticmethod
    def load_metadata(data_path=None):
        if data_path is None:
            if is_kaggle():
                data_path = data_path=Path(KAGGLE_INPUT) / CORD_CHALLENGE_PATH
            else:
                data_path = data_path = Path(NON_KAGGLE_DATA_DIR) / CORD_CHALLENGE_PATH

        print('Loading metadata from', data_path)
        metadata_path = PurePath(data_path) / 'metadata.csv'
        dtypes = {'Microsoft Academic Paper ID': 'str', 'pubmed_id': str}
        renames = {'source_x': 'source', 'has_full_text': 'has_text'}
        metadata = pd.read_csv(metadata_path, dtype=dtypes, parse_dates=['publish_time']).rename(columns=renames)
        # category_dict = {'license': 'category', 'source_x': 'category',
        #                 'journal': 'category', 'full_text_file': 'category'}
        metadata = clean_metadata(metadata)
        return metadata

    @classmethod
    def load(cls, data_dir=None, index=None):
        if not data_dir:
            data_dir = KAGGLE_INPUT if is_kaggle() else NON_KAGGLE_DATA_DIR
        data_path = Path(data_dir) / 'CORD-19-research-challenge'
        metadata = cls.load_metadata(data_path)
        return cls(metadata, data_dir, index=index)

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
               end_date=None,
               display='html'):
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
        return SearchResults(results, self.data_path, display=display)

    def _search_papers(self, SearchTerms: str):
        search_results = self.search(SearchTerms, display=self.display)
        if len(search_results) > 0:
            display(search_results)
        return search_results

    def searchbar(self, search_terms='sars-cov-2 outbreak cruise ship', num_results=10, display=None):
        self.num_results = num_results
        if display:
            self.display = display
        return widgets.interactive(self._search_papers, SearchTerms=search_terms)


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
            return load_json_paper(str(text_path.resolve()))

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
        # return render_html('Paper', paper=self)


class SearchResults:

    def __init__(self, data: pd.DataFrame, data_path, display='html'):
        self.data_path = data_path
        self.results = data.dropna(subset=['title'])
        self.results.authors = self.results.authors.apply(str).replace("'", '').replace('[', '').replace(']', '')
        self.results['url'] = self.results.doi.apply(doi_url)
        self.columns = [col for col in ['sha', 'title', 'abstract', 'when', 'authors'] if col in data]
        self.display = display

    def __getitem__(self, item):
        return Paper(self.results.loc[item], self.data_path)

    def __len__(self):
        return len(self.results)

    def _view_html(self, search_results):
        _results = [{'title': rec['title'],
                     'authors': shorten(rec['authors'], 200),
                     'abstract': shorten(rec['abstract'], 300),
                     'when': rec['when'],
                     'url': rec['url'],
                     'is_kaggle': is_kaggle()
                     }
                    for rec in search_results.to_dict('records')]
        return render_html('SearchResultsHTML', search_results=_results)

    def _repr_html_(self):
        if self.display == 'html':
            return self._view_html(self.results)
        elif any([self.display == v for v in ['df', 'dataframe', 'table']]):
            display_cols = [col for col in self.columns if not col == 'sha']
            return self.results[display_cols]._repr_html_()
        else:
            return self._view_html(self.results)
