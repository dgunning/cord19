import pickle
import re
import time
from pathlib import Path, PurePath

import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import display, clear_output
from rank_bm25 import BM25Okapi
from requests import HTTPError

from cord.core import render_html, show_common, describe_dataframe, is_kaggle, CORD_CHALLENGE_PATH, \
    JSON_CATALOGS, find_data_dir, SARS_DATE, SARS_COV_2_DATE, listify
from cord.dates import add_date_diff
from cord.jsonpaper import load_json_paper, json_cache_exists, load_json_cache, PDF_JSON, PMC_JSON, \
    get_json_paths, get_token_df
from cord.text import preprocess, shorten, summarize
from cord.vectors import show_2d_chart, similar_papers

_MINIMUM_SEARCH_SCORE = 2


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
    data.abstract = data.abstract.fillna('')
    return data


def rename_publish_time(data):
    return data.rename(columns={'publish_time': 'published'})


COVID_TERMS = ['covid', 'sars-?n?cov-?2', '2019-ncov', 'novel coronavirus', 'sars coronavirus 2']
COVID_SEARCH = f".*({'|'.join(COVID_TERMS)})"
NOVEL_CORONAVIRUS = '.*novel coronavirus'
WUHAN_OUTBREAK = 'wuhan'


def tag_covid(data):
    """
    Tag all the records that match covid
    :param data:
    :return: data
    """
    abstract = data.abstract.fillna('')
    since_covid = (data.published > SARS_COV_2_DATE) | (data.published.isnull())
    covid_term_match = since_covid & abstract.str.match(COVID_SEARCH, case=False)
    wuhan_outbreak = since_covid & abstract.str.match('.*(wuhan|hubei)', case=False)
    covid_match = covid_term_match | wuhan_outbreak
    data['covid_related'] = False
    data.loc[covid_match, 'covid_related'] = True
    return data


def tag_virus(data):
    VIRUS_SEARCH = f".*(virus|viruses|viral)"
    viral_cond = data.abstract.str.match(VIRUS_SEARCH, case=False)
    data['virus'] = False
    data.loc[viral_cond, 'virus'] = True
    return data


def tag_coronavirus(data):
    corona_cond = data.abstract.str.match(".*corona", case=False)
    data['coronavirus'] = False
    data.loc[corona_cond, 'coronavirus'] = True
    return data


def tag_sars(data):
    sars_cond = data.abstract.str.match(".*sars", case=False)
    sars_not_covid = ~(data.covid_related) & (sars_cond)
    data['sars'] = False
    data.loc[sars_not_covid, 'sars'] = True
    return data


def apply_tags(data):
    print('Applying tags to metadata')
    data = data.pipe(tag_covid) \
        .pipe(tag_virus) \
        .pipe(tag_coronavirus) \
        .pipe(tag_sars)
    return data


def clean_metadata(metadata):
    print('Cleaning metadata')
    return metadata.pipe(start) \
        .pipe(clean_title) \
        .pipe(clean_abstract) \
        .pipe(rename_publish_time) \
        .pipe(add_date_diff) \
        .pipe(drop_missing) \
        .pipe(fill_nulls) \
        .pipe(apply_tags)


def get_json_path(data_path, full_text_file, sha, pmcid):
    if pmcid and isinstance(pmcid, str):
        return Path(data_path) / full_text_file / full_text_file / PMC_JSON / f'{pmcid}.xml.json'
    elif sha and isinstance(sha, str):
        return Path(data_path) / full_text_file / full_text_file / PDF_JSON / f'{sha}.json'


def get_pdf_json_path(data_path, full_text_file, sha):
    """
    :return: The path to the json file if the sha is present
    """
    if sha and isinstance(sha, str):
        return Path(data_path) / full_text_file / full_text_file / PDF_JSON / f'{sha}.json'


def get_pmcid_json_path(data_path, full_text_file, pmcid):
    """
    :return: the path to the json file if the pmcid json is available
    """
    if pmcid and isinstance(pmcid, str):
        return Path(data_path) / full_text_file / full_text_file / PMC_JSON / f'{pmcid}.xml.json'


def _get_bm25Okapi(index_tokens):
    has_tokens = index_tokens.apply(len).sum() > 0
    if not has_tokens:
        index_tokens.loc[0] = ['no', 'tokens']
    return BM25Okapi(index_tokens.tolist())


def _set_index_from_text(metadata, data_path):
    print('Creating the BM25 index from the text contents of the papers')
    for catalog in JSON_CATALOGS:
        catalog_idx = metadata.full_text_file == catalog
        if json_cache_exists():
            json_tokens = load_json_cache(catalog).set_index('cord_uid')
        else:
            json_tokens = get_token_df(metadata.loc[catalog_idx], data_path)
        token_lookup = json_tokens.to_dict()['index_tokens']
        metadata.loc[catalog_idx, 'index_tokens'] = \
            metadata.loc[catalog_idx, 'cord_uid'].apply(lambda c: token_lookup.get(c, np.nan))

    # If the index tokens are still null .. use the abstracts
    null_tokens = metadata.index_tokens.isnull()
    print('There are', null_tokens.sum(), 'papers that will be indexed using the abstract instead of the contents')
    metadata.loc[null_tokens, 'index_tokens'] = metadata.loc[null_tokens].abstract.apply(preprocess)
    missing_index_tokens = len(metadata.loc[catalog_idx & metadata.index_tokens.isnull()])
    if missing_index_tokens > 0:
        print('There still are', missing_index_tokens, 'index tokens')

    return metadata


def create_annoy_index(document_vectors):
    print('Creating Annoy document index')
    tick = time.time()
    from annoy import AnnoyIndex
    annoy_index = AnnoyIndex(20, 'angular')
    for i in range(len(document_vectors)):
        annoy_index.add_item(i, document_vectors.loc[i])
    annoy_index.build()
    tock = time.time()
    print('Finished creating Annoy document index in', round(tock - tick, 0), 'seconds')
    return annoy_index


class ResearchPapers:

    def __init__(self, metadata, data_dir='data', index='abstract', view='html'):
        self.data_path = Path(data_dir)
        self.num_results = 10
        self.view = view
        self.metadata = metadata
        if 'index_tokens' not in metadata:
            print('\nIndexing research papers')
            if any([index == t for t in ['text', 'texts', 'content', 'contents']]):
                tick = time.time()
                _set_index_from_text(self.metadata, data_dir)
                print("Finished indexing in", int(time.time() - tick), 'seconds')
            else:
                print('Creating the BM25 index from the abstracts of the papers')
                print('Use index="text" if you want to index the texts of the paper instead')
                tick = time.time()
                self.metadata['index_tokens'] = metadata.abstract.apply(preprocess)
                tock = time.time()
                print('Finished Indexing in', round(tock - tick, 0), 'seconds')

        # Create BM25 search index
        self.bm25 = _get_bm25Okapi(self.metadata.index_tokens)

        if 'antivirals' not in self.metadata:
            # Add antiviral column
            self.metadata['antivirals'] = self.metadata.index_tokens \
                .apply(lambda t:
                       ','.join([token for token in t if token.endswith('vir')]))

    def show_similar(self, paper_id):
        similar_paper_ids = similar_papers(paper_id)
        self.display(*similar_paper_ids)

    def similar_to(self, paper_id):
        """
        Find and displays papers similar to the paper
        :param paper_id: the cord_uid
        :return: None
        """
        similar_paper_ids = similar_papers(paper_id)
        original_paper = self[paper_id]
        style = 'color: #008B8B; font-weight: bold; font-size: 0.9em;'
        display(widgets.HTML(
            f'<h4>Papers similar to <span style="{style}">{original_paper.title}</span></h4>'))
        return self.display(*similar_paper_ids)

    def show(self, *paper_ids):
        return self.display(*paper_ids)

    def display(self, *paper_ids):
        if len(paper_ids) == 1:
            paper_ids = listify(paper_ids[0])

        _recs = []
        for id in paper_ids:
            paper = self[id]
            _recs.append({'published': paper.metadata.published,
                          'title': paper.title,
                          'summary': paper.summary,
                          'when': paper.metadata.when,
                          'cord_uid': paper.cord_uid})
        df = pd.DataFrame(_recs).sort_values(['published'], ascending=False).drop(columns=['published'])

        def highlight_cols(s):
            return 'font-size: 1.1em; color: #008B8B; font-weight: bold'

        return df.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['title']]).hide_index()

    def create_document_index(self):
        print('Indexing research papers')
        tick = time.time()
        index_tokens = self._create_index_tokens()
        # Add antiviral column
        self.metadata['antivirals'] = index_tokens.apply(lambda t:
                                                         ','.join([token for token in t if token.endswith('vir')]))
        # Does it have any covid term?
        self.bm25 = BM25Okapi(index_tokens.tolist())
        tock = time.time()
        print('Finished Indexing in', round(tock - tick, 0), 'seconds')

    def get_json_paths(self):
        return get_json_paths(self.metadata, self.data_path)

    def describe(self):
        cols = [col for col in self.metadata if not col in ['sha', 'index_tokens']]
        return describe_dataframe(self.metadata, cols)

    def __getitem__(self, item):
        if isinstance(item, int):
            paper = self.metadata.iloc[item]
        else:
            paper = self.metadata[self.metadata.cord_uid == item]

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
                              view=self.view)

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

    def contains(self, search_str, column='abstract'):
        cond = self.metadata[column].fillna('').str.contains(search_str)
        return self._make_copy(self.metadata[cond])

    def match(self, search_str, column='abstract'):
        cond = self.metadata[column].fillna('').str.match(search_str)
        return self._make_copy(self.metadata[cond])

    def head(self, n):
        return self._make_copy(self.metadata.head(n))

    def tail(self, n):
        return self._make_copy(self.metadata.tail(n).copy())

    def sample(self, n):
        return self._make_copy(self.metadata.sample(n).copy())

    def abstracts(self):
        return pd.Series([self.__getitem__(i).abstract() for i in range(len(self))])

    def titles(self):
        return pd.Series([self.__getitem__(i).title() for i in range(len(self))])

    def get_summary(self):
        summary_df = pd.DataFrame({'Papers': [len(self.metadata)],
                                   'Oldest': [self.metadata.published.min()],
                                   'Newest': [self.metadata.published.max()],
                                   'SARS-COV-2': [self.metadata.covid_related.sum()],
                                   'SARS': [self.metadata.sars.sum()],
                                   'Coronavirus': [self.metadata.coronavirus.sum()],
                                   'Virus': [self.metadata.virus.sum()],
                                   'Antivirals': [self.metadata.antivirals.apply(lambda a: len(a) > 0).sum()]},
                                  index=[''])
        summary_df.Newest = summary_df.Newest.fillna('')
        summary_df.Oldest = summary_df.Oldest.fillna('')
        return summary_df

    def _repr_html_(self):
        display_cols = ['title', 'abstract', 'journal', 'authors', 'published', 'when']
        return render_html('ResearchPapers', summary=self.get_summary()._repr_html_(),
                           research_papers=self.metadata[display_cols]._repr_html_())

    @staticmethod
    def load_metadata(data_path=None):
        if not data_path:
            data_path = find_data_dir()

        print('Loading metadata from', data_path)
        metadata_path = PurePath(data_path) / 'metadata.csv'
        dtypes = {'Microsoft Academic Paper ID': 'str', 'pubmed_id': str}
        renames = {'source_x': 'source', 'has_full_text': 'has_text'}
        metadata = pd.read_csv(metadata_path, dtype=dtypes, low_memory=False,
                               parse_dates=['publish_time']).rename(columns=renames)
        metadata = clean_metadata(metadata)
        return metadata

    @classmethod
    def load(cls, data_dir=None, index=None):
        if data_dir:
            data_path = Path(data_dir) / CORD_CHALLENGE_PATH
        else:
            data_path = find_data_dir()
        metadata = cls.load_metadata(data_path)
        return cls(metadata, data_path, index=index)

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

    def search_2d(self, search_string,
                  num_results=25,
                  covid_related=False,
                  start_date=None,
                  end_date=None):
        search_results = self.search(search_string, num_results, covid_related, start_date, end_date)
        show_2d_chart(search_results.results, query=search_string)

    def search(self, search_string,
               num_results=10,
               covid_related=False,
               start_date=None,
               end_date=None,
               view='html'):
        n_results = num_results or self.num_results
        search_terms = preprocess(search_string)
        doc_scores = self.bm25.get_scores(search_terms)

        # Get the index from the doc scores
        ind = np.argsort(doc_scores)[::-1]
        results = self.metadata.iloc[ind].copy()
        results['Score'] = doc_scores[ind].round(1)
        # paper_ids = find_similar_papers(search_string, num_items=50)
        # results = self.metadata[self.metadata.cord_uid.isin(paper_ids)]

        # Filter covid related
        if covid_related:
            results = results[results.covid_related]

        # Filter by dates
        if start_date:
            results = results[results.published >= start_date]

        if end_date:
            results = results[results.published < end_date]

        # Show only up to n_results
        results = results.head(num_results)

        # Create the final results
        results = results.drop_duplicates(subset=['title'])

        # Return Search Results
        return SearchResults(results, self.data_path, view=view)

    def _search_papers(self, output, SearchTerms: str, num_results=None, view=None,
                       start_date=None, end_date=None, covid_related=False):
        if len(SearchTerms) < 5:
            return
        search_results = self.search(SearchTerms, num_results=num_results, view=view,
                                     start_date=start_date, end_date=end_date, covid_related=covid_related)
        if len(search_results) > 0:
            with output:
                clear_output()
                display(search_results)
        return search_results

    def searchbar(self, initial_search_terms='', num_results=10, view=None):
        text_input = widgets.Text(layout=widgets.Layout(width='400px'), value=initial_search_terms)

        search_button = widgets.Button(description='Search', button_style='primary',
                                       layout=widgets.Layout(width='100px'))
        search_box = widgets.HBox(children=[text_input, search_button])

        # A COVID-related checkbox
        covid_related_CheckBox = widgets.Checkbox(description='Covid-19 related', value=False, disable=False)
        checkboxes = widgets.HBox(children=[covid_related_CheckBox])

        # A date slider to limit research papers to a date range
        search_dates_slider = SearchDatesSlider()

        search_widget = widgets.VBox([search_box, search_dates_slider, checkboxes])

        output = widgets.Output()

        def do_search():
            search_terms = text_input.value.strip()
            if search_terms and len(search_terms) >= 4:
                start_date, end_date = search_dates_slider.value
                self._search_papers(output=output, SearchTerms=search_terms, num_results=num_results, view=view,
                                    start_date=start_date, end_date=end_date,
                                    covid_related=covid_related_CheckBox.value)

        def button_search_handler(btn):
            with output:
                clear_output()
            do_search()

        def text_search_handler(change):
            if len(change['new'].split(' ')) != len(change['old'].split(' ')):
                do_search()

        def date_handler(change):
            do_search()

        def checkbox_handler(change):
            do_search()

        search_button.on_click(button_search_handler)
        text_input.observe(text_search_handler, names='value')
        search_dates_slider.observe(date_handler, names='value')
        covid_related_CheckBox.observe(checkbox_handler, names='value')

        display(search_widget)
        display(output)

        # Show the initial terms
        if initial_search_terms:
            do_search()


def SearchDatesSlider():
    options = [(' 1951 ', '1951-01-01'), (' SARS 2003 ', '2002-11-01'),
               (' H1N1 2009 ', '2009-04-01'), (' COVID 19 ', '2019-11-30'),
               (' 2020 ', '2020-12-31')]
    return widgets.SelectionRangeSlider(
        options=options,
        description='Dates',
        disabled=False,
        value=('2002-11-01', '2020-12-31'),
        layout={'width': '480px'}
    )


class Paper:
    '''
    A single research paper
    '''

    def __init__(self, item, data_path):
        if isinstance(item, pd.DataFrame):
            # convert to a series
            item = item.T.iloc[:, 0]

        self.metadata = item
        self.sha = item.sha
        self.pmcid = item.pmcid
        self.cord_uid = item.cord_uid
        self.catalog = item.full_text_file
        self.data_path = data_path
        self.has_pmc = self.metadata.pmcid

    def get_json_paper(self):
        if self.metadata.has_pmc_xml_parse and self.pmcid and isinstance(self.pmcid, str):
            return self.get_pmc_json()
        elif self.metadata.has_pdf_parse and self.sha and isinstance(self.sha, str):
            return self.get_sha_json()

    def get_sha_path(self):
        if self.metadata.has_pdf_parse and self.sha and isinstance(self.sha, str):
            return self.data_path / self.metadata.full_text_file / \
                   self.metadata.full_text_file / PDF_JSON / f'{self.sha}.json'

    def get_sha_json(self):
        path = self.get_sha_path()
        if path and path.exists():
            return load_json_paper(path)

    def get_pmc_path(self):
        if self.metadata.full_text_file and self.metadata.pmcid:
            return self.data_path / self.metadata.full_text_file / \
                   self.metadata.full_text_file / PMC_JSON / f'{self.metadata.pmcid}.xml.json'

    def get_pmc_json(self):
        path = self.get_pmc_path()
        if path and path.exists():
            return load_json_paper(path)

    @property
    def url(self):
        return self.metadata.url

    @property
    def html(self):
        json_paper = self.get_json_paper()
        if json_paper:
            return json_paper.html

    @property
    def text(self):
        '''
        Load the paper from doi.org and display as text. Requires Internet to be ON
        '''
        json_paper = self.get_json_paper()
        if json_paper:
            return json_paper.text

    @property
    def abstract(self):
        return self.metadata.abstract

    @property
    def summary(self):
        return summarize(self.abstract)

    @property
    def text_summary(self):
        return summarize(self.text, word_count=300)

    @property
    def title(self):
        return self.metadata.title

    def has_text(self):
        return self.paper.has_text

    @property
    def authors(self, split=False):
        json_paper = self.get_json_paper()
        if json_paper:
            return ', '.join(json_paper.authors)
        '''
        Get a list of authors
        '''
        authors = self.metadata.authors
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
        paper_meta = self.metadata.to_frame().T
        paper_meta = paper_meta[['published', 'authors', 'cord_uid', 'url', ]]
        paper_meta.index = ['']

        return render_html('Paper', paper=self, meta=paper_meta)


class SearchResults:

    def __init__(self, data: pd.DataFrame, data_path, view='html'):
        self.data_path = data_path
        self.results = data.dropna(subset=['title'])
        self.results.authors = self.results.authors.apply(str).replace("'", '').replace('[', '').replace(']', '')
        self.results['url'] = self.results.doi.apply(doi_url)
        self.results['summary'] = self.results.abstract.apply(summarize)
        self.columns = [col for col in ['sha', 'title', 'summary', 'when'] if col in self.results]
        self.view = view

    def __getitem__(self, item):
        return Paper(self.results.loc[item], self.data_path)

    def __len__(self):
        return len(self.results)

    def _view_html(self, search_results):
        _results = [{'title': rec['title'],
                     'authors': shorten(rec['authors'], 200),
                     'abstract': shorten(rec['abstract'], 300),
                     'summary': shorten(summarize(rec['abstract']), 500),
                     'when': rec['when'],
                     'url': rec['url'],
                     'cord_uid': rec['cord_uid'],
                     'is_kaggle': is_kaggle()
                     }
                    for rec in search_results.to_dict('records')]
        return render_html('SearchResultsHTML', search_results=_results)

    def get_results_df(self):
        display_cols = [col for col in self.columns if not col == 'sha']
        return self.results[display_cols]

    def _repr_html_(self):
        if self.view == 'html':
            return self._view_html(self.results)
        elif any([self.view == v for v in ['df', 'dataframe', 'table']]):
            return self.get_results_df()._repr_html_()
        else:
            return self._view_html(self.results)
