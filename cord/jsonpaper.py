import collections
from functools import partial, reduce, lru_cache
import simplejson as json
import pandas as pd
import numpy as np
from .core import parallel, render_html, listify, add, CORD_CHALLENGE_PATH, BIORXIV_MEDRXIV, \
    NONCOMM_USE_SUBSET, CUSTOM_LICENSE, COMM_USE_SUBSET, find_data_dir
from pathlib import Path, PurePath
import pickle
from .text import preprocess
import ipywidgets as widgets
from typing import Dict, List
from gensim.corpora import Dictionary
import time

_JSON_CATALOG_SAVEFILE = 'JsonCatalog'
PDF_JSON = 'pdf_json'
PMC_JSON = 'pmc_json'


def get_text_sections(paper_json, text_key) -> Dict:
    """
    :param paper_json: The json
    :param text_key: the text_key - "body_text" or "abstract"
    :return: a dict with the sections
    """
    body_dict = collections.defaultdict(list)
    for rec in paper_json[text_key]:
        body_dict[rec['section'].strip()].append(rec['text'])
    return body_dict


get_body_sections = partial(get_text_sections, text_key='body_text')
get_abstract_sections = partial(get_text_sections, text_key='abstract')


def get_text(paper_json, text_key) -> str:
    """
    :param paper_json: The json
    :param text_key: the text_key - "body_text" or "abstract"
    :return: a text string with the sections
    """
    if not text_key in paper_json:
        return ''
    body_dict = collections.defaultdict(list)
    for rec in paper_json[text_key]:
        body_dict[rec['section']].append(rec['text'])

    body = ''
    for section, text_sections in body_dict.items():
        body += section + '\n\n'
        for text in text_sections:
            body += text + '\n\n'
    return body


get_body = partial(get_text, text_key='body_text')
get_abstract = partial(get_text, text_key='abstract')


def author_name(author_json):
    first = author_json.get('first')
    middle = "".join(author_json.get('middle'))
    last = author_json.get('last')
    if middle:
        return ' '.join([first, middle, last])
    return ' '.join([first, last])


def get_affiliation(author_json):
    affiliation = author_json['affiliation']
    institution = affiliation.get('institution', '')
    location = affiliation.get('location')
    if location:
        location = ' '.join(location.values())
    return f'{institution}, {location}'


def get_authors(paper_json, include_affiliation=False):
    if include_affiliation:
        return [f'{author_name(a)}, {get_affiliation(a)}'
                for a in paper_json['metadata']['authors']]
    else:
        return [author_name(a) for a in paper_json['metadata']['authors']]


def get_pdf_json_paths(metadata: str, data_path: str) -> pd.Series:
    """
    :param metadata: The CORD Research Metadata
    :param data_path: The path to the CORD data
    :return: a Series containing the PDF JSON paths
    """
    def path_fn(full_text_file, sha):
        if sha and isinstance(sha, str) and isinstance(full_text_file, str):
                return Path(data_path) / full_text_file / full_text_file / PDF_JSON / f'{sha}.json'

    sha_paths = metadata.apply(lambda m: [path_fn(m.full_text_file, sha.strip()) for sha in m.sha.split(';')]
                                                        if m.has_pdf_parse else np.nan, axis=1)
    return sha_paths


def get_first_json(jsons: List):
    if isinstance(jsons, list) and len(jsons) > 0:
        return jsons[0]
    return jsons


def get_pmcid_json_paths(metadata: pd.DataFrame, data_path: str) -> pd.Series:
    """
    :param metadata: The CORD Research Metadata
    :param data_path: The path to the CORD data
    :return: a series containing the paths to the PMC JSONS .. will contain nans
    """
    def path_fn(full_text_file, pmcid):
        if pmcid and isinstance(pmcid, str) and isinstance(full_text_file, str):
            pmcid_path= Path(data_path) / full_text_file / full_text_file / PMC_JSON / f'{pmcid}.xml.json'
            if pmcid_path.exists():
                return pmcid_path
        return np.nan
    pmc_paths = metadata.apply(lambda m: path_fn(m.full_text_file, m.pmcid), axis=1)
    return pmc_paths


def get_json_paths(metadata: pd.DataFrame, data_path: str, first=True, tolist=False) -> pd.Series:
    """
    :param metadata: The CORD Research Metadata
    :param data_path: The path to the CORD data
    :return: a series containing the paths to the JSONS .. will contain nans
    """
    has_pmc = metadata.has_pmc_xml_parse
    paths_df = metadata[['has_pmc_xml_parse']].copy()
    paths_df.loc[has_pmc, 'json_path'] = get_pmcid_json_paths(metadata.loc[paths_df.has_pmc_xml_parse], data_path)
    paths_df.loc[~has_pmc, 'json_path'] = get_pdf_json_paths(metadata.loc[~paths_df.has_pmc_xml_parse],data_path)

    if tolist:
        if first:
            paths_df.loc[~has_pmc, 'json_path'] = paths_df.loc[~has_pmc, 'json_path'].apply(get_first_json)
            return paths_df.json_path.dropna().tolist()
        else:
            return paths_df.loc[has_pmc, 'json_path'].tolist() + \
                [p for lst in paths_df.loc[~has_pmc, 'json_path'].dropna().tolist() for p in lst]
    else:
        return paths_df.json_path.apply(get_first_json)


class JsonPaper:

    def __init__(self, paper_json):
        self.paper_json = paper_json

    @property
    def sha(self):
        return self.paper_json['paper_id']

    @property
    def title(self):
        return self.paper_json['metadata']['title']

    @property
    def text(self):
        return get_body(self.paper_json)

    @property
    def abstract(self):
        return get_abstract(self.paper_json)

    @property
    def html(self):
        sections = get_body_sections(self.paper_json)
        html = render_html('JsonPaperBody', title=self.title, sections=sections)
        return widgets.HTML(html)

    @property
    def abstract_html(self):
        sections = get_abstract_sections(self.paper_json)
        html = render_html('JsonPaperBody', title=self.title, sections=sections)
        return widgets.HTML(html)

    @property
    def authors(self):
        return get_authors(self.paper_json)

    @classmethod
    def from_json(cls, paper_json):
        return cls(paper_json=paper_json)

    @classmethod
    def from_dict(cls, paper_dict):
        sha = paper_dict['sha']
        text = paper_dict['text']
        abstract = paper_dict['abstract']
        title = paper_dict['title']
        authors = paper_dict['authors']
        return cls(sha=sha, text=text, abstract=abstract, title=title, authors=authors)

    def to_dict(self):
        return {'sha': self.sha, 'abstract': self.abstract,
                'title': self.title, 'authors': ' '.join(self.authors)}

    def _repr_html_(self):
        return render_html('JPaper', paper=self)

    def __repr__(self):
        return 'JsonPaper'


@lru_cache(maxsize=1024)
def load_json_file(json_file):
    with Path(json_file).open('r') as f:
        return json.load(f)


def load_json_paper(json_file):
    with Path(json_file).open('r') as f:
        contents = json.load(f)
    return JsonPaper(contents)


def load_text_body_from_file(json_path):
    with json_path.open('r') as f:
        json_content = json.load(f)
        body_text = get_text(json_content, 'body_text')
        authors = get_authors(json_content)
    sha = json_path.stem
    return sha, body_text, authors


def load_text(json_path):
    """
    Load the text from the Json file
    :param json_path:
    :return: the text body of the json file
    """
    with json_path.open('r') as f:
        json_content = json.load(f)
        body_text = get_text(json_content, 'body_text')
    return body_text


def load_tokens_from_file(json_path):
    sha, text, authors = load_text_body_from_file(json_path)
    return sha, preprocess(text), authors


def list_json_files_in(json_path):
    # As of April 4th the json files are separated into two directories
    all_json_files = []
    for sub_dir in ['pdf_json', 'pmc_json']:
        json_sub_path = json_path / sub_dir
        if json_sub_path.exists():
            all_json_files = all_json_files + list(json_sub_path.glob('*.json'))
    return all_json_files


def load_json_texts(json_dirs=None, tokenize=False):
    data_path = Path(find_data_dir())
    json_dirs = json_dirs or [BIORXIV_MEDRXIV, NONCOMM_USE_SUBSET, COMM_USE_SUBSET, CUSTOM_LICENSE]
    json_dirs = listify(json_dirs)

    text_dfs = []
    for json_dir in json_dirs:
        json_path = Path(data_path) / json_dir / json_dir
        print('Loading json from', json_path.stem)
        load_fn = load_tokens_from_file if tokenize else load_text_body_from_file
        sha_texts_authors = parallel(load_fn, list_json_files_in(json_path))
        text_dfs.append(pd.DataFrame(sha_texts_authors, columns=['sha', 'text', 'authors']))
    text_df = pd.concat(text_dfs, ignore_index=True)

    # PCMID is  now in the name of the json file, insteadt of just being the sha
    text_df['pmcid'] = text_df.sha.str.extract('(PMC[0-9]+)\.xml')
    text_df.loc[~text_df.pmcid.isnull(), 'sha'] = np.nan
    text_df = text_df[['sha', 'pmcid', 'text']]

    if tokenize:
        return text_df.rename(columns={'text': 'index_tokens'})
    return text_df


def load_dictionary(catalog):
    json_cache_dir = Path(find_data_dir()).parent / 'json-cache'
    dictionary_path = json_cache_dir / f'jsoncache_{catalog}.dict'
    dictionary = Dictionary.load((str(dictionary_path.resolve())))
    return dictionary


def get_json_cache_dir():
    return Path(find_data_dir()).parent / 'json-cache'


def json_cache_exists():
    return get_json_cache_dir().exists()


def load_json_cache(catalog):
    print('Loading json cache files for', catalog)
    tick = time.time()
    json_cache_dir = get_json_cache_dir()
    file_paths = [PurePath(p) for p in json_cache_dir.glob(f'jsoncache_{catalog}*.pq')]
    if len(file_paths) == 1:
        json_cache = pd.read_parquet(file_paths[0])
    else:
        dfs = parallel(pd.read_parquet, file_paths)
        json_cache = pd.concat(dfs, ignore_index=True)
    dictionary: Dictionary = load_dictionary(catalog)
    json_cache['index_tokens'] \
        = json_cache.token_int.apply(lambda token_int: [dictionary[ti] for ti in token_int])
    df = json_cache.drop(columns=['token_int'])
    tock = time.time()
    print('Loaded', catalog, 'json cache in', int(tock - tick), 'seconds')
    return df


def get_tokens(cord_path):
    cord_uid, path = cord_path
    if isinstance(path, Path):
        tokens = preprocess(load_text(path))
        return cord_uid, tokens
    return cord_uid, np.nan


def get_token_df(metadata: pd.DataFrame, data_path:Path) -> pd.DataFrame:
    """
    This create a dataframe with the index_tokens
    :param metadata:
    :param data_path:
    :return:
    """
    catalog_paths = metadata.copy()
    catalog_paths['json_path'] = get_json_paths(catalog_paths, data_path)
    catalog_paths = catalog_paths[['cord_uid', 'json_path']]
    cord_paths = catalog_paths.to_records(index=False)
    cord_tokens = parallel(get_tokens, cord_paths)
    token_df = pd.DataFrame(cord_tokens, columns=['cord_uid', 'index_tokens'])
    return token_df.dropna()

class JsonCatalog:

    def __init__(self, papers, json_catalog):
        self.papers = pd.DataFrame([paper.to_dict() for paper in papers])
        self.papers['catalog'] = json_catalog

    def get_index_tokens(self):
        return self.papers.abstract.apply(preprocess())

    @classmethod
    def load(cls, json_dirs=None, data_path='data'):
        data_path = Path(data_path) / CORD_CHALLENGE_PATH
        json_dirs = json_dirs or [BIORXIV_MEDRXIV, NONCOMM_USE_SUBSET, COMM_USE_SUBSET, CUSTOM_LICENSE]
        json_paths = [Path(data_path) / json_dir / json_dir for json_dir in listify(json_dirs)]

        _catalogs = []
        catalog = None
        for json_path in json_paths:
            print('Loading json from', json_path.stem)
            papers = parallel(load_json_paper, list_json_files_in(json_path))
            if not catalog:
                catalog = cls(papers=papers, json_catalog=json_path.stem)
            else:
                catalog = catalog + cls(papers=papers, json_catalog=json_path.stem)
        return catalog

    def save(self, sub_catalog="", save_dir='data', ):
        save_file = _JSON_CATALOG_SAVEFILE
        if sub_catalog:
            save_file = f'{save_file}_{sub_catalog}'
        save_path = PurePath(save_dir) / f'{save_file}.pickle'
        print('Saving to', save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(sub_catalog="", save_dir='data'):
        print(f'Loading catalog {sub_catalog if sub_catalog else "all"}')
        save_file = _JSON_CATALOG_SAVEFILE
        if sub_catalog:
            save_file = f'{save_file}_{sub_catalog}'
        save_path = PurePath(save_dir) / f'{save_file}.pickle'
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, item):
        if isinstance(item, int):
            paper_rec = self.papers.iloc[item]
            sha = paper_rec.sha
        else:
            sha = item
        return self.get_paper(sha=sha)

    def get_paper(self, sha):
        paper_rec = self.papers[self.papers.sha == sha]
        if len(paper_rec) == 1:
            rec = paper_rec.to_dict('records')[0]
            sha, catalog = rec['sha'], rec['catalog']
            json_path = Path(find_data_dir()) / catalog / catalog / f'{sha}.json'
            jpaper = load_json_paper(json_path)
            return jpaper

    def __len__(self):
        return len(self.papers)

    def __add__(self, o):
        return JsonCatalog(self.papers + o.papers)

    def _repr_html_(self):
        display_cols = ['title', 'abstract', 'authors']
        _df = self.papers[display_cols]
        return _df._repr_html_()
