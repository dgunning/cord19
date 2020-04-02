import collections
from functools import partial, reduce, lru_cache
import simplejson as json
import pandas as pd
from .core import parallel, render_html, listify, add, CORD_CHALLENGE_PATH, BIORXIV_MEDRXIV, \
    NONCOMM_USE_SUBSET, CUSTOM_LICENSE, COMM_USE_SUBSET, find_data_dir
from pathlib import Path, PurePath
import pickle
from .text import preprocess
import ipywidgets as widgets
from typing import Dict
from gensim.corpora import Dictionary

_JSON_CATALOG_SAVEFILE = 'JsonCatalog'


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


def load_tokens_from_file(json_path):
    sha, text, authors = load_text_body_from_file(json_path)
    return sha, preprocess(text), authors


def load_json_texts(json_dirs=None, tokenize=False):
    data_path = Path(find_data_dir())
    json_dirs = json_dirs or [BIORXIV_MEDRXIV, NONCOMM_USE_SUBSET, COMM_USE_SUBSET, CUSTOM_LICENSE]
    json_dirs = listify(json_dirs)

    text_dfs = []
    for json_dir in json_dirs:
        json_path = Path(data_path) / json_dir / json_dir
        print('Loading json from', json_path.stem)
        load_fn = load_tokens_from_file if tokenize else load_text_body_from_file
        sha_texts_authors = parallel(load_fn, list(json_path.glob('*.json')))
        text_dfs.append(pd.DataFrame(sha_texts_authors, columns=['sha', 'text', 'authors']))
    text_df = pd.concat(text_dfs, ignore_index=True)

    if tokenize:
        return text_df.rename(columns={'text': 'index_tokens'})
    return text_df


def load_dictionary(catalog):
    json_cache_dir = Path(find_data_dir()).parent / 'json_cache'
    dictionary_path = json_cache_dir / f'jsoncache_{catalog}.dict'
    dictionary = Dictionary.load((str(dictionary_path.resolve())))
    return dictionary


def get_json_cache_dir():
    return Path(find_data_dir()).parent / 'json_cache'

def json_cache_exists():
    return get_json_cache_dir().exists()


def load_json_cache(catalog):
    print('Loading json cache files for', catalog)
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
    return json_cache.drop(columns=['token_int'])


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
            papers = parallel(load_json_paper, list(json_path.glob('*.json')))
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
