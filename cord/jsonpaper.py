import collections
from functools import partial, reduce
import json
import pandas as pd
from .core import parallel, render_html, listify, add, CORD_CHALLENGE_PATH, BIORXIV_MEDRXIV,\
    NONCOMM_USE_SUBSET, CUSTOM_LICENSE, COMM_USE_SUBSET
from pathlib import Path, PurePath
import pickle
from .text import preprocess


_JSON_CATALOG_SAVEFILE = 'JsonCatalog'


def get_text(paper_json, text_key):
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


class JPaper:

    def __init__(self, sha, text, abstract, title, authors):
        self.sha = sha
        self.text = text
        self.abstract = abstract
        self.title = title
        self.authors = authors

    @classmethod
    def from_json(cls, paper_json):
        sha = paper_json['paper_id']
        text = get_body(paper_json)
        abstract = get_abstract(paper_json)
        title = paper_json['metadata']['title']
        authors = get_authors(paper_json)
        return cls(sha=sha, text=text, abstract=abstract, title=title, authors=authors)

    @classmethod
    def from_dict(cls, paper_dict):
        sha = paper_dict['sha']
        text = paper_dict['text']
        abstract = paper_dict['abstract']
        title = paper_dict['title']
        authors = paper_dict['authors']
        return cls(sha=sha, text=text, abstract=abstract, title=title, authors=authors)

    def to_dict(self):
        return {'sha': self.sha, 'text': self.text, 'abstract': self.abstract,
                'title': self.title, 'authors': self.authors}

    def _repr_html_(self):
        return render_html('JPaper', paper=self)

    def __repr__(self):
        return 'JPaper'


def load_json_file(json_file):
    with open(json_file, 'r') as f:
        contents = json.load(f)
    return JPaper.from_json(contents)


class JCatalog:

    def __init__(self, papers):
        self.papers = pd.DataFrame([paper.to_dict() for paper in papers])

    def nlp(self):
        print('Creating index tokens')
        self.papers['index_tokens'] = self.papers.text.apply(preprocess)

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
            papers = parallel(load_json_file, list(json_path.glob('*.json')))
            if not catalog:
                catalog = cls(papers)
            else:
                catalog = catalog + cls(papers)
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
            j_dict = self.papers.iloc[item]
        else:
            j_dict = self.papers.loc[self.papers.sha == item].to_dict()
        return JPaper.from_dict(j_dict)

    def __len__(self):
        return len(self.papers)

    def __add__(self, o):
        return JCatalog(self.papers + o.papers)

    def _repr_html_(self):
        display_cols = ['title', 'abstract', 'authors']
        _df = self.papers[display_cols]
        return _df._repr_html_()