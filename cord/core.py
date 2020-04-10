import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Collection, Any
import math
import pandas as pd
import numpy as np
from jinja2 import Template
from pathlib import PurePath
from annoy import AnnoyIndex

CORD_CHALLENGE_PATH = 'CORD-19-research-challenge'
KAGGLE_INPUT = '../input'
NON_KAGGLE_DATA_DIR = 'data'
COMM_USE_SUBSET = 'comm_use_subset'
NONCOMM_USE_SUBSET = 'noncomm_use_subset'
CUSTOM_LICENSE = 'custom_license'
BIORXIV_MEDRXIV = 'biorxiv_medrxiv'
JSON_CATALOGS = [COMM_USE_SUBSET, BIORXIV_MEDRXIV, NONCOMM_USE_SUBSET, CUSTOM_LICENSE]
SARS_DATE = '2002-11-01'
SARS_COV_2_DATE = '2019-11-30'
DOCUMENT_VECTOR_LENGTH = 20


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (NameError, ImportError):
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def is_kaggle():
    return (str(Path('.').resolve())).startswith('/kaggle')


def find_data_dir():
    input_dir = KAGGLE_INPUT if is_kaggle() else NON_KAGGLE_DATA_DIR
    input_path = Path() / input_dir
    if input_path.exists():
        return str(input_path / CORD_CHALLENGE_PATH)
    else:
        input_path = Path('..') / input_dir
        if input_path.exists():
            return str(input_path / CORD_CHALLENGE_PATH)
    assert input_path.exists(), f'Cannot find the input dir should be {input_dir}/{CORD_CHALLENGE_PATH}'


def cord_support_dir():
    return Path(__file__).parent / 'cordsupport'


DOCUMENT_VECTOR_PATH = cord_support_dir() / f'DocumentVectors_{DOCUMENT_VECTOR_LENGTH}.pq'
SIMILARITY_INDEX_PATH = str((cord_support_dir() / 'PaperSimilarity.ann').resolve())
SIMILARITY_INDEX = AnnoyIndex(DOCUMENT_VECTOR_LENGTH, 'angular')
SIMILARITY_INDEX.load(SIMILARITY_INDEX_PATH)
METADATA_LOOKUP = pd.read_csv(PurePath(cord_support_dir() / 'Metadata.csv.gz'))


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def listify(o):
    if isinstance(o, list):
        return o
    return [o]


# @lru_cache(maxsize=16)
def load_template(template):
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    template_file = os.path.join(template_dir, f'{template}.template')
    with open(template_file, 'r') as f:
        return Template(f.read())


def render_html(template_name, **kwargs):
    template = load_template(template_name)
    return template.render(kwargs)


def parallel(func, arr: Collection, max_workers: int = None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, multiprocessing.cpu_count())
    progress_bar = tqdm(arr)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures_to_index = {ex.submit(func, o): i for i, o in enumerate(arr)}
        results = []
        for f in as_completed(futures_to_index):
            results.append((futures_to_index[f], f.result()))
            progress_bar.update()
        for n in range(progress_bar.n, progress_bar.total):
            time.sleep(0.1)
            progress_bar.update()
        results.sort(key=lambda x: x[0])
    return [result for i, result in results]


def add(cat1, cat2):
    return cat1 + cat2


def describe_column(series):
    col_counts = series.describe().loc[['count', 'unique', 'top']]
    col_counts.loc['null'] = series.isnull().sum()
    col_counts['duplicate'] = series.dropna().duplicated().sum()
    df = col_counts.to_frame().T
    df = df[['count', 'null', 'unique', 'duplicate', 'top']] \
        .rename(columns={'count': 'non-null', 'top': 'most common'}).T
    return df


def describe_dataframe(df, columns=None):
    columns = columns or df.columns
    column_descs = [describe_column(df[col]).T for col in columns]
    return pd.concat(column_descs)


def show_common(data, column, head=20):
    common_column = data[column].value_counts().to_frame()
    common_column = common_column[common_column[column] > 1]
    return common_column.head(head)


# From http://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
def index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)


def split_df(df, chunk_size):
    indices = index_marks(df.shape[0], chunk_size)
    return np.split(df, indices)


def lookup_by_sha(shas, sha_map, not_found=[]):
    '''
        Lookup a value in the map by the sha, handling cases of multiple shas
    '''
    if not isinstance(shas, str): return not_found
    for sha in shas.split(';'):
        sha_value = sha_map.get(sha.strip())
        if sha_value is not None:
            return sha_value
    return not_found


def get_index(cord_uid):
    return METADATA_LOOKUP.index[METADATA_LOOKUP.cord_uid == cord_uid].values[0]


def similar_papers(paper_id, num_items=6):
    index = paper_id if isinstance(paper_id, int) else get_index(paper_id)
    similar_indexes = SIMILARITY_INDEX.get_nns_by_item(index, num_items)
    return similar_indexes


def image(image_path):
    from ipywidgets import Image
    from IPython.display import display
    with open(image_path, "rb") as f:
        display(Image(value=f.read()))