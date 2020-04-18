from pathlib import Path
from typing import Dict, List

import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from annoy import AnnoyIndex

from .config import Config
from .core import cord_support_dir, find_data_dir

config = Config()

SPECTER_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16
RANDOM_STATE = 42
NUM_ANNOY_TREES = 30
SPECTOR_CSV_PATH = Path(find_data_dir()) / config.spector_csv_path
DOCUMENT_VECTOR_LENGTH = 192


def kmean_labels(docvectors, n_clusters=6, random_state=RANDOM_STATE):
    print('Setting cluster labels')
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(docvectors)
    return kmeans.labels_


import altair as alt


def chartEmbeddings2D(embeddings, width=500, height=350, color_column='color'):
    chart = alt.Chart(embeddings).mark_circle(opacity=0.4, size=80).encode(
        x=alt.X('x', axis=None),
        y=alt.Y('y', axis=None),
        color=alt.Color(color_column, scale=alt.Scale(scheme='set1'))
    ).properties(
        width=width,
        height=height,
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )
    return chart


def show_2d_chart(results, query=''):
    # Get the records from the metadata_coord df that match
    cord_uids = results.cord_uid.to_list()
    cord_matches = document_vectors.loc[cord_uids].copy()
    cord_matches['Search Results'] = 'Matches'

    title_lookup = results.set_index('cord_uid').to_dict()['title']

    chart_data = pd.concat([cord_matches, document_vectors.sample(800).copy()], sort=True).reset_index()
    chart_data['Search Results'] = chart_data['Search Results'].fillna('Non Matches')
    chart_data['Title'] = chart_data.cord_uid.apply(lambda id: title_lookup.get(id, ''))

    display(widgets.HTML(f'<h4>{query}</h4>'))

    chart = alt.Chart(chart_data).mark_circle(opacity=0.4, size=80).encode(
        x=alt.X('x', axis=None),
        y=alt.Y('y', axis=None),
        color=alt.Color('Search Results', scale=alt.Scale(scheme='set1'))
    ).properties(
        width=500,
        height=350,
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    )
    display(chart)


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def get_embeddings_for_papers(papers: List[Dict[str, str]]):
    """
    Get the specter embeddings for the paper

    :param papers: A list of dictionaries of the form
                    paper = {"paper_id": "paper", "title": title, "abstract": abstract}
    :return: a 768 dimension vector from the specter embedding url
    """
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(config.spector_url, json=chunk)

        if response.status_code != 200:
            print("Something went wrong on the spector API side .. try again")
            return None

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


def get_embeddings(title: str, abstract: str = None):
    """
    :param title: The title of the paper
    :param abstract: The abstract of the paper. Will be set to the title if not provided
    :return: The embeddings for the single paper
    """
    abstract = abstract or title
    paper = {"paper_id": "paper", "title": title, "abstract": abstract}
    embeddings = get_embeddings_for_papers([paper])
    return embeddings['paper'] if embeddings else None


def load_specter_embeddings():
    """
    Load the specter embeddings from the specter csv path
    :return:
    """
    print('Loading specter embeddings')
    VECTOR_COLS = [str(i) for i in range(768)]
    COLUMNS = ['cord_uid'] + VECTOR_COLS
    df = pd.read_csv(SPECTOR_CSV_PATH, names=COLUMNS).set_index('cord_uid')
    print('Loaded embeddings of shape', df.shape)
    return df


SPECTOR_INDEX_PATH = str((cord_support_dir() / config.annoy_index_path).resolve())
print(SPECTOR_INDEX_PATH)
SPECTOR_SIMILARITY_INDEX = AnnoyIndex(config.document_vector_length, 'angular')
SPECTOR_SIMILARITY_INDEX.load(SPECTOR_INDEX_PATH)
DOCUMENT_VECTOR_PATH = cord_support_dir() / config.document_vector_path
document_vectors = pd.read_parquet(DOCUMENT_VECTOR_PATH)


def get_index(cord_uid):
    row_match = np.where(document_vectors.index == cord_uid)
    if len(row_match[0]) > 0:
        return np.where(document_vectors.index == cord_uid)[0][0]


def similar_papers(paper_id, num_items=config.num_similar_items):
    from .vectors import SPECTOR_SIMILARITY_INDEX
    index = paper_id if isinstance(paper_id, int) else get_index(paper_id)
    if not index:
        return []
    similar_indexes = SPECTOR_SIMILARITY_INDEX.get_nns_by_item(index, num_items, search_k=config.search_k)
    similar_cord_uids = document_vectors.iloc[similar_indexes].index.values.tolist()
    return [id for id in similar_cord_uids if not id == paper_id]
