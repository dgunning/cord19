import pickle
from functools import lru_cache
from pathlib import Path, PurePath
from annoy import AnnoyIndex
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
import requests
from typing import Dict, List
from .core import cord_support_dir, find_data_dir, cord_cache_dir

SPECTER_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16
RANDOM_STATE = 42

SPECTOR_PATH = Path(find_data_dir()) / f"cord19_specter_embeddings_2020-04-10/cord19_specter_embeddings_2020-04-10.csv"
SPECTOR_DIMENSIONS = 768


def kmean_labels(docvectors, n_clusters=6, random_state=RANDOM_STATE):
    print('Setting cluster labels')
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(docvectors)
    return kmeans.labels_


def tsne_embeddings(docvectors, dimensions=2):
    print(f'Creating {dimensions}D  embeddings')
    from sklearn.manifold import TSNE
    tsne = TSNE(verbose=1,
                perplexity=15,
                early_exaggeration=24,
                n_components=dimensions,
                n_jobs=8,
                random_state=RANDOM_STATE,
                learning_rate=600)
    embeddings = tsne.fit_transform(docvectors)
    return embeddings


@lru_cache(maxsize=4)
def tsne_model(dimension):
    print('Loading TSNE model for', dimension, 'dimensions')
    TSNE_Path = Path(cord_support_dir()) / f'TSNE{dimension}d.pickle'
    with TSNE_Path.open('rb') as f:
        return pickle.load(f)


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
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(SPECTER_URL, json=chunk)

        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


def get_embeddings(title: str, abstract: str = None):
    abstract = abstract or title
    paper = {"paper_id": "paper", "title": title, "abstract": abstract}
    embeddings = get_embeddings_for_papers([paper])
    return embeddings['paper']


def load_specter_embeddings():
    print('Loading specter embeddings')
    VECTOR_COLS = [str(i) for i in range(768)]
    COLUMNS = ['cord_uid'] + VECTOR_COLS
    df = pd.read_csv(SPECTOR_PATH, names=COLUMNS).set_index('cord_uid')
    print('Loaded embeddings of shape', df.shape)
    return df


SPECTOR_INDEX_PATH = str((cord_cache_dir() / 'SpectorSimilarity.ann').resolve())
SPECTOR_SIMILARITY_INDEX = AnnoyIndex(SPECTOR_DIMENSIONS, 'angular')
SPECTOR_SIMILARITY_INDEX.load(SPECTOR_INDEX_PATH)
DOCUMENT_VECTOR_PATH = cord_support_dir() / f'DocumentVectors.pq'
document_vectors = pd.read_parquet(DOCUMENT_VECTOR_PATH)


def find_similar_papers(search_string, num_items=10, covid_related=True):
    vector = get_embeddings(search_string, search_string)
    similar_indexes = SPECTOR_SIMILARITY_INDEX.get_nns_by_vector(vector, n=num_items)
    similar_cord_uids = document_vectors.iloc[similar_indexes].index.values.tolist()
    return similar_cord_uids