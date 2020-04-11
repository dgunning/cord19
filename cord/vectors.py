import pickle
from functools import lru_cache
from pathlib import Path, PurePath

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from .core import cord_support_dir, document_vectors

RANDOM_STATE = 42


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


def to_2d(vector):
    tsne = tsne_model(2)


def to_1d(vector):
    pass


import altair as alt


def chartEmbeddings2D(embeddings, width=500, height=350, color_column='color'):
    chart = alt.Chart(embeddings).mark_circle(opacity=0.4, size=80).encode(
        x=alt.X('x', axis=None),
        y=alt.Y('y', axis=None),
        color = alt.Color(color_column, scale=alt.Scale(scheme='set1'))
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
