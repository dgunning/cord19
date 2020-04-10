
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
