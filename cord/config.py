

class Config:

    def __init__(self):
        self.document_vector_length = 192
        self.spector_csv_path = "cord_19_embeddings_4_17/cord_19_embeddings_4_17.csv"
        self.num_document_clusters = 6
        self.num_annoy_trees = 30
        self.num_similar_items = 10
        self.search_k = self.num_annoy_trees * self.num_similar_items * 2  # more accuracy
        self.annoy_index_path = f'DocumentIndex{self.document_vector_length}.ann'
        self.document_vector_path = 'DocumentVectors.pq'
        self.spector_url = "https://model-apis.semanticscholar.org/specter/v1/invoke"
