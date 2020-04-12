# Technical Notes

This notebook builds a simple yet powerful search engine on top of the CORD19 research papers, allowing for the rapid narrowing of search results to the most promising papers for each query.

The core technology is based on the BM25 search algorithm with more advanced NLP sprinkled in. 

The notebook and the library upon which it is built uses more advanced NLP techniques such as **summarization** and **entity recognition** only to assist a user to find exactly the right solutions in the right papers as quickly as they can. 

The code for this notebook is maintained at https://github.com/dgunning/cord19. For technical details, including code walkthrough, see the Technical Notes section

### BM25 Search Index
We extract the paper texts from the JSON file and convert them to word tokens for each paper. This document tokens are then used to create a **BM25** index using the **rank_bm25** python library. The sarch index is used to find papers that match a given search term. Search is used as the first screening of papers that match a given search term, and given that search terms are usually short, we find this a reasibable approach. 

### Similarity Index
The document token used in the step above are used to train a gensim **Doc2Vec** model. This model is then used to create document vectors of length 20 for each document. These vectors are then added to an **Annoy** index, which returns the papers which are most similar to the a given paper. We chose to use this approach for similarity since here we are comparing the full text content and document vectors gives us dimensionality reduction. That being said, we are considering using the document vectors/Annoy index for search.

## Technical Design

For current details on the design of the **cord** library, check the project on [github/dgunning/cord19](https://github.com/dgunning/cord19)

The **ResearchPapers** class is a container for the metadata, and the BM25 search index. It contains functions to find papers using **index** `research_papers[0]`,  **cord_uid** `research_papers["4nmc356g"]`, **OR** to create subsets of ResearchPapers like `papers.since_sarscov2`, **OR** to run `search()` or display the `searchbar()`

Because the ResearchPapers class is simply a container for the metadata dataframe, and all useful information about each paper is on the dataframe as a column, including the **index_tokens**, tags such as **covid_related** etc, subsetting ResearchPapers is simply a matter of subsetting the **metadata** dataframe, then creating a new ResearchPapers instance. To create a ResearchPapers instance after a date means 

    ::python
    def after(self, date, include_null_dates=False):
        cond = self.metadata.published >= date
        if include_null_dates:
            cond = cond | self.metadata.published.isnull()
        return self._make_copy(self.metadata[cond])
    

Thus, we implement functions such as **head**, **tail**, **sample**, **query**, which just delegate to the metadata dataframe function and then create a new ResearchPapers instance.


### Load Metadata

    ::python
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

### Clean Metadata

    ::python
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


3. **Create the BM25 Search Index**

**ResearchPapers** can be indexed with the metadata *abstracts* OR with the *text* content of the paper. Indexing from the abstracts is straightforward - we just apply a **preprocess** function to clean and tokenize the abstract. Indexing from the texts - if no json-cache exists - happens by loading the JSON files and, tokenizing the texts and setting the **index_tokens** on the metadata. However, there is now a **json_cache** dataset comprised of the preprocessed text tokens, along with the JSOn file's sha - which we use to merge into the metadata.

After the metadata is loaded and cleaned we create the **BM25** index inside of **ResearchPapers.__init__()**

    ::python
    if 'index_tokens' not in metadata:
        print('\nIndexing research papers')
        if any([index == t for t in ['text', 'texts', 'content', 'contents']]):
            _set_index_from_text(self.metadata, data_dir)
        else:
            print('Creating the BM25 index from the abstracts of the papers')
            print('Use index="text" if you want to index the texts of the paper instead')
            tick = time.time()
            self.metadata['index_tokens'] = metadata.abstract.apply(preprocess)
            tock = time.time()
            print('Finished Indexing in', round(tock - tick, 0), 'seconds')
