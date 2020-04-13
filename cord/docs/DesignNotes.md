## Technical Design Notes

The **CORD Research Engine** is a tool for locating research papers and surfacing insights on the **SARS-COV-2** virus
and the disease that it causes **COVID-19**. It follows a Search Engine design metaphor, since it is a design that is
well known by non-technical users. The tool is designed to help a user quickly assemble readable reports on COVID-19 areas
 and there is a heavy focus on ease of use. All of this is designed to run inside Jupyter notebooks, meaning data discovery, assembly and output will all run inside of
 a Kaggle or Jupyter notebook, rather than a separate web based took. This is mean to significantly increase the speed of information
 retrieval and presentation, and gets around the significant drawback of web tools which cannot easily produce ad-hoc reports 
 or render in multiple formats in the same way that notebooks can.

There are two search indexes - a **BM25 based Search index**, and a **Similarity Index** built from document vectors.
These two search indexes complement each other and are used for certain purposes.

### Search Index

BM25 is chosen because it is a robust, well-known set of algorithms for text search and information extract used by
popular search engine technology. It is the default algorithm set used in **ElasticSearch**.

We use a simple implementation of BM25 provided by the **rank_bm25** Python library. The index is populated with the preprocessed
and tokenized documents. Internally **rank_bm25** lemmatizes the tokens, and performs other optimizations required
for good search performance.
 
<br/>
#### Indexing Abstracts vs Paper Content
The CORD library provides the option of indexing the *metadata abstracts* vs indexing the *full paper content*. 
The trade-off here is speed vs accuracy, with indexing the abstracts resulting in a load time of about a minute,
while indexing the text content taking a few minutes on Kaggle. Generally, when we are performing research we may want
to use the full text index, while for other use cases we can just use the index from the abstracts. In practice, 
the difference in accuracy is not as severe as you would think - we do have the second, more powerful similarity index 
built from the vectors built on the full paper content.


### Similarity Index
The similarity index is built by down-sampling the 768-dimension specter document vectors to 100 dimensions and adding them
to an **Annoy** index. This annoy index will then return the most similar papers to a given document. 
Annoy is a simple and very performant library for nearest neighbour search - originally developed at Spotify.

We downsample to reduce the size of the Annoy index, and 100 dimensions gives us a 40 MB index. 
Downsampling is done using **PCA**. We chose to use PCA over **TSNE** since it is much faster.
In addition to downsampling to 100 dimensions, we also add **1d** and **2d** vectors. The 1d and 2d vectors are used for visualizing 
where a paper fits in the overall vector space, in the function `papers.search_2d`. 

<br/>

### Search vs Similarity in subsets of Research Papers
The Search and Similarity indexes also operate differently when we subset research papers. If for example, 
we subset ResearchPaper as follows  `papers.since_sarscov2()` then searches will happen on just that subset.
Similarity will still happen on the entire dataset, since a user will likely want to get the papers that are most similar
to a given one regardless of the subset criteria.

```python
papers = ResearchPapers.load()
```