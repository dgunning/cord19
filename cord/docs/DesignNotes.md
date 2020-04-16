## Design Notes

The **CORD Research Engine** is a tool for locating research papers and surfacing insights on the **SARS-COV-2** virus
and the disease that it causes **COVID-19**. It follows a Search Engine design metaphor, since it is a design that is
well known by non-technical users. The tool is designed to help a user quickly assemble readable reports on COVID-19 areas
 and there is a heavy focus on ease of use. All of this is designed to run inside Jupyter notebooks, meaning data discovery, assembly and output will all run inside of
 a Kaggle or Jupyter notebook, rather than a separate web based took. This is mean to significantly increase the speed of information
 retrieval and presentation, and gets around the significant drawback of web tools which cannot easily produce ad-hoc reports 
 or render in multiple formats in the same way that notebooks can.

There are two search indexes - a **BM25 based Search index**, and a **Similarity Index** built from the 768 dimension 
document vectors. These two search indexes complement each other and are used for certain purposes.

### Search Index

BM25 is chosen because it is a robust, well-known set of algorithms for text search and information extract used by
popular search engine technology. It is the default algorithm set used in **ElasticSearch**.

Interestingly we found that the **BM25** search index significantly outperformed the **Specter document vectors** on normal search queries.
However Specter vectors performed very well when comparing full documents. The reason could be that BM25 is very well tuned for short queries
while short queries when vectorized into 768 dimensions are too sparse to be accurate. On the other hand with full document text the
vectors have enough information in each dimension to be accurate.


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
The similarity index is built by adding the 768-dimension specter document vectors to an **Annoy** index. 
This annoy index will then return the most similar papers to a given document. 
Annoy is a simple and very performant library for nearest neighbour search - originally developed at Spotify.

In addition to the 768 dimension vector we use PCA to downsample to **1d** and **2d** vectors.
The 1d and 2d vectors are used for visualizing where a paper fits in the overall vector space, in the function `papers.search_2d`. 
This gives the user a nice visual context about the topic they are currently looking at

<br/>

### Search vs Similarity in subsets of Research Papers
The Search and Similarity indexes also operate differently when we subset research papers. If for example, 
we subset ResearchPaper as follows  `papers.since_sarscov2()` then searches will happen on just that subset.
Similarity will still happen on the entire dataset, since a user will likely want to get the papers that are most similar
to a given one regardless of the subset criteria.


## Summarizing Research Papers

We use **gensim's TextRank summarizer** to create a summary of the paper's abstract for display in the search results.
This can be easily switched to being a summary of the paper's content - indeed there is a function in **cord.text** 
that can summarize any text, but for now we use the abstract.

## Selecting Subsets of Research Papers

Because the **ResearchPapers** class is a wrapper around the dataframe loaded from **metadata.csv**, we can create subsets
of ResearchPapers by subsetting the dataframe and creating a new ResearchPapers instance. This gives us the ability to
provide convenience functions for selecting only sets of research papers that a user might be interested in. This includes

- **Papers since SARS** `research_papers.since_sars()`
- **Papers since SARS-COV-2** `research_papers.since_sarscov2()`
- **Papers before SARS** `research_papers.before_sars()`
- **Papers before SARS-COV-2** `research_papers.before_sarscov2()`
- **Papers before a date** `research_papers.before('1989-09-12')`
- **Papers after a date** `research_papers.after('1989-09-12')`
- **Papers that contains a string** `research_papers.contains("Fauci", column='authors')`
- **Papers that match a string** (using regex) `research_papers.match('H[0-9]N[0-9]')`