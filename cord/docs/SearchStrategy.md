## Search Strategy

For each of the questions in a topic we may create an alternative phrasing of the question to improve search performance
by reducing ambiguity, eliminating redundant words, or adding terms with a known relevance to the topic. This called a 
**SeedQuestion**. We use the SeedQuestion to search.

The **SeedQuestion** and the **OriginalQuestion** are maintained in the **Tasks** object, which is loaded from the 
**TaskDefinitions.csv** file. Thus we can refer launch a searchbar using 
`papers.searchbar(Tasks.Vaccines[11].SeedQuestion, num_results=50)`.

If we don't get good initial results we may find a paper or an article from some other source e.g. web search and then
use a paragraph from it to find papers related to that topic. If we find good search results we may then modify the SeedQuestion
based on terms that lead to good results, or use `papers.similar_to` to find papers that best match the topic.

Since we intend the tool to be a research engine, we find this to be a good approximation of what a research would do.
Once a user finds what they like they can assemble the report using
`papers.display('95r8swye', 'odcteqg8', 't8bobwzg')` by copying the **cord_uids**. For now the cord_uids are exposed, 
but we plan to add user interface widgets to make it more convenient. Regardless, research results can be assembled very rapidly.


## Search Technology BM25 vs Specter Search
We currently use **BM25** to index the preprocessed tokens of the full research paper content (with option index='text') or abstract.
We found **BM25** to be very accurate, returning relevant papers almost all the time.

We also implemented a search strategy that takes a search query, calls the **Specter API** to convert that query into a Specter Vector, 
then use the **Similarity index** to find related papers. We found this approach performed poorly compared with BM25.
For example, in one search for papers related to *"Artificial intelligence or machine learning"* **BM25** returned papers that refer to 
**AI** or **ML**, while the specter vector search returned papers with the general use of *"learning"*. BM25 algorithms have also been tuned
over the past 26 years to not have affinity to results which are too short, which seemed to be a shortcoming of the specter search.

On the other hand, the specter vectors were very accurate with full paper similarity, likely because this gives enough information to fill 
the **768 dimensions** and properly separate the papers in vector space.

