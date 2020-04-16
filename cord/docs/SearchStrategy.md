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