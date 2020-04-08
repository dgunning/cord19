import streamlit as st
from cord import ResearchPapers


@st.cache(suppress_st_warning=True)
def load_research_papers():
    return ResearchPapers.load().since_sarscov2()


@st.cache
def search(search_terms):
    search_results = research_papers.search(search_terms, view='df')
    return search_results.get_results_df()[['summary', 'when', 'authors']]


research_papers = load_research_papers()

st.write("""
# CORD-19 Research Papers
""")
search_terms = st.text_input('Search', 'cruise ship coronavirus')
search_results = search(search_terms)
search_results
