import pytest
from cord.cord19 import ResearchPapers

def test_load_research_papers():
    print('Loading research papers')
    reseearch_paper = ResearchPapers.load()