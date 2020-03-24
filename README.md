# COVID 19 Open Research Data Challenge Code
This repo contains code and notebooks for the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
on kaggle


## Usage

```{python}
from cord import ResearchPapers

#data_dir = 'data'
data_dir = '..\input'

research_papers = ResearchPapers.from_data_dir(data_dir)
```
