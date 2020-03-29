# COVID 19 Open Research Data Challenge Code
This repo contains code and notebooks for the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
on kaggle

## Installing

```{bash}
pip install git+https://github.com/dgunning/cord19.git
```

###
Login to **Kaggle** and download the [CORD Research Challenge data](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
and extract to a folder called **data**. 

You can also use the [Kaggle python api](https://github.com/Kaggle/kaggle-api) to download this datatet.

```bash
dir data\CORD-19-research-challenge
```
![Cord Searchbar](images/datadir.png)

## Usage

```{python}
from cord import ResearchPapers

research_papers = ResearchPapers.load()
```

### Search Bar
```{python}
research_papers.searchbar('vaccine transmission')
```
![Cord Searchbar](images/searchbar.png)
