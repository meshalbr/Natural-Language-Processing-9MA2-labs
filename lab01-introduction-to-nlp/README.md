# Lab 1 - Introduction to Natural Language Processing

An overview lab: what NLP is, why it is hard, common text file types, the main NLP libraries, typical applications, and a first look at an NLP pipeline. The hands-on part installs NLTK and loads a movie-review dataset with pandas.

## Files

| File | Content |
|------|---------|
| [Lab1_Introduction_to_NLP.ipynb](Lab1_Introduction_to_NLP.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab1_Introduction_to_NLP.py](Lab1_Introduction_to_NLP.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, figures, code, outputs) |
| [images/](images/) | Figures used in the lab |
| [dataset/README.md](dataset/README.md) | Course link to the dataset files |

## Topics covered

- Definition of NLP and its raw materials: data, compute, algorithms
- Why NLP is hard: ambiguity, spelling errors, phonetics, word order
- Common text sources: CSV, text files, JSON, SQL, HTML, PDF, DOCX
- NLTK and the idea of a corpus; Arabic libraries (Camel-Tools, PyArabic, Farasa)
- NLP applications: information extraction, machine translation, sentiment analysis
- The NLP pipeline: acquisition, preprocessing, feature extraction, model building
- Loading the IMDB reviews dataset with `kagglehub` and inspecting it with pandas (`head`, `describe`, `info`, class distribution)

## Tasks

1. In groups, search online for a dataset in one of the file types above (CSV, text, JSON, SQL, HTML, PDF, DOCX).
2. Download and open the dataset.

## Running

```bash
pip install nltk pandas kagglehub
jupyter notebook Lab1_Introduction_to_NLP.ipynb
```

The dataset used in the notebook is the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Place `IMDB Dataset.csv` next to the notebook or update the path in the loading cell.
