# Natural Language Processing (9MA2) - Labs

Lab work for the Natural Language Processing course (section 9MA2) at Imam Abdulrahman Bin Faisal University (IAU).
Each lab lives in its own folder with the notebook (run it cell by cell), a Markdown version of the whole lab, and the figures and datasets it uses.

## Labs

| # | Folder | Topic | Main tools |
|---|--------|-------|------------|
| 1 | [lab01-introduction-to-nlp](lab01-introduction-to-nlp/) | What NLP is, why it is hard, the NLP pipeline, NLTK and corpora, loading a first dataset | `nltk`, `pandas`, `kagglehub` |
| 2 | [lab02-text-preprocessing-and-regex](lab02-text-preprocessing-and-regex/) | Regular expressions, tokenization, lower casing, stemming, lemmatization, stop words | `re`, `nltk`, `spacy` |
| 3 | [lab03-n-grams](lab03-n-grams/) | Unigrams, bigrams, trigrams, padding, MLE language model, perplexity | `nltk.lm`, `nltk.util.ngrams` |
| 4 | [lab04-classification-and-evaluation](lab04-classification-and-evaluation/) | Sentiment analysis with TF-IDF features and an SVM classifier, evaluation metrics | `scikit-learn`, `pandas` |
| 5 | [lab05-text-representation](lab05-text-representation/) | TF-IDF, cosine similarity, Word2Vec (skip-gram) embeddings | `scikit-learn`, `gensim` |

## Folder layout

Every lab folder follows the same structure:

```
labNN-topic/
├── README.md          what the lab covers, the tasks, and how to run it
├── LabN_Topic.ipynb   the lab notebook: text, code and outputs, run cell by cell
├── LabN_Topic.py      the same notebook as one Python file, cell by cell (# %% cells)
├── notes.md           the whole lab as one readable Markdown document (text, code, outputs)
├── images/            figures used in the lab (only where the lab has any)
└── dataset/           dataset files or a link to download them (only where needed)
```

The notebook is the main file. Open it in VS Code or Jupyter and run each cell with Shift+Enter; the outputs from the lab session are saved inside it. The `.py` file is the same notebook in VS Code's cell format: open it and press Shift+Enter on any `# %%` cell to run it in the interactive window. `notes.md` is the same content as plain Markdown, so it reads well on GitHub without opening the notebook.

## Setup

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"
```

Then open a notebook, for example:

```bash
jupyter notebook lab02-text-preprocessing-and-regex/Lab2_Text_Preprocessing_and_Regex.ipynb
```

## Datasets

| Lab | Dataset | Where to get it |
|-----|---------|-----------------|
| 1 | IMDB Dataset of 50K Movie Reviews | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or the course link in `lab01-introduction-to-nlp/dataset/README.md` |
| 2 (task) | Apple Twitter Sentiment Texts | [Kaggle](https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts) |
| 3 (task) | Large Random Tweets from Pakistan | [Kaggle](https://www.kaggle.com/datasets/adizafar/large-random-tweets-from-pakistan) |
| 4 | Disneyland Reviews | [Kaggle](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) |
| 4 (task) | Amazon Reviews of Unlocked Mobile Phones | [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones) |
| 5 | Fake and Real News (True.csv) | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| 5 (task) | The Simpsons script lines | included in `lab05-text-representation/dataset/simpsons_script_lines.csv` |

Large Kaggle datasets are not committed to this repo. Download them and adjust the file path in the loading cell (the notebooks were written for Google Colab and use `/content/...` paths).

## Sources

The lab notebooks were provided by the course instructor and originally published in these repositories:

- Lab 1: https://github.com/MohammedMosuily/NLP_Lab1
- Lab 2: https://github.com/MohammedMosuily/NLP_Lab2
- Lab 3: https://github.com/MohammedMosuily/NLP_Lab3
- Lab 4: https://github.com/MohammedMosuily/NLP_Lab4
- Lab 5: https://github.com/MohammedMosuily/NLP_Lab5
