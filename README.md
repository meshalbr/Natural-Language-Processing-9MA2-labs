# Natural Language Processing (9MA2) - Labs

Lab work for the Natural Language Processing course (section 9MA2) at Imam Abdulrahman Bin Faisal University (IAU).
Each lab lives in its own folder with the original notebook, the lecture notes as Markdown, and the code split into small, runnable Python files.

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
├── LabN_Topic.ipynb   the original notebook as given in the lab
├── notes.md           the whole lab as one readable document: explanations, code and outputs
├── code/              the notebook's code split into small numbered .py files
│   ├── 01_....py
│   └── 02_....py
├── images/            figures used in the notes (only where the lab has any)
└── dataset/           dataset files or a link to download them (only where needed)
```

The files in `code/` are numbered in the order they appear in the notebook. Each file starts with a comment naming the section it comes from, and `# %%` markers separate the original cells so the files also run cell-by-cell in VS Code or Spyder.

## Setup

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"
```

Open any notebook with Jupyter, or run a code file directly, for example:

```bash
jupyter notebook lab02-text-preprocessing-and-regex/Lab2_Text_Preprocessing_and_Regex.ipynb
python lab02-text-preprocessing-and-regex/code/01_search.py
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
