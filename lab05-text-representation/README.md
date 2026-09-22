# Lab 5 - Text Representation

How to turn text into numbers a model can use. Two techniques are covered: TF-IDF vectors (with cosine similarity to compare documents) and Word2Vec word embeddings trained with gensim in skip-gram mode.

## Files

| File | Content |
|------|---------|
| [Lab5_Text_Representation.ipynb](Lab5_Text_Representation.ipynb) | Original notebook |
| [notes.md](notes.md) | The lecture notes with the TF-IDF and cosine similarity formulas |
| [code/01_tf-idf.py](code/01_tf-idf.py) | `TfidfVectorizer` on three short documents, shown as a DataFrame |
| [code/02_cosine-similarity.py](code/02_cosine-similarity.py) | TF-IDF vectors plus `cosine_similarity` between documents |
| [code/03_word2vec.py](code/03_word2vec.py) | Tokenize a news dataset, train a skip-gram `Word2Vec`, query `wv.similarity` and `wv.most_similar` |
| [dataset/simpsons_script_lines.csv](dataset/simpsons_script_lines.csv) | The Simpsons script lines used in the tasks (columns: `raw_character_text`, `spoken_words`) |

## Topics covered

- Embeddings: representing words as real-valued vectors
- TF-IDF: term frequency times inverse document frequency, discounting common words
- Cosine similarity: dot product over the product of vector lengths, 1 = same direction, 0 = orthogonal
- Word2Vec: two-layer neural network, CBOW vs skip-gram architectures

## Tasks

1. Cosine similarity: compute how similar these sentences are: "This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"
2. TF-IDF: find the important words in "data science is one of the most important fields of science", "this is one of the best data science courses", "data scientists analyze data".
3. Word2Vec: load `dataset/simpsons_script_lines.csv`, clean the `spoken_words` column with the given `clean_text` function (lowercase, remove digits and punctuation), tokenize, and train `gensim.models.Word2Vec(tokens, min_count=1, vector_size=100, window=5, sg=1)`.
4. With `wv.most_similar()`, find the words most similar to "homer", "marge" and "bart".
5. With `wv.doesnt_match()`, find the odd one out in ['jimbo', 'milhouse', 'kearney'], ['nelson', 'bart', 'milhouse'] and ['homer', 'patty', 'selma'].

## Running

```bash
pip install pandas scikit-learn nltk gensim
python code/01_tf-idf.py
```

The Word2Vec example loads `True.csv` from the [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) dataset (the notebook uses the Colab path `/content/True.csv`). The Simpsons dataset for the tasks is included in `dataset/`.
