# Lab 2 - Text Pre-processing and Regular Expressions

Two parts. The first covers regular expressions with Python's `re` module. The second walks through the standard text pre-processing steps: tokenization, lower casing, stemming, lemmatization, and stop word removal with NLTK and spaCy.

## Files

| File | Content |
|------|---------|
| [Lab2_Text_Preprocessing_and_Regex.ipynb](Lab2_Text_Preprocessing_and_Regex.ipynb) | Original notebook |
| [notes.md](notes.md) | The whole lab as one readable document: explanations (including the regex symbol reference table), code and outputs |
| [code/01_search.py](code/01_search.py) | `re.search` |
| [code/02_match.py](code/02_match.py) | `re.match` and how it differs from `search` |
| [code/03_findall.py](code/03_findall.py) | `re.findall` |
| [code/04_sub.py](code/04_sub.py) | `re.sub` with flags and count |
| [code/05_tokenization.py](code/05_tokenization.py) | Sentence and word tokenization with NLTK, compared with `str.split` |
| [code/06_lower-casting.py](code/06_lower-casting.py) | Lower casing |
| [code/07_what-is-porterstemmer.py](code/07_what-is-porterstemmer.py) | Porter stemmer |
| [code/08_what-is-snowballstemmer.py](code/08_what-is-snowballstemmer.py) | Snowball stemmer |
| [code/09_comparing-between-porter-and-snowball.py](code/09_comparing-between-porter-and-snowball.py) | Porter vs Snowball on the same words |
| [code/10_lemmatization.py](code/10_lemmatization.py) | WordNet lemmatizer, with and without a part-of-speech tag |
| [code/11_stop-words-in-spacy.py](code/11_stop-words-in-spacy.py) | spaCy stop words: list them, check a word, add and remove stop words |
| [code/12_stop-words-in-nltk.py](code/12_stop-words-in-nltk.py) | NLTK stop words |

## Topics covered

- Regex use cases and the main symbols: anchors, quantifiers, character classes, sets
- `search`, `match`, `findall`, `sub`
- Tokenization (sentence and word), lower casing
- Stemming with Porter and Snowball; lemmatization with WordNet and POS tags
- Stop words: why to remove them, when not to (sentiment analysis), spaCy vs NLTK lists

## Tasks

1. Hashtags: using the [Apple Twitter Sentiment Texts](https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts) dataset, count how many hashtags there are and find the top 10.
2. `re.compile()`: on the text "This year is 2021", compile a pattern matching one or more digits, print its type, replace 2021 with 2022, and explain what `re.compile()` does.
3. `re.split()`: split "a 11 b 2 3 c 4" wherever one or more digits appear, print the list, and explain `re.split()`.
4. Tokenization with spaCy: load `en_core_web_sm`, print each token of "I'm enjoying the NLP course!" on its own line, explain `spacy.load()`, and compare with NLTK's tokenizer.

## Running

```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"
python code/01_search.py
```
