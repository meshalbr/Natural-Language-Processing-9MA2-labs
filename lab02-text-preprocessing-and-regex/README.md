# Lab 2 - Text Pre-processing and Regular Expressions

Two parts. The first covers regular expressions with Python's `re` module. The second walks through the standard text pre-processing steps: tokenization, lower casing, stemming, lemmatization, and stop word removal with NLTK and spaCy.

## Files

| File | Content |
|------|---------|
| [Lab2_Text_Preprocessing_and_Regex.ipynb](Lab2_Text_Preprocessing_and_Regex.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs, including the regex symbol reference table) |

## Topics covered

- Regex use cases and the main symbols: anchors, quantifiers, character classes, sets
- `re.search`, `re.match`, `re.findall`, `re.sub`
- Tokenization (sentence and word) with NLTK, compared with `str.split`; lower casing
- Stemming with Porter and Snowball, and a comparison of the two
- Lemmatization with WordNet, with and without a part-of-speech tag
- Stop words: why to remove them, when not to (sentiment analysis), spaCy vs NLTK lists, adding and removing stop words in spaCy

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
jupyter notebook Lab2_Text_Preprocessing_and_Regex.ipynb
```
