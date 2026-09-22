# Lab 3 - N-Grams

Building n-gram language models with NLTK: unigrams, bigrams and trigrams, how to count and estimate their probabilities, why padding is needed, training a Maximum Likelihood Estimation (MLE) model, and evaluating it with perplexity.

## Files

| File | Content |
|------|---------|
| [Lab3_N_Grams.ipynb](Lab3_N_Grams.ipynb) | Original notebook |
| [notes.md](notes.md) | The lecture notes with the probability formulas |
| [code/01_unigram.py](code/01_unigram.py) | Generate unigrams with `nltk.util.ngrams` |
| [code/02_calculating-probabilities-in-unigram-mod.py](code/02_calculating-probabilities-in-unigram-mod.py) | Count words and compute unigram probabilities |
| [code/03_bigram-method-1.py](code/03_bigram-method-1.py) | Bigrams with `nltk.bigrams` |
| [code/04_bigram-method-2.py](code/04_bigram-method-2.py) | Bigrams with `ngrams(tokens, 2)` |
| [code/05_trigram-method-1.py](code/05_trigram-method-1.py) | Trigrams with `nltk.trigrams` |
| [code/06_trigram-method-2.py](code/06_trigram-method-2.py) | Trigrams with `ngrams(tokens, 3)` |
| [code/07_what-is-padding-in-nlp.py](code/07_what-is-padding-in-nlp.py) | Padding with `pad_left`/`pad_right` symbols and with `pad_both_ends` |
| [code/08_training-the-maximum-likelihood-model.py](code/08_training-the-maximum-likelihood-model.py) | `padded_everygram_pipeline` and training an `MLE(2)` model |
| [code/09_calculating-probabilities.py](code/09_calculating-probabilities.py) | `lm.score()` for a word given its context |
| [code/10_perplexity.py](code/10_perplexity.py) | `lm.perplexity()` on a test set |
| [images/](images/) | Figure referenced from the notes |

## Topics covered

- Types of n-grams and their use cases (autocomplete, spell check, assistants)
- Unigram probability: count(A) / count(all words)
- Bigram probability: count(A B) / count(A)
- Padding: n-1 start and end symbols so every word has a context
- MLE training with `nltk.lm` and evaluating with perplexity (lower is better)

## Task: generating tweets with n-grams

1. Load the [Large Random Tweets from Pakistan](https://www.kaggle.com/datasets/adizafar/large-random-tweets-from-pakistan) dataset.
2. Pre-process: remove hashtags, "RT", websites, mentions, and emojis (`pip install emoji`).
3. Build a bigram MLE model.
4. Evaluate the model.
5. Compute the probability of the bigram "pakistan is".
6. Compute the perplexity of the word "pakistan".

## Running

```bash
pip install nltk emoji
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python code/08_training-the-maximum-likelihood-model.py
```
