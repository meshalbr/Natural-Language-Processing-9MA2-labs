# Lab 3 - N-Grams

Building n-gram language models with NLTK: unigrams, bigrams and trigrams, how to count and estimate their probabilities, why padding is needed, training a Maximum Likelihood Estimation (MLE) model, and evaluating it with perplexity.

## Files

| File | Content |
|------|---------|
| [Lab3_N_Grams.ipynb](Lab3_N_Grams.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab3_N_Grams.py](Lab3_N_Grams.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, formulas, code, outputs) |
| [images/](images/) | Figure used in the lab |

## Topics covered

- Types of n-grams and their use cases (autocomplete, spell check, assistants)
- Unigram probability: count(A) / count(all words)
- Bigram probability: count(A B) / count(A); bigrams and trigrams with `nltk.bigrams`, `nltk.trigrams` and `ngrams(tokens, n)`
- Padding: n-1 start and end symbols so every word has a context (`pad_left`/`pad_right`, `pad_both_ends`)
- MLE training with `padded_everygram_pipeline` and `nltk.lm.MLE`, scoring with `lm.score()`
- Evaluating with `lm.perplexity()` (lower is better)

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
jupyter notebook Lab3_N_Grams.ipynb
```
