# Lab 3 - N-Grams
# Section: Method 2
# Extracted from Lab3_N_Grams.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

string = "moses supposes his toeses are roses but moses supposes erroneously"

bigrams = ngrams(word_tokenize(string),2)

for i in bigrams:
    print(i)
