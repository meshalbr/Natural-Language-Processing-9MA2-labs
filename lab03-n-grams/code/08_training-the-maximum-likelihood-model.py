# Lab 3 - N-Grams
# Section: Training the Maximum Likelihood Model
# Extracted from Lab3_N_Grams.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.lm import smoothing

string = "moses supposes his toeses are roses but moses supposes erroneously"

#Create Bigram Language Model
train, vocab = padded_everygram_pipeline(2, [word_tokenize(string)])

# %% cell 2
from nltk.lm import MLE
lm = MLE(2)
lm.fit(train, vocab) #Train the Maximum Likelihood Model
