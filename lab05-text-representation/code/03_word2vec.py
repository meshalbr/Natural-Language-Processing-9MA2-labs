# Lab 5 - Text Representation
# Section: 2- What is Word2vec?
# Extracted from Lab5_Text_Representation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import pandas as pd
import nltk
import numpy as np
import gensim

from nltk.tokenize import word_tokenize

# %% cell 2
df = pd.read_csv('/content/True.csv', engine='python', on_bad_lines='skip') #fake-and-real-news-dataset-true.csv
df.head()

# %% cell 3
tokens = []

for i in df['text']:
    token = word_tokenize(i)
    tokens.append(token)

# %% cell 4
w2v = gensim.models.Word2Vec(tokens, min_count=1, vector_size=100, window=5, sg=1)

# %% cell 5
print("Cosine similarity between 'provide' and 'program' - Skip Gram : ",w2v.wv.similarity('provide', 'program'))

# %% cell 6
print("words that similar to 'program' - Skip Gram : ",w2v.wv.most_similar('program'))
