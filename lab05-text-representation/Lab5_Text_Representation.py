# Lab 5 - Text Representation
# The notebook Lab5_Text_Representation.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Text Representation

# %% [markdown]
# Text representation in NLP means converting text into a numerical form that a machine learning model can understand and process.

# %% [markdown]
# ## What is Embedding?
#
# Embeddings in NLP is a technique where individual words are represented as real-valued vectors and captures inter-word semantics.
#
#
# In this notebook, We going to intreduce 2 techniques for embedding. These techniques will be used for a machine learning models such as SVM, Random forest, ... ect.

# %% [markdown]
# ![](images/img01.png)

# %% [markdown]
# # 1- TF-IDF
#
# ![](images/img02.png)

# %% [markdown]
# TF-IDF stands for term frequency-inverse document frequency. It is a measure that discounts common words. Used in the fields of information retrieval (IR) and machine learning, that can quantify the importance of words in a document amongst a collection of documents (also known as a corpus).

# %% [markdown]
# ### Components of TF-IDF
#
# 1. **TF (Term Frequency):**
#    Measures how frequently a term appears in a document.
#
#
# $$ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} $$
#
#
# 2. **IDF (Inverse Document Frequency):**
#    Measures how important a term is across all documents. Words that appear in many documents get lower scores.
#
# $$    \text{IDF}(t) = \log \left(\frac{\text{Number of all documents N}}{\text{Number of documents containing the term } t}\right) $$

# %% [markdown]
# ![](images/img03.png)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
doc_1 = "Data is the oil of the digital economy"
doc_2 = "Data is a new oil"

data = [doc_1, doc_2]

# %%
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(data) # returns sparce matrix

# %%
df = pd.DataFrame(result.toarray(), columns=tfidf.get_feature_names_out())
df

# %% [markdown]
# # Cosine similarity
#
# In NLP, Cosine similarity is a metric used to measure how similar the documents are.
#
#
# $$ \text{cosine similarity} = \frac{A \cdot B}{\|A\| \times \|B\|} $$
#
# Where:
# $ A \cdot B $  = dot product of vectors A and B
# $ \|A\| $ = magnitude (length) of vector A
# $ \|B\| $ =  magnitude (length) of vector B
#
# #### Intuition
#
# - If vectors point in the **same direction**, cosine similarity = **1** (maximum similarity).
# - If vectors are **orthogonal (90° apart)**, cosine similarity = **0** (no similarity).
#
# ![](images/img04.png)

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Sample text data (replace with your own documents)

doc_1 = "Data is the oil of the digital economy"
doc_2 = "Data is a new oil"

data = [doc_1, doc_2]

# %%
tfidf_vectorizer = TfidfVectorizer() # Create a CountVectorizer instance
vector_matrix = tfidf_vectorizer.fit_transform(data) # Fit and transform the documents into numerical vectors

# %%
# Calculate the cosine similarity between the documents
cosine_similarity_matrix = cosine_similarity(vector_matrix)

df_cosine = pd.DataFrame(data=cosine_similarity_matrix, index=data, columns=data)

df_cosine

# %% [markdown]
# # 2- What is Word2vec?

# %% [markdown]
# Word2Vec consists of models for generating word embedding. These models are two-layer neural networks having one input layer, one hidden layer, and one output layer.
#
# Word2Vec utilizes two architectures :
# 1. CBOW (Continuous Bag of Words)
# 2. **Skip Gram**
# ![](images/img05.png)
#
#
# Run this command in terminal to install
# > pip install gensim
#
# We will use fake and real news dataset to do our expirament. You can find the dataset here: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv

# %%
import pandas as pd
import nltk
import numpy as np
import gensim

from nltk.tokenize import word_tokenize

# %%
df = pd.read_csv('/content/True.csv', engine='python', on_bad_lines='skip') #fake-and-real-news-dataset-true.csv
df.head()

# %%
tokens = []

for i in df['text']:
    token = word_tokenize(i)
    tokens.append(token)

# %%
w2v = gensim.models.Word2Vec(tokens, min_count=1, vector_size=100, window=5, sg=1)

# %%
print("Cosine similarity between 'provide' and 'program' - Skip Gram : ",w2v.wv.similarity('provide', 'program'))

# %%
print("words that similar to 'program' - Skip Gram : ",w2v.wv.most_similar('program'))

# %% [markdown]
# # Tasks

# %% [markdown]
# ### Task 1: Cosine Similarity
# Use the Cosine Similarity method to determine how similar the following sentences are.
#
# 'This is the first document.',
# 'This document is the second document.',
# 'And this is the third one.',
# 'Is this the first document?'

# %%

# %% [markdown]
# ### Task 2: TF-IDF
# Use tf-idf method on the sentences below to determine the important words.
#
# 'data science is one of the most important fields of science',
# 'this is one of the best data science courses',
# 'data scientists analyze data'

# %%

# %% [markdown]
# ## Word2vec

# %% [markdown]
# ### Task 3:
#
# Download the Simpsons dataset **(simpsons_script_lines.csv)** and apply the preprocessing procedure.
# Use the **'spoken_words'** column.
# ```
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[0-9]", '', text)
#     text = re.sub(r"[)(,”“.’$-]", '', text)
#     return text
# ```
# Create a skip gram Word2Vec model as below.
# ```
# Skip_gram_model = gensim.models.Word2Vec(tokens, min_count = 1, vector_size = 100, window = 5, sg = 1)
# ```

# %%

# %% [markdown]
# ### Task 4
#
# Use: wv.most_similar() method to :
#
# 1. Find the words similar to “homer”.
#
# 2. Find the words similar to “marge”.
#
# 3. Find the words similar to “bart”

# %%

# %% [markdown]
# ### Task 5
#
# Use the wv.doesnt_match() method to :
#
# 1. Find which of 'jimbo', 'milhouse’, and 'kearney’ does not belong to the list.
#
# 3. Find the odd one among "nelson", "bart", and "milhouse".
#
# 4. Find the odd one among ‘homer', 'patty', and ‘selma'.
#
# *Hint: You need to pass the strings as List*

# %%
