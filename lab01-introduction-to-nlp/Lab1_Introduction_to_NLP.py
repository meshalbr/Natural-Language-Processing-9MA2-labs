# Lab 1 - Introduction to Natural Language Processing
# The notebook Lab1_Introduction_to_NLP.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Introduction to Natural Language processing

# %% [markdown]
# **Natural language processing (NLP):** Subfield of linguistics, computer science, and information engineering concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.

# %% [markdown]
# ![venn.png](images/img01_venn.png)

# %% [markdown]
# ## What is it all about?

# %% [markdown]
# ![Image1.png](images/img02_image1.png)

# %% [markdown]
# ![not_all.png](images/img03_not-all.png)

# %% [markdown]
# ## What are the raw materials for NLP revolution?
# **1. Data:** There is a huge amount of human knowledge that has been digitized.
#
# **2. Compute:** Our personal computers are very powerful, and it has never been easier to rent more compute power.
#
# **3. Algorithms:** We use a combination of Computer Science, Machine Learning, Deep Learning, and linguist techniques.

# %% [markdown]
# ## Why NLP is hard?
# It’s the nature of the human language that makes NLP difficult.
#
# ### Some examples:

# %% [markdown]
# **1. Ambiguity:** One word/sentence can have different meanings.

# %% [markdown]
# ![CZnxd2KWIAEznKM.jpg](images/img04_cznxd2kwiaeznkm.jpg)

# %% [markdown]
# **2. Spelling Errors:** There are some words that are usually misspelled by people.

# %% [markdown]
# ![Spelling%20errors.png](images/img05_spelling-20errors.png)

# %% [markdown]
# **3. Phonetics and Phonology**: Different words that have the same pronunciation.
#
# - there / their
# - plane / plain
# - ice cream / I scream

# %% [markdown]
# **4. Order of Context:** "Dog bites man" and "Man bites dog" have vastly different meanings

# %% [markdown]
# ### In ML terms, why is NLP hard?

# %% [markdown]
# **1. Input:** Large, discrete, open-ended input state spaces
#
# **2. Output:** Variable length output spaces
#
# **3. Algorithms:** Recycled from other fields

# %% [markdown]
# ## Top Used Text File Types

# %% [markdown]
# - CSV
# - Text Files
# - JSON
# - SQL
# - HTML
# - PDF
# - DOCX

# %% [markdown]
# ## NLP Libraries

# %% [markdown]
# There are many popular NLP libraries that we will cover in the next labs. In this lab, we will get introduced to NLTK.

# %% [markdown]
# ### NLTK
# Natural Language Toolkit (nltk) is a Python package for NLP
# https://www.nltk.org/

# %% [markdown]
# #### What are Corpus / Corpora?
#
# a collection of written texts, especially the entire works of a particular author or a body of writing on a particular subject.

# %%
!pip install nltk

# %%
import nltk

# %% [markdown]
# ### Arabic Libraries (Extra)

# %% [markdown]
# Camel-Tools: https://camel-tools.readthedocs.io/en/latest/

# %% [markdown]
# PyArabic: https://pyarabic.readthedocs.io/ar/latest/

# %% [markdown]
# Farasa: https://farasa.qcri.org/

# %% [markdown]
# ## NLP Applications

# %% [markdown]
# **1. Information Extraction**

# %% [markdown]
# The automated retrieval of specific information related to a selected topic from a body or bodies of text.

# %% [markdown]
# **2. Machine Translation**

# %% [markdown]
# The process of using artificial intelligence to automatically translate text from one language to another without human involvement.

# %% [markdown]
# **3. Sentiment Analysis**

# %% [markdown]
# Identifying sentiments and opinions stated in a text.

# %% [markdown]
# ## Example of an NLP Pipeline

# %% [markdown]
# ![e322457d-6d42-4f82-999a-a2a446d5862e.png](images/img06_e322457d-6d42-4f82-999a-a2a446d5862e.png)

# %% [markdown]
# #### 1. Text Acquisition:
# Obtain the text data from a source, such as a file, a web page, or a database.
#
# #### 2. Text Preprocessing:
# - **Tokenization:** Split the text into individual words or tokens.
# - **Lowercasing:** Convert all tokens to lowercase to ensure uniformity.
# - **Removing Punctuation:** Strip away punctuation marks from tokens.
# - **Stopword Removal:** Eliminate common words (e.g., "and," "the," "is") that may not carry significant meaning.
# - **Stemming:** Removes prefixes/suffixes to get a root form. (Running -> run)
# - **Lemmatization:** Converts word to base/dictionary form (lemma). (better -> good).
#
# #### 3. Feature Extraction:
# - **Bag-of-Words (BoW):** Create a vector representation of the text using the frequency of each token in the document.
# - **TF-IDF (Term Frequency-Inverse Document Frequency):** Assign weights to tokens based on their importance in the document relative to the entire corpus.
#
# #### 4. Model Building:
# Choose a machine learning model for sentiment analysis, such as a Naive Bayes classifier, Support Vector Machine (SVM), or a neural network.

# %% [markdown]
# ---

# %% [markdown]
# ### Task#1:
#
# Students should work in groups and search online for one of the different types of datasets.
# - CSV
# - Text Files
# - JSON
# - SQL
# - HTML
# - PDF
# - DOCX
#
#
#
# ### Task#2:
#
# Download and open the dataset.

# %%
conda install kagglehub

# %%
import kagglehub

# IMDB Dataset of 50K Movie Reviews
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)

# %%
import pandas as pd

# load the CSV file in a dataframe
df = pd.read_csv("IMDB Dataset.csv")

# Show the first few rows
df.head()

# %%
df.describe()

# %%
df.info()

# %%
# Check class distribution
print(df['sentiment'].value_counts())

# %%
