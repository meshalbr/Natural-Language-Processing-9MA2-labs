# Lab 5 - Text Representation - Notes

Markdown notes extracted from the lab notebook ($(System.Collections.Hashtable.out)). Code lives in the code/ folder.

# Text Representation

Text representation in NLP means converting text into a numerical form that a machine learning model can understand and process.

## What is Embedding?

Embeddings in NLP is a technique where individual words are represented as real-valued vectors and captures inter-word semantics.


In this notebook, We going to intreduce 2 techniques for embedding. These techniques will be used for a machine learning models such as SVM, Random forest, ... ect.

<img src="https://drive.google.com/uc?export=view&id=1jd0u_sGppDKqBYbhtJfGxp14lnUWUTTw" width="900">

# 1- TF-IDF

<img src="https://drive.google.com/uc?export=view&id=1saSt0mMgOQ2ybbTms1lu_ruhUcBWkKzL" width="500">

TF-IDF stands for term frequency-inverse document frequency. It is a measure that discounts common words. Used in the fields of information retrieval (IR) and machine learning, that can quantify the importance of words in a document amongst a collection of documents (also known as a corpus).

### Components of TF-IDF

1. **TF (Term Frequency):**  
   Measures how frequently a term appears in a document.


$$ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} $$


2. **IDF (Inverse Document Frequency):**  
   Measures how important a term is across all documents. Words that appear in many documents get lower scores.

$$    \text{IDF}(t) = \log \left(\frac{\text{Number of all documents N}}{\text{Number of documents containing the term } t}\right) $$

<img src="https://drive.google.com/uc?export=view&id=1JqPILC8TTh3yDQZCSuPNXwm6ER5YeJFh" width="900">

# Cosine similarity

In NLP, Cosine similarity is a metric used to measure how similar the documents are.


$$ \text{cosine similarity} = \frac{A \cdot B}{\|A\| \times \|B\|} $$

Where:  
$ A \cdot B $  = dot product of vectors A and B  
$ \|A\| $ = magnitude (length) of vector A  
$ \|B\| $ =  magnitude (length) of vector B

#### Intuition

- If vectors point in the **same direction**, cosine similarity = **1** (maximum similarity).
- If vectors are **orthogonal (90° apart)**, cosine similarity = **0** (no similarity).

<img src="https://drive.google.com/uc?export=view&id=1b-o8CVfHBsjUGY_hXmxevU91gHidIuxO" width="900">

# 2- What is Word2vec?

Word2Vec consists of models for generating word embedding. These models are two-layer neural networks having one input layer, one hidden layer, and one output layer.

Word2Vec utilizes two architectures :
1. CBOW (Continuous Bag of Words)
2. **Skip Gram**
<img src="https://drive.google.com/uc?export=view&id=1K38nEu_KhgtSJuG2RySwc6U5AAjVAgWZ" width="600">


Run this command in terminal to install
> pip install gensim

We will use fake and real news dataset to do our expirament. You can find the dataset here: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv

# Tasks

### Task 1: Cosine Similarity
Use the Cosine Similarity method to determine how similar the following sentences are.

'This is the first document.',  
'This document is the second document.',  
'And this is the third one.',  
'Is this the first document?'

### Task 2: TF-IDF
Use tf-idf method on the sentences below to determine the important words.

'data science is one of the most important fields of science',  
'this is one of the best data science courses',  
'data scientists analyze data'

## Word2vec

### Task 3:

Download the Simpsons dataset **(simpsons_script_lines.csv)** and apply the preprocessing procedure.  
Use the **'spoken_words'** column.
```
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[0-9]", '', text)
    text = re.sub(r"[)(,”“.’$-]", '', text)
    return text
```
Create a skip gram Word2Vec model as below.
```
Skip_gram_model = gensim.models.Word2Vec(tokens, min_count = 1, vector_size = 100, window = 5, sg = 1)

### Task 4

Use: wv.most_similar() method to :

1.	Find the words similar to “homer”.

2.	Find the words similar to “marge”.

3. Find the words similar to “bart”

### Task 5

Use the wv.doesnt_match() method to :

1.	Find which of 'jimbo', 'milhouse’, and 'kearney’ does not belong to the list.

3.	Find the odd one among "nelson", "bart", and "milhouse".

4.	Find the odd one among ‘homer', 'patty', and ‘selma'.

*Hint: You need to pass the strings as List*
