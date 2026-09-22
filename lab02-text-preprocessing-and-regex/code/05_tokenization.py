# Lab 2 - Text Pre-processing and Regular Expressions
# Section: Tokenization
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
# (notebook shell/magic command) !pip install nltk
nltk.download('punkt') #Punkt is a pretrained unsupervised machine learning tokenizer model

# %% cell 2
import nltk

text = "I'm enjoying the NLP course! I am also learning new concepts."

print(nltk.sent_tokenize(text))

# %% cell 3
import nltk

text = "I'm enjoying the NLP course!"

print(nltk.word_tokenize(text)) #I'm is tokenized to two tokens [I,m], course! is also tokenized to two tokens [course,!]

# %% cell 4
text = "I'm enjoying the NLP course!"
text.split()
