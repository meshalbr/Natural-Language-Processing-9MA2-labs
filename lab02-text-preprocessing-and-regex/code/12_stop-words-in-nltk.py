# Lab 2 - Text Pre-processing and Regular Expressions
# Section: Stop words in NLTK
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import nltk
nltk.download('stopwords')

# %% cell 2
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(len(stop_words))

# %% cell 3
print(stop_words)
