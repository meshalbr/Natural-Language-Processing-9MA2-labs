# Lab 2 - Text Pre-processing and Regular Expressions
# Section: What is SnowballStemmer?
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import nltk
from nltk.stem.snowball import SnowballStemmer

sn = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word+' --> '+sn.stem(word))
