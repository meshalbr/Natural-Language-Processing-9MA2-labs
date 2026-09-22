# Lab 2 - Text Pre-processing and Regular Expressions
# Section: match
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.
# Note: this file uses notebook syntax (lines like !pip install or re.match?). Run it cell by cell in Jupyter or the VS Code interactive window (Shift+Enter).

# %% cell 1
import re
re.match?

# %% cell 2
import re

text = 'I am enjoying the NLP course.'

print(re.search("I.*", text)) #global
print(re.match("I.*", text)) #local >> search at the start of the string

print(re.search("enjoying.*", text))
print(re.match("enjoying.*", text))

# %% cell 3
import re

text = '1999 was the year I was born.'

print(re.search(r"[a-zA-Z]+", text))
print(re.match(r"[a-zA-Z]+", text))
