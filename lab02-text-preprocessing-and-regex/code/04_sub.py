# Lab 2 - Text Pre-processing and Regular Expressions
# Section: sub
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import re

# (IPython help) re.sub?

# %% cell 2
import re

text = 'I am enjoying the Math course.'

print(re.sub(r"Math", "NLP", text, count=1, flags=re.I)) #flags=re.I >> Ignore the case (no matter if M or m)
                                                 # count = 1 >> number of word I need it to change. count= 0 by default
