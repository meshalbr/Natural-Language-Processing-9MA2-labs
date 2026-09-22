# Lab 2 - Text Pre-processing and Regular Expressions
# Section: search
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import re

text = 'I am enjoying the NLP course.'

print(re.search("I", text)) #r"I" means it's a raw string literal
print(re.search("se.$", text)) #$ Search at the end of the string
print(re.search("am", text))
print(re.search("m", text))
print(re.search("AI", text))
