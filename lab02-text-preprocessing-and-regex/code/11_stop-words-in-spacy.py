# Lab 2 - Text Pre-processing and Regular Expressions
# Section: Do we always remove stopwords? **NO!**
# Extracted from Lab2_Text_Preprocessing_and_Regex.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
import spacy
nlp = spacy.load('en_core_web_sm')

# %% cell 2
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))

# %% cell 3
print(nlp.vocab['myself'].is_stop)
print(nlp.vocab['mystery'].is_stop)

# %% cell 4
print (nlp.vocab['btw'].is_stop) #print False

# %% cell 5
nlp.Defaults.stop_words.add('btw')

nlp.vocab['btw'].is_stop = True # You must update the is_stop flag for the word you added

print (nlp.vocab['btw'].is_stop)

# %% cell 6
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False

print(nlp.vocab['beyond'].is_stop)
