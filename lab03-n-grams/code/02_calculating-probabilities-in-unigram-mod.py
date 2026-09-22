# Lab 3 - N-Grams
# Section: Calculating probabilities in unigram models
# Extracted from Lab3_N_Grams.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
### Calculate counts of each word

unique_string = set(word_tokenize(string))

words_frq = []
for uniques in unique_string:
    frq = string.count(uniques)
    words_frq.append((uniques, frq))

words_frq

# %% cell 2
### Calculate probabilities:

probab = []

for count in range(len(words_frq)):
    pro = words_frq[count][1]/len(word_tokenize(string))
    probab.append((words_frq[count][0],pro))

print('The probabilities are: ', probab)
