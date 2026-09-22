# Lab 3 - N-Grams
# Section: Evaluating n_grams models using preplexity
# Extracted from Lab3_N_Grams.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
test = [('moses', 'supposes'), ('toeses','are')]
print(lm.perplexity(test)) #Perplexity for bigram model
