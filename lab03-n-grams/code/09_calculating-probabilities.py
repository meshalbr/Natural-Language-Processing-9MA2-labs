# Lab 3 - N-Grams
# Section: Calculating Probabilities
# Extracted from Lab3_N_Grams.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
print(lm.score('supposes', ['moses'])) #Calculate Bigram Probabilities for P('moses'|'supposes')
print(lm.score('moses', ['supposes'])) #Calculate Bigram Probabilities for P('supposes'|'moses') - Returns 0 because unseen
