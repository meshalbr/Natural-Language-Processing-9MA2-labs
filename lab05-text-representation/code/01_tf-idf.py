# Lab 5 - Text Representation
# Section: Components of TF-IDF
# Extracted from Lab5_Text_Representation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
from sklearn.feature_extraction.text import TfidfVectorizer

# %% cell 2
doc_1 = "Data is the oil of the digital economy"
doc_2 = "Data is a new oil"

data = [doc_1, doc_2]

# %% cell 3
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(data) # returns sparce matrix

# %% cell 4
df = pd.DataFrame(result.toarray(), columns=tfidf.get_feature_names_out())
df
