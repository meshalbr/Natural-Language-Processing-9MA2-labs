# Lab 1 - Introduction to Natural Language Processing
# Section: Task#1
# Extracted from Lab1_Introduction_to_NLP.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
# (run in a terminal) conda install kagglehub

# %% cell 2
import kagglehub

# IMDB Dataset of 50K Movie Reviews
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)

# %% cell 3
import pandas as pd

# load the CSV file in a dataframe
df = pd.read_csv("IMDB Dataset.csv")

# Show the first few rows
df.head()

# %% cell 4
df.describe()

# %% cell 5
df.info()

# %% cell 6
# Check class distribution
print(df['sentiment'].value_counts())
