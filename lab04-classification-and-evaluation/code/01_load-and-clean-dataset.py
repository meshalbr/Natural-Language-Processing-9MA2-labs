# Lab 4 - Classification and Evaluation
# Step: Load the Disneyland reviews dataset, drop missing rows, map ratings to sentiment labels
# Extracted from Lab4_Classification_and_Evaluation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
# Load the dataset
import pandas as pd

data = pd.read_csv('/content/DisneylandReviews.csv', encoding='latin-1')
data

# %% cell 2
#Check for missing values
data.isna().sum()

# %% cell 3
# Drop rows with missing values in the relevant columns
data = data.dropna(subset=['Review_Text', 'Rating'])

# %% cell 4
# Map star ratings to sentiments
data['Sentiment'] = data['Rating'].apply(lambda rating: 'positive' if rating > 3 else ('negative' if rating < 3 else 'neutral'))
data.head()
