# Lab 4 - Classification and Evaluation
# Step: Lowercase, strip punctuation and tokenize the review text
# Extracted from Lab4_Classification_and_Evaluation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
# Import necessary libraries
import re
import nltk

def preprocess_text(text):

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenize the text
    words = text.split()

    return ' '.join(words)

# Apply preprocessing to the review text
data['Review_Text'] = data['Review_Text'].apply(preprocess_text)

# Display cleaned reviews
data[['Review_Text', 'Sentiment']].head()
