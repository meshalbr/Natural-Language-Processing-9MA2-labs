# Lab 4 - Classification and Evaluation
# Step: Split into train/test sets and vectorize with TF-IDF
# Extracted from Lab4_Classification_and_Evaluation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data['Review_Text'], data['Sentiment'], test_size=0.2, random_state=42)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features based on your dataset size - limit vocabulary to the 1000 most informative tokens
train_vectors = tfidf_vectorizer.fit_transform(train_data)
test_vectors = tfidf_vectorizer.transform(test_data)

# %% cell 2
print(f"train_data size: {train_data.shape}")
print(f"test_data size: {test_data.shape}")
