# Lab 4 - Classification and Evaluation
# The notebook Lab4_Classification_and_Evaluation.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Classification and Evaluation in NLP
#
# **Text classification** is a common Natural Language Processing (NLP) task where a model assigns a predefined category or label to a piece of text.
#
# Examples include:
# - Spam detection
# - Sentiment analysis
# - News categorization
#
# In this lab, we will learn how to build and evaluate a text classification model.
#
# ## Case Study: Sentiment Analysis
#
# Sentiment analysis, also known as opinion mining, its goal is to understand the information present in the text and categorize it as positive, negative, or neutral.

# %% [markdown]
# ![1_e90_bvVf9Agxfk4DxWu7og.jpg](images/img01.jpg)

# %% [markdown]
# # Disneyland Reviews Dataset
# In this lab we'll look into Disneyland Reviews dataset.
# *Link: https://www.kaggle.com/datasets/arushchillar/disneyland-reviews*
#
# The dataset includes 42,000 reviews of 3 Disneyland branches - Paris, California and Hong Kong, posted by visitors on Trip Advisor.
#
# Column Description:
#
# - Review_ID: unique id given to each review
# - Rating: ranging from 1 (unsatisfied) to 5 (satisfied)
# - Year_Month: when the reviewer visited the theme park
# - Reviewer_Location: country of origin of visitor
# - Review_Text: comments made by visitor
# - Disneyland_Branch: location of Disneyland Park

# %%
# Load the dataset
import pandas as pd

data = pd.read_csv('/content/DisneylandReviews.csv', encoding='latin-1')
data

# %%
#Check for missing values
data.isna().sum()

# %%
# Drop rows with missing values in the relevant columns
data = data.dropna(subset=['Review_Text', 'Rating'])

# %%
# Map star ratings to sentiments
data['Sentiment'] = data['Rating'].apply(lambda rating: 'positive' if rating > 3 else ('negative' if rating < 3 else 'neutral'))
data.head()

# %%
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

# %%
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data['Review_Text'], data['Sentiment'], test_size=0.2, random_state=42)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features based on your dataset size - limit vocabulary to the 1000 most informative tokens
train_vectors = tfidf_vectorizer.fit_transform(train_data)
test_vectors = tfidf_vectorizer.transform(test_data)

# %%
print(f"train_data size: {train_data.shape}")
print(f"test_data size: {test_data.shape}")

# %%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_vectors, train_labels)

# Predictions on the test set
predictions = svm_classifier.predict(test_vectors)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(test_labels, predictions))

# %% [markdown]
# # Use Case:  Amazon reviews of unlocked phone analysis
#
# The objective of this use case is to conduct sentiment analysis on Amazon reviews of unlocked phones, categorizing reviews into three classes: positive, negative, and neutral. The goal is to gain a comprehensive understanding of customer opinions and sentiments regarding various unlocked phone models.
#
# - Dataset link: https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones

# %% [markdown]
# Task1: Load the data and do 5 different preprocessing steps on it
#
# Task2: Use a proper train and test split
#
# Task3: Do Feature Extraction Using TF-IDF
#
# Task4: Train a Naive Bayes Classifier
#
# Task5: Evaluate the model on the Test Set
#
# Task6: Print the Confusion Matrix

# %%
