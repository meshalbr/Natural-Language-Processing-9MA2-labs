# Lab 6 - Deep Learning for NLP
# The notebook Lab6_Deep_Learning_for_NLP.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Deep Learning for NLP
#
# **Deep Learning** is a type of Machine Learning that uses **Neural Networks with multiple layers** to learn patterns from data.
#
# Deep Learning can be used in NLP tasks such as:
# - Text Classification
# - Sentiment Analysis
# - Question Answering
#
# In this lab, we will build a **deep learning model** to classify Yelp restaurant reviews as:
#
# - **0 = Negative**
# - **1 = Positive**

# %% [markdown]
# ![](images/img01.png)
# ![](images/img02.png)

# %% [markdown]
# ## 1. Load the Yelp Dataset
#
# Each row contains:
#
# - a restaurant review
# - its sentiment label

# %%
import pandas as pd

df = pd.read_csv("07-yelp-dataset.txt", sep="\t", names=["review", "label"])

print("Number of reviews:", len(df))
df.head()

# %%
print(df["label"].value_counts())

# %% [markdown]
# ## 2. Preprocessing the Dataset

# %%
import re

def preprocess_text(text):

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

    df["review"] = df["review"].apply(preprocess_text)
    df.head()

# %% [markdown]
# ## 3. Split the Data
#
# We divide the dataset into:
#
# - **80% training data** — used to train the model
# - **20% testing data** — used to evaluate the model
#
# `stratify=y` keeps the proportion of positive and negative reviews similar in both sets.

# %%
from sklearn.model_selection import train_test_split

X = df["review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# %% [markdown]
# ## 4. Convert Text to TF-IDF Vectors
#
# A neural network cannot directly understand text.
#
# **TF-IDF converts each review into a vector of numbers.**

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# %% [markdown]
# ## 5. Convert the Data to PyTorch Tensors
#
# PyTorch neural networks work with **tensors**.

# %%
import torch
import torch.nn as nn

# Convert TF-IDF data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_tfidf.toarray())
X_test_tensor = torch.FloatTensor(X_test_tfidf.toarray())

y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

# %% [markdown]
# ## 6. Build a Simple Neural Network
#
# Our network has:
#
# 1. **Input layer** — receives the TF-IDF features
# 2. **Hidden layer** — learns useful patterns
# 3. **ReLU activation** — adds non-linearity
# 4. **Output layer** — produces one value for binary classification
#
# We do **not** put `Sigmoid` inside the model because `BCEWithLogitsLoss` applies it internally in a numerically stable way.

# %%
input_size = X_train_tensor.shape[1]
# torch.size([800, 1000])

model = nn.Sequential(
    nn.Linear(input_size, 64),   # Input layer
    nn.ReLU(),

    nn.Linear(64, 32),           # Hidden layer
    nn.ReLU(),

    nn.Linear(32, 1)             # Output layer
)

print(model)

# %% [markdown]
# ## 7. Choose the Loss Function and Optimizer
#
# - **BCEWithLogitsLoss** is used for binary classification.
# To measure the model’s mistakes.
# - **Adam** is responsible for changing the model's weights so that the loss becomes smaller.

# %%
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% [markdown]
# ## 8. Train the Neural Network
#
# One **epoch** means the model has seen the full training dataset once.
#
# For every epoch:
#
# 1. Make predictions
# 2. Calculate the loss
# 3. Calculate gradients with backpropagation
# 4. Update the weights

# %%
epochs = 10

for epoch in range(epochs):
    model.train()

    # 1. Forward pass
    outputs = model(X_train_tensor)

    # 2. Calculate loss
    loss = loss_function(outputs, y_train_tensor)

    # 3. Backpropagation
    loss.backward()

    # 4. Update model weights
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# %% [markdown]
# ## 9. Evaluate the Model
#
# The model outputs raw scores called **logits**.
#
# We apply `sigmoid` to convert them to probabilities between 0 and 1.
#
# - Probability **≥ 0.5 → Positive (1)**
# - Probability **< 0.5 → Negative (0)**

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model.eval()

with torch.no_grad():
    test = model(X_test_tensor)
    test_probabilities = torch.sigmoid(test)
    predictions = (test_probabilities >= 0.5).int().numpy().flatten()

accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy:.3f}")

# %% [markdown]
# ## 10. Classification Report and Confusion Matrix
#
# The classification report shows:
#
# - **Precision**
# - **Recall**
# - **F1-score**
#
# The confusion matrix shows how many reviews were classified correctly and incorrectly.

# %%
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# %% [markdown]
# ## Task 1 – Represent Reviews Using Word2Vec
#
# In the previous part of the lab, TF-IDF was used to convert each review into numerical features.
# Now, use Word2Vec instead.
#
# Requirements:
# Tokenize the training and testing reviews.
# Train a Word2Vec model using the training reviews.
# Use:
# vector_size = 200
# window = 5
# min_count = 1
# sg = 1
#
# Convert each review into a single vector.
# Convert the resulting vectors into PyTorch tensors.

# %%

# %% [markdown]
# ## Task 2 – Build a Different Neural Network
#
# Use the Word2Vec vectors from Task 1 as the input to a new neural network.
# Your network must contain:
# An input layer
# Three hidden layers
# ReLU activation functions
# One output neuron
# Use the following architecture:
#
# Input
#
#    ↓
#
# 128 neurons
#
#    ↓
#
# 64 neurons
#
#    ↓
#
# 32 neurons
#
#    ↓
#
# 1 output neuron
#
# Train the model using:
# , BCEWithLogitsLoss
# , Adam optimizer
# , Learning rate = 0.001
# , 15 epochs.
# Then calculate the test accuracy.

# %%
