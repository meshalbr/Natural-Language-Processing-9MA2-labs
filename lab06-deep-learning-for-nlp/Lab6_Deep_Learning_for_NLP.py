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

df = pd.read_csv("dataset/07-yelp-dataset.txt", sep="\t", names=["review", "label"])

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

# %% [markdown]
# **Step 1 - Tokenize the reviews.** Word2Vec needs each review as a list of words. We reuse the lab's `preprocess_text` (lowercase, remove punctuation) and split on spaces. The test reviews are tokenized the same way but are not used for training.

# %%
from gensim.models import Word2Vec
import numpy as np

train_tokens = [preprocess_text(review).split() for review in X_train]
test_tokens  = [preprocess_text(review).split() for review in X_test]

print("Example review:", X_train.iloc[0])
print("As tokens     :", train_tokens[0])

# %% [markdown]
# **Step 2 - Train Word2Vec on the training reviews.** The task's settings: `vector_size=200` (each word becomes a vector of 200 numbers), `window=5` (5 context words on each side), `min_count=1` (keep every word), `sg=1` (skip-gram). With the default 5 training epochs on only 800 short reviews the vectors come out almost identical (every word is 0.99 similar to every other), so Word2Vec is trained for 200 epochs to get usable vectors.

# %%
w2v = Word2Vec(sentences=train_tokens, vector_size=200, window=5, min_count=1, sg=1, epochs=200, seed=42)

print("Vocabulary size:", len(w2v.wv))
print("Vector of 'good' (first 5 numbers):", w2v.wv["good"][:5])
print("Most similar to 'good':", w2v.wv.most_similar("good", topn=3))
print("Most similar to 'bad' :", w2v.wv.most_similar("bad", topn=3))

# %% [markdown]
# **Step 3 - One vector per review.** A review has many words but the network needs one fixed-size input, so we take the **average** of the word vectors of the review. A test review whose words were never seen in training gets a zero vector.

# %%
def review_vector(tokens, model, size=200):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:            # no known word in this review
        return np.zeros(size)
    return np.mean(vectors, axis=0)  # average of the word vectors

X_train_w2v = np.array([review_vector(tokens, w2v) for tokens in train_tokens])
X_test_w2v  = np.array([review_vector(tokens, w2v) for tokens in test_tokens])

print("Training Word2Vec shape:", X_train_w2v.shape)   # 800 reviews x 200 numbers
print("Testing Word2Vec shape :", X_test_w2v.shape)    # 200 reviews x 200 numbers

# %% [markdown]
# **Step 4 - Standardize and convert to PyTorch tensors.** The averaged Word2Vec values are tiny numbers of very different scales, so each of the 200 features is standardized (subtract the training mean, divide by the training standard deviation; the test set uses the training statistics). Then, as in section 5 of the lab, the features become `FloatTensor`s and the labels a column vector of shape (800, 1).

# %%
mean = X_train_w2v.mean(axis=0)
std  = X_train_w2v.std(axis=0) + 1e-8

X_train_w2v_tensor = torch.FloatTensor((X_train_w2v - mean) / std)
X_test_w2v_tensor  = torch.FloatTensor((X_test_w2v - mean) / std)
y_train_w2v_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

print("Tensor shapes:", X_train_w2v_tensor.shape, X_test_w2v_tensor.shape, y_train_w2v_tensor.shape)

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

# %% [markdown]
# **Step 1 - Build the network.** The input is the 200-dimensional Word2Vec vector of a review. The architecture asked for is 200 → 128 → 64 → 32 → 1 with ReLU after each hidden layer. The last layer has one neuron and no sigmoid, because `BCEWithLogitsLoss` applies the sigmoid itself.

# %%
torch.manual_seed(42)

model2 = nn.Sequential(
    nn.Linear(X_train_w2v_tensor.shape[1], 128),   # input layer -> hidden layer 1 (128 neurons)
    nn.ReLU(),
    nn.Linear(128, 64),                            # hidden layer 2 (64 neurons)
    nn.ReLU(),
    nn.Linear(64, 32),                             # hidden layer 3 (32 neurons)
    nn.ReLU(),
    nn.Linear(32, 1)                               # output layer: one neuron (a logit)
)
print(model2)

# %% [markdown]
# **Step 2 - Loss function and optimizer.** Same loss as the lab (`BCEWithLogitsLoss`, binary classification) and Adam, but with the learning rate the task asks for: 0.001.

# %%
loss_function2 = nn.BCEWithLogitsLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# %% [markdown]
# **Step 3 - Train for 15 epochs with mini-batches.** One epoch is one pass over the 800 training reviews. The lab updated the weights once per epoch on all 800 reviews at once; with the small learning rate 0.001 that would be only 15 updates, far too few to learn. Here each epoch is split into mini-batches of 32 reviews (25 updates per epoch, 375 in total). Every update is the same four steps: forward pass, loss, backpropagation, optimizer step, and `zero_grad()` clears the previous gradients.

# %%
epochs = 15
batch_size = 32
torch.manual_seed(42)

for epoch in range(epochs):
    model2.train()
    order = torch.randperm(len(X_train_w2v_tensor))          # shuffle the reviews every epoch
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        optimizer2.zero_grad()                                # reset gradients
        outputs = model2(X_train_w2v_tensor[batch])           # 1. forward pass
        loss = loss_function2(outputs, y_train_w2v_tensor[batch])   # 2. loss
        loss.backward()                                       # 3. backpropagation
        optimizer2.step()                                     # 4. update the weights

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss (last batch): {loss.item():.4f}")

# %% [markdown]
# **Step 4 - Test accuracy.** Switch to evaluation mode, turn the logits into probabilities with the sigmoid, threshold at 0.5 and compare with the true labels. The TF-IDF model's accuracy from the lab is printed next to it for comparison.

# %%
model2.eval()
with torch.no_grad():
    probabilities2 = torch.sigmoid(model2(X_test_w2v_tensor))
    predictions2 = (probabilities2 >= 0.5).int().numpy().flatten()

accuracy2 = accuracy_score(y_test, predictions2)
print(f"Test Accuracy (Word2Vec + 3 hidden layers): {accuracy2:.3f}")
print(f"Test Accuracy (TF-IDF network from the lab) : {accuracy:.3f}")
print()
print(classification_report(y_test, predictions2, target_names=["Negative", "Positive"]))

# %% [markdown]
# **Observation.** The Word2Vec network reaches about the same test accuracy as the TF-IDF network of the lab (0.765 against 0.780 in this run; the lab model is not seeded, so its number moves a little between runs), with 200 input numbers instead of 1000. It only works after three changes: Word2Vec trained for 200 epochs (with the default 5 epochs on 800 short reviews every word vector is almost the same and the network predicts "negative" for everything, accuracy 0.48), standardized features, and mini-batch updates. Averaging word vectors also throws away word order and negations such as "not good", so Word2Vec cannot beat TF-IDF on this small dataset; pre-trained embeddings (GloVe, Google News Word2Vec) or far more text would be needed for that.
