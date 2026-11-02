# Lab 11 - Attention with RNN
# The notebook Lab11_Attention_with_RNN.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Lab 11: Attention

# %% [markdown]
# ### Attention is a mechanism that allows a neural network to focus more on relevant information when making a prediction.
#
# Encoder-decoder architecture with attention:
# ![](images/img01.png)

# %% [markdown]
# ## Step 1 — Import libraries

# %%
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from google.colab import files

torch.manual_seed(42)

# %% [markdown]
# ## Step 2 — Upload the dataset

# %%
df = pd.read_csv("/content/Lab_11_QnA_Dataset_Download.csv")
print("Number of questions:", len(df))
print("Number of answer groups:", df["answer_id"].nunique())
display(df.head())

# %% [markdown]
# ## Step 3 — Preprocessing
#
# 1. Lowercase each question and remove punctuation.
# 2. Give each word a number.
# 3. Make all questions the same length by adding zeros (**padding**).

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Lowercase
df["question"] = df["question"].str.lower()

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["question"])

# Convert words to numbers and add padding
X = pad_sequences(tokenizer.texts_to_sequences(df["question"]), padding="post")

# Answers
y = df["answer_id"].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# %% [markdown]
# ## Step 4 — Build an RNN with attention
#
# - **Embedding** turns word numbers into vectors.
# - **RNN** reads the words in sequence.
# - **Attention** gives each word a weight.
# - **Output** chooses one answer from the dataset.

# %%
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(tokenizer.word_index) + 1, 16)
        self.rnn = nn.RNN(16, 32)
        self.att = nn.Linear(32, 1)
        self.fc = nn.Linear(32, df["answer_id"].nunique())

    def forward(self, x):
        x, _ = self.rnn(self.emb(x))
        weights = torch.softmax(self.att(x), dim=1)
        return self.fc((x * weights).sum(1)), weights.squeeze(-1)

model = RNN()

# %% [markdown]
# ## Step 5 — Train the model
# The model learns which answer belongs to each question.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    loss = nn.functional.cross_entropy(model(X)[0], y) #calculate loss to reduce it
    loss.backward() #backpropagation to update parameters
    optimizer.step() #update parameters

print("Training finished!")

# %% [markdown]
# ## Step 6 — See attention
# The bar chart shows how much attention the model gives to each question word.

# %%
question = df["question"][0]
words = question.split()

with torch.no_grad():
    prediction, weights = model(X[:1])

plt.bar(words, weights[0, :len(words)].tolist())
plt.title("Attention Weights")
plt.show()

# %% [markdown]
# ## Tasks:

# %% [markdown]
# Task 1 — Understand Attention Weights
# Easy
#
# Modify Step 6 to visualize the attention weights for three different questions from the dataset instead of only the first question.
#
# Instructions:
# - Select three questions from the dataset.
# - Generate an attention bar chart for each question.
# - Identify the word receiving the highest attention weight in each question.
#
# Question: Do the words with the highest attention weights appear important for understanding the question? Explain your observations.

# %%

# %% [markdown]
# Task 2 — Compare Attention for Similar Questions
# Medium
#
# Select two questions from the dataset that have similar meanings but different wording.
#
# Instructions:
# - Pass both questions through the trained model.
# - Visualize their attention weights.
# - Compare the words receiving the highest attention weights.
#
# Question: Does the model focus on similar or different words when processing questions with similar meanings? What does this suggest about how attention works?

# %%
