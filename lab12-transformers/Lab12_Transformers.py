# Lab 12 - Transformers
# The notebook Lab12_Transformers.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Transformers
#
#
# #### A Transformer is a deep learning architecture designed to process sequences of text using self-attention.
#
# - Transformers are the foundation of many modern NLP models, including BERT, GPT, and T5.
#
# - Before Transformers, we used models such as RNNs and LSTMs for NLP tasks.
#
# ![](images/img01.png)

# %%
%pip -q install transformers torch matplotlib

# %% [markdown]
# ## 1. Load a Transformer
#
# We use **DistilBERT**, a  Transformer fine-tuned for positive/negative sentiment classification.

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Look inside the Transformer architecture.
print("Transformer layers:", model.config.n_layers)
print("Attention heads per layer:", model.config.n_heads)
print("Hidden size:", model.config.dim)

# %% [markdown]
#
# ![](images/img02.png)

# %% [markdown]
# ## 2. Tokenizing the text

# %%
sentence = "The movie was not good."

# Tokenization: converts text into token IDs
inputs = tokenizer(sentence, return_tensors="pt")

# [CLS] = start token, [SEP] = end token
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("Tokens:", tokens)

# %% [markdown]
# ## 3. Embeddings and self-attention
#
# Each token ID becomes a **vector (embedding)**. Inside each Transformer layer, **self-attention** lets a token use information from other tokens in the same sentence.

# %%
import torch

# Enable attention weights
model.set_attn_implementation("eager")

with torch.no_grad():
    embeddings = model.distilbert.embeddings(inputs["input_ids"])
    outputs = model(**inputs, output_attentions=True)

# Attention from the first layer and first head
attention = outputs.attentions[0][0, 0]

print("Embeddings shape:", embeddings.shape)
print("Attention shape:", attention.shape)
print("Tokens:", tokens)
print("Attention weights:\n", attention)

# %% [markdown]
# ## 4. Classification: From Transformer to prediction
#
# The Transformer produces contextual representations. The **classification head** turns them into two scores (*logits*). **Softmax** converts these into probabilities that sum to 1.

# %%
# Get prediction scores from the Transformer
with torch.no_grad():
    logits = model(**inputs).logits

# Convert scores into probabilities
probs = torch.softmax(logits, dim=1)[0]

# Display the prediction
prediction = probs.argmax().item()

print("Prediction:", model.config.id2label[prediction])
print("Confidence:", round(probs[prediction].item(), 3))

# %% [markdown]
#
# ### Task: Exploring Transformer Layers and Attention Heads
#
# Load the pretrained **`bert-base-uncased`** model using Hugging Face Transformers.
#
# 1. Write code to print the number of Transformer layers and the number of attention heads per layer.
# 2. Calculate and print the total number of attention heads across all Transformer layers.
# 3. Using the sentence `"Transformers use self-attention to understand language."`, extract the attention weights from the **third Transformer layer and second attention head**.
# 4. Explain what the rows and columns of the attention matrix represent.

# %% [markdown]
# ### Solution
#
# **Step 1 - Load BERT.** Load the tokenizer and the base `bert-base-uncased` encoder. `attn_implementation="eager"` makes the model return its attention weights, and `eval()` switches off dropout so the weights are deterministic.

# %%
from transformers import AutoTokenizer, AutoModel
import torch

bert_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name, attn_implementation="eager")
bert_model.eval()

print(type(bert_model).__name__)

# %% [markdown]
# **Step 2 - Task 1: layers and heads per layer.** Both numbers are stored in the model configuration.

# %%
num_layers = bert_model.config.num_hidden_layers
heads_per_layer = bert_model.config.num_attention_heads

print("Number of Transformer layers:", num_layers)
print("Attention heads per layer:", heads_per_layer)

# %% [markdown]
# **Step 3 - Task 2: total number of attention heads.** Every layer has the same number of heads, so the total is layers multiplied by heads per layer.

# %%
total_heads = num_layers * heads_per_layer
print("Total attention heads:", num_layers, "x", heads_per_layer, "=", total_heads)

# %% [markdown]
# **Step 4 - Task 3: tokenize the sentence.** The tokenizer adds `[CLS]` at the start and `[SEP]` at the end, and may split rare words into word pieces.

# %%
task_sentence = "Transformers use self-attention to understand language."
task_inputs = bert_tokenizer(task_sentence, return_tensors="pt")
task_tokens = bert_tokenizer.convert_ids_to_tokens(task_inputs["input_ids"][0])

print("Tokens:", task_tokens)
print("Number of tokens:", len(task_tokens))

# %% [markdown]
# **Step 5 - Run BERT and keep the attentions.** With `output_attentions=True` the model returns one attention tensor per layer, each shaped `(batch, heads, tokens, tokens)`.

# %%
with torch.no_grad():
    task_outputs = bert_model(**task_inputs, output_attentions=True)

print("Attention tensors returned:", len(task_outputs.attentions))
print("Shape of one layer's attention:", tuple(task_outputs.attentions[0].shape))

# %% [markdown]
# **Step 6 - Select the third layer and the second head.** Python counts from 0, so the third layer is index `2` and the second head is index `1`. The result is a square matrix with one row and one column per token.

# %%
import pandas as pd

layer_index = 2   # third Transformer layer
head_index = 1    # second attention head

attention_l3_h2 = task_outputs.attentions[layer_index][0, head_index]
print("Attention matrix shape:", tuple(attention_l3_h2.shape))

attention_table = pd.DataFrame(attention_l3_h2.numpy(), index=task_tokens, columns=task_tokens).round(3)
attention_table

# %% [markdown]
# **Step 7 - Check the rows.** Each row comes out of a softmax, so it should add up to 1. The code also prints which token each token attends to most.

# %%
row_sums = attention_l3_h2.sum(dim=-1)
print("Row sums:", [round(v, 4) for v in row_sums.tolist()])

for i, token in enumerate(task_tokens):
    j = int(attention_l3_h2[i].argmax())
    print(f"{token:>12} -> attends most to {task_tokens[j]:<12} ({attention_l3_h2[i, j].item():.3f})")

# %% [markdown]
# **Step 8 - Visualise the matrix.** A heatmap makes the pattern easier to read: rows are the tokens doing the attending, columns are the tokens being attended to.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(attention_l3_h2.numpy(), cmap="viridis")
ax.set_xticks(range(len(task_tokens)))
ax.set_xticklabels(task_tokens, rotation=90)
ax.set_yticks(range(len(task_tokens)))
ax.set_yticklabels(task_tokens)
ax.set_xlabel("Key token (attended to)")
ax.set_ylabel("Query token (attending)")
ax.set_title("bert-base-uncased: layer 3, head 2")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Step 9 - Task 4: what the rows and columns mean.**
#
# **Answer.**
#
# - **Rows are the query tokens:** the token that is "looking". Row *i* shows how token *i* spreads its attention when layer 3 builds its new representation.
# - **Columns are the key tokens:** the tokens being looked at. Column *j* is token *j* as a source of information.
# - **Each cell (i, j)** is the weight token *i* gives to token *j*. The matrix is 11 x 11 because the sentence becomes 11 tokens, including `[CLS]` and `[SEP]`.
# - **Each row is a probability distribution:** it comes out of a softmax, so it adds up to 1. Step 7 confirms this for every row. Columns do not have to add up to 1.
# - **The matrix is not symmetric:** `use` gives 0.569 to `transformers`, but `transformers` gives only 0.034 to `use`.
#
# **What this head does on this sentence:**
# - **Much of the weight goes to `[CLS]`:** for example `transformers` 0.889, `[SEP]` 0.828 and `.` 0.658. Heads often park attention on a special token when they have nothing useful to pick up.
# - **The rest links words that belong together:** `-` attends to `self` (0.639), `to` to `use` (0.622), `use` to `transformers` (0.569), and `language` to `use` (0.429) and `understand` (0.398).
