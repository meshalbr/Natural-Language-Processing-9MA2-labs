# Lab 7 - Large Language Models
# The notebook Lab7_Large_Language_Models.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Lab 7 — Large Language Models (LLMs)

# %% [markdown]
# ## The Three Main Transformer Model Families
#
# | Family | Example Models | Main Use |
# |---|---|---|
# | **Encoder-only** | BERT, RoBERTa | Classification, sentiment analysis, NER |
# | **Encoder–decoder** | T5, BART | Translation, summarization, question answering |
# | **Decoder-only** | GPT family | Text generation, conversation, completion |

# %% [markdown]
# ## 1. Setup

# %%
!pip -q install transformers sentencepiece torch

# %%
from transformers import pipeline, set_seed

print("Libraries loaded successfully!")

# %% [markdown]
# # Part A — Decoder-Only Model
#
# ## Decoder-only example: GPT-style Text Generation
#
# A decoder-only model predicts the next token based on previous tokens.
#
# We will use **DistilGPT-2**, a small GPT-style model.

# %%
generator = pipeline("text-generation", model="distilgpt2")

prompt = "Natural language processing is"

output = generator(prompt, max_new_tokens=30, do_sample=False)

print(output[0]["generated_text"])

# %% [markdown]
# # Part B — Encoder–Decoder Model
#
# ## Encoder–decoder example: Instruction / Text-to-Text Generation
#
# Encoder–decoder models contain two main parts:
#
# 1. The **encoder** reads the input.
# 2. The **decoder** produces the output.
#
# We will use **FLAN-T5-small**.

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

prompt = "Translate to French: I love natural language processing."

# Convert text into tokens
output = model.generate(**tokenizer(prompt, return_tensors="pt"))

# Convert tokens back to text
print(tokenizer.decode(output[0], skip_special_tokens=True))

# %% [markdown]
# # Part C — Encoder-Only Model
#
# ## Encoder-only example: Sentiment Classification
#
# Encoder-only models read the full input and create a representation of it.
#
# They are very useful when the task is to **understand or classify text**.
#
# Here we use a DistilBERT sentiment model.

# %%
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

text = "This NLP lab is easy and interesting."

result = classifier(text)
print(result)

# %% [markdown]
# # Tasks
#
# Complete the following tasks.
#
# ## Task 1 — Model Families
#
# For each task below, choose the most suitable model family:
#
# 1. Sentiment classification
# 2. Machine translation
# 3. Open-ended text generation
# 4. Named Entity Recognition
# 5. Text summarization

# %%

# %% [markdown]
# ## Task 2 — Encoder-only Model
#
# Use the sentiment classifier on these three sentences:
#
# ```text
# The movie was excellent.
# The lecture was difficult to understand.
# I enjoyed learning about transformers.
# ```
#
# Print the predicted label and score for each sentence.

# %%

# %% [markdown]
# ## Task 3 — Encoder–Decoder Model
#
# Use the FLAN-T5 model to perform the following:
#
# - Summarize a short paragraph.

# %%

# %% [markdown]
#
# ## Task 5 — Short Reflection
#
# Answer in 3–5 sentences:
#
# **Why is a decoder-only model suitable for a chatbot such as ChatGPT?**

# %% [markdown]
# # Optional Challenge
#
# Try generating text several times with the same prompt.
#
# Remove:
#
# ```python
# set_seed(42)
# ```
#
# and run the sampling cell multiple times.
#
# ### Question
#
# Why do you sometimes get different outputs even when the prompt is the same?
#
# **Hint:** Sampling introduces randomness when choosing the next token.
