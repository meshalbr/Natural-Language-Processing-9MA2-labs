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
%pip -q install transformers sentencepiece torch

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

# %% [markdown]
# **Answer.** The rule: *understand or classify text* → encoder-only; *text in, different text out* → encoder-decoder; *continue or create text* → decoder-only.

# %%
tasks = {
    "Sentiment classification":   "Encoder-only (BERT, RoBERTa, DistilBERT): read the whole input, output a class",
    "Machine translation":        "Encoder-decoder (T5, BART, MarianMT): read the source sentence, generate the target",
    "Open-ended text generation": "Decoder-only (GPT family): predict the next token from the previous ones",
    "Named Entity Recognition":   "Encoder-only (BERT): one label per token needs a full bidirectional view of the sentence",
    "Text summarization":         "Encoder-decoder (T5, BART, PEGASUS): long text in, short text out",
}

for task, family in tasks.items():
    print(f"{task:28s} -> {family}")

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

# %% [markdown]
# **Step 1 - The three sentences.** The sentiment classifier is the DistilBERT pipeline created in Part C. A pipeline accepts a list, so all three sentences are classified in one call.

# %%
sentences = [
    "The movie was excellent.",
    "The lecture was difficult to understand.",
    "I enjoyed learning about transformers.",
]

results = classifier(sentences)

# %% [markdown]
# **Step 2 - Print the label and the score.** Each result is a dictionary with the predicted label (POSITIVE or NEGATIVE) and the model's confidence between 0 and 1.

# %%
for sentence, result in zip(sentences, results):
    print(f"{sentence:45s} -> {result['label']:8s} (score = {result['score']:.4f})")

# %% [markdown]
# ## Task 3 — Encoder–Decoder Model
#
# Use the FLAN-T5 model to perform the following:
#
# - Summarize a short paragraph.

# %% [markdown]
# **Step 1 - Write the paragraph to summarize.** FLAN-T5 is an instruction-tuned model: we tell it what to do inside the prompt, so the prompt starts with `Summarize:` followed by the text.

# %%
paragraph = (
    "Natural language processing is a field of artificial intelligence that gives computers the ability "
    "to read, understand and generate human language. Early systems relied on hand-written rules, later "
    "systems used statistical models trained on large corpora, and today most systems are built on "
    "transformer neural networks such as BERT, T5 and GPT. These models are pre-trained on huge amounts "
    "of text and then adapted to tasks such as translation, question answering and summarization."
)

prompt = "Summarize: " + paragraph
print("Words in the paragraph:", len(paragraph.split()))

# %% [markdown]
# **Step 2 - Encode, generate, decode.** The tokenizer turns the prompt into token ids, the encoder reads them, the decoder generates the summary tokens (`max_new_tokens=60` caps the length), and the tokenizer turns the ids back into text.

# %%
inputs = tokenizer(prompt, return_tensors="pt")          # text -> token ids
output = model.generate(**inputs, max_new_tokens=60)      # encoder reads, decoder writes
summary = tokenizer.decode(output[0], skip_special_tokens=True)   # token ids -> text

print("Summary:", summary)
print("Words in the summary:", len(summary.split()))

# %% [markdown]
#
# ## Task 5 — Short Reflection
#
# Answer in 3–5 sentences:
#
# **Why is a decoder-only model suitable for a chatbot such as ChatGPT?**

# %% [markdown]
# **Answer.** A chatbot has to produce an open-ended reply one word after another, and that is exactly what a decoder-only model is trained to do: predict the next token given everything written so far. The whole conversation (system prompt, the user's messages, the model's earlier replies) is simply the context, so the same mechanism that completes a sentence also continues a dialogue, and no separate encoder is needed. Because the decoder is autoregressive it can keep generating until it decides to stop, which allows answers of any length, and the same model can be fine-tuned with instruction data and human feedback (as GPT-3.5 and GPT-4 were) to follow instructions. Encoder-only models cannot generate text at all, and an encoder-decoder adds an encoder that brings little when the input is just the previous dialogue text.

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

# %% [markdown]
# **Step 1 - Generate with sampling, three times.** `do_sample=True` makes the model pick the next token randomly from its predicted distribution (`top_k` keeps the 50 most likely tokens, `temperature` controls how random the choice is). No `set_seed`, so every run can differ.

# %%
prompt = "Natural language processing is"

for run in range(3):
    output = generator(prompt, max_new_tokens=20, do_sample=True, top_k=50, temperature=0.9)
    print(f"Run {run + 1}: {output[0]['generated_text']}\n")

# %% [markdown]
# **Answer.** With sampling the model does not take the single most probable next token; it draws the next token at random from the probability distribution it predicts. Each draw can land on a different word, and once one word differs the rest of the continuation changes with it, so the same prompt gives different texts. `set_seed(42)` fixes the random number generator, which is why the runs repeat exactly with the seed, and `do_sample=False` (greedy decoding) removes the randomness completely.
