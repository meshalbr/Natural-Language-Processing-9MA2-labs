# Lab 7 - Large Language Models

This file is the complete lab as a readable document: the explanations, the code, and the printed outputs, in the same order as the notebook [Lab7_Large_Language_Models.ipynb](Lab7_Large_Language_Models.ipynb).
Open the notebook to run the cells yourself.

---

# Lab 7 — Large Language Models (LLMs)

## The Three Main Transformer Model Families

| Family | Example Models | Main Use |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Classification, sentiment analysis, NER |
| **Encoder–decoder** | T5, BART | Translation, summarization, question answering |
| **Decoder-only** | GPT family | Text generation, conversation, completion |

## 1. Setup

```python
!pip -q install transformers sentencepiece torch
```

```python
from transformers import pipeline, set_seed

print("Libraries loaded successfully!")
```

Output:

```text
Libraries loaded successfully!
```

# Part A — Decoder-Only Model

## Decoder-only example: GPT-style Text Generation

A decoder-only model predicts the next token based on previous tokens.

We will use **DistilGPT-2**, a small GPT-style model.

```python
generator = pipeline("text-generation", model="distilgpt2")

prompt = "Natural language processing is"

output = generator(prompt, max_new_tokens=30, do_sample=False)

print(output[0]["generated_text"])
```

Output:

```text
Loading weights:   0%|          | 0/76 [00:00<?, ?it/s]
[transformers] Both `max_new_tokens` (=30) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Natural language processing is a common feature of the language processing process.
```

# Part B — Encoder–Decoder Model

## Encoder–decoder example: Instruction / Text-to-Text Generation

Encoder–decoder models contain two main parts:

1. The **encoder** reads the input.
2. The **decoder** produces the output.

We will use **FLAN-T5-small**.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

prompt = "Translate to French: I love natural language processing."

# Convert text into tokens
output = model.generate(**tokenizer(prompt, return_tensors="pt"))

# Convert tokens back to text
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Output:

```text
Loading weights:   0%|          | 0/190 [00:00<?, ?it/s]
[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
Je s'agissant de la technologie de linguage naturelle.
```

# Part C — Encoder-Only Model

## Encoder-only example: Sentiment Classification

Encoder-only models read the full input and create a representation of it.

They are very useful when the task is to **understand or classify text**.

Here we use a DistilBERT sentiment model.

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

text = "This NLP lab is easy and interesting."

result = classifier(text)
print(result)
```

Output:

```text
Loading weights:   0%|          | 0/104 [00:00<?, ?it/s]
[{'label': 'POSITIVE', 'score': 0.999778687953949}]
```

# Tasks

Complete the following tasks.

## Task 1 — Model Families

For each task below, choose the most suitable model family:

1. Sentiment classification
2. Machine translation
3. Open-ended text generation
4. Named Entity Recognition
5. Text summarization

```python
# your solution here
```

## Task 2 — Encoder-only Model

Use the sentiment classifier on these three sentences:

```text
The movie was excellent.
The lecture was difficult to understand.
I enjoyed learning about transformers.
```

Print the predicted label and score for each sentence.

```python
# your solution here
```

## Task 3 — Encoder–Decoder Model

Use the FLAN-T5 model to perform the following:

- Summarize a short paragraph.

```python
# your solution here
```

## Task 5 — Short Reflection

Answer in 3–5 sentences:

**Why is a decoder-only model suitable for a chatbot such as ChatGPT?**

# Optional Challenge

Try generating text several times with the same prompt.

Remove:

```python
set_seed(42)
```

and run the sampling cell multiple times.

### Question

Why do you sometimes get different outputs even when the prompt is the same?

**Hint:** Sampling introduces randomness when choosing the next token.
