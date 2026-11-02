# Lab 7 - Large Language Models

The three transformer model families, each tried with a small Hugging Face model: a decoder-only model (DistilGPT-2) for text generation, an encoder-decoder model (FLAN-T5-small) for text-to-text tasks such as translation, and an encoder-only model (DistilBERT fine-tuned on SST-2) for sentiment classification.

## Files

| File | Content |
|------|---------|
| [Lab7_Large_Language_Models.ipynb](Lab7_Large_Language_Models.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab7_Large_Language_Models.py](Lab7_Large_Language_Models.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs) |

## Topics covered

- Encoder-only (BERT, RoBERTa), encoder-decoder (T5, BART) and decoder-only (GPT) families and their main uses
- `transformers.pipeline` for text generation and sentiment analysis
- Decoder-only: next-token prediction with `distilgpt2`, `max_new_tokens`, greedy decoding with `do_sample=False`
- Encoder-decoder: `AutoTokenizer` and `AutoModelForSeq2SeqLM` with `google/flan-t5-small`, `generate` and `decode`
- Encoder-only: `distilbert-base-uncased-finetuned-sst-2-english` returning a label and a score
- Sampling and `set_seed`: why the same prompt can give different outputs

## Tasks

1. Model families: for sentiment classification, machine translation, open-ended text generation, named entity recognition and text summarization, choose the most suitable model family.
2. Encoder-only model: run the sentiment classifier on "The movie was excellent.", "The lecture was difficult to understand." and "I enjoyed learning about transformers." and print the predicted label and score for each.
3. Encoder-decoder model: use FLAN-T5 to summarize a short paragraph.
4. Short reflection (3 to 5 sentences): why is a decoder-only model suitable for a chatbot such as ChatGPT?
5. Optional challenge: remove `set_seed(42)`, run the sampling cell several times with the same prompt, and explain why the outputs differ.

All tasks are solved at the end of the notebook, step by step: a short explanation above each code cell, the code, its output, and a written answer where the task asks a question.

## Running

```bash
pip install transformers sentencepiece torch
jupyter notebook Lab7_Large_Language_Models.ipynb
```

The models are downloaded from the Hugging Face Hub on first use (a few hundred MB in total); no dataset is needed.
