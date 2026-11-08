# Lab 12 - Transformers

How a Transformer turns a sentence into a prediction. A pretrained DistilBERT sentiment model tokenizes a sentence, builds embeddings, uses self-attention inside each layer and classifies the sentence as positive or negative with a softmax. The task then opens up `bert-base-uncased` to count its layers and attention heads and to read one attention matrix.

## Files

| File | Content |
|------|---------|
| [Lab12_Transformers.ipynb](Lab12_Transformers.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab12_Transformers.py](Lab12_Transformers.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs) |
| [images/](images/) | Figures used in the lab and the attention heatmap from the task |

## Topics covered

- What a Transformer is and why it replaced RNNs and LSTMs for most NLP tasks
- Loading a pretrained model and tokenizer from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`)
- Tokenization with the special `[CLS]` and `[SEP]` tokens
- Embeddings and self-attention: reading attention weights with `output_attentions=True` and eager attention
- From hidden states to a prediction: the classification head, logits and softmax probabilities
- Model architecture from the config: layers, attention heads per layer, hidden size

## Tasks

Load the pretrained `bert-base-uncased` model using Hugging Face Transformers.

1. Print the number of Transformer layers and the number of attention heads per layer.
2. Calculate and print the total number of attention heads across all Transformer layers.
3. Using the sentence `"Transformers use self-attention to understand language."`, extract the attention weights from the third Transformer layer and second attention head.
4. Explain what the rows and columns of the attention matrix represent.

The task is solved at the end of the notebook, step by step: a short explanation above each code cell, the code, its output, a heatmap of the attention matrix, and a written answer for question 4.

## Running

```bash
pip install transformers torch matplotlib pandas
jupyter notebook Lab12_Transformers.ipynb
```

No dataset is needed. The first run downloads `distilbert-base-uncased-finetuned-sst-2-english` and `bert-base-uncased` from the Hugging Face Hub (about 700 MB together), so it needs an internet connection.
