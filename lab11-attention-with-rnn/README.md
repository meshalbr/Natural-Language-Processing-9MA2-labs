# Lab 11 - Attention with RNN

The attention mechanism on a small question-answering dataset. Questions are tokenized and padded, an RNN with an attention layer (PyTorch) is trained to pick the right answer group for each question, and the attention weights are plotted to see which words the model focuses on.

## Files

| File | Content |
|------|---------|
| [Lab11_Attention_with_RNN.ipynb](Lab11_Attention_with_RNN.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab11_Attention_with_RNN.py](Lab11_Attention_with_RNN.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs) |
| [images/](images/) | Figures used in the lab |
| [dataset/Lab_11_QnA_Dataset.csv](dataset/Lab_11_QnA_Dataset.csv) | 168 questions in 12 answer groups (columns: `question`, `answer_id`, `answer`) |

## Topics covered

- Attention: letting the network weight the relevant parts of the input when predicting
- Encoder-decoder architecture with attention
- Preprocessing: lowercase, Keras `Tokenizer`, `pad_sequences(padding="post")`, conversion to PyTorch tensors
- The model: `nn.Embedding` → `nn.RNN` → attention scores from `nn.Linear(32, 1)` with `softmax` over the time steps → weighted sum → `nn.Linear` classifier
- Training with `cross_entropy` and Adam for 50 epochs
- Reading the attention weights with a bar chart per question word

## Tasks

1. Understand attention weights (easy): modify Step 6 to visualize the attention weights for three different questions from the dataset, generate a bar chart for each, and identify the word with the highest attention weight in each question. Do the words with the highest weights appear important for understanding the question? Explain.
2. Compare attention for similar questions (medium): select two questions with similar meanings but different wording, pass both through the trained model, visualize their attention weights and compare the words receiving the highest weights. Does the model focus on similar or different words, and what does this suggest about how attention works?

## Running

```bash
pip install pandas torch tensorflow matplotlib
jupyter notebook Lab11_Attention_with_RNN.ipynb
```

The notebook was written for Google Colab (`from google.colab import files`, path `/content/Lab_11_QnA_Dataset_Download.csv`). Locally, remove the Colab import and load `dataset/Lab_11_QnA_Dataset.csv` instead.
