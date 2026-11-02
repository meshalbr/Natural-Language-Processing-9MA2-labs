# Lab 6 - Deep Learning for NLP

A first neural network for text classification. Yelp restaurant reviews are cleaned, split 80/20, turned into TF-IDF vectors and fed to a small PyTorch feed-forward network (1000 → 64 → 32 → 1) trained with `BCEWithLogitsLoss` and Adam; the model is then evaluated with accuracy, a classification report and a confusion matrix.

## Files

| File | Content |
|------|---------|
| [Lab6_Deep_Learning_for_NLP.ipynb](Lab6_Deep_Learning_for_NLP.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab6_Deep_Learning_for_NLP.py](Lab6_Deep_Learning_for_NLP.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs) |
| [images/](images/) | Figures used in the lab |
| [dataset/07-yelp-dataset.txt](dataset/07-yelp-dataset.txt) | 1000 Yelp reviews, tab-separated: `review`, `label` (0 = negative, 1 = positive) |

## Topics covered

- Deep learning for NLP tasks (text classification, sentiment analysis, question answering)
- Loading and preprocessing the Yelp dataset (lowercase, remove punctuation and extra spaces)
- Train/test split with `train_test_split`
- TF-IDF features (`max_features=1000`) and converting them to PyTorch tensors
- Building a feed-forward network with `nn.Sequential`, `nn.Linear` and `nn.ReLU`; why the sigmoid is left out of the model when using `BCEWithLogitsLoss`
- Training loop: forward pass, loss, backpropagation, optimizer step; epochs
- Evaluation: sigmoid on the logits, threshold 0.5, accuracy, precision/recall/F1, confusion matrix

## Tasks

1. Represent the reviews with Word2Vec instead of TF-IDF: tokenize the training and testing reviews, train a Word2Vec model on the training reviews with `vector_size=200`, `window=5`, `min_count=1`, `sg=1`, convert each review into a single vector, and convert the vectors into PyTorch tensors.
2. Build a different neural network on the Word2Vec vectors: input layer, three hidden layers of 128, 64 and 32 neurons with ReLU, and one output neuron. Train it with `BCEWithLogitsLoss`, Adam, learning rate 0.001 and 15 epochs, then calculate the test accuracy.

All tasks are solved at the end of the notebook, step by step: a short explanation above each code cell, the code, its output, and a written answer where the task asks a question.

## Running

```bash
pip install pandas scikit-learn torch gensim
jupyter notebook Lab6_Deep_Learning_for_NLP.ipynb
```

The loading cell reads `dataset/07-yelp-dataset.txt`, so the notebook runs from this folder as it is (on Colab, upload the file and change the path to `07-yelp-dataset.txt`).
