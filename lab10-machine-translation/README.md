# Lab 10 - Introduction to Machine Translation

Machine translation and the neural (encoder-decoder) approach. A tiny English-French example is tokenized with Keras `Tokenizer`, padded with `pad_sequences`, and used to train a sequence-to-sequence LSTM model (Embedding → LSTM encoder → RepeatVector → LSTM decoder → TimeDistributed softmax) that is then used to translate a sentence.

## Files

| File | Content |
|------|---------|
| [Lab10_Introduction_to_Machine_Translation.ipynb](Lab10_Introduction_to_Machine_Translation.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab10_Introduction_to_Machine_Translation.py](Lab10_Introduction_to_Machine_Translation.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, code, outputs) |
| [images/](images/) | Figures used in the lab |

## Topics covered

- What machine translation is; challenges (ambiguity, idiomatic expressions, low-resource languages) and applications
- Neural machine translation: one network learns the source language, the target language and the mapping between them
- Tokenizing each language with `Tokenizer`, `texts_to_sequences`, `word_index` (index 0 reserved for padding)
- Padding sequences to the same length with `pad_sequences`
- The LSTM encoder-decoder model in Keras: `Embedding`, `LSTM`, `RepeatVector`, `LSTM(return_sequences=True)`, `TimeDistributed(Dense(softmax))`, `sparse_categorical_crossentropy`
- Inference: predict probabilities per time step, `argmax` to word ids, `sequences_to_texts` back to words

## Tasks

1. Understanding the model layers, in your own words: what the Embedding layer does to a word index; the role of the encoder LSTM; what the decoder LSTM does; why `softmax` is used in the final layer.
2. Use case, English-French translation for a global e-commerce company that wants to translate product listings, customer reviews and other content for French-speaking customers. Dataset: [Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench) on Kaggle.

## Running

```bash
pip install tensorflow scikit-learn numpy
jupyter notebook Lab10_Introduction_to_Machine_Translation.ipynb
```

The example data is three sentence pairs written in the notebook. The Kaggle dataset for Task 2 is not included; download it and load it with pandas.
