# Lab 8 - Part-of-Speech Tagging

Part-of-speech (POS) tagging with NLTK and spaCy: the Penn Treebank tag set used by `nltk.pos_tag`, spaCy's coarse-grained (`pos_`) and fine-grained (`tag_`) tags with `spacy.explain`, how context changes a tag ("I read books" vs "I read a book"), dependency visualization with displaCy, and a small example that pulls the adjectives out of a product review.

## Files

| File | Content |
|------|---------|
| [Lab8_Part_of_Speech_Tagging.ipynb](Lab8_Part_of_Speech_Tagging.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab8_Part_of_Speech_Tagging.py](Lab8_Part_of_Speech_Tagging.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, tag tables, code, outputs) |
| [images/](images/) | Figures used in the lab |

## Topics covered

- What POS tagging is and where it helps: information retrieval (nouns), sentiment analysis (adjectives), NER, relation extraction, grammar and spelling correction, question answering
- NLTK: the full list of Penn Treebank tags (CC, CD, DT, ... VBZ, WDT, WP) and `pos_tag` on tokenized text
- spaCy: coarse-grained tags (ADJ, ADP, ADV, AUX, NOUN, VERB, ...) versus fine-grained tags (JJ, IN, RB, MD, NN, VBD, ...), `token.pos_`, `token.tag_`, `spacy.explain`
- Morphology and context: the same word gets different tags in different sentences
- displaCy dependency rendering inside Jupyter
- Filtering tokens by tag (all `JJ` adjectives in a review)

## Tasks

1. For `sen = nlp("I like watching football matches. I do not know how to play it though")` print the POS tag of "play" (`sen[#].pos_`), its fine-grained tag (`sen[#].tag_`), the meaning of that tag (`spacy.explain`), and then the text, coarse tag, fine tag and explanation for every word in the sentence.
2. Compare the explanation of the words "google" and "table" in: "Did you search on Google?", "Let us google it.", "The has placed the food on the table", "She plans to table a motion for debate in Parliament".
3. Visualize the part of speech of each word in "I like to play football. I hated it in my childhood though" with `displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})`.

## Running

```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
jupyter notebook Lab8_Part_of_Speech_Tagging.ipynb
```

No dataset is needed; all sentences are in the notebook.
