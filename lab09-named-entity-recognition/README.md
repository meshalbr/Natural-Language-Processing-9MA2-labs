# Lab 9 - Named Entity Recognition

Named Entity Recognition (NER) with spaCy: recognizing people, organizations, locations, dates and other entity types in text, reading the entity labels and positions, preparing custom annotations as BILUO tags with `offsets_to_biluo_tags`, training a custom NER model, and visualizing entities with displaCy.

## Files

| File | Content |
|------|---------|
| [Lab9_Named_Entity_Recognition.ipynb](Lab9_Named_Entity_Recognition.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [Lab9_Named_Entity_Recognition.py](Lab9_Named_Entity_Recognition.py) | The same notebook as one Python file, cell by cell (`# %%` cells, run with Shift+Enter in VS Code) |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, entity table, code, outputs) |
| [images/](images/) | Figures used in the lab |

## Topics covered

- NER as a two-step process: detect the entities, then classify them
- spaCy entity labels: PERSON, ORG, GPE/LOC, DATE, TIME, MONEY, PERCENT, PRODUCT, EVENT, LAW, LANGUAGE, NORP
- `doc.ents`, `ent.text`, `ent.label_`, `ent.start`, `ent.end`, `spacy.explain(ent.label_)`
- Custom entities: manual character-offset annotations, alignment check with `offsets_to_biluo_tags` (Begin, Inside, Last, Unit, Outside)
- Training a custom NER model (spaCy training guide) and loading it back with `spacy.load("custom_ner_model")`
- Entity visualization with `displacy.render(doc, style="ent", jupyter=True)`

## Tasks

1. Ask the user to enter a sentence, identify the named entities in it, and print the details of each entity.
2. For the paragraph "This agreement is made and entered into by and between 'Abc & Co.' and 'Bcd LLC' for term 1 year starting from April 1, 2020, hereinafter collectively referred to as the Parties. Samuel P. Jackson in the place (New York) and on the date written below, with the following terms and conditions.": find any entity that has not been identified and add it, count the PERSON and ORG entities, and visualize the entities.
3. Training a custom NER model for `text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. It is known for its Windows operating system and Office software products."`: create a training example with `Example.from_dict()` from the text and annotations, train the NER model for 20 iterations, save it in a folder named `custom_ner_model`, then load the saved model and use it to identify named entities in a sentence.

All tasks are solved at the end of the notebook, step by step: a short explanation above each code cell, the code, its output, and a written answer where the task asks a question.

## Running

```bash
pip install spacy
python -m spacy download en_core_web_sm
jupyter notebook Lab9_Named_Entity_Recognition.ipynb
```

The visualization cell loads `custom_ner_model` from the working directory, so run Task 3 (training and saving the model) before it. No dataset is needed.
