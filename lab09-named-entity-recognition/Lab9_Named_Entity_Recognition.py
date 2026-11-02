# Lab 9 - Named Entity Recognition
# The notebook Lab9_Named_Entity_Recognition.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Name Entity Recognition (NER)

# %% [markdown]
#
# Name Entity Recognition (NER) is a fundamental task in NLP that focuses on identifying and classifying entities such as names of people, organizations, locations, dates, and more within a given text. NER plays a crucial role in various NLP applications, including information retrieval, question answering, and sentiment analysis.

# %% [markdown]
# ![bert-based-named-entity-recognition-ner-1536x804.jpg](images/img01.jpg)

# %% [markdown]
# At its core, it is just a two-step process:
#
# 1.   Detecting the entities from the text
# 2.   Classifying them into different categories

# %% [markdown]
# ## NER using spaCy

# %% [markdown]
#
# SpaCy offers robust support for Named Entity Recognition (NER). It can recognize a variety of named entities, such as persons, organizations, dates, locations, and more. Here's more information about using spaCy for NER:

# %% [markdown]
# ![pwn-623x660.png](images/img02.png)

# %% [markdown]
# When performing Named Entity Recognition (NER) using spaCy or other NLP tools, you'll encounter various entity types, and they are represented as labels. Here are some common entity types and their labels:

# %% [markdown]
# | **Entity Type**                          | **Label**    | **Description**                                                          | **Examples**                       |
# | ---------------------------------------- | ------------ | ------------------------------------------------------------------------ | ---------------------------------- |
# | **Person**                               | `PERSON`     | Represents names of people.                                              | *John Smith*, *Barack Obama*       |
# | **Organization**                         | `ORG`        | Names of companies, institutions, agencies, or other organized entities. | *Microsoft*, *United Nations*      |
# | **Location**                             | `GPE`, `LOC` | Geographical locations such as cities, countries, or landmarks.          | *New York*, *Eiffel Tower*         |
# | **Date**                                 | `DATE`       | Calendar dates, time intervals, or relative dates.                       | *January 1, 2023*, *tomorrow*      |
# | **Time**                                 | `TIME`       | Clock times or time intervals.                                           | *2:30 PM*, *5 minutes*             |
# | **Money**                                | `MONEY`      | Monetary values, including currencies and amounts.                       | *$100*, *€50*                      |
# | **Percent**                              | `PERCENT`    | Percentage values.                                                       | *50%*, *10 percent*                |
# | **Product**                              | `PRODUCT`    | Names of products or goods.                                              | *iPhone*, *Coca-Cola*              |
# | **Event**                                | `EVENT`      | Names of events or occurrences.                                          | *World Cup*, *New Year's Eve*      |
# | **Law**                                  | `LAW`        | Legal references or statutes.                                            | *Copyright Act*, *First Amendment* |
# | **Language**                             | `LANGUAGE`   | Names of languages.                                                      | *English*, *Spanish*               |
# | **Nationality/Religion/Political Group** | `NORP`       | Nationalities, religious groups, or political affiliations.              | *American*, *Buddhist*             |

# %% [markdown]
# In this example, we will use spaCy to perform NER on some sample text. Make sure you have spaCy installed; if not, you can install it using :
# `pip install spacy`

# %%
# Import the spaCy library and load the English language model
import spacy
nlp = spacy.load("en_core_web_sm")

# Sample text for NER
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. It is known for its iPhone and Mac products."
doc = nlp(text)

print(f"Entities recognized in text : {doc.ents}") #Return a tuple of entities recognized in text

# Extract and print named entities
for entity in doc.ents:
    print(f"Entity: {entity.text}, Label: {entity.label_}")

# %%
# Iterate through recognized entities and print their details
for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"Label: {ent.label_}")
    print(f"Start Position: {ent.start}")
    print(f"End Position: {ent.end}")
    print(f"Entity Type: {spacy.explain(ent.label_)}")
    print("")

# %% [markdown]
# ## Create a custom Named Entity Model
# In spaCy, you can add custom entities that have not been identified in the pre-trained models by training your own Named Entity Recognition (NER) model. This involves providing labeled examples of your custom entities and then fine-tuning the model.

# %% [markdown]
# Here is an example of how to train a custom NER model in spaCy: https://spacy.io/usage/training#ner

# %%
from spacy.training.iob_utils import offsets_to_biluo_tags

# The text to annotate
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. It is known for its iPhone and Mac products."

# --- Add custom entities annotations manually ---
annotations = [
    (text.find("Apple Inc."), text.find("Apple Inc.") + len("Apple Inc."), "ORG"),
    (text.find("Steve Jobs"), text.find("Steve Jobs") + len("Steve Jobs"), "PERSON"),
    (text.find("Cupertino, California"), text.find("Cupertino, California") + len("Cupertino, California"), "GPE")]

# --- Step 2: Check entity alignment ---
doc = nlp.make_doc(text) #splitted to tokens
tags = offsets_to_biluo_tags(doc, annotations) #converts annotations into BILUO tags (Begin, Inside, Last, Unit, Outside).
print("Alignment Tags:", tags[:10])

# %% [markdown]
# ### Visualization

# %%
# Load and use the custom model for entity recognition
# (the model is trained and saved in Task 3 below; run that task first)
custom_nlp = spacy.load("custom_ner_model")
doc2 = custom_nlp(text)

# Entity visualization
from spacy import displacy
displacy.render(doc2, style="ent", jupyter=True)

# %% [markdown]
# ---

# %% [markdown]
# ### Task 1
#
# 1. Ask a user to enter a sentence, and identify the named entities in the sentence.
#
# 2. Print the details of each entity in the sentence provided by the user.

# %% [markdown]
# **Step 1 - A function that prints the details of every entity.** For each entity: text, label with its explanation, token positions and character positions.

# %%
def show_entities(sentence):
    doc = nlp(sentence)
    if not doc.ents:
        print("No named entities found.")
    for ent in doc.ents:
        print(f"Entity: {ent.text}")
        print(f"   Label      : {ent.label_} ({spacy.explain(ent.label_)})")
        print(f"   Start token: {ent.start}, End token: {ent.end}")
        print(f"   Characters : {ent.start_char} to {ent.end_char}")

# %% [markdown]
# **Step 2 - Ask the user for a sentence and show its entities.** `input()` reads from the keyboard. When the notebook is run without a keyboard (automatic execution) a default sentence is used instead, so the cell still works.

# %%
try:
    sentence = input("Enter a sentence: ")
except Exception:
    sentence = "Meshal studied NLP at Imam Abdulrahman Bin Faisal University in Dammam in November 2026 and paid 50 dollars for a book by O'Reilly."
    print("Enter a sentence:", sentence)

show_entities(sentence)

# %% [markdown]
# ### Task 2
#
# Use the paragraph below to answer the following questions:
#
# “This agreement is made and entered into by and between ‘Abc & Co.’  and ‘Bcd LLC’ for term 1 year starting from April 1, 2020,  hereinafter collectively referred to as the Parties. Samuel P. Jackson in the place (New York) and on the date written below, with the following terms and conditions.”
#
# 1. Find any entity that has not been identified and add it.
#
# 2. Find the count of ‘PERSON’ and ‘ORG’ in the paragraph above.
#
# 3. Visualize the entities in the paragraph.

# %% [markdown]
# **Step 1 - Run the pre-trained model on the paragraph** and list what it finds.

# %%
from spacy.tokens import Span
from collections import Counter

paragraph = ("This agreement is made and entered into by and between 'Abc & Co.' and 'Bcd LLC' for term 1 year "
             "starting from April 1, 2020, hereinafter collectively referred to as the Parties. Samuel P. Jackson "
             "in the place (New York) and on the date written below, with the following terms and conditions.")

doc = nlp(paragraph)

print("Entities found by the pre-trained model:")
for ent in doc.ents:
    print(f"   {ent.text!r:22} {ent.label_}")

# %% [markdown]
# **Step 2 - Add the entities that were missed (question 1).** The entities we expect are listed with their labels; any that the model did not find is turned into a `Span` with `char_span` and added with `doc.set_ents` (existing entities that overlap the new one are dropped).

# %%
wanted = {"Abc & Co.": "ORG", "Bcd LLC": "ORG", "Samuel P. Jackson": "PERSON", "New York": "GPE"}
found_texts = {ent.text for ent in doc.ents}

new_spans = []
for entity_text, label in wanted.items():
    if entity_text not in found_texts:
        start_char = paragraph.find(entity_text)
        new_span = doc.char_span(start_char, start_char + len(entity_text), label=label)
        if new_span is not None:
            new_spans.append(new_span)
            print(f"Adding missing entity: {entity_text!r} as {label}")

if new_spans:
    kept = [ent for ent in doc.ents if not any(ent.start < s.end and s.start < ent.end for s in new_spans)]
    doc.set_ents(kept + new_spans)

print("\nEntities after the correction:")
for ent in doc.ents:
    print(f"   {ent.text!r:22} {ent.label_}")

# %% [markdown]
# **Step 3 - Count PERSON and ORG (question 2).** `Counter` over the labels of the corrected entities.

# %%
counts = Counter(ent.label_ for ent in doc.ents)
print("PERSON count:", counts["PERSON"])
print("ORG count   :", counts["ORG"])

# %% [markdown]
# **Step 4 - Visualize the entities (question 3)** with displaCy in entity style.

# %%
from spacy import displacy

displacy.render(doc, style="ent", jupyter=True)

# %% [markdown]
# ### Task 3: Training a Custom NER Model
# text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. It is known for its Windows operating system and Office software products."
#
# 1. Create a training example using `Example.from_dict()` with the provided text and annotations.
# 2. Train the NER model for **20 iterations**
# 3. Save the trained model in a folder named `custom_ner_model`.
# 5. Load the saved model and use it to identify named entities in a sentence.

# %% [markdown]
# **Step 1 - The text and its annotations.** Each entity is given as `(start_char, end_char, label)`. The helper `span` finds the character positions so they are not typed by hand.

# %%
import random
from spacy.training import Example

text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. It is known for its Windows operating system and Office software products."

def span(s):
    start = text.find(s)
    return (start, start + len(s))

annotations = {"entities": [
    span("Microsoft")   + ("ORG",),
    span("Bill Gates")  + ("PERSON",),
    span("Paul Allen")  + ("PERSON",),
    span("Albuquerque") + ("GPE",),
    span("New Mexico")  + ("GPE",),
    span("Windows")     + ("PRODUCT",),
    span("Office")      + ("PRODUCT",),
]}
annotations

# %% [markdown]
# **Step 2 - Create the training example with `Example.from_dict()`.** It pairs the tokenized text with the gold entities and aligns the character offsets with the tokens. A fresh copy of the English model is loaded so the lab's `nlp` is not modified.

# %%
ner_nlp = spacy.load("en_core_web_sm")
example = Example.from_dict(ner_nlp.make_doc(text), annotations)

print("Aligned entity tags:", example.get_aligned("ENT_IOB", as_string=True))

# %% [markdown]
# **Step 3 - Train the NER component for 20 iterations.** The labels are registered on the `ner` pipe, the other pipes are disabled so only NER is updated, and `nlp.update` runs one training step per iteration (`drop=0.2` is dropout against overfitting).

# %%
ner = ner_nlp.get_pipe("ner")
for _, _, label in annotations["entities"]:
    ner.add_label(label)

other_pipes = [pipe for pipe in ner_nlp.pipe_names if pipe != "ner"]
with ner_nlp.disable_pipes(*other_pipes):
    optimizer = ner_nlp.resume_training()
    for iteration in range(20):
        random.seed(iteration)
        losses = {}
        ner_nlp.update([example], sgd=optimizer, drop=0.2, losses=losses)
        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1:2d}  loss = {losses['ner']:.3f}")

# %% [markdown]
# **Step 4 - Save the model.** `to_disk` writes the whole pipeline into the folder `custom_ner_model`.

# %%
ner_nlp.to_disk("custom_ner_model")
print("Model saved in the folder 'custom_ner_model'")

# %% [markdown]
# **Step 5 - Load the saved model and use it.** `spacy.load` reads the folder back; the model is tried on the training sentence and on a new sentence, and the entities are visualized with displaCy.

# %%
from spacy import displacy

custom_nlp = spacy.load("custom_ner_model")

for sentence in [text, "Paul Allen left Microsoft and Bill Gates kept improving Windows and Office in New Mexico."]:
    doc3 = custom_nlp(sentence)
    print(sentence)
    for ent in doc3.ents:
        print(f"   {ent.text:14} {ent.label_}")
    print()

displacy.render(custom_nlp(text), style="ent", jupyter=True)
