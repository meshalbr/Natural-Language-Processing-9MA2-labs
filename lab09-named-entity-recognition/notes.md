# Lab 9 - Named Entity Recognition

This file is the complete lab as a readable document: the explanations, the code, and the printed outputs, in the same order as the notebook [Lab9_Named_Entity_Recognition.ipynb](Lab9_Named_Entity_Recognition.ipynb).
Open the notebook to run the cells yourself.

---

# Name Entity Recognition (NER)

Name Entity Recognition (NER) is a fundamental task in NLP that focuses on identifying and classifying entities such as names of people, organizations, locations, dates, and more within a given text. NER plays a crucial role in various NLP applications, including information retrieval, question answering, and sentiment analysis.

![bert-based-named-entity-recognition-ner-1536x804.jpg](images/img01.jpg)

At its core, it is just a two-step process:

1.   Detecting the entities from the text
2.   Classifying them into different categories

## NER using spaCy

SpaCy offers robust support for Named Entity Recognition (NER). It can recognize a variety of named entities, such as persons, organizations, dates, locations, and more. Here's more information about using spaCy for NER:

![pwn-623x660.png](images/img02.png)

When performing Named Entity Recognition (NER) using spaCy or other NLP tools, you'll encounter various entity types, and they are represented as labels. Here are some common entity types and their labels:

| **Entity Type**                          | **Label**    | **Description**                                                          | **Examples**                       |
| ---------------------------------------- | ------------ | ------------------------------------------------------------------------ | ---------------------------------- |
| **Person**                               | `PERSON`     | Represents names of people.                                              | *John Smith*, *Barack Obama*       |
| **Organization**                         | `ORG`        | Names of companies, institutions, agencies, or other organized entities. | *Microsoft*, *United Nations*      |
| **Location**                             | `GPE`, `LOC` | Geographical locations such as cities, countries, or landmarks.          | *New York*, *Eiffel Tower*         |
| **Date**                                 | `DATE`       | Calendar dates, time intervals, or relative dates.                       | *January 1, 2023*, *tomorrow*      |
| **Time**                                 | `TIME`       | Clock times or time intervals.                                           | *2:30 PM*, *5 minutes*             |
| **Money**                                | `MONEY`      | Monetary values, including currencies and amounts.                       | *$100*, *€50*                      |
| **Percent**                              | `PERCENT`    | Percentage values.                                                       | *50%*, *10 percent*                |
| **Product**                              | `PRODUCT`    | Names of products or goods.                                              | *iPhone*, *Coca-Cola*              |
| **Event**                                | `EVENT`      | Names of events or occurrences.                                          | *World Cup*, *New Year's Eve*      |
| **Law**                                  | `LAW`        | Legal references or statutes.                                            | *Copyright Act*, *First Amendment* |
| **Language**                             | `LANGUAGE`   | Names of languages.                                                      | *English*, *Spanish*               |
| **Nationality/Religion/Political Group** | `NORP`       | Nationalities, religious groups, or political affiliations.              | *American*, *Buddhist*             |

In this example, we will use spaCy to perform NER on some sample text. Make sure you have spaCy installed; if not, you can install it using :
`pip install spacy`

```python
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
```

Output:

```text
Entities recognized in text : (Apple Inc., Steve Jobs, Cupertino, California, Mac)
Entity: Apple Inc., Label: ORG
Entity: Steve Jobs, Label: PERSON
Entity: Cupertino, Label: GPE
Entity: California, Label: GPE
Entity: Mac, Label: ORG
```

```python
# Iterate through recognized entities and print their details
for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"Label: {ent.label_}")
    print(f"Start Position: {ent.start}")
    print(f"End Position: {ent.end}")
    print(f"Entity Type: {spacy.explain(ent.label_)}")
    print("")
```

Output:

```text
Entity: Apple Inc.
Label: ORG
Start Position: 0
End Position: 2
Entity Type: Companies, agencies, institutions, etc.

Entity: Steve Jobs
Label: PERSON
Start Position: 5
End Position: 7
Entity Type: People, including fictional

Entity: Cupertino
Label: GPE
Start Position: 8
End Position: 9
Entity Type: Countries, cities, states

Entity: California
Label: GPE
Start Position: 10
End Position: 11
Entity Type: Countries, cities, states

Entity: Mac
Label: ORG
Start Position: 19
End Position: 20
Entity Type: Companies, agencies, institutions, etc.
```

## Create a custom Named Entity Model
In spaCy, you can add custom entities that have not been identified in the pre-trained models by training your own Named Entity Recognition (NER) model. This involves providing labeled examples of your custom entities and then fine-tuning the model.

Here is an example of how to train a custom NER model in spaCy: https://spacy.io/usage/training#ner

```python
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
```

Output:

```text
Alignment Tags: ['B-ORG', 'L-ORG', 'O', 'O', 'O', 'B-PERSON', 'L-PERSON', 'O', 'B-GPE', 'I-GPE']
```

### Visualization

```python
# Load and use the custom model for entity recognition
# (the model is trained and saved in Task 3 below; run that task first)
custom_nlp = spacy.load("custom_ner_model")
doc2 = custom_nlp(text)

# Entity visualization
from spacy import displacy
displacy.render(doc2, style="ent", jupyter=True)
```

Output:

```text
<IPython.core.display.HTML object>
```

---

### Task 1

1. Ask a user to enter a sentence, and identify the named entities in the sentence.

2. Print the details of each entity in the sentence provided by the user.

**Step 1 - A function that prints the details of every entity.** For each entity: text, label with its explanation, token positions and character positions.

```python
def show_entities(sentence):
    doc = nlp(sentence)
    if not doc.ents:
        print("No named entities found.")
    for ent in doc.ents:
        print(f"Entity: {ent.text}")
        print(f"   Label      : {ent.label_} ({spacy.explain(ent.label_)})")
        print(f"   Start token: {ent.start}, End token: {ent.end}")
        print(f"   Characters : {ent.start_char} to {ent.end_char}")
```

**Step 2 - Ask the user for a sentence and show its entities.** `input()` reads from the keyboard. When the notebook is run without a keyboard (automatic execution) a default sentence is used instead, so the cell still works.

```python
try:
    sentence = input("Enter a sentence: ")
except Exception:
    sentence = "Meshal studied NLP at Imam Abdulrahman Bin Faisal University in Dammam in November 2026 and paid 50 dollars for a book by O'Reilly."
    print("Enter a sentence:", sentence)

show_entities(sentence)
```

Output:

```text
Enter a sentence: Meshal studied NLP at Imam Abdulrahman Bin Faisal University in Dammam in November 2026 and paid 50 dollars for a book by O'Reilly.
Entity: NLP
   Label      : ORG (Companies, agencies, institutions, etc.)
   Start token: 2, End token: 3
   Characters : 15 to 18
Entity: Imam Abdulrahman Bin Faisal University
   Label      : ORG (Companies, agencies, institutions, etc.)
   Start token: 4, End token: 9
   Characters : 22 to 60
Entity: Dammam
   Label      : GPE (Countries, cities, states)
   Start token: 10, End token: 11
   Characters : 64 to 70
Entity: November 2026
   Label      : DATE (Absolute or relative dates or periods)
   Start token: 12, End token: 14
   Characters : 74 to 87
Entity: 50 dollars
   Label      : MONEY (Monetary values, including unit)
   Start token: 16, End token: 18
   Characters : 97 to 107
Entity: O'Reilly
   Label      : ORG (Companies, agencies, institutions, etc.)
   Start token: 22, End token: 23
   Characters : 122 to 130
```

### Task 2

Use the paragraph below to answer the following questions:

“This agreement is made and entered into by and between ‘Abc & Co.’  and ‘Bcd LLC’ for term 1 year starting from April 1, 2020,  hereinafter collectively referred to as the Parties. Samuel P. Jackson in the place (New York) and on the date written below, with the following terms and conditions.”

1. Find any entity that has not been identified and add it.

2. Find the count of ‘PERSON’ and ‘ORG’ in the paragraph above.

3. Visualize the entities in the paragraph.

**Step 1 - Run the pre-trained model on the paragraph** and list what it finds.

```python
from spacy.tokens import Span
from collections import Counter

paragraph = ("This agreement is made and entered into by and between 'Abc & Co.' and 'Bcd LLC' for term 1 year "
             "starting from April 1, 2020, hereinafter collectively referred to as the Parties. Samuel P. Jackson "
             "in the place (New York) and on the date written below, with the following terms and conditions.")

doc = nlp(paragraph)

print("Entities found by the pre-trained model:")
for ent in doc.ents:
    print(f"   {ent.text!r:22} {ent.label_}")
```

Output:

```text
Entities found by the pre-trained model:
   "between 'Abc & Co.'"  ORG
   "Bcd LLC'"             WORK_OF_ART
   '1 year'               DATE
   'April 1, 2020'        DATE
   'Samuel P. Jackson'    PERSON
   'New York'             GPE
```

**Step 2 - Add the entities that were missed (question 1).** The entities we expect are listed with their labels; any that the model did not find is turned into a `Span` with `char_span` and added with `doc.set_ents` (existing entities that overlap the new one are dropped).

```python
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
```

Output:

```text
Adding missing entity: 'Abc & Co.' as ORG
Adding missing entity: 'Bcd LLC' as ORG

Entities after the correction:
   'Abc & Co.'            ORG
   'Bcd LLC'              ORG
   '1 year'               DATE
   'April 1, 2020'        DATE
   'Samuel P. Jackson'    PERSON
   'New York'             GPE
```

**Step 3 - Count PERSON and ORG (question 2).** `Counter` over the labels of the corrected entities.

```python
counts = Counter(ent.label_ for ent in doc.ents)
print("PERSON count:", counts["PERSON"])
print("ORG count   :", counts["ORG"])
```

Output:

```text
PERSON count: 1
ORG count   : 2
```

**Step 4 - Visualize the entities (question 3)** with displaCy in entity style.

```python
from spacy import displacy

displacy.render(doc, style="ent", jupyter=True)
```

Output:

```text
<IPython.core.display.HTML object>
```

### Task 3: Training a Custom NER Model
text = "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. It is known for its Windows operating system and Office software products."

1. Create a training example using `Example.from_dict()` with the provided text and annotations.
2. Train the NER model for **20 iterations**
3. Save the trained model in a folder named `custom_ner_model`.
5. Load the saved model and use it to identify named entities in a sentence.

**Step 1 - The text and its annotations.** Each entity is given as `(start_char, end_char, label)`. The helper `span` finds the character positions so they are not typed by hand.

```python
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
```

Output:

```text
{'entities': [(0, 9, 'ORG'),
  (25, 35, 'PERSON'),
  (40, 50, 'PERSON'),
  (54, 65, 'GPE'),
  (67, 77, 'GPE'),
  (99, 106, 'PRODUCT'),
  (128, 134, 'PRODUCT')]}
```

**Step 2 - Create the training example with `Example.from_dict()`.** It pairs the tokenized text with the gold entities and aligns the character offsets with the tokens. A fresh copy of the English model is loaded so the lab's `nlp` is not modified.

```python
ner_nlp = spacy.load("en_core_web_sm")
example = Example.from_dict(ner_nlp.make_doc(text), annotations)

print("Aligned entity tags:", example.get_aligned("ENT_IOB", as_string=True))
```

Output:

```text
Aligned entity tags: [3, 2, 2, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2]
```

**Step 3 - Train the NER component for 20 iterations.** The labels are registered on the `ner` pipe, the other pipes are disabled so only NER is updated, and `nlp.update` runs one training step per iteration (`drop=0.2` is dropout against overfitting).

```python
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
```

Output:

```text
Iteration  5  loss = 0.005
Iteration 10  loss = 0.000
Iteration 15  loss = 0.000
Iteration 20  loss = 0.000
```

**Step 4 - Save the model.** `to_disk` writes the whole pipeline into the folder `custom_ner_model`.

```python
ner_nlp.to_disk("custom_ner_model")
print("Model saved in the folder 'custom_ner_model'")
```

Output:

```text
Model saved in the folder 'custom_ner_model'
```

**Step 5 - Load the saved model and use it.** `spacy.load` reads the folder back; the model is tried on the training sentence and on a new sentence, and the entities are visualized with displaCy.

```python
from spacy import displacy

custom_nlp = spacy.load("custom_ner_model")

for sentence in [text, "Paul Allen left Microsoft and Bill Gates kept improving Windows and Office in New Mexico."]:
    doc3 = custom_nlp(sentence)
    print(sentence)
    for ent in doc3.ents:
        print(f"   {ent.text:14} {ent.label_}")
    print()

displacy.render(custom_nlp(text), style="ent", jupyter=True)
```

Output:

```text
Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. It is known for its Windows operating system and Office software products.
   Microsoft      ORG
   Bill Gates     PERSON
   Paul Allen     PERSON
   Albuquerque    GPE
   New Mexico     GPE
   Windows        PRODUCT
   Office         PRODUCT

Paul Allen left Microsoft and Bill Gates kept improving Windows and Office in New Mexico.
   Paul Allen     PERSON
   Microsoft      ORG
   Bill Gates     PERSON
   Windows        PRODUCT
   Office         PRODUCT
   New Mexico     GPE
<IPython.core.display.HTML object>
```
