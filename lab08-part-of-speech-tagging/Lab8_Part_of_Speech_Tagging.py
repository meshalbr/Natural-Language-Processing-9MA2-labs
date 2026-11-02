# Lab 8 - Part-of-Speech Tagging
# The notebook Lab8_Part_of_Speech_Tagging.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Part-of-speech (POS)
#
# Part-of-speech (POS) tagging is a popular NLP technique which refers to categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context.

# %% [markdown]
# ![](images/img01.png)

# %% [markdown]
# ## Benefits of POS tagging

# %% [markdown]
# - For some applications, we want to focus on certain POS:
#
# E.g.
# Nouns are important for information retrival.
# Adjectives are important for sentiment analysis.
#
# - POS Taggning is mostly useful in Named Entity Recognition (NER) applications and extracting relations between words.
#
# ![](images/img02.png)
#
# **POS Tagging can be used in:**
# - Grammar correction
# - Spelling correction
# - Question-Answering

# %% [markdown]
# ## POS tagging using NLTK

# %% [markdown]
# ### List of POS tags and their meaning in NLTK
#
# | **Tag**  | **Part of Speech**                       | **Example Words**   | **Description**                     |
# | -------- | ---------------------------------------- | ------------------- | ----------------------------------- |
# | **CC**   | Coordinating conjunction                 | and, but, or        | Joins words or clauses              |
# | **CD**   | Cardinal number                          | one, two, 100       | Numbers                             |
# | **DT**   | Determiner                               | the, a, an, this    | Specifies a noun                    |
# | **EX**   | Existential there                        | there               | Indicates existence (“there is”)    |
# | **FW**   | Foreign word                             | bonjour, gracias    | Non-English word                    |
# | **IN**   | Preposition or subordinating conjunction | in, on, because     | Links nouns or clauses              |
# | **JJ**   | Adjective                                | happy, tall, blue   | Describes a noun                    |
# | **JJR**  | Adjective, comparative                   | bigger, faster      | Comparative adjective               |
# | **JJS**  | Adjective, superlative                   | biggest, fastest    | Superlative adjective               |
# | **LS**   | List item marker                         | 1., a., i.          | Used in lists                       |
# | **MD**   | Modal                                    | can, will, should   | Auxiliary expressing mood           |
# | **NN**   | Noun, singular or mass                   | dog, car, love      | Common singular noun                |
# | **NNS**  | Noun, plural                             | dogs, cars          | Common plural noun                  |
# | **NNP**  | Proper noun, singular                    | John, London        | Specific singular noun              |
# | **NNPS** | Proper noun, plural                      | Americans, Italians | Specific plural noun                |
# | **PDT**  | Predeterminer                            | all, both, half     | Comes before a determiner           |
# | **POS**  | Possessive ending                        | ’s, ’               | Indicates possession                |
# | **PRP**  | Personal pronoun                         | I, you, he, she     | Replaces a noun                     |
# | **PRP$** | Possessive pronoun                       | my, your, his       | Shows possession                    |
# | **RB**   | Adverb                                   | quickly, very       | Modifies verb, adjective, or adverb |
# | **RBR**  | Adverb, comparative                      | faster, better      | Comparative adverb                  |
# | **RBS**  | Adverb, superlative                      | fastest, best       | Superlative adverb                  |
# | **RP**   | Particle                                 | up, off             | Forms phrasal verbs                 |
# | **SYM**  | Symbol                                   | \$, %, +            | Mathematical or other symbols       |
# | **TO**   | to                                       | to                  | Infinitive marker or preposition    |
# | **UH**   | Interjection                             | wow, hey            | Expresses emotion                   |
# | **VB**   | Verb, base form                          | run, eat, go        | Base form of verb                   |
# | **VBD**  | Verb, past tense                         | ran, ate, went      | Past tense verb                     |
# | **VBG**  | Verb, gerund/present participle          | running, eating     | -ing form                           |
# | **VBN**  | Verb, past participle                    | eaten, gone         | Past participle                     |
# | **VBP**  | Verb, non-3rd person singular present    | run, eat            | Present tense (I/you/we/they)       |
# | **VBZ**  | Verb, 3rd person singular present        | runs, eats          | Present tense (he/she/it)           |
# | **WDT**  | Wh-determiner                            | which, that         | Introduces a clause                 |
# | **WP**   | Wh-pronoun                               | who, what           | Question or relative pronoun        |
# | **WP\$**  | Possessive wh-pronoun                    | whose               | Possessive interrogative            |

# %% [markdown]
# ### pos_tag method

# %%
import nltk
#nltk.download('averaged_perceptron_tagger_eng')

# %%
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "I received my new phone broken from Amazon."
tokens = word_tokenize(text)

for word in tokens:
    print(f'word: ({word}), part of speech: ({pos_tag([word])[0][1]})')

# %% [markdown]
# ## POS tagging using Spacy

# %% [markdown]
# **Coarse-grained**: Tells us the general grammatical category of a word.
# ![](images/img03.png)
#
# **Fine-grained**: Gives us more specific grammatical information.
#
# ![](images/img04.png)

# %% [markdown]
# ### Coarse-grained Part-of-speech Tags
# Every token is assigned a POS Tag from the following list:
#
#
# <table><tr><th>POS</th><th>DESCRIPTION</th><th>EXAMPLES</th></tr>
#
# <tr><td>ADJ</td><td>adjective</td><td>*big, old, green, incomprehensible, first*</td></tr>
# <tr><td>ADP</td><td>adposition</td><td>*in, to, during*</td></tr>
# <tr><td>ADV</td><td>adverb</td><td>*very, tomorrow, down, where, there*</td></tr>
# <tr><td>AUX</td><td>auxiliary</td><td>*is, has (done), will (do), should (do)*</td></tr>
# <tr><td>CONJ</td><td>conjunction</td><td>*and, or, but*</td></tr>
# <tr><td>CCONJ</td><td>coordinating conjunction</td><td>*and, or, but*</td></tr>
# <tr><td>DET</td><td>determiner</td><td>*a, an, the*</td></tr>
# <tr><td>INTJ</td><td>interjection</td><td>*psst, ouch, bravo, hello*</td></tr>
# <tr><td>NOUN</td><td>noun</td><td>*girl, cat, tree, air, beauty*</td></tr>
# <tr><td>NUM</td><td>numeral</td><td>*1, 2017, one, seventy-seven, IV, MMXIV*</td></tr>
# <tr><td>PART</td><td>particle</td><td>*'s, not,*</td></tr>
# <tr><td>PRON</td><td>pronoun</td><td>*I, you, he, she, myself, themselves, somebody*</td></tr>
# <tr><td>PROPN</td><td>proper noun</td><td>*Mary, John, London, NATO, HBO*</td></tr>
# <tr><td>PUNCT</td><td>punctuation</td><td>*., (, ), ?*</td></tr>
# <tr><td>SCONJ</td><td>subordinating conjunction</td><td>*if, while, that*</td></tr>
# <tr><td>SYM</td><td>symbol</td><td>*$, %, §, ©, +, −, ×, ÷, =, :), 😝*</td></tr>
# <tr><td>VERB</td><td>verb</td><td>*run, runs, running, eat, ate, eating*</td></tr>
# <tr><td>X</td><td>other</td><td>*sfpksdpsxmsa*</td></tr>
# <tr><td>SPACE</td><td>space</td></tr>

# %% [markdown]
# ### Fine-grained Part-of-speech Tags
# Tokens are subsequently given a fine-grained tag as determined by morphology:
#
# | Coarse-Grained Tag (`pos_`) | Fine-Grained Tag (`tag_`) | Description                              | Example(s)               |
# | ---------------------- | ------------------------- | ---------------------------------------- | ------------------------ |
# | **ADJ**                | JJ                        | Adjective                                | blue, quick, large       |
# |                        | JJR                       | Comparative adjective                    | bigger, faster           |
# |                        | JJS                       | Superlative adjective                    | biggest, fastest         |
# | **ADP**                | IN                        | Preposition or subordinating conjunction | in, on, because, after   |
# | **ADV**                | RB                        | Adverb                                   | quickly, very, too       |
# |                        | RBR                       | Comparative adverb                       | more, less               |
# |                        | RBS                       | Superlative adverb                       | most, least              |
# |                        | WRB                       | Wh-adverb                                | where, when, how         |
# | **AUX**                | MD                        | Modal auxiliary                          | can, will, should, may   |
# | **CCONJ**              | CC                        | Coordinating conjunction                 | and, but, or             |
# | **DET**                | DT                        | Determiner                               | the, a, this, those      |
# |                        | PDT                       | Predeterminer                            | all, both, half          |
# |                        | WDT                       | Wh-determiner                            | which, that              |
# | **INTJ**               | UH                        | Interjection                             | oh, wow, um, hey         |
# | **NOUN**               | NN                        | Noun, singular or mass                   | cat, water, information  |
# |                        | NNS                       | Noun, plural                             | cats, dogs               |
# | **NUM**                | CD                        | Cardinal number                          | one, 42, twenty          |
# | **PART**               | POS                       | Possessive ending (’s)                   | ’s (as in “John’s”)      |
# |                        | RP                        | Particle                                 | up, off                  |
# |                        | TO                        | “to” (infinitive marker or preposition)  | to (as in “to go”)       |
# | **PRON**               | EX                        | Existential “there”                      | there (as in “there is”) |
# |                        | PRP                       | Personal pronoun                         | I, you, he, she          |
# |                        | PRP\$                      | Possessive pronoun                       | my, your, his, her       |
# |                        | WP                        | Wh-pronoun                               | who, what, whom          |
# |                        | WP\$                       | Possessive wh-pronoun                    | whose                    |
# | **PROPN**              | NNP                       | Proper noun, singular                    | London, Mary             |
# |                        | NNPS                      | Proper noun, plural                      | Beatles, Americas        |
# | **PUNCT**              | ``                        | Opening quotation mark                   | “                        |
# |                        | ''                        | Closing quotation mark                   | ”                        |
# |                        | (                         | Left parenthesis                         | (                        |
# |                        | )                         | Right parenthesis                        | )                        |
# |                        | ,                         | Comma                                    | ,                        |
# |                        | :                         | Colon / semicolon / ellipsis             | :, ;, …                  |
# |                        | .                         | Sentence-final punctuation               | ., !, ?                  |
# |                        | --                        | Dash / double hyphen                     | --                       |
# | **SYM**                | SYM                       | Symbol                                   | %, +, =, &               |
# |                        | \$                         | Dollar sign                              | \$                        |
# |                        | #                         | Number sign                              | #                        |
# | **VERB**               | VB                        | Verb, base form                          | eat, go, see             |
# |                        | VBD                       | Verb, past tense                         | ate, went, saw           |
# |                        | VBG                       | Verb, gerund/present participle          | eating, going            |
# |                        | VBN                       | Verb, past participle                    | eaten, gone              |
# |                        | VBP                       | Verb, non-3rd person singular present    | run, eat (I run)         |
# |                        | VBZ                       | Verb, 3rd person singular present        | runs, eats               |
# | **X**                  | FW                        | Foreign word                             | déjà, d’où               |
# |                        | LS                        | List item marker                         | 1), a.                   |

# %%
import spacy
nlp = spacy.load('en_core_web_sm')

# %%
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# %% [markdown]
# - To view the coarse POS tag use `token.pos_`
# - To view the fine-grained tag use `token.tag_`
# - To view the description of either type of tag use `spacy.explain(tag)`

# %%
for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

# %% [markdown]
# In the English language, the same word can have different meanings, even within the same sentence.
#
# For this reason, morphology is important. **spaCy** uses machine learning algorithms to best predict the use of a token in a sentence. Is *"I read books on NLP"* present or past tense?

# %%
doc = nlp(u'I read books on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

# %%
doc = nlp(u'I read a book on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

# %% [markdown]
#
# What tag will be assigned if you added **'always'**?

# %%
doc = nlp(u'I always read a book on NLP.')
r = doc[2]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

# %% [markdown]
# ### Visualizing Parts of Speech
# spaCy offers an outstanding visualizer called **displaCy**:

# %%
from spacy import displacy

# %%
# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# %%
# Render the dependency parse immediately inside Jupyter:
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})

# %% [markdown]
# ## Example: Find the adjective words in a sentance

# %%
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

# %%
text =  "Mobile is very handy. Packed with good performances and nice camera. This is my first iPhone. Very good experience with Apple. Battery is worst. I am hardly getting 5 hours screen on time. Positive thing is I came to know how many hours I was spending on my mobile. Because of poor battery now I am only using mobile for 4 hours a day."

print(text, end='\n\n')

for sent in sent_tokenize(text):
    tokens = word_tokenize(sent)
    part_of_speech = pos_tag(tokens)

    for i in part_of_speech:
        if 'JJ' in i:
            print(i)
    print('============================')

# %% [markdown]
# ---

# %% [markdown]
# ### Task 1
#
# sen = nlp(u"I like watching football matches. I do not know how to play it though")
#
# Use the spacy document object below to print out the following:
#
# 1. POS tag of the word "play". (Use sen[#].pos_)
# 2. Fine-grained POS tag for the word "play". (Use sen[#].tag_)
# 3. The meaning of the Fine-grained POS tag produced in (ii). (Use spacy.explain(sen[#].tag_)
# 4. Print the text, coarse-grained POS tags, fine-grained POS tags, and the explanation for the tags for all the words in the sentence.
#
# **Hint:**
# ```
# for word in sen:
#  print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
# ```

# %%

# %% [markdown]
# ### Task 2
#
# Compare the explanation of the words “google”  and “table” in the following sentences.
# 1. “Did you search on Google?”
# 2. “Let us google it.”
# 3. “The has placed the food on the table”
# 4. “She plans to table a motion for debate in Parliament”
#
# **Hint:**
# `word = sen[#]`

# %%

# %% [markdown]
# ### Task 3
#
# Visualize the part of speech of each word in the sentence below.
#
# “I like to play football. I hated it in my childhood though”
#
# **Hint:** `displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})`

# %%
