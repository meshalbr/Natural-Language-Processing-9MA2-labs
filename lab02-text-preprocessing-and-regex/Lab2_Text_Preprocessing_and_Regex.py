# Lab 2 - Text Pre-processing and Regular Expressions
# The notebook Lab2_Text_Preprocessing_and_Regex.ipynb as one Python file, cell by cell, in the same order.
# Open it in VS Code (Python + Jupyter extensions) and press Shift+Enter on a cell to run it in the
# interactive window. '# %% [markdown]' cells are the notebook's text, '# %%' cells are its code.

# %% [markdown]
# # Lab2: Text Pre-processing and Regular Expressions

# %% [markdown]
#  In this lab, you will learn essential techniques for preparing and cleaning text data using various pre-processing steps. Additionally, you will gain hands-on experience with regular expressions, a powerful tool for pattern matching and text manipulation in natural language processing (NLP) tasks.

# %% [markdown]
# ---

# %% [markdown]
# ## Regular Expressions

# %% [markdown]
# Regular expressions (regex) extremely useful in extracting information from any text by searching for one or more matches of a specific search pattern
#
# https://www.regexpal.com/

# %% [markdown]
# #### Regular Expressions use cases
#
# - Data pre-processing
# - Pattern matching
# - Text feature Engineering
# - Web scraping
# - Data validation
# - Data extraction
#
# An example use case is extracting all hashtags from a tweet, or getting email addresses or phone numbers from large unstructured text content.

# %% [markdown]
# ## Regular Expressions with Python
#
# Python provides a convenient built-in module for managing regular expressions:
#
# > import re
#
# We can import it as we import any other Python library.

# %% [markdown]
# ### Important symbols in Regular Expression (For your reference)
#
# Let’s start with the basic regular expression characters and some examples:
#
# - (^) : Matches the expression to its right, at the start of a string before it finds a line break.
# - ($) : Matches the expression to its left, at the end of a string before it finds a line break.
# - (.) : Matches any character except newline.
# - (a) : Matches exactly one character.
# - (ab) : Matches the string ab.
#
# ----------------------------------------------------------------------------
# Quantifiers:
#
# - (a|b) : Matches expression a or b. If a is matched first, b is not checked;
# - (+) : Matches the expression to its left 1 or more times;
# - (*) : Matches the expression to its left 0 or more times;
# - (?) : Matches the expression to its left 0 or 1 times.
# ----------------------------------------------------------------------------
# Character Classes:
#
# - \w : Matches alphanumeric characters, that is a-z, A-Z, 0–9, and underscore(_);
# - \W : Matches non-alphanumeric characters, that is except a-z, A-Z, 0–9, and _;
# - \d : Matches digits, from 0–9;
# - \D : Matches any non-digits;
# - \s : Matches whitespace characters, which also include the \t, \n, \r, and space characters;
# - \S : Matches non-whitespace characters.
# - \n : Matches a newline character;
# - \t : Matches a tab character;
# - \b : Matches the word boundary (or empty string) at the start and end of a word;
# - \B : Matches where \b does not, that is non-word boundary.
#
# ----------------------------------------------------------------------------
# Sets:
#
# - [abc] : Matches either a, b, or c. It does not match abc;
# - [a-z] : Matches any alphabet from a to z;
# - [A-Z] : Matches any alphabets in capital from A to Z;
# - [a\-p] : Matches a, -, or p. It matches - because \ escapes it;
# - [-z] : Matches - or z;
# - [a-z0–9] : Matches characters from a to z or from 0 to 9.
# - [(+*)] : Special characters become literal inside a set, so this matches (, +, *, or );
# - [^ab5] : Adding ^ excludes any character in the set. Here, it matches characters that are not a, b, or 5;
# ----------------------------------------------------------------------------
#
#
# ### Regular Expression functions

# %% [markdown]
# #### search
#
# Search for pattern occurrences in a string using the search function of the **re module**. This function returns a match object, containing the first matched substring (or None, if it doesn’t exist) and its position inside the original string.

# %%
import re

text = 'I am enjoying the NLP course.'

print(re.search("I", text)) #r"I" means it's a raw string literal
print(re.search("se.$", text)) #$ Search at the end of the string
print(re.search("am", text))
print(re.search("m", text))
print(re.search("AI", text))

# %% [markdown]
# #### match
#
# The match function is similar to search, but it only tries to match the pattern at the beginning of the target string.

# %%
import re
re.match?

# %%
import re

text = 'I am enjoying the NLP course.'

print(re.search("I.*", text)) #global
print(re.match("I.*", text)) #local >> search at the start of the string

print(re.search("enjoying.*", text))
print(re.match("enjoying.*", text))

# %%
import re

text = '1999 was the year I was born.'

print(re.search(r"[a-zA-Z]+", text))
print(re.match(r"[a-zA-Z]+", text))

# %% [markdown]
# Search does a global search, whereas match does a local search!
#
# **Match** is often used when you need to check if the string starts with a specific pattern.
#
#
# **Search** is used when you want to find a pattern anywhere within the string.

# %% [markdown]
# #### findall
#
# The findall function looks for all the pattern matches in the target string (whereas search and match look only for the first occurrence).
#
# - `re.findall()` returns a list of all matches.
#
# - `re.finditer()` returns an iterator of match objects.
#
# - `re.search() and re.match()` both return a single match object for the first occurrence (but differ in where they start matching: match() matches only at the start of the string, search() anywhere).

# %%
import re

text1 = "a 1 b 2 c 3"
text2 = "a1 b2 c 3"
text3 = "a 1 b 222 c 3"

print(re.findall(r"\d+", text1)) #match any digit
print(re.findall(r"\d+", text2))
print(re.findall(r"\d+", text3))

# %% [markdown]
# #### sub
#
# The sub is a function that finds patterns in the target string and substitute them with another string.
#
# flags=re.I --> non-case-sensitive
#
# count = specify number of matches you want to replace

# %%
import re

re.sub?

# %%
import re

text = 'I am enjoying the Math course.'

print(re.sub(r"Math", "NLP", text, count=1, flags=re.I)) #flags=re.I >> Ignore the case (no matter if M or m)
                                                 # count = 1 >> number of word I need it to change. count= 0 by default

# %% [markdown]
# ---

# %% [markdown]
# ## Text pre-processing steps

# %% [markdown]
# Text preprocessing involves transforming text into a clean and consistent format that can then be fed into a model for further analysis and learning. Raw text data might contain unwanted or unimportant text due to which our results might not give efficient accuracy, and might make it hard to understand and analyze.

# %% [markdown]
# **The various text preprocessing steps are:**
#
# 1. Tokenization.
# 2. Lower casing.
# 3. Stop words removal.
# 4. Stemming.
# 5. Lemmatization.
#
# Before implementing the pre-processing, let us understand the concept first.

# %% [markdown]
# ### Tokenization
#
# Tokenization is used in NLP to split paragraphs and sentences into smaller units that can be more easily assigned meaning.
#
# The first step of the NLP process is gathering the data (a sentence) and breaking it into understandable parts (words).

# %% [markdown]
# - **Sentence Tokenization**

# %%
!pip install nltk
nltk.download('punkt') #Punkt is a pretrained unsupervised machine learning tokenizer model

# %%
import nltk

text = "I'm enjoying the NLP course! I am also learning new concepts."

print(nltk.sent_tokenize(text))

# %% [markdown]
# - **Word Tokenzitaion**

# %%
import nltk

text = "I'm enjoying the NLP course!"

print(nltk.word_tokenize(text)) #I'm is tokenized to two tokens [I,m], course! is also tokenized to two tokens [course,!]

# %%
text = "I'm enjoying the NLP course!"
text.split()

# %% [markdown]
# ### Lower Casting

# %% [markdown]
# Converting a word to lower case (NLP -> nlp). Words like Book and book mean the same but when not converted to the lower case those two are represented as two different words in the vector space model (resulting in more dimensions).

# %%
text = "I'm enjoying the NLP course!"
text = text.lower()
print(text)

# %% [markdown]
# ### Stemming

# %% [markdown]
# Stemming is basically removing the suffix from a word and reduce it to its root word. For example: “Flying” is a word and its suffix is “ing”, if we remove “ing” from “Flying” then we will get base word or root word which is “Fly”. We uses these suffix to create a new word from original stem word.

# %% [markdown]
# https://www.nltk.org/howto/stem.html

# %% [markdown]
# #### What is PorterStemmer?
#
# It is one of the most popular stemming methods proposed in 1980. It is based on the idea that the suffixes in the English language are made up of a combination of smaller and simpler suffixes. This stemmer is known for its speed and simplicity. The main applications of Porter Stemmer include data mining and Information retrieval. However, its applications are only limited to English words. Also, the group of stems is mapped on to the same stem and the output stem is not necessarily a meaningful word. The algorithms are fairly lengthy in nature and are known to be the oldest stemmer.
#
# > Advantage: It produces the best output as compared to other stemmers and it has less error rate.
#
# > Limitation:  Morphological variants produced are not always real words.

# %%
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer() # create a PorterStemmer object

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word + ' --> ' + ps.stem(word))

# %% [markdown]
# #### What is SnowballStemmer?
#
# It is a stemming algorithm which is also known as the Porter2 stemming algorithm as it is a better version of the Porter Stemmer since some issues of it were fixed in this stemmer.
#
# > Advantage: It is slightly faster computation time than porter, with a reasonably large community around it.

# %%
import nltk
from nltk.stem.snowball import SnowballStemmer

sn = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word+' --> '+sn.stem(word))

# %% [markdown]
# #### Comparing between porter and snowball

# %%
words = ['generous','generation','generously','generate']

for word in words:
    print(word+' --> '+ps.stem(word))
    print(word+' --> '+sn.stem(word))
    print('---------------------------------------')

# %% [markdown]
# **Note**: Spacy does not provide stemming

# %% [markdown]
# ### Lemmatization

# %% [markdown]
# Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meanings to one word.
#
# > One major difference with stemming is that lemmatize takes a part of speech parameter (pos) If not supplied, the default is “noun.”

# %% [markdown]
# | Part of Speech          | Description                | Example Words           |
# | ----------------------- | -------------------------- | ----------------------- |
# | **Noun (N)**            | Person, place, thing, idea | dog, city, love, Python |
# | **Verb (V)**            | Action or state            | run, eat, is, was, jump |
# | **Adjective (ADJ)**     | Describes a noun           | happy, blue, fast       |
# | **Adverb (ADV)**        | Describes a verb/adjective | quickly, very, well     |
# | **Pronoun (PRON)**      | Replaces a noun            | he, she, it, they       |
# | **Preposition (ADP)**   | Shows relationship         | in, on, under, with     |
# | **Conjunction (CONJ)**  | Connects words/clauses     | and, but, or            |
# | **Determiner (DET)**    | Introduces a noun          | a, an, the              |
# | **Interjection (INTJ)** | Expresses emotion          | wow, oh, hey            |

# %%
#nltk.download('wordnet')

# %%
import nltk
from nltk.stem import WordNetLemmatizer

text = 'I am enjoying AI courses. I am taking NLP course.'
lemmatizer = WordNetLemmatizer()

for word in nltk.word_tokenize(text):
    print(f"{word}: ", lemmatizer.lemmatize(word))

# %%
# Changing pos to verb
for word in nltk.word_tokenize(text):
    print(f"{word}: ", lemmatizer.lemmatize(word, pos='v'))

# %% [markdown]
# You may go through Spacy's lemmatizer: https://spacy.io/api/lemmatizer

# %% [markdown]
# ### Stop Words Removal

# %% [markdown]
# The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”.

# %% [markdown]
# #### Why do we need to remove stopwords?

# %% [markdown]
# By removing these words, we remove the low-level information from our text in order to give more focus to the important information. In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task.
#
# - reduces the dataset size
# - reduces the training time

# %% [markdown]
# #### Do we always remove stopwords? **NO!**

# %% [markdown]
# We do not always remove the stop words. The removal of stop words is highly dependent on the task we are performing and the goal we want to achieve. For example, if we are training a model that can perform the sentiment analysis task, we might not remove the stop words.
#
# Movie review: “The movie was not good at all.”
#
# Text after removal of stop words: “movie good”

# %%
import spacy
nlp = spacy.load('en_core_web_sm')

# %%
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))

# %%
print(nlp.vocab['myself'].is_stop)
print(nlp.vocab['mystery'].is_stop)

# %%
print (nlp.vocab['btw'].is_stop) #print False

# %%
nlp.Defaults.stop_words.add('btw')

nlp.vocab['btw'].is_stop = True # You must update the is_stop flag for the word you added

print (nlp.vocab['btw'].is_stop)

# %%
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False

print(nlp.vocab['beyond'].is_stop)

# %% [markdown]
# ### Stop words in NLTK

# %%
import nltk
nltk.download('stopwords')

# %%
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(len(stop_words))

# %%
print(stop_words)

# %% [markdown]
# ---

# %% [markdown]
# # Text pre-processing and Regular Expression Tasks:

# %% [markdown]
# # Task 1:
# To make you understand better about the need of text pre-processing. We will do a small project. In this project, you need to find how many hashtags and the top 10 hashtags used in the dataset.
#
# The dataset is about people tweets about apple company on Twitter. You can find the dataset here:
#
# https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts

# %%

# %% [markdown]
# # Task 2: Using re.compile()
# Given the following text: "This year is 2021"
#
# Write a Python program that:
#
# 1. Use re.compile() to create a regular expression pattern that matches one or more digits.
# 2. Print the type of the compiled pattern.
# 3. Use the compiled pattern to replace 2021 with 2022.
# 4. Print the updated text.
# 5. Briefly explain what re.compile() does and why it is useful when working with regular expressions.

# %%

# %% [markdown]
# # Task 3: Using re.split()
# Given the following text = "a 11 b 2 3 c 4"
#
# Write a Python program that:
#
# 1. Use re.split() with a regular expression pattern to split the text whenever one or more digits appear.
# 2. Print the resulting list.
# 3. Briefly explain: what re.split() does.

# %%

# %% [markdown]
# # Task 4: Tokenization Using spaCy
#
# Processes the following sentence: "I'm enjoying the NLP course!"
#
# Write a Python program that:
#
# 1. Use spaCy library and load the small English model en_core_web_sm.
# 2. Iterate through the text and print each token on a separate line.
# 3. Briefly explain: What spacy.load() does.
# 5. Tokenize the same text using nltk library and notice the differences compared to spaCy output.

# %%
