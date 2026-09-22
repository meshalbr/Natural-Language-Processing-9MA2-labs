# Lab 2 - Text Pre-processing and Regular Expressions

This file is the complete lab as a readable document: the explanations, the code, and the printed outputs, in the same order as the notebook [Lab2_Text_Preprocessing_and_Regex.ipynb](Lab2_Text_Preprocessing_and_Regex.ipynb).
Open the notebook to run the cells yourself.

---

# Lab2: Text Pre-processing and Regular Expressions

In this lab, you will learn essential techniques for preparing and cleaning text data using various pre-processing steps. Additionally, you will gain hands-on experience with regular expressions, a powerful tool for pattern matching and text manipulation in natural language processing (NLP) tasks.

---

## Regular Expressions

Regular expressions (regex) extremely useful in extracting information from any text by searching for one or more matches of a specific search pattern

https://www.regexpal.com/

#### Regular Expressions use cases

- Data pre-processing
- Pattern matching
- Text feature Engineering
- Web scraping
- Data validation
- Data extraction

An example use case is extracting all hashtags from a tweet, or getting email addresses or phone numbers from large unstructured text content.

## Regular Expressions with Python

Python provides a convenient built-in module for managing regular expressions:

> import re

We can import it as we import any other Python library.

### Important symbols in Regular Expression (For your reference)

Let’s start with the basic regular expression characters and some examples:

- (^) : Matches the expression to its right, at the start of a string before it finds a line break.
- ($) : Matches the expression to its left, at the end of a string before it finds a line break.
- (.) : Matches any character except newline.
- (a) : Matches exactly one character.
- (ab) : Matches the string ab.

----------------------------------------------------------------------------
Quantifiers:

- (a|b) : Matches expression a or b. If a is matched first, b is not checked;
- (+) : Matches the expression to its left 1 or more times;
- (*) : Matches the expression to its left 0 or more times;
- (?) : Matches the expression to its left 0 or 1 times.
----------------------------------------------------------------------------
Character Classes:

- \w : Matches alphanumeric characters, that is a-z, A-Z, 0–9, and underscore(_);
- \W : Matches non-alphanumeric characters, that is except a-z, A-Z, 0–9, and _;
- \d : Matches digits, from 0–9;
- \D : Matches any non-digits;
- \s : Matches whitespace characters, which also include the \t, \n, \r, and space characters;
- \S : Matches non-whitespace characters.
- \n : Matches a newline character;
- \t : Matches a tab character;
- \b : Matches the word boundary (or empty string) at the start and end of a word;
- \B : Matches where \b does not, that is non-word boundary.

----------------------------------------------------------------------------
Sets:

- [abc] : Matches either a, b, or c. It does not match abc;
- [a-z] : Matches any alphabet from a to z;
- [A-Z] : Matches any alphabets in capital from A to Z;
- [a\-p] : Matches a, -, or p. It matches - because \ escapes it;
- [-z] : Matches - or z;
- [a-z0–9] : Matches characters from a to z or from 0 to 9.
- [(+*)] : Special characters become literal inside a set, so this matches (, +, *, or );
- [^ab5] : Adding ^ excludes any character in the set. Here, it matches characters that are not a, b, or 5;
----------------------------------------------------------------------------

### Regular Expression functions

#### search

Search for pattern occurrences in a string using the search function of the **re module**. This function returns a match object, containing the first matched substring (or None, if it doesn’t exist) and its position inside the original string.

```python
import re

text = 'I am enjoying the NLP course.'

print(re.search("I", text)) #r"I" means it's a raw string literal
print(re.search("se.$", text)) #$ Search at the end of the string
print(re.search("am", text))
print(re.search("m", text))
print(re.search("AI", text))
```

Output:

```text
<re.Match object; span=(0, 1), match='I'>
<re.Match object; span=(26, 29), match='se.'>
<re.Match object; span=(2, 4), match='am'>
<re.Match object; span=(3, 4), match='m'>
None
```

#### match

The match function is similar to search, but it only tries to match the pattern at the beginning of the target string.

```python
import re
re.match?
```

Output:

```text
Signature: re.match(pattern, string, flags=0)
Docstring:
Try to apply the pattern at the start of the string, returning
a Match object, or None if no match was found.
File:      c:\users\mesha\miniconda3\lib\re\__init__.py
Type:      function
```

```python
import re

text = 'I am enjoying the NLP course.'

print(re.search("I.*", text)) #global
print(re.match("I.*", text)) #local >> search at the start of the string

print(re.search("enjoying.*", text))
print(re.match("enjoying.*", text))
```

Output:

```text
<re.Match object; span=(0, 29), match='I am enjoying the NLP course.'>
<re.Match object; span=(0, 29), match='I am enjoying the NLP course.'>
<re.Match object; span=(5, 29), match='enjoying the NLP course.'>
None
```

```python
import re

text = '1999 was the year I was born.'

print(re.search(r"[a-zA-Z]+", text))
print(re.match(r"[a-zA-Z]+", text))
```

Output:

```text
<re.Match object; span=(5, 8), match='was'>
None
```

Search does a global search, whereas match does a local search!

**Match** is often used when you need to check if the string starts with a specific pattern.

**Search** is used when you want to find a pattern anywhere within the string.

#### findall

The findall function looks for all the pattern matches in the target string (whereas search and match look only for the first occurrence).

- `re.findall()` returns a list of all matches.

- `re.finditer()` returns an iterator of match objects.

- `re.search() and re.match()` both return a single match object for the first occurrence (but differ in where they start matching: match() matches only at the start of the string, search() anywhere).

```python
import re

text1 = "a 1 b 2 c 3"
text2 = "a1 b2 c 3"
text3 = "a 1 b 222 c 3"

print(re.findall(r"\d+", text1)) #match any digit
print(re.findall(r"\d+", text2))
print(re.findall(r"\d+", text3))
```

Output:

```text
['1', '2', '3']
['1', '2', '3']
['1', '222', '3']
```

#### sub

The sub is a function that finds patterns in the target string and substitute them with another string.

flags=re.I --> non-case-sensitive

count = specify number of matches you want to replace

```python
import re

re.sub?
```

Output:

```text
Signature: re.sub(pattern, repl, string, count=0, flags=0)
Docstring:
Return the string obtained by replacing the leftmost
non-overlapping occurrences of the pattern in string by the
replacement repl.  repl can be either a string or a callable;
if a string, backslash escapes in it are processed.  If it is
a callable, it's passed the Match object and must return
a replacement string to be used.
File:      c:\users\sumay\anaconda3\lib\re\__init__.py
Type:      function
```

```python
import re

text = 'I am enjoying the Math course.'

print(re.sub(r"Math", "NLP", text, count=1, flags=re.I)) #flags=re.I >> Ignore the case (no matter if M or m)
                                                 # count = 1 >> number of word I need it to change. count= 0 by default
```

Output:

```text
I am enjoying the NLP course.
```

---

## Text pre-processing steps

Text preprocessing involves transforming text into a clean and consistent format that can then be fed into a model for further analysis and learning. Raw text data might contain unwanted or unimportant text due to which our results might not give efficient accuracy, and might make it hard to understand and analyze.

**The various text preprocessing steps are:**

1. Tokenization.
2. Lower casing.
3. Stop words removal.
4. Stemming.
5. Lemmatization.

Before implementing the pre-processing, let us understand the concept first.

### Tokenization

Tokenization is used in NLP to split paragraphs and sentences into smaller units that can be more easily assigned meaning.

The first step of the NLP process is gathering the data (a sentence) and breaking it into understandable parts (words).

- **Sentence Tokenization**

```python
!pip install nltk
nltk.download('punkt') #Punkt is a pretrained unsupervised machine learning tokenizer model
```

Output:

```text
True
```

```python
import nltk

text = "I'm enjoying the NLP course! I am also learning new concepts."

print(nltk.sent_tokenize(text))
```

Output:

```text
["I'm enjoying the NLP course!", 'I am also learning new concepts.']
```

- **Word Tokenzitaion**

```python
import nltk

text = "I'm enjoying the NLP course!"

print(nltk.word_tokenize(text)) #I'm is tokenized to two tokens [I,m], course! is also tokenized to two tokens [course,!]
```

Output:

```text
['I', "'m", 'enjoying', 'the', 'NLP', 'course', '!']
```

```python
text = "I'm enjoying the NLP course!"
text.split()
```

Output:

```text
["I'm", 'enjoying', 'the', 'NLP', 'course!']
```

### Lower Casting

Converting a word to lower case (NLP -> nlp). Words like Book and book mean the same but when not converted to the lower case those two are represented as two different words in the vector space model (resulting in more dimensions).

```python
text = "I'm enjoying the NLP course!"
text = text.lower()
print(text)
```

Output:

```text
i'm enjoying the nlp course!
```

### Stemming

Stemming is basically removing the suffix from a word and reduce it to its root word. For example: “Flying” is a word and its suffix is “ing”, if we remove “ing” from “Flying” then we will get base word or root word which is “Fly”. We uses these suffix to create a new word from original stem word.

https://www.nltk.org/howto/stem.html

#### What is PorterStemmer?

It is one of the most popular stemming methods proposed in 1980. It is based on the idea that the suffixes in the English language are made up of a combination of smaller and simpler suffixes. This stemmer is known for its speed and simplicity. The main applications of Porter Stemmer include data mining and Information retrieval. However, its applications are only limited to English words. Also, the group of stems is mapped on to the same stem and the output stem is not necessarily a meaningful word. The algorithms are fairly lengthy in nature and are known to be the oldest stemmer.

> Advantage: It produces the best output as compared to other stemmers and it has less error rate.

> Limitation:  Morphological variants produced are not always real words.

```python
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer() # create a PorterStemmer object

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word + ' --> ' + ps.stem(word))
```

Output:

```text
run --> run
runner --> runner
running --> run
ran --> ran
runs --> run
easily --> easili
fairly --> fairli
```

#### What is SnowballStemmer?

It is a stemming algorithm which is also known as the Porter2 stemming algorithm as it is a better version of the Porter Stemmer since some issues of it were fixed in this stemmer.

> Advantage: It is slightly faster computation time than porter, with a reasonably large community around it.

```python
import nltk
from nltk.stem.snowball import SnowballStemmer

sn = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs','easily','fairly']

for word in words:
    print(word+' --> '+sn.stem(word))
```

Output:

```text
run --> run
runner --> runner
running --> run
ran --> ran
runs --> run
easily --> easili
fairly --> fair
```

#### Comparing between porter and snowball

```python
words = ['generous','generation','generously','generate']

for word in words:
    print(word+' --> '+ps.stem(word))
    print(word+' --> '+sn.stem(word))
    print('---------------------------------------')
```

Output:

```text
generous --> gener
generous --> generous
---------------------------------------
generation --> gener
generation --> generat
---------------------------------------
generously --> gener
generously --> generous
---------------------------------------
generate --> gener
generate --> generat
---------------------------------------
```

**Note**: Spacy does not provide stemming

### Lemmatization

Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meanings to one word.

> One major difference with stemming is that lemmatize takes a part of speech parameter (pos) If not supplied, the default is “noun.”

| Part of Speech          | Description                | Example Words           |
| ----------------------- | -------------------------- | ----------------------- |
| **Noun (N)**            | Person, place, thing, idea | dog, city, love, Python |
| **Verb (V)**            | Action or state            | run, eat, is, was, jump |
| **Adjective (ADJ)**     | Describes a noun           | happy, blue, fast       |
| **Adverb (ADV)**        | Describes a verb/adjective | quickly, very, well     |
| **Pronoun (PRON)**      | Replaces a noun            | he, she, it, they       |
| **Preposition (ADP)**   | Shows relationship         | in, on, under, with     |
| **Conjunction (CONJ)**  | Connects words/clauses     | and, but, or            |
| **Determiner (DET)**    | Introduces a noun          | a, an, the              |
| **Interjection (INTJ)** | Expresses emotion          | wow, oh, hey            |

```python
#nltk.download('wordnet')
```

```python
import nltk
from nltk.stem import WordNetLemmatizer

text = 'I am enjoying AI courses. I am taking NLP course.'
lemmatizer = WordNetLemmatizer()

for word in nltk.word_tokenize(text):
    print(f"{word}: ", lemmatizer.lemmatize(word))
```

Output:

```text
I:  I
am:  am
enjoying:  enjoying
AI:  AI
courses:  course
.:  .
I:  I
am:  am
taking:  taking
NLP:  NLP
course:  course
.:  .
```

```python
# Changing pos to verb
for word in nltk.word_tokenize(text):
    print(f"{word}: ", lemmatizer.lemmatize(word, pos='v'))
```

Output:

```text
I:  I
am:  be
enjoying:  enjoy
AI:  AI
courses:  course
.:  .
I:  I
am:  be
taking:  take
NLP:  NLP
course:  course
.:  .
```

You may go through Spacy's lemmatizer: https://spacy.io/api/lemmatizer

### Stop Words Removal

The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. Examples of a few stop words in English are “the”, “a”, “an”, “so”, “what”.

#### Why do we need to remove stopwords?

By removing these words, we remove the low-level information from our text in order to give more focus to the important information. In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task.

- reduces the dataset size
- reduces the training time

#### Do we always remove stopwords? **NO!**

We do not always remove the stop words. The removal of stop words is highly dependent on the task we are performing and the goal we want to achieve. For example, if we are training a model that can perform the sentiment analysis task, we might not remove the stop words.

Movie review: “The movie was not good at all.”

Text after removal of stop words: “movie good”

```python
import spacy
nlp = spacy.load('en_core_web_sm')
```

```python
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))
```

Output:

```text
{'twenty', '‘re', 'become', 'her', 'thereby', 'through', 'hereafter', 'else', 'front', 'during', 'own', 'something', 'becoming', 'whoever', 'please', 'call', 'which', 'such', 'part', 'throughout', 'may', 'by', 'wherein', 'whereas', 'its', 'who', 'down', 'off', 'all', 'though', 'therefore', 'make', 'thru', 'four', 'have', 'always', 'beforehand', 'top', 'at', 'why', 'and', 'even', 'no', 'this', 'too', 'still', 'under', 'ever', 'am', 'whose', 'hereupon', 'enough', 'eleven', 'with', 'others', '‘m', 'as', 'yours', 'never', 'say', 'whereupon', "'s", 'any', 'last', 'me', 'our', 'thus', 'becomes', 'now', '’ve', 'therein', '’s', 'out', 'nine', 'also', 'few', 'hers', 'these', 'used', 'whither', 'six', 'former', 'us', 'anyhow', 'latter', 'when', 'everyone', 'around', 'into', 'not', 'from', 'alone', 'his', 'you', 'them', 'amongst', 'amount', 'within', 'afterwards', 'until', 'hundred', 'must', 'another', 'regarding', 'are', 'least', 'does', 'would', 'beside', 'how', 'together', 'wherever', "n't", 'seems', 'so', 'further', 'empty', 'has', 'beyond', 'see', 'again', 'two', 'seemed', 'anything', 'n‘t', 'themselves', 'ours', 'were', 'seeming', 'whole', '’d', 'nowhere', 'without', 'whence', 'show', 'both', 'really', 'could', 'first', 'about', 'side', 'three', 'more', 'anyway', 'otherwise', 'above', 'perhaps', 'had', 'but', 'should', 'across', 'except', 'much', 'give', 'do', 'n’t', 'yourself', 'mine', 'some', 'on', "'m", 'your', 'very', 'anywhere', 'can', 'himself', 'five', 'ourselves', 'serious', 'in', "'re", 'often', 'that', 'toward', 'everything', 'none', 'their', 'using', 'if', "'ve", 'then', 'already', '’ll', 'move', 'upon', 'put', 'nevertheless', 'rather', 'towards', 'namely', 'someone', 'since', 'other', 'whereby', 'might', 'hence', 'twelve', 'yet', 'been', 'many', 'made', 'everywhere', '‘ll', '‘ve', 'there', '‘s', 'forty', 'before', 'itself', 'meanwhile', 'each', 'anyone', "'d", 'only', 'ca', 'myself', 'to', '’re', 'along', 'the', 'elsewhere', 'a', 'over', '’m', 'full', 'i', 'was', 'she', 'once', 'an', 'against', 'take', 'between', 'what', 'formerly', 'fifty', 'nothing', 'neither', 'it', 'herself', 'will', 'fifteen', 'here', 'next', 'than', 'sometimes', 'same', 'less', 'of', 'due', 'nobody', 'indeed', 'ten', 'sometime', 'seem', 'via', 'well', 'for', 'most', 'be', 'although', 'various', 'where', 'third', 'below', 'thereafter', 'almost', 'go', 'because', 'or', 'several', 'latterly', 'is', 'being', 'somewhere', 'back', 'whom', 'one', 'herein', 'after', 'every', 'among', 'while', 'noone', 'those', 'whereafter', 'either', 'yourselves', 'whether', 'quite', 'somehow', 'onto', '‘d', 'cannot', 'up', "'ll", 'became', 'did', 're', 'whatever', 'hereby', 'unless', 'whenever', 'mostly', 'we', 'just', 'done', 'thence', 'they', 'name', 'he', 'nor', 'eight', 'him', 'keep', 'doing', 'sixty', 'bottom', 'however', 'thereupon', 'my', 'behind', 'per', 'besides', 'moreover', 'get'}
326
```

```python
print(nlp.vocab['myself'].is_stop)
print(nlp.vocab['mystery'].is_stop)
```

Output:

```text
True
False
```

```python
print (nlp.vocab['btw'].is_stop) #print False
```

Output:

```text
False
```

```python
nlp.Defaults.stop_words.add('btw')

nlp.vocab['btw'].is_stop = True # You must update the is_stop flag for the word you added

print (nlp.vocab['btw'].is_stop)
```

Output:

```text
True
```

```python
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False

print(nlp.vocab['beyond'].is_stop)
```

Output:

```text
False
```

### Stop words in NLTK

```python
import nltk
nltk.download('stopwords')
```

Output:

```text
True
```

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(len(stop_words))
```

Output:

```text
198
```

```python
print(stop_words)
```

Output:

```text
{'about', "they're", 'themselves', 'then', 'over', 'hers', 'were', 'not', 'what', 'how', 'it', "shouldn't", 're', "should've", 'their', "we're", 'itself', 'needn', "you're", "you'd", 'can', 'more', 'couldn', 'ours', 'having', "you've", 'than', "doesn't", "hasn't", 'from', 'into', 'is', 's', 'there', 'you', "we'll", 'herself', 'does', 'a', "it'd", 'have', 'theirs', 'don', 'below', 'again', 'wouldn', 'few', 'they', 'when', 'off', 'shan', 'down', 'with', 'both', 'until', 'hasn', 'in', 'o', 'an', 'aren', "it's", 'mightn', 'whom', 'had', 'them', 'didn', 'her', 'him', 'between', 've', 'on', 'doesn', 'haven', "i've", "hadn't", 'its', 'm', 'my', 'same', "wasn't", "aren't", 'do', 'but', 'as', 'after', "haven't", 'ain', 'if', 'myself', "she's", "didn't", "don't", 'above', "they'll", 'now', "they've", 'some', 'nor', 'out', "weren't", 'those', 'ma', "he's", 'here', 'too', 'won', 'she', 'ourselves', 'for', "he'd", 'just', 'while', 'our', 'wasn', "won't", 'before', 'yours', 'was', "mustn't", "we've", 'isn', "she'd", 'yourselves', 'so', 'i', 'your', 'and', 'very', "she'll", 'at', 'been', "wouldn't", 'each', 'why', "it'll", "he'll", 'the', "i'd", 'of', "they'd", 'against', 'will', 'are', 'during', 'should', 'only', 'be', 'once', 'such', 'most', 'through', 'which', 'me', 'up', 'am', "you'll", 'no', 'd', "isn't", 'where', 'that', "couldn't", 'under', "i'll", 'doing', 'has', 'his', 'because', 'these', "shan't", "i'm", 'hadn', "needn't", 'weren', 'any', 'did', 'further', 'or', 'own', "that'll", 'himself', "mightn't", 'shouldn', 'who', 'y', 'yourself', 'all', 'being', 'this', 'mustn', 't', 'he', 'we', 'll', "we'd", 'other', 'to', 'by'}
```

---

# Text pre-processing and Regular Expression Tasks:

# Task 1:
To make you understand better about the need of text pre-processing. We will do a small project. In this project, you need to find how many hashtags and the top 10 hashtags used in the dataset.

The dataset is about people tweets about apple company on Twitter. You can find the dataset here:

https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts

```python
# your solution here
```

# Task 2: Using re.compile()
Given the following text: "This year is 2021"

Write a Python program that:

1. Use re.compile() to create a regular expression pattern that matches one or more digits.
2. Print the type of the compiled pattern.
3. Use the compiled pattern to replace 2021 with 2022.
4. Print the updated text.
5. Briefly explain what re.compile() does and why it is useful when working with regular expressions.

```python
# your solution here
```

# Task 3: Using re.split()
Given the following text = "a 11 b 2 3 c 4"

Write a Python program that:

1. Use re.split() with a regular expression pattern to split the text whenever one or more digits appear.
2. Print the resulting list.
3. Briefly explain: what re.split() does.

```python
# your solution here
```

# Task 4: Tokenization Using spaCy

Processes the following sentence: "I'm enjoying the NLP course!"

Write a Python program that:

1. Use spaCy library and load the small English model en_core_web_sm.
2. Iterate through the text and print each token on a separate line.
3. Briefly explain: What spacy.load() does.
5. Tokenize the same text using nltk library and notice the differences compared to spaCy output.

```python
# your solution here
```
