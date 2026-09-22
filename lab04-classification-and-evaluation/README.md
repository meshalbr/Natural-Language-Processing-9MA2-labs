# Lab 4 - Classification and Evaluation

A complete text classification pipeline for sentiment analysis. The notebook takes the Disneyland reviews dataset, maps star ratings to positive / neutral / negative labels, cleans the text, converts it to TF-IDF features, trains a linear SVM, and evaluates it with accuracy and a classification report.

## Files

| File | Content |
|------|---------|
| [Lab4_Classification_and_Evaluation.ipynb](Lab4_Classification_and_Evaluation.ipynb) | Original notebook |
| [notes.md](notes.md) | The lecture notes and dataset description |
| [code/01_load-and-clean-dataset.py](code/01_load-and-clean-dataset.py) | Load the CSV, check and drop missing values, map ratings to sentiment labels |
| [code/02_preprocess-text.py](code/02_preprocess-text.py) | Lowercase, remove punctuation, tokenize the reviews |
| [code/03_train-test-split-and-tfidf.py](code/03_train-test-split-and-tfidf.py) | 80/20 train-test split and TF-IDF vectorization (1000 features) |
| [code/04_train-and-evaluate-svm.py](code/04_train-and-evaluate-svm.py) | Train `SVC(kernel='linear')`, predict, print accuracy and classification report |
| [images/](images/) | Figure referenced from the notes |

The four code files are steps of one pipeline: run them in order in the same session, or run the notebook.

## Dataset

[Disneyland Reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews): about 42,000 Trip Advisor reviews of the Paris, California and Hong Kong parks.

| Column | Meaning |
|--------|---------|
| Review_ID | unique id |
| Rating | 1 (unsatisfied) to 5 (satisfied) |
| Year_Month | visit date |
| Reviewer_Location | visitor's country |
| Review_Text | the review |
| Disneyland_Branch | park location |

Ratings above 3 become "positive", below 3 "negative", and 3 "neutral".

## Task: Amazon reviews of unlocked phones

Using the [Amazon Reviews of Unlocked Mobile Phones](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones) dataset, classify reviews as positive, negative or neutral:

1. Load the data and apply 5 different pre-processing steps.
2. Use a proper train / test split.
3. Extract features with TF-IDF.
4. Train a Naive Bayes classifier.
5. Evaluate the model on the test set.
6. Print the confusion matrix.

## Running

```bash
pip install pandas scikit-learn nltk
```

Download `DisneylandReviews.csv` and update the path in `code/01_load-and-clean-dataset.py` (the notebook uses the Colab path `/content/DisneylandReviews.csv`).
