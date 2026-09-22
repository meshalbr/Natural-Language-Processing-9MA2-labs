# Lab 4 - Classification and Evaluation

A complete text classification pipeline for sentiment analysis. The notebook takes the Disneyland reviews dataset, maps star ratings to positive / neutral / negative labels, cleans the text, converts it to TF-IDF features, trains a linear SVM, and evaluates it with accuracy and a classification report.

## Files

| File | Content |
|------|---------|
| [Lab4_Classification_and_Evaluation.ipynb](Lab4_Classification_and_Evaluation.ipynb) | The lab notebook: text, code and outputs, run cell by cell |
| [notes.md](notes.md) | The whole lab as one readable Markdown document (text, dataset description, code, outputs) |
| [images/](images/) | Figure used in the lab |

## Pipeline in the notebook

1. Load the CSV, check and drop missing values, map ratings to sentiment labels.
2. Lowercase the reviews, remove punctuation, tokenize.
3. 80/20 train-test split and TF-IDF vectorization (1000 features).
4. Train `SVC(kernel='linear')`, predict, print accuracy and the classification report.

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
jupyter notebook Lab4_Classification_and_Evaluation.ipynb
```

Download `DisneylandReviews.csv` and update the path in the first code cell (the notebook uses the Colab path `/content/DisneylandReviews.csv`).
