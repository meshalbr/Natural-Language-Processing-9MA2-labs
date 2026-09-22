# Lab 4 - Classification and Evaluation
# Step: Train a linear SVM and evaluate it (accuracy, classification report)
# Extracted from Lab4_Classification_and_Evaluation.ipynb; each '# %%' marks one original notebook cell.

# %% cell 1
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_vectors, train_labels)

# Predictions on the test set
predictions = svm_classifier.predict(test_vectors)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(test_labels, predictions))
