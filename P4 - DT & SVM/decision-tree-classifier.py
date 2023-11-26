"""
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.
The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
A tree can be seen as a piecewise constant approximation.

This program will evaluate data provided in 1 argument and will print performance summary.

Author: Mateusz Budzisz
"""
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load input data
input_file = sys.argv[1]
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Decision Trees classifier
params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

# Evaluate classifier performance
class_names = ['Class 0', 'Class 1']
print("#" * 40 + "\n")
print("Classifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")
print("Classifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")
