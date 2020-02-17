# XGBoost
# Part1 - Data Preprocessing
# Importing the libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
print("----------Matrix of features X--------------")
print(X)
print("-----------Dependent Variable vector Y-----------")
print(Y)

# Encoding Categorical Data

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print("--------After Encoding data------------")
print(X)

# Avoiding the dummy variable trap by removing one column
X = X[:, 1:]
print("-----Avoiding dummy variable trap----------")
print(X)

# Splitting the dataset into training set and test set

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print("----------X Train----------")
print(X_train)
print("----------Y Train----------")
print(Y_train)
print("----------X Test--------")
print(X_test)
print("----------Y Test---------")
print(Y_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)
print("------Predicted results-----------")
print(Y_pred)

# Making the confusion matrix

# The predict method above returns probabilities that customers leave the bank, however for the matrix we don't need probabilities but we need results in boolean form ie true or false, so we convert the probabilities in the form true or false.

cm = confusion_matrix(Y_test, Y_pred)
print("------Confusion Matrix---------")
print(cm)

TP, TN, FP, FN = cm[1][1], cm[0][0], cm[0][1], cm[1][0]

# Computing the Accuracy
accuracy = (TP + TN)/(TP + TN + FP + FN)  # (TP + TN)/ (TP+TN+FP+FN)


# Computing the Precision   (Measuring Exactness)
precision = TP/(TP + FP)  # TP / (TP + FP)


# Computing the Recall (Measuring Completeness)
recall = TP/(TP + FN)  # TP / (TP + FN)


# Computing the F1 Score (Compromise between precision and recall)
f1_score = (2 * precision * recall)/(precision + recall)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Applying K-fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
print("--------Accuracies after K-fold Cross Validation--------")
print(accuracies)
print("Mean accuracy:", accuracies.mean())
print("Standard Deviation:", accuracies.std())
