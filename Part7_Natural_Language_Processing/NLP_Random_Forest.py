import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
nltk.download('stopwords')

N = 1000
corpus = []

for i in range(0, N):
    review = re.sub('^a-zA-Z', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print("-------Reviews after cleaning---------")
print(corpus)

# Creating the Bag of Words Model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
print("---------Sparse Matrix of independent variables---------")
print(X)
Y = dataset.iloc[:, 1].values
print("_----------Dependent Variable Vector---------")
print(Y)

# Splitting the dataset into training and test set

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)
print("---------------X train------------")
print(X_train)
print("---------------Y train---------------")
print(Y_train)
print("--------------X test------------------")
print(X_test)
print("----------------Y test-----------------")
print(Y_test)


# Fitting the classifier to the training set

classifier = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)
print("-------------Vector of predicitons Y_pred--------------")
print(Y_pred)


# Making the confusion matrix
# A confusion matrix shows us how many correct and incorrect predictions did our model make on the test set. It helps us evaluate the performnce of our model to see if our model got trained on the training set correctly or not.

cm = confusion_matrix(Y_test, Y_pred)
print("----------------Confusion Matrix------------")
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
