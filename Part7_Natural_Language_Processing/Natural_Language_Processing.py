# Natural Language Processing
# Making a 'Bag of Words' model

# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset 
# Here, the function is expecting a csv file but we are using a tsv file as some commas or quotes can cause problems in case of csv files. 
# So we change the delimiter to tabspace by putting \t in delimiter parameter and to ignore the double quotes, we use the code 3 in quoting parameter.
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
# print("---------Dataset----------")
# print(dataset)

# Cleaning the texts 
import re               #Contains tools to review text and clean
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
N = 1000

corpus = []
for i in range(0, N):
    # print("------S1 Reviews (Not Cleaned)-------")
    # print(dataset['Review'][i])
# Removing all other characters except capital and small letters and also adding spaces as Step 1 of cleaning
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    # print("----------S1 Reviews (Cleaned)----------")
    # print(review)
# Step 2 of cleaning involves putting the entire string in lowercase 
    review = review.lower()
    # print("---------S2 Reviews (Cleaned)(Lowercase)------------")
    # print(review)
# Step 3 of cleaning is to remove the non significant words(stopwords) like 'the', 'this' etc.
# The Split function will first split the review which is a string and convert it into a list of its different words
    review = review.split()
    # print("------------S3 Reviews (Cleaned)(Lowercase)(List form)--------")
    # print(review)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # print("----S3, S4 Reviews (Cleaned)(Lowercase)(List Form)(Stopwords Removed)(Stemmed)-------")
    # print(review)
# Step 4 is Stemming, meaning taking the root form of the words. So in first review as eg, 'Loved' becomes 'Love' which is done in the above process itself
# Step 5 is to reverse this cleaned list and convert it back to a cleaned string
    review = ' '.join(review)
    # print("---------S5 Reviews (Cleaned Completely and Joined back to string --------")
    # print(review)
    corpus.append(review)
print("------The Entire Corpus of all 1000 reviews in the dataset(Cleaned Completely)----")
print(corpus)


# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer #For tokenisation
cv = CountVectorizer(max_features = 1500)
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

classifier = GaussianNB()
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

# Computing the Accuracy
accuracy = (55-91)/200
print("----------Accuracy-----------")
print(accuracy)


