# Importing the libraries for data preprocessing template "Data.csv"

#This library contains mathematical tools and we need it to include any mathematics in our code
import numpy as np  
#This library will help us to plot charts
import matplotlib.pyplot as plt 
#This is the best library to import and manage datasets
import pandas as pd 
   

# Import dataset 
dataset = pd.read_csv("Data.csv")
# print(dataset)

#We take all the columns, first : is to take all columns, :-1 means we take all columns except the last one and make the matrix of features.
X = dataset.iloc[:, :-1].values
print("---------------------------------------------")
print(X)

# Now we create the dependent variable vector
Y = dataset.iloc[:, 3].values
print("---------------------------------------------")
print(Y)

# Splitting the dataset into training set and test set
print("-------Splitting dataset into training set and test set--------------")

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print("X_train: ", X_train)
print("-----------------------")
print("Y_train: ", Y_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")
print("Y_test: ", Y_test)


"""# Feature Scaling
print("----------------------Feature Scaling ----------------------")
# We can see in the dataset that the age and salary columns are not on the same scale ie age columns goes from 27 to 40 and salary column goes from 40k to 79k. This can cause problems in the ML models.
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X_train: ", X_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")"""
