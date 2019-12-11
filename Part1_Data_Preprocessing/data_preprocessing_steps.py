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


# Handling missing data

# sklearn is scikitlearn and it is a preprocessing library that contains methods and classes to preprocess data
from sklearn.impute import SimpleImputer 

# Check documentation by clicking the Imputer for future reference of parameters
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose = 0)
# 
imputer = imputer.fit(X[:, 1:3]) #Lowerbound 1 is included, upperbound 3 is excluded so we are taking columns 1 and 2.
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("-------------------Handling missing data------------------")
print(X)

# Encoding categorical data
# In our dataset, the country column has 3 categories ie France, Spain and Germany and the Purchased column has two namely Yes and no
# It will be difficult to solve equations if there is categorical variables, so we need to encode these categorical variables in to numbers

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder().fit_transform(X[:, 0])
#We fitted the labelencoder_X into our country column with foll line  
print("------------------------Encoding categorical data-----------------")
print(labelencoder_X)
print("-------")

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print (X)
print("-----------------------------------------------")

Y = LabelEncoder().fit_transform(Y)
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

# Feature Scaling
print("----------------------Feature Scaling ----------------------")
# We can see in the dataset that the age and salary columns are not on the same scale ie age columns goes from 27 to 40 and salary column goes from 40k to 79k. This can cause problems in the ML models.
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X_train: ", X_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")

# The above are the steps that are done during data preprocessing however when we are making our template for this dataset, we will not include some steps as some libraries in Py take care of that, so the final template is in the other file numbered 1.

