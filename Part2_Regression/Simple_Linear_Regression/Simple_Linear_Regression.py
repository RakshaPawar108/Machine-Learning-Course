# Importing the libraries for data preprocessing template "Data.csv"

#This library contains mathematical tools and we need it to include any mathematics in our code
from sklearn.model_selection import train_test_split
import numpy as np
#This library will help us to plot charts
import matplotlib.pyplot as plt
#This is the best library to import and manage datasets
import pandas as pd


# Import dataset
dataset = pd.read_csv("Salary_Data.csv")
# print(dataset)

#We take all the columns, first : is to take all columns, :-1 means we take all columns except the last one and make the matrix of features.
X = dataset.iloc[:, :-1].values
print("---------------------------------------------")
print(X)

# Now we create the dependent variable vector
Y = dataset.iloc[:, 1].values
print("---------------------------------------------")
print(Y)

# Splitting the dataset into training set and test set
print("-------Splitting dataset into training set and test set--------------")


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 1/3, random_state = 0)
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

# Fitting Simple Linear Regression to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results

# This is vector of predictions for dependent variable
Y_pred = regressor.predict(X_test)
print("----------------Predictions of Salary------------------------")
print(Y_pred)

# Visualising the training set results

plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the test set results

plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
