# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Position_Salaries.csv")


X = dataset.iloc[:, 1:2].values
# By doing :2, X will now be considered as a matrix instead of as a vector.
print("-------------------------------------------------")
print(X)

Y = dataset.iloc[:, 2].values
print("---------------------------------------------")
print(Y)

'''from sklearn.model_selection import train_test_split

print("------------Splitting dataset into training and test--------------")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

print("X_train: ", X_train)
print("------------------------")
print("X_test: ", X_test)
print("-----------------------")
print("Y_train: ", Y_train)
print("------------------------")
print("Y_test: ", Y_test)'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures

# This object is going to transform our matrix of features X into a new matrix of features X_poly containing not only independent variabe X1 but also X1^2 and powers of X1.
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
print("---------------New Matrix of features X_poly--------------")
print(X_poly)

# We created a new Linear regression object to fit the new matrix of features X_poly with the dependent variable Y

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Linear Regression Results

plt.scatter(X, Y, color = "Red")
plt.plot(X, lin_reg.predict(X), color = "Blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression Results

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, Y, color = "Red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="Blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression

print("Prediction of Salary Level 6.5 using Linear Regression: ", lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression

print("Prediction of Salary for level 6.5 using Polynomial Regression: ", lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

