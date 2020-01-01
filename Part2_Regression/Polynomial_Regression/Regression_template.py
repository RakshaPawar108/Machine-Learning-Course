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


# Fitting Regression Model to the dataset 
# Create a new regressor

# Predicting a new result 

Y_pred = regressor.predict(([[6.5]]))

print("Prediction of Salary for level 6.5 using Polynomial Regression: ", Y_pred)

# Visualising the Regression Results

plt.scatter(X, Y, color = "Red")
plt.plot(X, regressor.predict(X), color="Blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Regression Results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, Y, color = "Red")
plt.plot(X_grid, regressor.predict(X_grid), color="Blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

