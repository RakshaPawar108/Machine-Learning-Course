# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

print("------------Matrix of features--------------")
print(X)

print("------------Dependent Variable Vector -------------")
print(Y)

# Splitting the dataset into training set and test set

'''from sklearn.model_selection import train_test_split

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

print("X_train: ", X_train)
print("------------------------")
print("X_test: ", X_test)
print("-----------------------")
print("Y_train: ", Y_train)
print("------------------------")
print("Y_test: ", Y_test)'''

# Feature Scaling
print("----------------------Feature Scaling ----------------------")
# We can see in the dataset that the age and salary columns are not on the same scale ie age columns goes from 27 to 40 and salary column goes from 40k to 79k. This can cause problems in the ML models.
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = np.squeeze(sc_Y.fit_transform(Y.reshape(-1, 1)))
print("X: ", X)
print("-----------------------")
print("Y: ", Y)
print("-----------------------")


# Fitting the SVR to the dataset

from sklearn.svm import SVR  

regressor = SVR(kernel = "rbf")

regressor.fit(X, Y)


# Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print("Predicted Result: ", Y_pred)


# Visualising the SVR results

plt.scatter(X, Y, color = "Red")
plt.plot(X, regressor.predict(X), color = "Blue")
plt.title("Truth vs Bluff (Support Vector Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the SVR Results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color="Red")
plt.plot(X_grid, regressor.predict(X_grid), color="Blue")
plt.title("Truth or Bluff (SVR Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
