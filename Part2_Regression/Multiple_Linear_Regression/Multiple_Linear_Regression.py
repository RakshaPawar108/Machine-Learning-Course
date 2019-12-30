# Multiple linear regression

# Importing the libraries for data preprocessing template "Data.csv"

#This library contains mathematical tools and we need it to include any mathematics in our code
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
#This library will help us to plot charts
import matplotlib.pyplot as plt
#This is the best library to import and manage datasets
import pandas as pd


# Import dataset
dataset = pd.read_csv("50_Startups.csv")
# print(dataset)

#We take all the columns, first : is to take all columns, :-1 means we take all columns except the last one and make the matrix of features.
X = dataset.iloc[:, :-1].values
print("---------------------------------------------")
print(X)

# Now we create the dependent variable vector
Y = dataset.iloc[:, 4].values
print("---------------------------------------------")
print(Y)

# Encoding categorical data
# In our dataset, the country column has 3 categories ie France, Spain and Germany and the Purchased column has two namely Yes and no
# It will be difficult to solve equations if there is categorical variables, so we need to encode these categorical variables in to numbers


labelencoder_X = LabelEncoder().fit_transform(X[:, 3])
#We fitted the labelencoder_X into our country column with foll line
print("------------------------Encoding categorical data-----------------")
print(labelencoder_X)
print("-------")

ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X)

# Avoiding the dummy variable trap

# Removing first column of X. 1: means take all columns of X starting from index 1
X = X[:, 1:]



# Splitting the dataset into training set and test set
print("-------Splitting dataset into training set and test set--------------")


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print("X_train: ", X_train)
print("-----------------------")
print("Y_train: ", Y_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")
print("Y_test: ", Y_test)


# """# Feature Scaling
# print("----------------------Feature Scaling ----------------------")
# # We can see in the dataset that the age and salary columns are not on the same scale ie age columns goes from 27 to 40 and salary column goes from 40k to 79k. This can cause problems in the ML models.
# from sklearn.preprocessing import StandardScaler

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# print("X_train: ", X_train)
# print("-----------------------")
# print("X_test: ", X_test)
# print("-----------------------")"""

# Fitting out Multiple Linear Regression model to training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results

Y_pred = regressor.predict(X_test)
print("----------------Predictions of Profit-------------")
print(Y_pred)

# BACKWARD ELIMINATION
# Building optimal model using Backward Elimination

import statsmodels.regression.linear_model as sm

# We will add a column of x0 = 1 associated with the constant b0 so the library will understand that we are using the equation for multiple linear regression y=b0x0 + b1x1+....+bnxn.

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X = np.vstack(X).astype(np.float)
# We appended X to column of ones instead of the opposite so it becomes the first column by interchanging the values of parameters arr and values.
print("--------------Adding column of ones at the beginning---------------")
print (X)

# We make an optimal matrix of features

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Step 2: Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

# Step 3: 
print(regressor_OLS.summary())

print("--------------------REMOVING INDEX WITH THE HIGHEST P VALUE---------------------")

# Building optimal model using Backward Elimination

# We make an optimal matrix of features

X_opt = X[:, [0, 1, 3, 4, 5]]
# Step 2: Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

# Step 3:
print(regressor_OLS.summary())

# Building optimal model using Backward Elimination

# We make an optimal matrix of features

X_opt = X[:, [0, 3, 4, 5]]
# Step 2: Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

# Step 3:
print(regressor_OLS.summary())

# Building optimal model using Backward Elimination

# We make an optimal matrix of features

X_opt = X[:, [0, 3, 5]]
# Step 2: Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

# Step 3:
print(regressor_OLS.summary())

# Building optimal model using Backward Elimination

# We make an optimal matrix of features

X_opt = X[:, [0, 3]]
# Step 2: Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

# Step 3:
print(regressor_OLS.summary())


