# Logistic Regression

# Importing the libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv("Social_Network_Ads.csv")
# print("--------------------Dataset-----------------")
# print(dataset)
X = dataset.iloc[:, [2,3]].values
print("------------Matrix of Features X------------")
print(X)
Y = dataset.iloc[:, 4].values
print("-------------Dependent variable vector Y -----------")
print(Y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
print("---------------X train------------")
print(X_train)
print("---------------Y train---------------")
print(Y_train)
print("--------------X test------------------")
print(X_test)
print("----------------Y test-----------------")
print(Y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
print("--------------Feature Scaling--------------------")

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X_train: ", X_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")

# Fitting the Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)
print("-------------Vector of predicitons Y_pred--------------")
print(Y_pred)

# Making the Confusion Matrix
# A confusion matrix shows us how many correct and incorrect predictions did our model make on the test set. It helps us evaluate the performnce of our model to see if our model got trained on the training set correctly or not.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("----------------Confusion Matrix------------")
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('yellow', 'pink')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c = ListedColormap(('red','green'))(i),label=j)


plt.title("Logistic Regression (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


# Visualising the Test set results
X_set, Y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('yellow', 'pink')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):

    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Logistic Regression (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()




