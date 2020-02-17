# Grid Search
# Kernel SVM Model

from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, [2, 3]].values
print("------------Matrix of Features X------------")
print(X)
Y = dataset.iloc[:, 4].values
print("-------------Dependent variable vector Y -----------")
print(Y)

# Splitting the dataset into training and test set

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

print("----------------------Feature Scaling -----------------")
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("X_train: ", X_train)
print("-----------------------")
print("X_test: ", X_test)
print("-----------------------")

# Fitting the classifier to the training set

classifier = SVC(kernel='rbf', random_state=0)
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


# Applying K-fold Cross Validation (To evaluate performance of our model)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
print("--------Accuracies after K-fold Cross Validation--------")
print(accuracies)
print("Mean accuracy:", accuracies.mean())
print("Standard Deviation:", accuracies.std())

# Applying Grid Search (To find the best model and best parameters)
print("--------Grid Search------------")
from sklearn.model_selection import GridSearchCV
parameters = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]} ]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
print("Best Accuracy:", best_accuracy)
best_parameters = grid_search.best_params_
print("Best Parameters:", best_parameters)

# Visualising the Training set results

X_set, Y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('yellow', 'pink')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):

    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)


plt.title("SVM (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


# Visualising the test set results
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

plt.title("SVM (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
