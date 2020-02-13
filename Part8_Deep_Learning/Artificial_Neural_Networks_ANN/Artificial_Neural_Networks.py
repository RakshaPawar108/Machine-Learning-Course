# Artificial Neural Networks

# Part1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
print("----------Matrix of features X--------------")
print(X)
print("-----------Dependent Variable vector Y-----------")
print(Y)

# Encoding Categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print("--------After Encoding data------------")
print (X)

# Avoiding the dummy variable trap by removing one column
X = X[:, 1:]
print("-----Avoiding dummy variable trap----------")
print(X)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print("----------X Train----------")
print(X_train)
print("----------Y Train----------")
print(Y_train)
print("----------X Test--------")
print(X_test)
print("----------Y Test---------")
print(Y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print("------Applying Feature scaling-------")
print("-------X Train---------")
print(X_train)
print("---------X Test---------")
print(X_test)


# Part2 - Now let's make the ANN!!!!!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# print("--------Reached HERE!!!!-----------")

# Initializing the ANN!!
classifier = Sequential()

# Adding the input layer and the first hidden layer

# First parameter is the number of nodes in the hidden layer. Here we take the average of the no. of input and output nodes.
# No. of input nodes = 11, no. of output nodes = 1, total = 12, avg = 6

# Second parameter is for ANN using Stochastic Gradient Descent which is to input random weights to inputs...so we use uniform.

# Third parameter is the activation function that we want to choose in our HIDDEN LAYER. We are using the rectifier function for hidden layer

# The last argument is a compulsory argument because we are creating our first hidden layer and it does not know which input nodes to expect. As we will create the next hidden layer, this parameter will not be necessary as the following layers will know what inputs they are getting

        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

# Adding the second hidden layer
        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# MAKING THE layers using loop 
input_nodes = 11
output_nodes = 1
no_of_layers = 2
hidden_nodes = int((input_nodes + output_nodes) / 2)    #Taking avg

for layers in range(0, no_of_layers):
        if(layers == 0):
                # for first layer ie. input layer
            classifier.add(Dense(output_dim = hidden_nodes, init = 'uniform', activation = 'relu', input_dim = input_nodes))
        else:
            classifier.add(Dense(output_dim = hidden_nodes, init = 'uniform', activation = 'relu'))


# Adding the output layer

# There is only one node in the output layer so we put first parameter as 1
# Since we are making a geodemographic segmentation model, we need to have probabilities for the outcome, so for that we change the activation function

# We are using the sigmoid activation function for the output layer

# Here we have one category but if our problem has more than two categories, we have to change the output dim  to that number and activation function to softmax which is a sigmoid function for dependent variables having more than 2 categories.

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

# Parameter 1 is the stochastic gradient descent algorithm
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Part3 - Making the predictions and evaluating the model

# Predicting the test set results
Y_pred = classifier.predict(X_test)
print("------Predicted results-----------")
print(Y_pred)

# Making the confusion matrix

# The predict method above returns probabilities that customers leave the bank, however for the matrix we don't need probabilities but we need results in boolean form ie true or false, so we convert the probabilities in the form true or false.

from sklearn.metrics import confusion_matrix

Y_pred = (Y_pred > 0.5)

cm = confusion_matrix(Y_test, Y_pred)
print("------Confusion Matrix---------")
print(cm)

TP, TN, FP, FN = cm[1][1], cm[0][0], cm[0][1], cm[1][0]

# Computing the Accuracy
accuracy = (TP + TN)/(TP + TN + FP + FN)  # (TP + TN)/ (TP+TN+FP+FN)


# Computing the Precision   (Measuring Exactness)
precision = TP/(TP + FP)  # TP / (TP + FP)


# Computing the Recall (Measuring Completeness)
recall = TP/(TP + FN)  # TP / (TP + FN)


# Computing the F1 Score (Compromise between precision and recall)
f1_score = (2 * precision * recall)/(precision + recall)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


