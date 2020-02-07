# Apriori for Market Basket Optimisation

# We need the apyori.py file in the folder which is an implementation of the apriori model by the python foundation. This file contains some classes which we will use to build the rules for our business problem.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset using pandas
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
print("------------Dataset----------------")
print(dataset)

# Making a list of lists because the dataset is not a list but a dataframe, so we convert it to lists of various customers, so the first row is the first list, second row is second list and so on...
# So for this business problem, the following list 'transactions' is a list of 7501 lists.

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)]) 

# print("---------A list of lists for the apriori algorithm---------")
# print(transactions)

# Training Apriori on the dataset 
from apyori import apriori
rules =  apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results 
results = list(rules)
print("---------Printing the rules-------------")
print(results)

