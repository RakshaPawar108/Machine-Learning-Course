# Random Selection...selects random ads to show to the users and get results 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Imprting the dataset 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random 
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range (0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
print("-------Ads Selected at random--------")
print(ads_selected)
print("---------Total Reward -----------")
print(total_reward)

# Visualising the results in a Histogram
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()