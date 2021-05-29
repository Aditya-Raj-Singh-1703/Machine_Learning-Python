# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j])for j in range(0,20)])

# Training the Apriori Model on dataset
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2,min_lift=3,min_length = 2, max_length=2)

# Visualising Results
# Displaying the first results coming directly from the apriori function
results = list(rules)
print(results)

# Putting results well organised into pandas dataframe
def inspect(results):
    lhs = [tuple(i[2][0][0])[0] for i in results]
    rhs = [tuple(i[2][0][1])[0] for i in results]
    support = [i[1] for i in results]
    confidences = [i[2][0][2] for i in results]
    lifts = [i[2][0][3] for i in results]
    return list(zip(lhs, rhs, support, confidences, lifts))

resultinDataFrame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side','Support', 'Confidence','Lift'])

# Displaying Results non sorted
print(resultinDataFrame)

# Displaying Results sorted by descending lifts

print(resultinDataFrame.nlargest(n=10, columns='Lift'))