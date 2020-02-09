# In this demo, we practice importing files from the local file system to demonstrate supervised learning
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Load the iris dataset
iris = datasets.load_iris()

# Check type
print("Type: ", type(iris))
print()
# Result should read:
# class 'sklearn.utils.Bunch' which is similar to a dictionary with its key/value pairs

# Printing the keys
print(iris.keys())
print()
# Results should read:
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

# Inspection of each key
print("Iris data: ", type(iris.data))
print("Iris target: ", type(iris.target))
print("Iris target_names: ", type(iris.target_names))
print("Iris DESCR: ", type(iris.DESCR))
print("Iris feature_names: ", type(iris.data))
print("Iris filename: ", type(iris.filename))
print()

# Inspect the number of rows and columns
# Remember: samples are in rows, features are in columns
print("Number of samples and rows of the Iris data set: ", iris.data.shape)

# Inspect the target values
print("Names of target variables: ", iris.target_names)
print()

# In order to explore to perform some initial exploratory data analysis (EDA)
# assign the feature and target data to X and Y respectively
X = iris.data
Y = iris.target

# Next, create a data frame of the feature data
df = pd.DataFrame(X, columns=iris.feature_names)
print()
print("Currently viewing the first 5 rows of the data frame:")
print(df.head())

# Visual EDA using pandas scatter_matrix(data frame, color, figure size, marker size and shape)
visual_EDA = pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(8, 8), diagonal='hist')

