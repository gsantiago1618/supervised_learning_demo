# import
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# load the training data set
iris = datasets.load_iris()

# instantiate classifier and assign number of neighbors to six
knn = KNeighborsClassifier(n_neighbors=6)

# Fit classifier to training set using two arguments, features and target, as NumPy arrays
# Requires features to be continuous vice categorical
# Requires that their are no missing values
knn.fit(iris['data'], iris['target'])

# create an observation of unlabeled data
X_new = ([2.5, 4.8, 3.4, 1.2],
         [2.9, 4.2, 3.7, 1.7],
         [2.7, 4.5, 3.3, 1.4])

# Predicting on unlabeled data
prediction = knn.predict(X_new)

print('Prediction: {}'.format(prediction))

