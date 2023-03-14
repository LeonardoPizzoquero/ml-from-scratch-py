import numpy as np
from collections import Counter
from utils.euclidean_distance import euclidean_distance

# This class implements the k-nearest neighbor algorithm. The
# constructor takes the number of neighbors to use as an argument.
class KNN:
  
  def __init__(self, k):
    self.k = k # k is the number of clusters
  
  def fit(self, X, y):
    # Save the training data for use in `predict`
    self.X_train = X
    self.y_train = y
  
  # This function uses the _predict function to predict the labels of the given X. 
  # It returns an array of predicted labels.
  def predict(self, X):
      predicted_labels = [self._predict(x) for x in X]
      return np.array(predicted_labels)
    
  def _predict(self, x):
    #calculates distance between x and all training data points
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    #finds the k nearest training data points
    k_indices = np.argsort(distances)[:self.k]
    #finds the labels of the k nearest training data points
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    #finds the most common label among the k nearest training data points
    most_common = Counter(k_nearest_labels).most_common(1)
    #returns the most common label
    return most_common[0][0]