import numpy as np

class LinearRegression:
  
  def __init__(self, lr=0.001, n_iters=1000):
    if not isinstance(lr, float):
        raise TypeError("lr must be a float")
    if lr < 0:
        raise ValueError("lr must be positive")
    if not isinstance(n_iters, int):
        raise TypeError("n_iters must be an integer")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
      
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    
  def fit(self, X, y):
    # X is the training data, y is the target data
    # n_samples is the number of rows in X, n_features is the number of columns in X
    n_samples, n_features = X.shape
    # Initialize weights and bias
    self.weights = np.zeros(n_features)
    self.bias = 0
    
    # Perform gradient descent for self.n_iters iterations
    for _ in range(self.n_iters):
      # Predict the y values for the current weights and bias
      y_predicted = np.dot(X, self.weights) + self.bias
      # Calculate the derivative of the weights and bias
      dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1 / n_samples) * np.sum(y_predicted - y)
      # Update the weights and bias
      self.weights -= self.lr * dw
      self.bias -= self.lr * db
  
  # this function takes the input and returns the predicted output by multiplying the input with weights and adding the bias
  def predict(self, X):
      y_predicted = np.dot(X, self.weights) + self.bias
      return y_predicted