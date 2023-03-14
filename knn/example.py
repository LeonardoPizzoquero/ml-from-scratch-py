from knn import KNN
import numpy as np

# The data is a list of 5x2 matrices. Each matrix represents a 2D point.
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train = np.array([0, 0, 0, 1, 1])

# The data is a list of 5x2 matrices. Each matrix represents a 2D point.
X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_test = np.array([0, 0, 0, 1, 1])

# Create a KNN classifier with k=4
model = KNN(k=4)
  
# Train the classifier on the training data
model.fit(X_train, y_train)

# Use the trained classifier to predict labels for the test data
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy*100:.2f}%")