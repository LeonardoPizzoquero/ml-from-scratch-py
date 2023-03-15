import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression

X , y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LinearRegression(lr=0.1, n_iters=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, predictions)

print(mse_value)

