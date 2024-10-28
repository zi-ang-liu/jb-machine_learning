"""
Implementation of the Perceptron algorithm
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit the model to the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        y : array-like, shape = [n_samples]
            Target values

        Returns
        -------
        self : object
        """

        # Initialize weights and bias
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.errors = []

        # Train
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                if self.predict(xi) != target:
                    update = self.learning_rate * target
                    self.w += update * xi
                    self.b += update
                    errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def plot_errors(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Number of updates")
        plt.show()


# Load data (linearly separable), blobs

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# plot data
figure = plt.figure()
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=40, c="red", marker="o", label="0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=40, c="blue", marker="x", label="1")
plt.legend()
plt.show()

# make labels -1 and 1
y = np.where(y == 0, -1, 1)

# Train model
ppn = Perceptron(learning_rate=0.1, n_iter=10)
ppn.fit(X, y)

# plot errors
ppn.plot_errors()

# plot decision boundary
w = ppn.w
b = ppn.b

plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], s=40, c="red", marker="o", label="0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=40, c="blue", marker="x", label="1")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min = (-w[0] * x_min - b) / w[1]
y_max = (-w[0] * x_max - b) / w[1]

plt.plot([x_min, x_max], [y_min, y_max], "k-")
plt.legend()
plt.show()
