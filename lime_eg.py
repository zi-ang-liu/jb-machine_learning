"""
LIME: Local Interpretable Model-agnostic Explanations
"""

from sklearn.datasets import make_moons
import sklearn.neural_network
from sklearn.model_selection import train_test_split

import numpy as np

import lime
import lime.lime_tabular

import matplotlib.pyplot as plt

np.random.seed(1)

# Load data
dataset = make_moons(noise=0.3)

# plot data
figure = plt.figure()
plt.scatter(dataset[0][:, 0], dataset[0][:, 1], s=40, c=dataset[1], cmap=plt.cm.Spectral)

# Split data
X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.01, max_iter=1000)
mlp.fit(X_train, y_train)

print('Train accuracy:', mlp.score(X_test, y_test))

# plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.Spectral)
plt.title('Decision Boundary')
plt.show()

# Explain modela
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['x1', 'x2'], class_names=['0', '1'])
exp = explainer.explain_instance(X_test[0], mlp.predict_proba, num_features=2)
exp.show_in_notebook(show_table=True, show_all=False)
exp.as_list()

# plot explanation
figure = plt.figure()
plt.scatter(dataset[0][:, 0], dataset[0][:, 1], s=40, c=dataset[1], cmap=plt.cm.Spectral)
plt.scatter(X_test[0][0], X_test[0][1], s=40, c='black')
plt.title('Explanation')
plt.show()

intercept = exp.intercept[1]
coefficient_1 = exp.local_exp[1][0][1]
coefficient_2 = exp.local_exp[1][1][1]
print('Intercept:', intercept)
print('Coefficient 1:', coefficient_1)
print('Coefficient 2:', coefficient_2)

# plot explanation
figure = plt.figure()
plt.scatter(dataset[0][:, 0], dataset[0][:, 1], s=40, c=dataset[1], cmap=plt.cm.Spectral)
plt.scatter(X_test[0][0], X_test[0][1], s=40, c='black')
# plot linear model
x = np.linspace(-2, 3, 100)
y = intercept + coefficient_1 * x + coefficient_2 * x
plt.plot(x, y, color='black')
plt.title('Explanation')
plt.show()