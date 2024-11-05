# Perceptron

## Introduction

Perceptron is one of the oldest algorithm for supervised learning of binary classifiers. The idea of the perceptron was invented in 1943 by Warren McCulloch and Walter Pitts, and it was further developed by Frank Rosenblatt in 1957. 

A perceptron represents a binary linear classifier that maps its input $\mathbf{x} \in \mathbb{R}^n$ to an output value $f(\mathbf{x}) \in \{-1, 1\}$. The perceptron can be represented as follows:

$$
f(\mathbf{x}) = h(\mathbf{w} \cdot \mathbf{x} + b)
$$

where $\mathbf{w}$ is the weight vector, $\mathbf{w} \in \mathbb{R}^n$, $\mathbf{b}$ is the bias, and $h(\cdot)$ is defined as follows:

$$
h(z) =
\begin{cases}
1 & \text{if } z \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$$

By setting appropriate values for the weight vector $\mathbf{w}$ and the bias $b$, the perceptron can learn to classify the input data $\mathbf{x}$ into two classes, represented by the output values $-1$ and $1$.

## Loss Function

To simplify the illustration, we use $\hat{y}^{(i)} = f(\mathbf{x}^{(i)})$ to represent the prediction of the perceptron for the input $\mathbf{x}^{(i)}$. The true label of the input $\mathbf{x}^{(i)}$ is denoted as $y^{(i)}$. The perceptron makes a correct prediction if $\hat{y}^{(i)} = y^{(i)}$ and an incorrect prediction if $\hat{y}^{(i)} \neq y^{(i)}$.

The loss function used in the perceptron learning algorithm is the hinge loss, which is defined as follows:

$$
\mathcal{L}(\mathbf{w}, b) = \sum_{i=1}^{n} \max(0, -y^{(i)} \hat{y}^{(i)})
$$

We consider the hinge loss for a single sample $(\mathbf{x}^{(i)}, y^{(i)})$:

$$
\mathcal{L}_i(\mathbf{w}, b) = \max(0, -y^{(i)} \hat{y}^{(i)})
$$

Both $y^{(i)}$ and $\hat{y}^{(i)}$ are either $-1$ or $1$. Therefore, 

- If $y^{(i)} = f(\mathbf{x}^{(i)})$, the prediction is correct (i.e., $y^{(i)} \hat{y}^{(i)} = 1$), and the hinge loss is zero.
- If $y^{(i)} \neq f(\mathbf{x}^{(i)})$, the prediction is incorrect (i.e., $y^{(i)} \hat{y}^{(i)} = -1$), and the hinge loss is $-y^{(i)} \hat{y}^{(i)}$.

Note that when we have the "perfect" classifier, the hinge loss $\mathcal{L}(\mathbf{w}, b) = 0$.

## Optimization

The goal of the perceptron learning algorithm is to minimize the hinge loss by updating the weight vector $\mathbf{w}$ and the bias $b$. To achieve this, we use gradient descent to update the parameters in the direction that reduces the loss. The update rule for the weight vector $\mathbf{w}$ and the bias $b$ is as follows:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b) \\
b &\leftarrow b - \eta \nabla_{b} \mathcal{L}_i(\mathbf{w}, b)
\end{align*}
$$

where $\nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b)$ and $\nabla_{b} \mathcal{L}_i(\mathbf{w}, b)$ are the gradients of the hinge loss for sample $(\mathbf{x}^{(i)}, y^{(i)})$ with respect to the weight vector $\mathbf{w}$ and the bias $b$.

If $-y^{(i)} \hat{y}^{(i)} \leq 0$, the gradient of the hinge loss is zero, and the weight vector $\mathbf{w}$ and the bias $b$ remain unchanged.

If $-y^{(i)} \hat{y}^{(i)} > 0$, the gradient of the hinge loss for sample $(\mathbf{x}^{(i)}, y^{(i)})$ with respect to the weight vector $\mathbf{w}$ and the bias $b$ is:

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}, b) &= -y^{(i)} \mathbf{x}^{(i)} \\
\nabla_{b} \mathcal{L}(\mathbf{w}, b) &= -y^{(i)}
\end{align*}
$$

Substituting the gradients into the update rule, we get:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta y^{(i)} \mathbf{x}^{(i)} \\
b &\leftarrow b + \eta y^{(i)}
\end{align*}
$$

There is another way to update the weight vector $\mathbf{w}$ and the bias $b$:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta (y^{(i)} - \hat{y}^{(i)}) \mathbf{x}^{(i)} \\
b &\leftarrow b + \eta (y^{(i)} - \hat{y}^{(i)})
\end{align*}
$$

This update rule is equivalent to the previous one. When $y^{(i)} = \hat{y}^{(i)}$, the update is zero, and the weight vector $\mathbf{w}$ and the bias $b$ remain unchanged. When $y^{(i)} \neq \hat{y}^{(i)}$, the update is $y^{(i)} - \hat{y}^{(i)}$, which is equivalent to $2 y^{(i)}$. By adjusting the learning rate $\eta$, we can consider these two update rules as equivalent.

## Algorithm

In the implementation of the perceptron learning algorithm, by setting $x_0 = 1$ and $w_0 = b$, we can combine the bias $b$ with the weight vector $\mathbf{w}$ as follows:

$$
\mathbf{w} = [b, w_1, w_2, \ldots, w_n]
$$

and

$$
\mathbf{x} = [1, x_1, x_2, \ldots, x_n]
$$

Note that $\mathbf{w} \in \mathbb{R}^{n+1}$ and $\mathbf{x} \in \mathbb{R}^{n+1}$. The perceptron can then be represented as:

$$
f(\mathbf{x}) = h(\mathbf{w} \cdot \mathbf{x})
$$

where $\mathbf{w} \cdot \mathbf{x} = \sum_{i=0}^{n} w_i x_i$.

Since we set $x_0 = 1$, the update rule can be simplified as follows:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta y^{(i)} \mathbf{x}^{(i)} 
$$

The perceptron learning algorithm is summarized in the following steps:

```{prf:algorithm} Perceptron Learning Algorithm
:label: perceptron-learning-algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, Learning rate $\eta$, Number of epochs $T$   
**Output**: Weight vector $\mathbf{w}$

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ 
2. **For** $t = 1$ to $T$
    1. **For** $i = 1$ to $n$
        1. Compute the prediction $f(\mathbf{x}^{(i)}) = h(\mathbf{w} \cdot \mathbf{x}^{(i)})$
        2. **If** $y^{(i)} f(\mathbf{x}^{(i)}) \leq 0$
            1. Update the weight vector $\mathbf{w} \leftarrow \mathbf{w} + \eta y^{(i)} \mathbf{x}^{(i)}$
3. **Return** $\mathbf{w}$
```

In the perceptron learning algorithm, the weight vector $\mathbf{w}$ and is updated iteratively for each training example $(\mathbf{x}^{(i)}, y^{(i)})$ in the training data $\mathcal{D}$. The learning rate $\eta$ controls the step size of the updates. The algorithm continues for a fixed number of epochs $T$ or until convergence.

## Python Implementation

```python
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
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        # Train
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                if self.predict(xi) != target:
                    update = self.learning_rate * target
                    self.w[1:] += update * xi
                    self.w[0] += update
                    errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

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
w = ppn.w[1:]
b = ppn.w[0]

plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], s=40, c="red", marker="o", label="0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=40, c="blue", marker="x", label="1")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min = (-w[0] * x_min - b) / w[1]
y_max = (-w[0] * x_max - b) / w[1]

plt.plot([x_min, x_max], [y_min, y_max], "k-")
plt.legend()
plt.show()
```