# Perceptron

Perceptron is an algorithm for supervised learning of binary classifiers. The idea of the perceptron was invented in 1943 by Warren McCulloch and Walter Pitts, and it was further developed by Frank Rosenblatt in 1957. 

A perceptron represents a binary linear classifier that maps its input $\mathbf{x} \in \mathbb{R}^n$ to an output value $f(\mathbf{x}) \in \{0, 1\}$ by the following function:

$$
f(\mathbf{x}) = H(\mathbf{w} \cdot \mathbf{x} + b)
$$

where $\mathbf{w}$ is the weight vector, $\mathbf{w} \in \mathbb{R}^n$, $\mathbf{b}$ is the bias, and $H$ is the Heaviside step function. The Heaviside step function is defined as:

$$
H(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

The weight vector $\mathbf{w}$ and the bias $b$ are learned from the training data using the perceptron learning algorithm. The learning algorithm can be summarized as follows:

```{prf:algorithm} Perceptron Learning Algorithm
:label: perceptron-learning-algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, Learning rate $\eta$, Number of epochs $T$

**Output**: Weight vector $\mathbf{w}$ and bias $b$

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ and $b \leftarrow 0$
2. **For** $t = 1$ to $T$
    1. **For** $i = 1$ to $n$
        1. Compute the predicted output $f(\mathbf{x}_i) = H(\mathbf{w} \cdot \mathbf{x}_i + b)$
        2. $\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - f(\mathbf{x}_i)) \mathbf{x}_i$
        3. $b \leftarrow b + \eta (y_i - f(\mathbf{x}_i))$
3. **Return** $\mathbf{w}$ and $b$
```

In the perceptron learning algorithm, the weight vector $\mathbf{w}$ and the bias $b$ are updated iteratively for each training example $(\mathbf{x}_i, y_i)$ in the training data $\mathcal{D}$. The learning rate $\eta$ controls the step size of the updates. The algorithm continues for a fixed number of epochs $T$ or until convergence.

When the training data is linearly separable, the perceptron learning algorithm is guaranteed to converge and find a separating hyperplane that correctly classifies all training examples. That is, the perceptron will find a weight vector $\mathbf{w}$ and a bias $b$ such that $f(\mathbf{x}_i) = y_i$ for all training examples $(\mathbf{x}_i, y_i) \in \mathcal{D}$.

Therefore, the objective of the perceptron learning algorithm is to find a weight vector $\mathbf{w}$ and a bias $b$ that minimize $(y_i - f(\mathbf{x}_i))^2$ for all training examples $(\mathbf{x}_i, y_i) \in \mathcal{D}$. This can be achieved by minimizing the following loss function:

$$
L(\mathbf{w}, b) = \sum_{i=1}^{n} (y_i - f(\mathbf{x}_i))^2
$$

The loss function $L(\mathbf{w}, b)$ can be minimized using gradient descent or stochastic gradient descent. The gradient of the loss function with respect to the weight vector $\mathbf{w}$ and the bias $b$ can be computed as follows:

$$
\frac{\partial L}{\partial \mathbf{w}} = -2 \sum_{i=1}^{n} (y_i - f(\mathbf{x}_i)) \mathbf{x}_i
$$

$$
\frac{\partial L}{\partial b} = -2 \sum_{i=1}^{n} (y_i - f(\mathbf{x}_i))
$$