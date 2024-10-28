# Perceptron

Perceptron is an algorithm for supervised learning of binary classifiers. The idea of the perceptron was invented in 1943 by Warren McCulloch and Walter Pitts, and it was further developed by Frank Rosenblatt in 1957. 

A perceptron represents a binary linear classifier that maps its input $\mathbf{x} \in \mathbb{R}^n$ to an output value $f(\mathbf{x}) \in \{0, 1\}$ by the following function:

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

As a result, the output of the perceptron is binary, i.e., $f(\mathbf{x}) \in \{1, -1\}$. 

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

The update rule for the weight vector $\mathbf{w}$ and the bias $b$ is as follows:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta (y_i - f(\mathbf{x}_i)) \mathbf{x}_i \\
b &\leftarrow b + \eta (y_i - f(\mathbf{x}_i))
\end{align*}
$$

The update rule is derived from the gradient descent algorithm. For each training example $(\mathbf{x}_i, y_i)$, the perceptron learning algorithm updates the weight vector $\mathbf{w}$ and the bias $b$ in the direction that reduces the error between the predicted output $f(\mathbf{x}_i)$ and the true output $y_i$. Therefore, the objective can be formulated as minimizing the following loss function:

$$
\min (y_i - f(\mathbf{x}_i))^2
$$

When the training data is linearly separable, the perceptron learning algorithm is guaranteed to converge and find a separating hyperplane that correctly classifies all training examples. That is, the perceptron will find a weight vector $\mathbf{w}$ and a bias $b$ such that $f(\mathbf{x}_i) = y_i$ for all training examples $(\mathbf{x}_i, y_i) \in \mathcal{D}$.