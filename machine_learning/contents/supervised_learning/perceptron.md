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

```{algorithm} Perceptron Learning Algorithm
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

```{prf:algorithm} Policy Evaluation for estimating $V \approx v_{\pi}$
:label: policy-evaluation

**Input**: $\theta, \pi$

**Output**: $V \approx v_{\pi}$

1. Initialize $V(s)$ arbitrarily, for all $s \in \mathcal{S}$
2. $\Delta \leftarrow 2\theta$
3. **while** $\Delta \geq \theta$ **do**
    1. $\Delta \leftarrow 0$
    2. **for** $s \in \mathcal{S}$ **do**
        1. $v \leftarrow V(s)$
        2. $V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma V(s')]$
        3. $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
```

In the perceptron learning algorithm, the weight vector $\mathbf{w}$ and the bias $b$ are updated iteratively for each training example $(\mathbf{x}_i, y_i)$ in the training data $\mathcal{D}$. The learning rate $\eta$ controls the step size of the updates. The algorithm continues for a fixed number of epochs $T$ or until convergence.

The update rule for the weight vector $\mathbf{w}$ and the bias $b$ is derived from the perceptron criterion, which aims to minimize the mean squared error (MSE) loss function:

$$
\min \sum_{i=1}^{n} (y_i - f(\mathbf{x}_i))^2
$$



