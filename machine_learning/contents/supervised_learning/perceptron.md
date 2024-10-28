# Perceptron

## Introduction

Perceptron is an algorithm for supervised learning of binary classifiers. The idea of the perceptron was invented in 1943 by Warren McCulloch and Walter Pitts, and it was further developed by Frank Rosenblatt in 1957. 

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

The weight vector $\mathbf{w}$ and the bias $b$ are learned from the training data using the perceptron learning algorithm. 

## Algorithm

The learning algorithm can be summarized as follows:

```{prf:algorithm} Perceptron Learning Algorithm
:label: perceptron-learning-algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, Learning rate $\eta$, Number of epochs $T$

**Output**: Weight vector $\mathbf{w}$ and bias $b$

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ and $b \leftarrow 0$
2. **For** $t = 1$ to $T$
    1. **For** $i = 1$ to $n$
        1. Compute the prediction $f(\mathbf{x}_i) = h(\mathbf{w} \cdot \mathbf{x}_i + b)$
        2. **If** $y_i \neq f(\mathbf{x}_i)$
            1. Update the weight vector $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$
            2. Update the bias $b \leftarrow b + \eta y_i$
**Return** $\mathbf{w}$ and $b$
```

In the perceptron learning algorithm, the weight vector $\mathbf{w}$ and the bias $b$ are updated iteratively for each training example $(\mathbf{x}_i, y_i)$ in the training data $\mathcal{D}$. The learning rate $\eta$ controls the step size of the updates. The algorithm continues for a fixed number of epochs $T$ or until convergence.

The update rule for the weight vector $\mathbf{w}$ and the bias $b$ is as follows:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i \\
b &\leftarrow b + \eta y_i
\end{align*}
$$

## Loss Function

The loss function used in the perceptron learning algorithm is the hinge loss, which is defined as follows:

$$
\mathcal{L}(\mathbf{w}, b) = \sum_{i=1}^{n} \max(0, -y_i (\mathbf{w} \cdot \mathbf{x}_i + b))
$$

The goal of the perceptron learning algorithm is to minimize the hinge loss by updating the weight vector $\mathbf{w}$ and the bias $b$. To achieve this, we use gradient descent to update the parameters in the direction that reduces the loss.

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}, b) \\
b &\leftarrow b + \eta \nabla_{b} \mathcal{L}(\mathbf{w}, b)
\end{align*}
$$

When $-y_i (\mathbf{w} \cdot \mathbf{x}_i + b) > 0$, the gradient of the hinge loss for sample $(\mathbf{x}_i, y_i)$ with respect to the weight vector $\mathbf{w}$ and the bias $b$ is:

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}, b) &= -y_i \mathbf{x}_i \\
\nabla_{b} \mathcal{L}(\mathbf{w}, b) &= -y_i
\end{align*}
$$

Substituting the gradients into the update rule, we get:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i \\
b &\leftarrow b + \eta y_i
\end{align*}
$$
