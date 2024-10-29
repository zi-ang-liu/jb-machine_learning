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

To simplify the illustration, we use $\hat{y}_i = f(\mathbf{x}_i)$ to represent the prediction of the perceptron for the input $\mathbf{x}_i$. The true label of the input $\mathbf{x}_i$ is denoted as $y_i$. The perceptron makes a correct prediction if $\hat{y}_i = y_i$ and an incorrect prediction if $\hat{y}_i \neq y_i$.

The loss function used in the perceptron learning algorithm is the hinge loss, which is defined as follows:

$$
\mathcal{L}(\mathbf{w}, b) = \sum_{i=1}^{n} \max(0, -y_i \hat{y}_i)
$$

We consider the hinge loss for a single sample $(\mathbf{x}_i, y_i)$:

$$
\mathcal{L}_i(\mathbf{w}, b) = \max(0, -y_i \hat{y}_i)
$$

Both $y_i$ and $\hat{y}_i$ are either $-1$ or $1$. Therefore, 

- If $y_i = f(\mathbf{x}_i)$, the prediction is correct (i.e., $y_i \hat{y}_i = 1$), and the hinge loss is zero.
- If $y_i \neq f(\mathbf{x}_i)$, the prediction is incorrect (i.e., $y_i \hat{y}_i = -1$), and the hinge loss is $-y_i \hat{y}_i$.

Note that when we have the "perfect" classifier, the hinge loss $\mathcal{L}(\mathbf{w}, b) = 0$.

## Optimization

The goal of the perceptron learning algorithm is to minimize the hinge loss by updating the weight vector $\mathbf{w}$ and the bias $b$. To achieve this, we use gradient descent to update the parameters in the direction that reduces the loss. The update rule for the weight vector $\mathbf{w}$ and the bias $b$ is as follows:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b) \\
b &\leftarrow b - \eta \nabla_{b} \mathcal{L}_i(\mathbf{w}, b)
\end{align*}
$$

where $\nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b)$ and $\nabla_{b} \mathcal{L}_i(\mathbf{w}, b)$ are the gradients of the hinge loss for sample $(\mathbf{x}_i, y_i)$ with respect to the weight vector $\mathbf{w}$ and the bias $b$.

If $-y_i \hat{y}_i \leq 0$, the gradient of the hinge loss is zero, and the weight vector $\mathbf{w}$ and the bias $b$ remain unchanged.

If $-y_i \hat{y}_i > 0$, the gradient of the hinge loss for sample $(\mathbf{x}_i, y_i)$ with respect to the weight vector $\mathbf{w}$ and the bias $b$ is:

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

There is another way to update the weight vector $\mathbf{w}$ and the bias $b$:

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i \\
b &\leftarrow b + \eta (y_i - \hat{y}_i)
\end{align*}
$$

This update rule is equivalent to the previous one. When $y_i = \hat{y}_i$, the update is zero, and the weight vector $\mathbf{w}$ and the bias $b$ remain unchanged. When $y_i \neq \hat{y}_i$, the update is $y_i - \hat{y}_i$, which is equivalent to $2 y_i$. By adjusting the learning rate $\eta$, we can consider these two update rules as equivalent.

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
\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i 
$$

The perceptron learning algorithm is summarized in the following steps:

```{prf:algorithm} Perceptron Learning Algorithm
:label: perceptron-learning-algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, Learning rate $\eta$, Number of epochs $T$

**Output**: Weight vector $\mathbf{w}$

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ 
2. **For** $t = 1$ to $T$
    1. **For** $i = 1$ to $n$
        1. Compute the prediction $f(\mathbf{x}_i) = h(\mathbf{w} \cdot \mathbf{x}_i)$
        2. **If** $y_i f(\mathbf{x}_i) \leq 0$
            1. Update the weight vector $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$
**Return** $\mathbf{w}$
```

In the perceptron learning algorithm, the weight vector $\mathbf{w}$ and is updated iteratively for each training example $(\mathbf{x}_i, y_i)$ in the training data $\mathcal{D}$. The learning rate $\eta$ controls the step size of the updates. The algorithm continues for a fixed number of epochs $T$ or until convergence.

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

# パセプトロン

## はじめに

パーセプトロンは、バイナリ分類器の教師あり学習のための最も古いアルゴリズムの1つです。パーセプトロンは、1943年にWarren McCullochとWalter Pittsによって提案され、1957年にFrank Rosenblattによって実装されました。

パーセプトロンは、入力$\mathbf{x} \in \mathbb{R}^n$を出力値$f(\mathbf{x}) \in \{-1, 1\}$にマッピングするバイナリ線形分類器です。パーセプトロンは次のように表すことができます。

$$
f(\mathbf{x}) = h(\mathbf{w} \cdot \mathbf{x} + b)
$$

ここで、$\mathbf{w}$は重みベクトル、$\mathbf{w} \in \mathbb{R}^n$、$\mathbf{b}$はバイアス、$h(\cdot)$は次のように定義されます。

$$
h(z) =
\begin{cases}
1 & \text{if } z \geq 0 \\
-1 & \text{otherwise}
\end{cases}
$$

適切な重みベクトル$\mathbf{w}$とバイアス$b$の値を設定することで、パーセプトロンは入力データ$\mathbf{x}$を$-1$と$1$の出力値で表される2つのクラスに分類することを学習することができます。

## 損失関数

説明を簡略化するために、$\hat{y}_i = f(\mathbf{x}_i)$を使用して、入力$\mathbf{x}_i$のパーセプトロンの予測を表します。入力$\mathbf{x}_i$の真のラベルを$y_i$とします。パーセプトロンは、$\hat{y}_i = y_i$の場合は正しい予測を行い、$\hat{y}_i \neq y_i$の場合は誤った予測を行います。

パーセプトロン学習アルゴリズムで使用される損失関数は、ヒンジ損失と呼ばれ、次のように定義されます。

$$
\mathcal{L}(\mathbf{w}, b) = \sum_{i=1}^{n} \max(0, -y_i \hat{y}_i)
$$

単一のサンプル$(\mathbf{x}_i, y_i)$のヒンジ損失は次のようになります。

$$
\mathcal{L}_i(\mathbf{w}, b) = \max(0, -y_i \hat{y}_i)
$$

$y_i$と$\hat{y}_i$はどちらも$-1$または$1$のいずれかです。したがって、

- $y_i = f(\mathbf{x}_i)$の場合、予測は正しい（すなわち、$y_i \hat{y}_i = 1$）ため、ヒンジ損失はゼロです。
- $y_i \neq f(\mathbf{x}_i)$の場合、予測は誤っています（すなわち、$y_i \hat{y}_i = -1$）ため、ヒンジ損失は$-y_i \hat{y}_i = 1$です。

「完璧な」分類器の場合、ヒンジ損失$\mathcal{L}(\mathbf{w}, b) = 0$です。 

## 最適化

パーセプトロン学習アルゴリズムの目標は、ヒンジ損失を最小化することで、重みベクトル$\mathbf{w}$とバイアス$b$を更新することです。これを達成するために、損失を減少させる方向にパラメータを更新するために勾配降下法を使用します。重みベクトル$\mathbf{w}$とバイアス$b$の更新ルールは次のとおりです。

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b) \\
b &\leftarrow b - \eta \nabla_{b} \mathcal{L}_i(\mathbf{w}, b)
\end{align*}
$$

ここで、$\nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}, b)$と$\nabla_{b} \mathcal{L}_i(\mathbf{w}, b)$は、サンプル$(\mathbf{x}_i, y_i)$に対するヒンジ損失の重みベクトル$\mathbf{w}$とバイアス$b$に関する勾配です。

$-y_i \hat{y}_i \leq 0$の場合、ヒンジ損失の勾配はゼロであり、重みベクトル$\mathbf{w}$とバイアス$b$は変更されません。

$-y_i \hat{y}_i > 0$の場合、サンプル$(\mathbf{x}_i, y_i)$に対するヒンジ損失の重みベクトル$\mathbf{w}$とバイアス$b$の勾配は次のようになります。

$$
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}, b) &= -y_i \mathbf{x}_i \\
\nabla_{b} \mathcal{L}(\mathbf{w}, b) &= -y_i
\end{align*}
$$

これらの勾配を更新ルールに代入すると、次のようになります。

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i \\
b &\leftarrow b + \eta y_i
\end{align*}
$$

重みベクトル$\mathbf{w}$とバイアス$b$を更新する別の方法があります。

$$
\begin{align*}
\mathbf{w} &\leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i \\
b &\leftarrow b + \eta (y_i - \hat{y}_i)
\end{align*}
$$

この更新ルールは、前のものと同等です。$y_i = \hat{y}_i$の場合、更新はゼロであり、重みベクトル$\mathbf{w}$とバイアス$b$は変更されません。$y_i \neq \hat{y}_i$の場合、更新は$y_i - \hat{y}_i$であり、これは$2 y_i$と等価です。学習率$\eta$を調整することで、これら2つの更新ルールを同等と見なすことができます。

## アルゴリズム

パーセプトロン学習アルゴリズムの実装では、$x_0 = 1$および$w_0 = b$を設定することで、以下のように表現することができます。

$$
\mathbf{w} = [b, w_1, w_2, \ldots, w_n]
$$

および

$$
\mathbf{x} = [1, x_1, x_2, \ldots, x_n]
$$

ここで、$\mathbf{w} \in \mathbb{R}^{n+1}$および$\mathbf{x} \in \mathbb{R}^{n+1}$です。パーセプトロンは次のように表現することができます。

$$
f(\mathbf{x}) = h(\mathbf{w} \cdot \mathbf{x})
$$

ここで、$\mathbf{w} \cdot \mathbf{x} = \sum_{i=0}^{n} w_i x_i$です。

$x_0 = 1$と設定したため、更新ルールは次のように簡略化されます。

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i
$$

パーセプトロン学習アルゴリズムは、以下の通りです。

```{prf:algorithm} Perceptron Learning Algorithm
:label: perceptron-learning-algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, Learning rate $\eta$, Number of epochs $T$

**Output**: Weight vector $\mathbf{w}$

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ 
2. **For** $t = 1$ to $T$
    1. **For** $i = 1$ to $n$
        1. Compute the prediction $f(\mathbf{x}_i) = h(\mathbf{w} \cdot \mathbf{x}_i)$
        2. **If** $y_i f(\mathbf{x}_i) \leq 0$
            1. Update the weight vector $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$
**Return** $\mathbf{w}$
```

パーセプトロン学習アルゴリズムでは、重みベクトル$\mathbf{w}$は、トレーニングデータ$\mathcal{D}$の各トレーニング例$(\mathbf{x}_i, y_i)$に対して反復的に更新されます。学習率$\eta$は、更新のステップサイズを制御します。アルゴリズムは、固定されたエポック数$T$または収束まで続行されます。

## Python実装

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