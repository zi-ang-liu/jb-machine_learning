# Local Interpretable Model-agnostic Explanations (LIME)

LIME is a model-agnostic method for explaining the predictions of machine learning models {cite:p}`pRibeiro2016-vc`.

## Notation
* $f$ is the model to be explained
* $g$ is the interpretable model
* $\pi_x$ is the proximity measure
* $\Omega(g)$ is the complexity measure
* $\mathcal{L}$ is the fidelity measure
* $G$ is the set of interpretable models

The objective is to find an interpretable model $g$ that is locally faithful to the model $f$ at the instance $x$ and is simple. $\mathcal{L}$ is the fidelity measure that measures how well the interpretable model $g$ approximates the model $f$ at the instance $x$. The smaller the value of $\mathcal{L}$, the better the approximation. $\Omega(g)$ is the complexity measure that measures the complexity of the interpretable model $g$. The smaller the value of $\Omega(g)$, the simpler the model.

$$
\xi(x) = \arg\min_{g \in G}\;\;\mathcal{L}(f, g, \pi_x) + \Omega(g)
$$

$\pi_x$ is the proximity measure that measures the similarity between the instance $x$ and the samples $z$ in the neighborhood. The proximity measure is defined as follows:

$$
\pi_x(z) = \exp(-\frac{D(x, z)^2}{\sigma^2})
$$

$\mathcal{L}$ is the weighted least squares loss function. The fidelity measure is defined as follows:

$$
\mathcal{L}(f, g, \pi_x) = \sum_{i=1}^{N} \pi_x(z_i)(f(z_i) - g(z'_i))^2
$$

The intuition of $\mathcal{L}$ is that the data points $z_i$ that are closer to the instance $x$ should have more weight in the loss function.

$\Omega(g)$ is the complexity measure. In the original paper, for text classification, it can be defined as follows:

$$
\Omega(g) = \infty \mathbb{1} [ \|w_g\|_0 > K ]
$$

where $w_g$ is the weight vector of the interpretable model $g$ and $K$ is the maximum number of non-zero weights. $\|w_g\|_0$ is the number of non-zero weights in the weight vector $w_g$. The complexity measure $\Omega(g)$ is the number of non-zero weights in the weight vector $w_g$.

Let $\mathbf{x}$ represent a vector. Then $\|\mathbf{x}\|_0$ is:

$$
\|\mathbf{x}\|_0 = \sum_{i=1}^{N} \mathbb{1}[x_i \neq 0]
$$

## Algorithm

```{prf:algorithm} Sparse Linear Explanations using LIME
:label: sle-lime

**Input**: Classifier $f$, Number of samples $N$, Instance $x$ and it's interpretable version$x'$, Similarity kernel $\pi_x$, Length of explanation $K$   
**Output**: $w$

1. $\mathcal{Z} \leftarrow \emptyset$
2. **For** $i = 1$ to $N$
    1. $z'_i \leftarrow \text{sample}(x')$
    2. $\mathcal{Z} \leftarrow \mathcal{Z} \cup (z_i, f(z_i), \pi_x(z_i))$
3. $w \leftarrow \text{K-LASSO}(\mathcal{Z}, K)$ 
```

## References
```{bibliography}
:style: unsrt
```
