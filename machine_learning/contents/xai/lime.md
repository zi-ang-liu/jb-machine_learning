# Local Interpretable Model-agnostic Explanations (LIME)

## Notation
* $f$ is the model to be explained
* $g$ is the interpretable model
* $\pi_x$ is the proximity measure
* $\Omega(g)$ is the complexity measure
* $\mathcal{L}$ is the fidelity measure
* $G$ is the set of interpretable models

The objective is to find an interpretable model $g$ that is locally faithful to the model $f$ at the instance $x$ and is simple. $\mathcal{L}$ is the fidelity measure that measures how well the interpretable model $g$ approximates the model $f$ at the instance $x$. The smaller the value of $\mathcal{L}$, the better the approximation. $\Omega(g)$ is the complexity measure that measures the complexity of the interpretable model $g$. The smaller the value of $\Omega(g)$, the simpler the model.
$$
\xi(x) = \argmin_{g \in G}\;\;\mathcal{L}(f, g, \pi_x) + \Omega(g)
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

$$
\Omega(g) = \Omega(\boldsymbol{w}) = \sum_{i=1}^{M} w_i