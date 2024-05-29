# Dynamic Programming

## Policy Evaluation

The objective of _policy evaluation_ is to compute the state-value function $v\_{\pi}$ for an arbitrary policy $\pi$. Recall that the state-value function for $s \in \mathcal{S}$ is defined as

$$
\begin{aligned} v_{\pi}(s) &= \mathbb{E}_{\pi}[G_t | S_t = s] \\ &= \mathbb{E}_{\pi}[{R_{t+1} + \gamma G_{t+1} | S_t = s}] \\ &= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma \mathbb{E}_{\pi}[G_{t+1} | S_{t+1} = s']] \\ &= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')] \end{aligned}
$$

In MDPs, the environment's dynamics are completely known, given an arbitrary policy $\pi$, so the only unknown in the above equation is the state-value function $v_{\pi}(s), \forall s \in \mathcal{S}$. Consequently, the equation below is a system of $|\mathcal{S}|$ linear equations in $|\mathcal{S}|$ unknowns,

$$
v_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')], \forall s \in \mathcal{S}
$$

This system of equations can be solved straightforwardly using linear algebra techniques.

In addition, iterative solution methods can also be used to solve the system of equations. First, we initialize the state-value function arbitrarily, say $v_0(s) = 0, \forall s \in \mathcal{S}$. Then, we iteratively update the state-value function using the following update rule,

$$
v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')], \forall s \in \mathcal{S}
$$

Iteratively updating the state-value function using the above update rule will generate a sequence of state-value functions, $v_0, v_1, v_2, \ldots$, which will converge to the true state-value function $v_{\pi}$ as $k \rightarrow \infty$. This algorithm is known as _iterative policy evaluation_.

<figure><img src="../.gitbook/assets/policy_evaluation.png" alt=""><figcaption></figcaption></figure>

\begin{algorithm}[H]
  \caption{Policy Evaluation for estimating $V \approx v_{\pi}$}\label{alg:policy iteration}
  \KwIn{$\theta, \pi$}
  \KwOut{$V \approx v_{\pi}$}
  Initialize $V(s)$ arbitrarily, for all $s \in \mathcal{S}$\;
  $\Delta \leftarrow 2\theta$\;
    \While{$\Delta \geq \theta$}{
      $\Delta \leftarrow 0$\;
      \For{$s \in \mathcal{S}$}{
        $v \leftarrow V(s)$\;
        $V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma V(s')]$\;
        $\Delta \leftarrow \max(\Delta, |v - V(s)|)$\;
      }
    }
\end{algorithm}

```{algorithm} Policy Evaluation for estimating $V \approx v_{\pi}$

```

## Policy Improvement

Given a policy $\pi$, the iterative policy evaluation algorithm can be used to estimate the state-value function $v_{\pi}$. The state-value function $v_{\pi}$ describes expected return from each state under policy $\pi$.

Once the state-value function $v_{\pi}$ is estimated, can we improve the policy to get better expected return? The answer is yes.

For a given state $s \in \mathcal{S}$, we choose the action $a \in \mathcal{A}$ and use $\pi$ thereafter. The value is given by

$$
q_{\pi}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]
$$

{% hint style="info" %}
**Policy Improvement Theorem**:

Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in \mathcal{S}$,

$$
q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s)
$$

Then the policy $\pi'$ must be as good as, or better than, policy $\pi$. Therefore, for all $s \in \mathcal{S}$,

$$
v_{\pi'}(s) \geq v_{\pi}(s)
$$

Furthermore, if $q_{\pi}(s, \pi'(s)) > v_{\pi}(s)$ for at least one state $s \in \mathcal{S}$, then the policy $\pi'$ is strictly better than policy $\pi$.
{% endhint %}

**Proof**:

$$
\begin{aligned} v_{\pi}(s) & \leq q_{\pi}(s, \pi'(s)) \\ & = \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s, A_t = \pi'(s)] \\ & = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s] \\ & \leq \mathbb{E}_{\pi'}[{R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1})) | S_t = s}] \\ & = \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2}) | S_t = s] \\ & \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 q_{\pi}(S_{t+2}, \pi'(S_{t+2})) | S_t = s] \\ & \dots \\ & \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots | S_t = s] \\ & = v_{\pi'}(s) \end{aligned}
$$

Note that $\mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2}) | S_t = s]$ means that we choose actions according to $\pi'$ and get $R_{t+1}$ and $R_{t+2}$ and then use $\pi$ thereafter.

With the policy improvement theorem, given a policy $\pi$ and its state-value function $v_{\pi}$, we can construct a new policy $\pi'$ that is as good as, or better than, policy $\pi$. The new policy $\pi'$ is constructed by selecting the action that maximizes the state-action value function $q_{\pi}(s, a)$ for each state $s \in \mathcal{S}$,

$$
\begin{aligned} \pi'(s) &= \arg\max_{a} q_{\pi}(s, a) \\ &= \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')] \end{aligned}
$$

This algorithm is known as _policy improvement_.

<figure><img src="../.gitbook/assets/policy_improvement.png" alt=""><figcaption></figcaption></figure>

If the new policy $\pi'$ is the same as the old policy $\pi$, $\pi' = \pi$, then we have the following equation for all $s \in \mathcal{S}$,

$$
\begin{aligned} \pi'(s) &= \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')] \\ &= \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi'}(s')] \\ \end{aligned}
$$

Hence,

$$
\begin{aligned} v_{\pi'}(s) &= \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi'}(s')] \\ \end{aligned}
$$

This equation is the same as the Bellman optimality equation. Therefore, $v_{\pi'}(s) = v_{*}(s)$, and $\pi' = \pi_{*}$.

## Policy Iteration

Given an arbitrary policy $\pi$, we can use the iterative policy evaluation algorithm to estimate the state-value function $v_{\pi}$. Then, we can use the policy improvement algorithm to construct a new policy $\pi'$ that is as good as, or better than, policy $\pi$. Iteratively applying the policy evaluation and policy improvement algorithms will generate a sequence of policies, $\pi_0, \pi_1, \pi_2, \ldots$. If the new policy $\pi'$ is the same as the old policy $\pi$, then we have found the optimal policy $\pi_{*}$.

This algorithm is known as _policy iteration_. The pseudo-code for policy iteration is as follows,

<figure><img src="../.gitbook/assets/policy_iteration.png" alt=""><figcaption></figcaption></figure>

## Value Iteration

The policy iteration algorithm iteratively applies the policy evaluation and policy improvement to find the optimal policy $\pi_{*}$.

An alternative to policy iteration is the _value iteration_ algorithm. The value iteration iteratively applies the Bellman optimality equation as an update rule to find the optimal state-value function $v_{*}$.

$$
v_{k+1}(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_{k}(s')], \forall s \in \mathcal{S}
$$

From an arbitrary initial state-value function $v_0$, the value iteration algorithm generates a sequence of state-value functions, $v_0, v_1, v_2, \ldots$, which will eventually converge to the optimal state-value function $v_{*}$.

The pseudo-code for value iteration is as follows,

<figure><img src="../.gitbook/assets/value_iteration.png" alt=""><figcaption></figcaption></figure>

## Python Implementation

### Value Iteration for Jack's Car Rental

The goal is to find the optimal policy for moving cars between two locations based on the expected number of cars at each location at the end of the day. The problem is formulated as a Markov Decision Process (MDP) and solved using value iteration

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

def build_car_rental_dynamics():
    # parameters
    CREDIT = 10
    MOVING_COST = 2
    LAMBDA_RENTAL_REQUEST_1 = 3
    LAMBDA_RENTAL_REQUEST_2 = 4
    LAMBDA_RETURN_1 = 3
    LAMBDA_RETURN_2 = 2
    MAX_CARS = 20
    MAX_MOVE = 5
    EPSILON = 1e-4

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RENTAL_REQUEST_1) < EPSILON:
            MAX_RENTAL_REQUEST_1 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RENTAL_REQUEST_2) < EPSILON:
            MAX_RENTAL_REQUEST_2 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RETURN_1) < EPSILON:
            MAX_RETURN_1 = i
            break

    for i in range(MAX_CARS + 1):
        if poisson.pmf(i, LAMBDA_RETURN_2) < EPSILON:
            MAX_RETURN_2 = i
            break

    # initialize state space, action space, dynamics, and probability dictionary
    state_space = [(i, j) for i in range(MAX_CARS + 1)
                   for j in range(MAX_CARS + 1)]
    action_space = list(range(-MAX_MOVE, MAX_MOVE + 1))
    dynamics = {}
    prob_dict = {}

    # build probability dictionary
    for rental_request_1 in range(MAX_RENTAL_REQUEST_1 + 1):
        prob_dict[rental_request_1, LAMBDA_RENTAL_REQUEST_1] = poisson.pmf(
            rental_request_1, LAMBDA_RENTAL_REQUEST_1)
    for rental_request_2 in range(MAX_RENTAL_REQUEST_2 + 1):
        prob_dict[rental_request_2, LAMBDA_RENTAL_REQUEST_2] = poisson.pmf(
            rental_request_2, LAMBDA_RENTAL_REQUEST_2)
    for return_1 in range(MAX_RETURN_1 + 1):
        prob_dict[return_1, LAMBDA_RETURN_1] = poisson.pmf(
            return_1, LAMBDA_RETURN_1)
    for return_2 in range(MAX_RETURN_2 + 1):
        prob_dict[return_2, LAMBDA_RETURN_2] = poisson.pmf(
            return_2, LAMBDA_RETURN_2)

    # build dynamics
    for state in state_space:
        for action in action_space:
            dynamics[state, action] = {}
            # invalid action
            if not ((0 <= action <= state[0]) or (-state[1] <= action <= 0)):
                reward = -np.inf
                next_state = state
                dynamics[state, action][next_state, reward] = 1
                continue

            for rental_request_1 in range(MAX_RENTAL_REQUEST_1 + 1):
                for rental_request_2 in range(MAX_RENTAL_REQUEST_2 + 1):
                    for return_1 in range(MAX_RETURN_1 + 1):
                        for return_2 in range(MAX_RETURN_2 + 1):
                            # moving cars
                            next_state = (
                                min(state[0] - action, MAX_CARS), min(state[1] + action, MAX_CARS))
                            reward = -MOVING_COST * abs(action)

                            prob = prob_dict[rental_request_1, LAMBDA_RENTAL_REQUEST_1] * \
                                prob_dict[rental_request_2, LAMBDA_RENTAL_REQUEST_2] * \
                                prob_dict[return_1, LAMBDA_RETURN_1] * \
                                prob_dict[return_2, LAMBDA_RETURN_2]
                            valid_rental_1 = min(next_state[0], rental_request_1)
                            valid_rental_2 = min(next_state[1], rental_request_2)
                            reward = reward + CREDIT * \
                                (valid_rental_1 + valid_rental_2)
                            next_state = (
                                next_state[0] - valid_rental_1, next_state[1] - valid_rental_2)

                            # return cars
                            next_state = (min(next_state[0] + return_1, MAX_CARS), min(
                                next_state[1] + return_2, MAX_CARS))

                            if (next_state, reward) in dynamics[state, action]:
                                dynamics[state, action][next_state, reward] += prob
                            else:
                                dynamics[state, action][next_state, reward] = prob

    init_value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    init_policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    return dynamics, state_space, action_space, init_value, init_policy

def value_iteration(dynamics, state_space, action_space, value, policy, theta=1e-4, gamma=0.9):
    # initialize value
    delta = np.inf
    k = 0
    while delta >= theta:
        k = k + 1
        value_old = value.copy()
        for state in state_space:
            # Update V[s].
            value[state] = max([sum([prob * (reward + gamma * value_old[next_state]) for (
                next_state, reward), prob in dynamics[state, action].items()]) for action in action_space])
            # print('State {}, value = {}'.format(state, value[state]))
        delta = np.max(np.abs(value - value_old))
        print('Iteration {}, delta = {}'.format(k, delta))

    for state in state_space:
        q_max_value = -np.inf
        for action in action_space:
            q_value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if q_value_temp > q_max_value:
                q_max_value = q_value_temp
                policy[state] = action
    return value, policy


if __name__ == '__main__':

    dynamics, state_space, action_space, init_value, init_policy = build_car_rental_dynamics()

    value, policy = value_iteration(
        dynamics, state_space, action_space, init_value, init_policy)

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(policy, cmap='viridis', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('# of cars at second location', fontsize=16)
    ax.set_ylabel('# of cars at first location', fontsize=16)
    ax.set_title('Policy', fontsize=20)
    plt.savefig('policy.png')
    plt.close()

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(value, cmap='viridis', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('# of cars at second location', fontsize=16)
    ax.set_ylabel('# of cars at first location', fontsize=16)
    ax.set_title('Value', fontsize=20)
    plt.savefig('Value.png')
    plt.close()
```

## Summary

- Dynamic programming methods can be used to solve MDPs.
- Policy evaluation estimates the state-value function for an arbitrary policy.
- Policy improvement constructs a new policy that is as good as, or better than, the old policy.
- Policy iteration iteratively applies policy evaluation and policy improvement to find the optimal policy.
- Value iteration iteratively applies the Bellman optimality equation to find the optimal state-value function.