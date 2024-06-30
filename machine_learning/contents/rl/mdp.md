# Markov Decision Processes

The multi-armed bandit problem is a nonassociative problem, that is it does not involve learning to action in more than one situation. Consider the newsvendor problem, the number of newspapers in the morning is always 0, regardless of the previous day's orders and sales. However, many real-world problems are associative. Therefore, we need to learn to choose different actions in different situations.

Markov decision processes (MDPs) are associative problems in which the action taken in the current period affects the future states and rewards. MDPs are idealized models of reinforcement learning problems. In MDPs, complete knowledge of the environment is available.

## Formulation of MDP

A Markov decision process (MDP) is a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ where:

* $\mathcal{S}$: Set of states
* $\mathcal{A}$: Set of actions
* $p(s', r | s, a)$: Dynamics function $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$
* $r(s, a)$: Reward function $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$
* $\gamma$: Discount factor $\gamma \in [0, 1]$


### Agent-Environment Interface

MDPs are used to formulate sequential decision-making problems. The _agent_ interacts with the _environment_ to achieve a goal.

* Agent: The learner and decision-maker
* Environment: Everything outside the agent

```{figure} ../images/mdp/agent_env.svg
---
width: 400px
name: agent_env
---
Agent-Environment Interface
```

The agent interacts with the environment in discrete time $t = 0, 1, 2, \ldots$. At each time step $t$, the agent receives a representation of the environment's _state_ $S_t \in \mathcal{S}$, selects an _action_ $A_t \in \mathcal{A}(S_t)$, and receives a _reward_ $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ and the next state $S_{t+1}$.

The interaction between the agent and the environment can be summarized as _trajectory_:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots$$

### Dynamics function

The function function $p$ defines the probability of transitioning to state $s'$ and receiving reward $r$ given state $s$ and action $a$.

$$
p(s', r | s, a) = \Pr\{S_{t} = s', R_{t} = r | S_{t-1} = s, A_{t-1} = a\}
$$

The state $s_t$ is Markov if and only if:

$$
\Pr\{S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} | S_t = s_t, A_t = a_t\}\\
= \Pr\{S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} | S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \ldots\}
$$

<!-- 
\Pr\{S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} | S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \ldots\} = \Pr\{S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} | S_t = s_t, A_t = a_t\} -->

<!-- The state has the _Markov property_ if the state includes all relevant information from the interaction history that may affect the future. -->

Note that $p$ captures all the environment's dynamics completely. The possibility of each possible $S_t$ and $R_t$ depends only on the preceding state $S_{t-1}$ and action $A_{t-1}$.

### Reward function

The reward function $r(s, a)$ defines the expected rewards given state $s$ and action $a$.

$$
\begin{aligned} r(s, a) &= \mathbb{E}[R_{t} | S_{t} = s, A_{t} = a] \\ &= \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r | s, a) \end{aligned}
$$

### Return

The objective of the agent is to maximize the expected value of the cumulative sum of a received scalar signal (reward).

To formalize the objective, we define the _return_ $G_t$ as certain function of the rewards received after time $t$. The simplest form of return is the sum of rewards:

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + \ldots + R_T
$$

where $T$ is the final time step. Note that this definition of return is suitable for tasks that will eventually end. In such cases, each _episode_ ends in a _terminal state_.

There are also MDPs that do not have terminal states. We call such MDPs _continuing tasks_. In continuing tasks, the return that we defined above could be infinite. To handle such cases, we introduce the concept of _discounted return_:

$$
\begin{aligned} G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\ &= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \end{aligned}
$$

where $\gamma$ is the _discount factor_ that satisfies $0 \leq \gamma \leq 1$.

The discount return can be represented as the following recursive form:

$$
\begin{aligned} G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots \\ &= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \ldots) \\ &= R_{t+1} + \gamma G_{t+1} \end{aligned}
$$

## Policy and Value function

### Policy

The agent interacts with the environment by selecting actions. To describe the agent's behavior, we introduce the concept of _policy_. A policy is a mapping from states to probabilities of selecting each possible action. A policy is denoted by $\pi$.

$$
\pi(a | s) = \Pr\{A_t = a | S_t = s\}
$$

### Value function

The value function is the expected return when starting in state $s$ and following policy $\pi$ thereafter. The _state-value function for policy_ $\pi$ is denoted by $v_{\pi}(s)$.

$$
v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]
$$

Similarly, we use a action-value function to represent the expected return when starting in state $s$, taking action $a$, and following policy $\pi$ thereafter. The _action-value function for policy_ $\pi$ is denoted by $q_{\pi}(s, a)$.

$$
q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]
$$

An important property of value functions is that they satisfy recursive relationships. The value of a state can be expressed in terms of the value of its possible successor states:

$$
\begin{aligned} v_{\pi}(s) &= \mathbb{E}_{\pi}[G_t | S_t = s] \\ &= \mathbb{E}_{\pi}[{R_{t+1} + \gamma G_{t+1} | S_t = s}] \\ &= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma \mathbb{E}_{\pi}[G_{t+1} | S_{t+1} = s']] \\ &= \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')] \end{aligned}
$$

The last equation is known as the _Bellman equation_.

$$
v_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]
$$

It is write in recursive form that indicates the relationship between $v_{\pi}(s)$ and all the possible successor states' values $v_{\pi}(s')$.


## Optimal Policies and Optimal Value Functions

For all $s \in \mathcal{S}$, if $v_{\pi}(s) \geq v_{\pi'}(s)$, then $\pi$ is better than or equal to $\pi'$, denoted by $\pi \geq \pi'$. There is always at least one policy that is better than or equal to all other policies. This policy is called the _optimal policy_ and denoted by $\pi_*$.

Using the concept of optimal policy, we can define the _optimal state-value function_ and _optimal action-value function_ as follows:

$$
v_*(s) = \max_{\pi} v_{\pi}(s), \quad \forall s \in \mathcal{S}
$$

$$
q_*(s, a) = \max_{\pi} q_{\pi}(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}
$$

### Bellman Optimality Equation

$$
\begin{aligned} v_*(s) &= \max_{a} q_*(s, a) \\ &= \max_{a} \mathbb{E}_{\pi_*}[G_t | S_t = s, A_t = a] \\ &= \max_{a} \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\ &= \max_{a} \mathbb{E}[R_{t+1} + \gamma v_*({S_{t+1}}) | S_t = s, A_t = a] \\ &= \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')] \end{aligned}
$$

The last equation is known as the _Bellman optimality equation_ for the state-value function.

$$
v_*(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]
$$

The Bellman optimality equation is a system of nonlinear equations. The solution to the system of equations is the optimal value function. For a finite MDP that has $n$ states, the system of equations has $n$ equations and $n$ unknowns.

In addition, the Bellman optimality equation for the action-value function is:

$$
q_*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_*(s', a')]
$$

Note that if we know the optimal value function $v_*(s)$, for all $s \in \mathcal{S}$, we can easily find the optimal policy $\pi_*(s)$ by selecting the action that maximizes the right-hand side of the Bellman optimality equation.

$$
\pi_*(s) = \arg\max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]
$$

### Linear Programming

The Bellman optimality equation can be solved using linear programming. This is a less frequently used method for solving MDP. The idea is that

* If $v(s) \geq r(s, a) + \gamma \sum_{s'} p(s' | s, a) v(s')$ for all $s \in S$ and $a \in A$, then $v(s)$ is an upper bound on $v_*(s)$.
* $v_*(s)$ must be the smallest such solution

The linear programming formulation of the Bellman optimality equation is as follows:

$$
\begin{aligned} \textnormal{minimize } & \sum_{s \in S} \alpha_s v(s) \\ \textnormal{s.t. } & v(s) \geq r(s, a) + \gamma \sum_{s'} p(s' | s, a) v(s'), \forall s \in S, \forall a \in A \\ & \textnormal{The constants $\alpha_s$ are arbitrary positive numbers.} \end{aligned}
$$

Notes:

* Linear programming methods can also be used to solve MDPs
* Linear programming methods become impractical at a much smaller number of states than do DP methods (by a factor of about 100).

## Python Implementation
### Linear Programming for Cliff Walking Problem

The agent is placed in a 4x12 grid world. The agent can move in four directions: up, down, left, and right. The agent receives a reward of -1 for each step taken. The agent receives a reward of -100 for falling off the cliff. The agent receives a reward of 0 for reaching the goal. Figure below shows the cliff walking problem implemented in OpenAI Gym.

```{figure} ../images/mdp/cliff_walking.jpg
---
width: 400px
name: agent_env
---
Cliff Walking Problem
```

The agent starts at the bottom-left corner [3, 0] and the goal is at the bottom-right corner [3, 11]. The cliff is at the bottom row [3, 1] to [3, 10]. For simplicity, the state is represented as a single integer from 0 to 47. The state is computed as `current_row * n_col + current_col`.

The goal is to find the optimal policy for moving an agent from a starting position to a goal position as quickly as possible while avoiding falling off a cliff.

```python
# r: reward matrix, n_state * n_action
# p: transition probability matrix, n_state * n_action * n_state
# gamma: discount factor

from gurobipy import GRB, Model, quicksum

def lp_solver(r, p, gamma):

    action_set = set(range(r.shape[1]))
    state_set = set(range(r.shape[0]))
    n_state = len(state_set)

    # create a model instance
    model = Model()

    # create variables
    for s in range(n_state):
        model.addVar(name=f'v_{s}', lb=-GRB.INFINITY)
    
    # update the model
    model.update()

    # create constraints
    for state in state_set:
        for action in action_set:
            model.addConstr(model.getVarByName(f'v_{state}') >= quicksum(
                gamma * p[state, action, next_state] * model.getVarByName(f'v_{next_state}') for next_state in state_set ) + r[state, action])

    # set objective
    model.setObjective(quicksum(model.getVarByName(
        f'v_{state}') for state in state_set ), GRB.MINIMIZE)

    # optimize
    model.optimize()

    return model
```

```python
from gurobipy import GRB, Model, quicksum
import gymnasium as gym
import numpy as np
from lp_solver import lp_solver

# create an environment
env = gym.make('CliffWalking-v0')
n_state = env.unwrapped.nS
n_action = env.unwrapped.nA
state_set = set(range(n_state))
action_set = set(range(n_action))
# The player cannot be at the cliff, nor at the goal 
terminal_state_set = [47] 
unreachable_state_set = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
# the reachable state set is the set of all states except the cliff and the goal.
# only the states in the reachable state set are considered in the optimization problem
reachable_state_set = set(set(state_set) - set(terminal_state_set) - set(unreachable_state_set))

# set parameters
gamma = 1

# initialize reward and transition probability
r = np.zeros((n_state, n_action))
p = np.zeros((n_state, n_action, n_state))

for state in reachable_state_set:
    for action in action_set:
        for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
            r[state, action] += prob * reward
            p[state, action, next_state] += prob

# solve the mdp problem using linear programming
model = lp_solver(r, p, gamma)

# state value
value_function = {}
for state in reachable_state_set:
    value_function[state] = model.getVarByName(f'v_{state}').x

policy = {}
for state in terminal_state_set:
    value_function[47] = 0
    
for state in reachable_state_set:
    q_max_value = -np.inf
    for action in action_set:
        q_value_temp = sum([prob * (reward + gamma * value_function[next_state])
                            for prob, next_state, reward, terminated in env.unwrapped.P[state][action]])
        if q_value_temp > q_max_value:
            q_max_value = q_value_temp
            policy[state] = action
        
# print value function 4*12, 1 digital after decimal point

print('value function = ')
for i in range(4):
    for j in range(12):
        if i * 12 + j in value_function:
            print('{:.1f}'.format(value_function[i * 12 + j]), end='\t')
        else:
            print('x', end='\t')
    print()

print('optimal policy = ')
for i in range(4):
    for j in range(12):
        if i * 12 + j in policy:
            print(policy[i * 12 + j], end='\t')
        else:
            print('x', end='\t')
    print()

model.write("model.lp")
```

## Summary

* Markov decision processes (MDPs) are used to formulate sequential decision-making problems.
* MDPs are a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$.
* The Bellman optimality equation is a system of nonlinear equations that can be solved to find the optimal value function.
* Linear programming can be used to find the optimal value function.
