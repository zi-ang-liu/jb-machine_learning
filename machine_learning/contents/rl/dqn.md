# Deep Q Networks

## Algorithm

```{prf:algorithm} Deep Q-Network
:label: deep-q-network

**Input**: replay buffer capacity $N$, the number of steps $C$ to perform a target update, a small $\epsilon > 0$, a small $\alpha \in (0,1]$   
**Output**: output a deterministic policy $\pi \approx \pi_{*}$

1. Initialize empty replay memory $D$ to capacity $N$
2. Initialize action-value function $Q$ with random weights $\theta$
3. Initialize target action-value function $\hat{Q}$ with weights $\theta^{-} \leftarrow \theta$
4. **for** $\texttt{episode} = 1, 2, \dots M$ **do**
    1. $t \leftarrow 0$
    2. Initialize $S_t$
    3. **while** $S_t$ is not terminal **do**
        1. with probability $\epsilon$ select a random action $A_t$
        2. otherwise take action $A_t$ using policy derived from $Q$
        3. Execute action $A_t$ and observe reward $R_t$ and $S_{t+1}$
        4. Store transition $(S_t, A_t, R_t, S_{t+1})$ in $D$
        5. $t \leftarrow t+1$
        6. Sample random minibatch of transitions $(S_j, A_j, R_j, S_{j+1})$ from $D$
        7. Set $y_j = \begin{cases}
            r_j                                                    & \text{for terminal } S_{j+1}     \\
            r_j + \gamma \max_{a'}\hat{Q}(S_{j+1}, a'; \theta^{-}) & \text{for non-terminal } S_{j+1}
          \end{cases}$
        8. Perform a gradient descent step on $(y_j - Q(S_j, a_j; \theta))^2$ with respect to the network parameters $\theta$
        9. **if** $t \mod C = 0$ **then**
            1. Reset $\hat{Q} \leftarrow Q$
```

## Python Implementation

```python
```