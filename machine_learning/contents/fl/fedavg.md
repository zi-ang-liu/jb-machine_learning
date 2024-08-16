# The FederatedAveraging Algorithm

## Algorithm

```{prf:algorithm} Federated Averaging
:label: federated-averaging

**Input**: $K$ clients, the number of local epochs $E$, the learning rate $\alpha$, the batch size $B$, fraction $C$ of clients to sample 
**Output**: output a global model $w$

**Server executes:**   
1. Initialize global model $w_0$
2. **for** each round $t = 1, 2, \dot$ **do**
    1. $m \leftarrow \max(C \cdot K, 1)$
    2. $S_t \leftarrow$ (random set of $m$ clients)
    3. **for** each client $k \in S_t$ **do**
        1. $w^k_{t+1} \leftarrow \text{ClientUpdate}(k, w_t)$
    4. $n \leftarrow \sum_{k \in S_t} n_k$
    5. $w_{t+1} \leftarrow \sum_{k \in S_t} \frac{n_k}{n} w^k_{t+1}$
   
**ClientUpdate($k, w$):**
1. $\mathcal{B} \leftarrow$ (split $\mathcal{P}_k$ into batches of size $B$)
2. **for** $e = 1, 2, \dots, E$ **do**
    1. **for** batch $b \in \mathcal{B}$ **do**
        1. $w \leftarrow w - \alpha \nabla \ell(w; b)$
3. **return** $w$
```