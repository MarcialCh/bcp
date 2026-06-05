### Lemma 1 (Total Bid Bound under Lazy Proportional-Share Allocation)

Let

$$
W=\{i_1,i_2,\ldots,i_m\}
$$

be the winner set returned by the lazy-threshold proportional-share allocation rule.

Suppose that every selected seller satisfies

$$
b_{i_t}
\le
\frac{(1-\varepsilon)B}{2}
\cdot
\frac{U_{i_t\mid W_{t-1}}}
     {U(W_t)},
$$

where

$$
W_t=\{i_1,\ldots,i_t\}.
$$

Then,

$$
\sum_{i\in W} b_i
\le
\frac{B}{2}.
$$

### Proof

Consider the last winner $i_m$.

By the proportional-share condition,

$$
b_{i_m}
\le
\frac{(1-\varepsilon)B}{2}
\frac{U_{i_m\mid W_{m-1}}}
     {U(W)}.
$$

Rearranging gives

$$
\frac{U_{i_m\mid W_{m-1}}}{b_{i_m}} \ge \frac{2U(W)}{(1-\varepsilon)B}. (1)
$$

Since the allocation rule selects a $(1-\varepsilon)$-approximate
maximum-density seller at every iteration,

$$
\frac{U_{i_t\mid W_{t-1}}}
     {b_{i_t}}
\ge
(1-\varepsilon)
\frac{U_{i_m\mid W_{m-1}}}
     {b_{i_m}}
\qquad
\forall t\le m.
$$

Combining with (1),

$$
\frac{U_{i_t\mid W_{t-1}}}
     {b_{i_t}}
\ge
(1-\varepsilon)
\frac{2U(W)}
     {(1-\varepsilon)B}
= \frac{2U(W)}{B}.
$$

Therefore,

$$
b_{i_t}
\le
\frac{B}{2}
\frac{U_{i_t\mid W_{t-1}}}
     {U(W)}.
$$

Summing over all winners,

$$
\sum_{i\in W} b_i
\le
\frac{B}{2U(W)}
\sum_{t=1}^{m}
U_{i_t\mid W_{t-1}}.
$$

By telescoping,

$$
\sum_{t=1}^{m}
U_{i_t\mid W_{t-1}}
= U(W).
$$

Hence,

$$
\sum_{i\in W} b_i
\le
\frac{B}{2}.
$$

$\square$


### Theorem 1.1 (Budget Feasibility under Lazy Evaluation)

Let $W$ be the winner set returned by the lazy-threshold proportional-share allocation rule. Suppose that each selected seller satisfies

$$
\frac{U_{i\mid S}}{b_i}
\ge
(1-\varepsilon)
\max_{j\in A\setminus S}
\frac{U_{j\mid S}}{b_j}.
$$

The winner selection rule accepts seller $i$ only if

$$
b_i
\le
\frac{(1-\varepsilon)B}{2}
\cdot
\frac{U_{i\mid S}}
     {U(S\cup\{i\})}.
$$

For every winner $k\in W$, the payment is defined as

$$
p_k=
\min\{b_{k(r)}^{\varepsilon},
\beta_{k(r)}^{\varepsilon}\}.
$$

where

$$
b_{k(j)}^{\varepsilon}=
(1-\varepsilon)b_j
\frac{U_{k\mid S_{j-1}}}
     {U_{j\mid S_{j-1}}},
$$

and

$$
\beta_{k(j)}^{\varepsilon}=
\frac{(1-\varepsilon)B}{2}
\frac{U_{k\mid S_{j-1}}}
     {U(S_{j-1}\cup\{k\})}.
$$

Then the mechanism is budget feasible, i.e.,

$$
\sum_{k\in W} p_k \le B.
$$

### Proof

We first establish an upper bound on the payment of every winner.

Consider an arbitrary winner $k\in W$. By the definition of $p_k$, there exists an index $r$ such that

$$
p_k=
\min\{b_{k(r)}^{\varepsilon},
\beta_{k(r)}^{\varepsilon}\}.
$$

Therefore,

$$
p_k
\le
b_{k(r)}^{\varepsilon}
=(1-\varepsilon)b_r
\frac{U_{k\mid S_{r-1}}}
     {U_{r\mid S_{r-1}}}.
$$

Rearranging gives

$$
\frac{U_{r\mid S_{r-1}}}{b_r}
\le
(1-\varepsilon)
\frac{U_{k\mid S_{r-1}}}{p_k}.
(1)
$$

Similarly,

$$
p_k
\le
\beta_{k(r)}^{\varepsilon}
=\frac{(1-\varepsilon)B}{2}
\frac{U_{k\mid S_{r-1}}}
     {U(S_{r-1}\cup\{k\})}.
$$

Hence,

$$
\frac{U_{k\mid S_{r-1}}}{p_k}
\ge
\frac{2U(S_{r-1}\cup\{k\})}
     {(1-\varepsilon)B}.
(2)
$$

We now prove that

$$
p_k
\le
\frac{B U_{k\mid S_{r-1}}}{U(W)}.
(3)
$$

Assume otherwise that

$$
p_k > \frac{B U_{k\mid S_{r-1}}}{U(W)}.
(4)
$$

Then

$$
\frac{U_{k\mid S_{r-1}}}{p_k}
<
\frac{U(W)}{B}.
(5)
$$

Define

$$
S_1=S_{r-1}\cup\{k\},
\qquad
S_2=S_{r-1}\cup W.
$$

Since $W\subseteq S_2$, by monotonicity,

$$
U(W)\le U(S_2).
(6)
$$

#### Case 1: $S_1=S_2$

In this case, $W\subseteq S_1$. Therefore,

$$
U(W)\le U(S_1)=U(S_{r-1}\cup\{k\}).
(7)
$$

Combining (2) and (7), we obtain

$$
\frac{U_{k\mid S_{r-1}}}{p_k} \ge \frac{2U(W)}{(1-\varepsilon)B}>\frac{U(W)}{B},
$$

which contradicts (5).

#### Case 2: $S_1\subset S_2$

By the lazy version of the average-density lemma,

$$
\frac{U(S_2)-U(S_1)}
     {\sum_{i\in S_2\setminus S_1} b_i}
\le
\frac{1}{1-\varepsilon}
\frac{U_{r\mid S_{r-1}}}
     {b_r}.
(8)
$$

Substituting (1) into (8) yields

$$
\frac{U(S_2)-U(S_1)}
     {\sum_{i\in S_2\setminus S_1} b_i}
\le
\frac{U_{k\mid S_{r-1}}}
     {p_k}.
(9)
$$

Since $S_2\setminus S_1\subseteq W$, and the total bid of all winners satisfies

$$
\sum_{i\in W} b_i
\le
\frac{B}{2},
(10)
$$

we have

$$
\sum_{i\in S_2\setminus S_1} b_i
\le
\frac{B}{2}.
(11)
$$

Combining (9) and (11),

$$
U(S_2)-U(S_1)
\le
\frac{B}{2}
\frac{U_{k\mid S_{r-1}}}
     {p_k}.
(12)
$$

Applying (5), we get

$$
U(S_2)-U(S_1)
<
\frac{U(W)}{2}.
(13)
$$

Since $U(W)\le U(S_2)$,

$$
U(W)-U(S_1)
\le
U(S_2)-U(S_1)
<
\frac{U(W)}{2}.
$$

Thus,

$$
U(S_1)>\frac{1}{2}U(W).
(14)
$$

Substituting (14) into (2),

$$
\frac{U_{k\mid S_{r-1}}}{p_k} \ge \frac{2U(S_1)}{(1-\varepsilon)B} > \frac{U(W)}{(1-\varepsilon)B} \ge \frac{U(W)}{B},
$$

which again contradicts (5).

Therefore, both cases lead to contradictions, and (3) must hold:

$$
p_k
\le
\frac{B U_{k\mid S_{r-1}}}{U(W)}.
(15)
$$

Summing over all winners, we obtain

$$
\sum_{k\in W} p_k
\le
\frac{B}{U(W)}
\sum_{k\in W}
U_{k\mid S_{r(k)-1}}.
$$

By monotonicity and submodularity, the marginal contributions can be charged to the winners in $W$, yielding

$$
\sum_{k\in W}
U_{k\mid S_{r(k)-1}}
\le
U(W).
$$

Therefore,

$$
\sum_{k\in W} p_k
\le
\frac{B}{U(W)}U(W)
=B.
$$

Hence, the mechanism is budget feasible:

$$
\sum_{k\in W}p_k\le B.
$$

$\square$

## Lemma 1.2: Approximation of Lazy Fractional Greedy without Inactive-Element Loss

Ignoring the inactive-element loss, the lazy fractional greedy solution $S_{l+1}'$ satisfies

$$
U(W^*) \leq \frac{e}{(e-1)(1-\varepsilon)} U(S_{l+1}').
$$

### Proof

Let $W^*$ denote the optimal budget-feasible solution, where

$$
\sum_{i\in W^*} b_i \leq B.
$$

Let $S_{k-1}'$ be the set constructed by the lazy greedy algorithm before iteration $k$, and let $i_k'$ be the seller selected in iteration $k$.

Under lazy evaluation, the selected seller has density at least a $(1-\varepsilon)$ fraction of the maximum available density. Hence,

$$
\frac{U_{i_k'\mid S_{k-1}'}}{b_{i_k'}}
\geq
(1-\varepsilon)
\max_{j\in A\setminus S_{k-1}'}
\frac{U_{j\mid S_{k-1}'}}{b_j}.
$$


Therefore, there exists at least one seller in $W^*\setminus S_{k-1}'$ whose density is at least the average density of the remaining optimal marginal utility, namely,

$$
\max_{j\in A\setminus S_{k-1}'}
\frac{U_{j\mid S_{k-1}'}}{b_j}
\geq
\frac{U(W^*) - U(S_{k-1}')}{B}.
$$

Combining the above inequalities gives

$$
\frac{U_{i_k'\mid S_{k-1}'}}{b_{i_k'}}
\geq
(1-\varepsilon)
\frac{U(W^*) - U(S_{k-1}')}{B}.
$$

Equivalently,

$$
U_{i_k'\mid S_{k-1}'}
\geq
(1-\varepsilon)
\frac{b_{i_k'}}{B}
\left(
U(W^*) - U(S_{k-1}')
\right).
$$

Let

$$
\Delta_k = U(W^*) - U(S_k').
$$

Then

$$
\Delta_k
\leq
\left(
1 - (1-\varepsilon)\frac{b_{i_k'}}{B}
\right)\Delta_{k-1}.
$$

For the fractional greedy solution, the budget is exactly filled after adding the fractional seller $i^+$. That is,

$$
\sum_{k=1}^{l+1} b_{i_k'} = B.
$$

Applying the above recurrence over all fractional greedy steps yields

$$
\Delta_{l+1}
\leq
\prod_{k=1}^{l+1}
\left(
1 - (1-\varepsilon)\frac{b_{i_k'}}{B}
\right)
U(W^*).
$$

Using $1-x\leq e^{-x}$, we obtain
$e^{-(1-\varepsilon)}U(W^*)$.

Thus,

$$
U(S_{l+1}')
\geq
\left(1-e^{-(1-\varepsilon)}\right)U(W^*).
$$

Since for $0<\varepsilon<1$,

$$
1-e^{-(1-\varepsilon)}
\geq
(1-\varepsilon)\left(1-\frac{1}{e}\right),
$$

we further have

$$
U(S_{l+1}')
\geq
(1-\varepsilon)\left(1-\frac{1}{e}\right)U(W^*).
$$

Rearranging gives

$$
U(W^*)
\leq
\frac{1}{(1-\varepsilon)(1-1/e)}U(S_{l+1}').
$$

Therefore,

$$
U(W^*)
\leq
\frac{e}{(e-1)(1-\varepsilon)}U(S_{l+1}').
$$

This completes the proof.

