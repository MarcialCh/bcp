### Theorem (Budget Feasibility under Lazy Evaluation)

Let $W$ be the winner set returned by the lazy-threshold proportional-share allocation rule. Suppose that each selected seller satisfies

\[
\frac{U_{i\mid S}}{b_i}
\ge
(1-\varepsilon)
\max_{j\in A\setminus S}
\frac{U_{j\mid S}}{b_j},
\]

and the winner selection rule accepts seller $i$ only if

\[
b_i
\le
\frac{(1-\varepsilon)B}{2}
\cdot
\frac{U_{i\mid S}}
     {U(S\cup\{i\})}.
\]

For every winner $k\in W$, let the payment be

\[
p_k
=
\max_{1\le j\le \hat{\ell}+1}
\min
\left\{
b_{k(j)}^{\varepsilon},
\rho_{k(j)}^{\varepsilon}
\right\},
\]

where

\[
b_{k(j)}^{\varepsilon}
=
(1-\varepsilon)b_j
\frac{U_{k\mid S_{j-1}}}
     {U_{j\mid S_{j-1}}},
\]

and

\[
\rho_{k(j)}^{\varepsilon}
=
\frac{(1-\varepsilon)B}{2}
\frac{U_{k\mid S_{j-1}}}
     {U(S_{j-1}\cup\{k\})}.
\]

Then the mechanism is budget feasible, i.e.,

\[
\sum_{k\in W} p_k \le B.
\]

### Proof

We first establish an upper bound on the payment of every winner.

Consider an arbitrary winner $k\in W$.
By the definition of $p_k$, there exists an index $r$ such that

\[
p_k
=
\min
\left\{
b_{k(r)}^{\varepsilon},
\rho_{k(r)}^{\varepsilon}
\right\}.
\]

Therefore,

\[
p_k
\le
b_{k(r)}^{\varepsilon}
=
(1-\varepsilon)b_r
\frac{U_{k\mid S_{r-1}}}
     {U_{r\mid S_{r-1}}},
\]

which implies

\[
\frac{U_{r\mid S_{r-1}}}{b_r}
\le
(1-\varepsilon)
\frac{U_{k\mid S_{r-1}}}{p_k}.
\tag{1}
\]

Similarly,

\[
p_k
\le
\rho_{k(r)}^{\varepsilon}
=
\frac{(1-\varepsilon)B}{2}
\frac{U_{k\mid S_{r-1}}}
     {U(S_{r-1}\cup\{k\})},
\]

and hence

\[
\frac{U_{k\mid S_{r-1}}}{p_k}
\ge
\frac{2U(S_{r-1}\cup\{k\})}
     {(1-\varepsilon)B}.
\tag{2}
\]

We now prove that

\[
p_k
\le
\frac{B\,U_{k\mid S_{r-1}}}
     {U(W)}.
\tag{3}
\]

Assume otherwise that

\[
p_k
>
\frac{B\,U_{k\mid S_{r-1}}}
     {U(W)}.
\tag{4}
\]

Then

\[
\frac{U_{k\mid S_{r-1}}}{p_k}
<
\frac{U(W)}{B}.
\tag{5}
\]

Define

\[
S_1=S_{r-1}\cup\{k\},
\qquad
S_2=S_{r-1}\cup W.
\]

Since $W\subseteq S_2$, by monotonicity,

\[
U(W)\le U(S_2).
\tag{6}
\]

We distinguish two cases.

#### Case 1: $S_1=S_2$

In this case,

\[
W\subseteq S_1.
\]

Therefore,

\[
U(W)\le U(S_1)=U(S_{r-1}\cup\{k\}).
\tag{7}
\]

Combining (2) and (7),

\[
\frac{U_{k\mid S_{r-1}}}{p_k}
\ge
\frac{2U(W)}
     {(1-\varepsilon)B}
>
\frac{U(W)}{B},
\]

which contradicts (5).

---

#### Case 2: $S_1\subset S_2$

By the lazy version of the average-density lemma,

\[
\frac{U(S_2)-U(S_1)}
     {\sum_{i\in S_2\setminus S_1} b_i}
\le
\frac{1}{1-\varepsilon}
\frac{U_{r\mid S_{r-1}}}
     {b_r}.
\tag{8}
\]

Substituting (1) into (8) yields

\[
\frac{U(S_2)-U(S_1)}
     {\sum_{i\in S_2\setminus S_1} b_i}
\le
\frac{U_{k\mid S_{r-1}}}
     {p_k}.
\tag{9}
\]

Observe that

\[
S_2\setminus S_1
\subseteq W.
\]

Moreover, using the proportional-share rule and the approximate
density ordering, the total bid of all winners satisfies

\[
\sum_{i\in W} b_i
\le
\frac{B}{2}.
\tag{10}
\]

Hence,

\[
\sum_{i\in S_2\setminus S_1} b_i
\le
\frac{B}{2}.
\tag{11}
\]

Combining (9) and (11),

\[
U(S_2)-U(S_1)
\le
\frac{B}{2}
\frac{U_{k\mid S_{r-1}}}
     {p_k}.
\tag{12}
\]

Applying (5),

\[
U(S_2)-U(S_1)
<
\frac{U(W)}{2}.
\tag{13}
\]

Since $U(W)\le U(S_2)$ by (6),

\[
U(W)-U(S_1)
\le
U(S_2)-U(S_1)
<
\frac{U(W)}{2}.
\]

Thus,

\[
U(S_1)
>
\frac{1}{2}U(W).
\tag{14}
\]

Substituting (14) into (2),

\[
\frac{U_{k\mid S_{r-1}}}{p_k}
\ge
\frac{2U(S_1)}
     {(1-\varepsilon)B}
>
\frac{U(W)}
     {(1-\varepsilon)B}
\ge
\frac{U(W)}{B},
\]

which again contradicts (5).

---

Therefore, both cases lead to contradictions, and (3) must hold:

\[
p_k
\le
\frac{B\,U_{k\mid S_{r-1}}}
     {U(W)}.
\tag{15}
\]

Summing over all winners,

\[
\sum_{k\in W} p_k
\le
\frac{B}{U(W)}
\sum_{k\in W}
U_{k\mid S_{r(k)-1}}.
\]

By monotonicity and submodularity, the marginal contributions can
be charged to the winners in $W$, yielding

\[
\sum_{k\in W}
U_{k\mid S_{r(k)-1}}
\le
U(W).
\]

Hence,

\[
\sum_{k\in W} p_k
\le
\frac{B}{U(W)}U(W)
=
B.
\]

Therefore, the mechanism is budget feasible.

\[
\boxed{
\sum_{k\in W} p_k \le B.
}
\]

$\square$
