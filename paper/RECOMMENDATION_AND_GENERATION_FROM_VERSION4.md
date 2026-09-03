# Recommendation and basket generation from the Version-4 law

## Purpose

This note explains, in one continuous argument, how the fitted Version-4 basket law
supports:

1. basket-completion recommendation;
2. non-empty basket generation; and
3. price and promotion counterfactuals.

The theory is not changed for any of these tasks. They are different questions asked of
the same fitted probability distribution.

The central distinction is stated at the beginning:

- recommendation conditions on a revealed basket and exactly one missing product, so the
  global normalizer cancels;
- generation must sample an entire basket from the joint probability, so its size,
  categories, products, and interaction state must all be sampled correctly; and
- the interaction bridge is not a fifth generation stage or a correction applied after a
  basket is produced. It is the selected positive numerical method for obtaining the
  correct interaction-state distribution required by the first generation stage.

The model generates sets of distinct products conditional on a non-empty purchase
occasion. It does not, by itself, generate store visits, no-purchase occasions, or unit
quantities.

---

## 1. The modeling problem

For a purchase context $x$, the retailer observes a catalogue of offered products and
wants a probability for every feasible non-empty basket.

Let:

- $x=(h,t,s,\ldots)$ denote household, time, store, prices, promotions, and other declared
  contextual variables;
- $\mathcal A_x$ denote the products offered in context $x$;
- $J_x=|\mathcal A_x|$, with up to 5,455 products in the current catalogue;
- $S\subseteq\mathcal A_x$ denote an unordered set of distinct purchased products;
- $N=|S|$ denote total basket size;
- $c(j)$ denote the category of product $j$;
- $N_c(S)=|\{j\in S:c(j)=c\}|$ denote the number of selected products from category $c$;
- $\phi_j\in\mathbb R^r$ denote the interaction embedding of product $j$; and
- $n_{\max}=120$ denote the maximum supported basket size.

The support is

\[
\Omega_x^+
=
\left\{
S\subseteq\mathcal A_x:
1\le |S|\le n_{\max}
\right\}.
\tag{1}
\]

The superscript $+$ indicates non-empty support. More explicitly, the fitted distribution
is

\[
p_\Theta(S\mid x,S\neq\varnothing),
\tag{2}
\]

although the non-empty conditioning will usually be suppressed.

The model therefore answers:

> Given that this household has a purchase occasion in context $x$, what basket is
> purchased?

It does not answer whether a purchase occasion occurs. Section 10 explains how a separate
purchase-occurrence model can be composed with this law when no-purchase simulation is
required.

---

## 2. The Version-4 energy and probability law

For every $S\in\Omega_x^+$, Version-4 defines

\[
\boxed{
\begin{aligned}
E_\Theta(S;x)
={}&
\sum_{j\in S}b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k\\
&-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|).
\end{aligned}
}
\tag{3}
\]

The terms have separate interpretations:

- $b_j(x)$ is the contextual utility of product $j$;
- $\phi_j^\top\phi_k$ is the low-rank interaction between products $j$ and $k$;
- $\rho_c{N_c\choose2}$ governs broad within-category crowding or attraction; and
- $\rho_0(N)$ governs the distribution of total basket size.

The joint basket probability is

\[
\boxed{
p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S;x)\}}{Z_+(x)}
},
\qquad
Z_+(x)
=
\sum_{T\in\Omega_x^+}\exp\{E_\Theta(T;x)\}.
\tag{4}
\]

Every result below begins from Equation (4). There is no separate recommendation model
and no separate basket-generation model.

---

## 3. Contextual utility and the log-price variable

The Version-4 contextual utility can be written

\[
\begin{aligned}
b_{jht}
={}&
\lambda_j
+\theta_h^\top\alpha_j
-g_{hj}\,\Delta\log\operatorname{Price}_{jt}\\
&+w_j^{\mathrm{dsp}}D_{jt}
+w_j^{\mathrm{mlr}}M_{jt}
+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}
+\psi_j^\top r_{jht},
\end{aligned}
\tag{5}
\]

with

\[
g_{hj}
=
\operatorname{softplus}(\gamma_h)^\top
\operatorname{softplus}(\beta_j)
\ge0.
\tag{6}
\]

The notation $\operatorname{Price}_{jt}$ is used deliberately: it is a monetary price,
not a probability. The price feature is

\[
\Delta\log\operatorname{Price}_{jt}
=
\log\left(
\frac{\operatorname{Price}_{jt}}
{\operatorname{Price}^{\mathrm{ref}}_j}
\right).
\tag{7}
\]

### 3.1 Why use log price?

Log price measures proportional change. A move from 2 to 2.20 and a move from 20 to 22
are both 10% increases:

\[
\log(2.20/2)=\log(22/20)=\log(1.1).
\tag{8}
\]

It is also invariant to currency units. For any $a>0$,

\[
\log\left(
\frac{a\operatorname{Price}_{jt}}
{a\operatorname{Price}^{\mathrm{ref}}_j}
\right)
=
\log\left(
\frac{\operatorname{Price}_{jt}}
{\operatorname{Price}^{\mathrm{ref}}_j}
\right).
\tag{9}
\]

For a small relative change,

\[
\Delta\log\operatorname{Price}
\approx
\frac{\Delta\operatorname{Price}}{\operatorname{Price}},
\tag{10}
\]

so $g_{hj}$ has an elasticity-like interpretation on the energy scale.

### Proposition 1 — the direct own-price effect has the correct sign

Holding other inputs fixed,

\[
\frac{\partial b_{jht}}
{\partial\,\Delta\log\operatorname{Price}_{jt}}
=-g_{hj}\le0.
\tag{11}
\]

#### Proof

The price term in Equation (5) is
$-g_{hj}\Delta\log\operatorname{Price}_{jt}$, and Equation (6) guarantees
$g_{hj}\ge0$. Differentiation proves Equation (11). $\square$

For a fractional discount $d\in[0,1)$,

\[
\operatorname{Price}'=(1-d)\operatorname{Price}
\tag{12}
\]

and therefore

\[
\Delta b_j
=
-g_{hj}\log(1-d)
\ge0.
\tag{13}
\]

This is the direct effect on product utility. The final change in marginal incidence also
passes through basket size, categories, interactions, and competition from other
products.

---

## 4. Recommendation follows by conditioning the joint law

Suppose one product is hidden from an observed basket and $R$ is the revealed remainder.
The evaluation question is:

> Conditional on the basket containing all products in $R$ and exactly one additional
> product, which product is that addition?

Define

\[
\mathcal C_R
=
\left\{
R\cup\{k\}:
k\in\mathcal A_x\setminus R
\right\}.
\tag{14}
\]

For candidate $j\notin R$, the desired probability is

\[
P_\Theta\left(
S=R\cup\{j\}
\mid
S\in\mathcal C_R,x
\right).
\tag{15}
\]

Define the add-one energy increment

\[
s_j(R,x)
=
E_\Theta(R\cup\{j\};x)-E_\Theta(R;x).
\tag{16}
\]

### Proposition 2 — exact basket-completion softmax

\[
\boxed{
P_\Theta\left(
S=R\cup\{j\}
\mid S\in\mathcal C_R,x
\right)
=
\frac{e^{s_j(R,x)}}
{\displaystyle\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}}
}.
\tag{17}
\]

#### Proof

By conditional probability,

\[
P_\Theta(S=R\cup\{j\}\mid S\in\mathcal C_R,x)
=
\frac{
P_\Theta(S=R\cup\{j\}\mid x)
}{
\displaystyle\sum_{k\notin R}
P_\Theta(S=R\cup\{k\}\mid x)
}.
\tag{18}
\]

Substituting Equation (4) gives

\[
\frac{
\exp\{E_\Theta(R\cup\{j\};x)\}/Z_+(x)
}{
\displaystyle\sum_{k\notin R}
\exp\{E_\Theta(R\cup\{k\};x)\}/Z_+(x)
}.
\tag{19}
\]

The common $Z_+(x)$ cancels. Using

\[
E_\Theta(R\cup\{j\};x)
=E_\Theta(R;x)+s_j(R,x),
\tag{20}
\]

the common factor $e^{E_\Theta(R;x)}$ also cancels, leaving Equation (17).
$\square$

This cancellation is exact. Recommendation under this one-hidden-item protocol needs no
quadrature, no Monte Carlo sample, and no recommendation-specific training.

### 4.1 Expanding the score

Adding candidate $j$ changes the interaction energy by

\[
\sum_{k\in R}\phi_j^\top\phi_k.
\tag{21}
\]

Its category penalty changes by

\[
{N_{c(j)}(R)+1\choose2}
-{N_{c(j)}(R)\choose2}
=N_{c(j)}(R).
\tag{22}
\]

Its size penalty changes by

\[
\rho_0(|R|+1)-\rho_0(|R|).
\tag{23}
\]

Hence

\[
\boxed{
\begin{aligned}
s_j(R,x)
={}&
b_j(x)
+\sum_{k\in R}\phi_j^\top\phi_k
-\rho_{c(j)}N_{c(j)}(R)\\
&-\left[
\rho_0(|R|+1)-\rho_0(|R|)
\right].
\end{aligned}
}
\tag{24}
\]

All candidates create the same final size, so the last line is common to all candidates
and cancels from their relative ranking. Let

\[
m_R=\sum_{k\in R}\phi_k.
\tag{25}
\]

The ranking score is therefore

\[
\boxed{
s_j^{\mathrm{rank}}(R,x)
=
b_j(x)
+\phi_j^\top m_R
-\rho_{c(j)}N_{c(j)}(R).
}
\tag{26}
\]

Contextual relevance, learned product interaction, and category competition all remain in
the recommendation score.

### 4.2 Recommendation is not marginal incidence

The unconditional incidence probability is

\[
\pi_j(x)
=
P_\Theta(j\in S\mid x)
=
\frac{\partial\log Z_+(x)}{\partial b_j(x)}.
\tag{27}
\]

Equation (27) averages over all supported baskets. Equation (17) conditions on one
specific revealed basket and exactly one missing product. Thus:

- use $\pi_j(x)$ for unconditional trip-level incidence ranking; and
- use $s_j(R,x)$ for one-item basket completion.

### 4.3 MRR

If the hidden true product has rank $r_i$ in test case $i$, then

\[
\operatorname{MRR}
=
\frac1M\sum_{i=1}^M\frac1{r_i},
\tag{28}
\]

and

\[
\operatorname{MRR@K}
=
\frac1M\sum_{i=1}^M
\frac{\mathbf 1\{r_i\le K\}}{r_i}.
\tag{29}
\]

MRR measures rank, not full probability calibration. A likelihood gain can arise from
better size, category, or tail calibration without changing the leading candidate ranks.

Recommendation is now complete. Basket generation begins again from Equation (4), not
from the recommendation softmax.

---

## 5. What basket generation must accomplish

Generation asks for

\[
S\sim p_\Theta(S\mid x)
=
\frac{e^{E_\Theta(S;x)}}{Z_+(x)},
\qquad S\in\Omega_x^+.
\tag{30}
\]

The final operational sequence will be

\[
\boxed{
z
\longrightarrow
N
\longrightarrow
(N_c)_c
\longrightarrow
\text{products within categories}
\longrightarrow
S.
}
\tag{31}
\]

After the products are selected, $S$ is complete. There is no subsequent interaction
correction.

The learned interaction embeddings already affect the process through the distribution
of $z$ and the conditional product weights. The only reason additional numerical
machinery may be needed is that the first-stage distribution of $z$ is not generally a
standard Gaussian or another closed-form distribution.

A repeated add-one recommendation softmax is not an exact generator. Version-4 assigns
probability to an unordered final set, not an ordering of purchases. Such a procedure
would also omit the stopping probability and the mass of all possible future
completions.

To derive Equation (31), the quadratic interaction must first be represented in a form
that is additive conditional on a low-dimensional auxiliary variable.

---

## 6. The H--S representation of the interaction

The pair interaction satisfies

\[
\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
=
\frac12
\left\|
\sum_{j\in S}\phi_j
\right\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2.
\tag{32}
\]

Define

\[
\widetilde b_j(x)
=
b_j(x)-\frac12\|\phi_j\|^2
\tag{33}
\]

and

\[
w_j(z;x)
=
\exp\left\{
\widetilde b_j(x)+z^\top\phi_j
\right\},
\qquad
z\in\mathbb R^r.
\tag{34}
\]

For $z\sim\mathcal N(0,I_r)$, the Gaussian moment-generating identity is

\[
\mathbb E\left[e^{z^\top v}\right]
=e^{\|v\|^2/2}.
\tag{35}
\]

### Proposition 3 — positive augmented law with the Version-4 basket marginal

Let $\varphi_r(z)$ be the standard $r$-dimensional Gaussian density. Define

\[
\boxed{
\begin{aligned}
p_\Theta(S,z\mid x)
=
\frac{\varphi_r(z)}{Z_+(x)}
\exp\Bigg\{
&\sum_{j\in S}
\left[
b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j
\right]\\
&-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|)
\Bigg\}.
\end{aligned}
}
\tag{36}
\]

Then

\[
\int_{\mathbb R^r}p_\Theta(S,z\mid x)\,dz
=p_\Theta(S\mid x).
\tag{37}
\]

#### Proof

Let

\[
v_S=\sum_{j\in S}\phi_j.
\tag{38}
\]

Integrating the $z$-dependent part of Equation (36) gives

\[
\begin{aligned}
&\int\varphi_r(z)
\exp\left\{
z^\top v_S
-\frac12\sum_{j\in S}\|\phi_j\|^2
\right\}dz\\
&\quad=
\exp\left\{
\frac12\|v_S\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2
\right\}\\
&\quad=
\exp\left\{
\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
\right\},
\end{aligned}
\tag{39}
\]

using Equations (32) and (35). Every remaining factor in Equation (36) is the
corresponding term in the original Version-4 energy. The marginal is therefore Equation
(4). $\square$

The augmentation has not modified the model. It has only introduced an auxiliary
variable that makes product contributions additive conditional on $z$.

---

## 7. The exact four-stage generation factorization

The next task is to normalize the conditional basket weights at fixed $z$.

For category $c$, let $\mathcal A_{xc}$ be its offered products. Define the elementary
symmetric polynomial

\[
e_{c,k}(z,x)
=
\sum_{\substack{A\subseteq\mathcal A_{xc}\\|A|=k}}
\prod_{j\in A}w_j(z;x),
\qquad e_{c,0}=1.
\tag{40}
\]

Attach the original category term:

\[
q_c(k;z,x)
=
e^{-\rho_c{k\choose2}}e_{c,k}(z,x).
\tag{41}
\]

Define the category polynomial

\[
G_c(u;z,x)
=
\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
q_c(k;z,x)u^k.
\tag{42}
\]

Multiplying over categories gives

\[
\prod_cG_c(u;z,x)
=
\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{43}
\]

### Proposition 4 — $A_n$ is the complete size-$n$ mass at fixed $z$

\[
A_n(z,x)
=
\sum_{\substack{S\subseteq\mathcal A_x\\|S|=n}}
\left[
\prod_c e^{-\rho_c{N_c(S)\choose2}}
\right]
\left[
\prod_{j\in S}w_j(z;x)
\right].
\tag{44}
\]

#### Proof

Choosing degree $k$ from $G_c$ chooses $k$ distinct products from category $c$. The ESP
sums their product weights, while $e^{-\rho_c{k\choose2}}$ supplies the Version-4 category
term. Multiplication over categories combines disjoint category subsets. The power of
$u$ records total basket size, so the coefficient of $u^n$ contains every supported
size-$n$ basket exactly once. $\square$

Define the positive non-empty mass

\[
F_+(z;x)
=
\sum_{n=1}^{n_{\max}}
e^{-\rho_0(n)}A_n(z,x).
\tag{45}
\]

Summing the augmented law over baskets gives the first-stage distribution

\[
\boxed{
p_\Theta(z\mid x)
=
\frac{\varphi_r(z)F_+(z;x)}{Z_+(x)}.
}
\tag{46}
\]

Dividing the augmented joint law by Equation (46) gives

\[
\boxed{
p_\Theta(S\mid z,x)
=
\frac{
e^{-\rho_0(|S|)}
\prod_c e^{-\rho_c{N_c(S)\choose2}}
\prod_{j\in S}w_j(z;x)
}{
F_+(z;x)
}.
}
\tag{47}
\]

Therefore

\[
p_\Theta(S\mid x)
=
\int
p_\Theta(z\mid x)\,
p_\Theta(S\mid z,x)\,dz.
\tag{48}
\]

Equations (46)--(48) lead directly to the four generation stages.

### Stage 1 — draw the interaction state

\[
\boxed{
z\sim p_\Theta(z\mid x)
\propto
\varphi_r(z)F_+(z;x).
}
\tag{49}
\]

This is the correct first-stage law. It depends on every fitted interaction embedding
through $w_j(z;x)$ and $F_+(z;x)$.

It is important not to replace Equation (49) with
$z\sim\mathcal N(0,I_r)$. Let $\widetilde q(S,z;x)$ denote the positive unnormalized
basket factor in Equation (36), excluding $\varphi_r(z)$. Then

\[
F_+(z;x)=\sum_S\widetilde q(S,z;x).
\tag{50}
\]

The correct joint factorization is

\[
\underbrace{
\frac{\varphi_r(z)F_+(z;x)}{Z_+(x)}
}_{p_\Theta(z\mid x)}
\times
\underbrace{
\frac{\widetilde q(S,z;x)}{F_+(z;x)}
}_{p_\Theta(S\mid z,x)}
=
\frac{\varphi_r(z)\widetilde q(S,z;x)}{Z_+(x)}.
\tag{51}
\]

If $z$ were instead drawn directly from $\varphi_r(z)$, the resulting factor would be

\[
\varphi_r(z)
\frac{\widetilde q(S,z;x)}{F_+(z;x)},
\tag{52}
\]

whose basket marginal is generally not Equation (4). The standard Gaussian is the base
measure in the H--S integral; after conditioning on a non-empty basket, the marginal of
$z$ is tilted by $F_+(z;x)$.

### Stage 2 — draw total basket size

Conditional on the Stage-1 draw,

\[
\boxed{
P_\Theta(N=n\mid z,x)
=
\frac{
e^{-\rho_0(n)}A_n(z,x)
}{
F_+(z;x)
},
\qquad 1\le n\le n_{\max}.
}
\tag{53}
\]

This is where $\rho_0$ controls generated basket size. It cancelled from recommendation
because all recommendation candidates had the same final size; it cannot cancel when size
itself is being sampled.

### Stage 3 — draw the count from every category

Conditional on $N=n$, the category-count vector $r=(r_1,\ldots,r_C)$ has probability

\[
\boxed{
P_\Theta((N_c)_c=r\mid N=n,z,x)
=
\frac{
\mathbf 1\{\sum_cr_c=n\}
\prod_cq_c(r_c;z,x)
}{
A_n(z,x)
}.
}
\tag{54}
\]

The count vector can be drawn without enumerating all possibilities. Define suffix
continuation masses

\[
H_c(m)
=
[u^m]\prod_{d=c}^{C}G_d(u;z,x),
\tag{55}
\]

with $H_{C+1}(0)=1$ and $H_{C+1}(m)=0$ for $m\neq0$. If $m$ products remain to be
allocated at category $c$, draw $N_c=k$ with

\[
\boxed{
P(N_c=k\mid m,z,x)
=
\frac{
q_c(k;z,x)H_{c+1}(m-k)
}{
H_c(m)
}.
}
\tag{56}
\]

The numerator is the mass of every valid continuation taking $k$ products from category
$c$ and $m-k$ from later categories. Summing over feasible $k$ recovers $H_c(m)$.
Multiplying the successive backtracking probabilities yields Equation (54).

### Stage 4 — draw the actual products within each category

Given $N_c=k$, a category subset $B_c\subseteq\mathcal A_{xc}$ with $|B_c|=k$ has
probability

\[
\boxed{
P_\Theta(B_c\mid |B_c|=k,z,x)
=
\frac{
\prod_{j\in B_c}w_j(z;x)
}{
e_{c,k}(z,x)
}.
}
\tag{57}
\]

One exact procedure uses suffix ESP values. Index products in category $c$ by
$1,\ldots,M_c$. If $k$ selections remain before product $i$, then

\[
P(i\text{ included}\mid k\text{ remain},z,x)
=
\frac{
w_i(z;x)e_{k-1}(w_{i+1},\ldots,w_{M_c})
}{
e_k(w_i,\ldots,w_{M_c})
}.
\tag{58}
\]

The identity

\[
e_k(w_i,\ldots,w_M)
=
e_k(w_{i+1},\ldots,w_M)
+w_i e_{k-1}(w_{i+1},\ldots,w_M)
\tag{59}
\]

shows that the inclusion and exclusion probabilities sum to one. Following these
transitions produces Equation (57).

An equivalent exact conditional construction uses independent Bernoulli variables

\[
Q_j
=
\frac{e^{\eta_j+a}}{1+e^{\eta_j+a}},
\qquad
\eta_j=\log w_j,
\tag{60}
\]

and conditions on exactly $k$ successes. For any size-$k$ set $B_c$,

\[
\prod_{j\in B_c}Q_j
\prod_{j\notin B_c}(1-Q_j)
\propto
\prod_{j\in B_c}e^{\eta_j}.
\tag{61}
\]

The scalar shift $a$ cancels under fixed-count conditioning and can be chosen to improve
acceptance. If a rejection limit is exhausted, the algorithm must report failure or use
exact suffix-ESP sampling; silently dropping the context would alter the target
distribution.

After drawing every $B_c$, return

\[
\boxed{
S=\bigcup_cB_c.
}
\tag{62}
\]

The basket is complete at Equation (62).

### Proposition 5 — the four stages reproduce the Version-4 basket law

For a particular basket $S$, let $n=|S|$, $r_c=N_c(S)$, and
$B_c=S\cap\mathcal A_{xc}$. Conditional on $z$, multiplying Stages 2--4 gives

\[
\begin{aligned}
&\frac{e^{-\rho_0(n)}A_n}{F_+}
\times
\frac{\prod_c[
e^{-\rho_c{r_c\choose2}}e_{c,r_c}
]}{A_n}
\times
\prod_c
\frac{\prod_{j\in B_c}w_j}{e_{c,r_c}}\\
&\quad=
\frac{
e^{-\rho_0(n)}
\prod_c e^{-\rho_c{r_c\choose2}}
\prod_{j\in S}w_j
}{
F_+
}
=p_\Theta(S\mid z,x).
\end{aligned}
\tag{63}
\]

Integrating this result with respect to the Stage-1 distribution in Equation (46) gives

\[
\int p_\Theta(z\mid x)p_\Theta(S\mid z,x)\,dz
=p_\Theta(S\mid x),
\tag{64}
\]

by Equation (48). Thus the four stages generate the original Version-4 law. $\square$

All discrete stages retain the full offered catalogue and every declared size
$1,\ldots,n_{\max}$. There is no 20-product support restriction.

---

## 8. Numerically implementing Stage 1

The mathematical four-stage sampler is complete, but Stage 1 requires sampling

\[
p_\Theta(z\mid x)
\propto
\varphi_r(z)F_+(z;x).
\tag{65}
\]

The learned parameters determine this density completely. Its normalizing constant is
$Z_+(x)$, but the selected unnormalized-target sampler does not need to know that
constant.

The selected implementation uses a positive sequential Monte Carlo interaction bridge.
This choice has a precise theoretical basis: it starts from an exactly sampleable law,
uses positive incremental weights, applies mutations that preserve every intermediate
target, and terminates at the full Version-4 interaction law. Other samplers could target
Equation (65), but each would require its own mixing and convergence certification; they
are not the method specified by this generation pipeline.

Smolyak quadrature is appropriate for deterministic integration, but its combination
weights may be negative. Signed quadrature nodes cannot be treated as samples from
Equation (65).

### 8.1 Why the interaction bridge is useful

The bridge converts a difficult direct interaction-state draw into a sequence of positive,
exactly defined intermediate distributions. Its starting distribution can be sampled
exactly, its endpoint is the fitted Version-4 law, and Proposition 6 proves that the
blocked mutation preserves every intermediate target. Define

\[
p_\beta(S\mid x)
\propto
\exp\{
E_0(S;x)+\beta V_\Phi(S)
\},
\qquad 0\le\beta\le1,
\tag{66}
\]

where

\[
E_0(S;x)
=
\sum_{j\in S}b_j(x)
-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|)
\tag{67}
\]

and

\[
V_\Phi(S)
=
\sum_{\substack{j<k\\j,k\in S}}
\phi_j^\top\phi_k.
\tag{68}
\]

At $\beta=0$, the model retains contextual utility, category structure, basket size, and
the full catalogue, but removes the Gram interaction. This base law is sampled exactly
using the size, category-count, and product dynamic program. At $\beta=1$, Equation (66)
is exactly the fitted Version-4 law.

Choose

\[
0=\beta_0<\beta_1<\cdots<\beta_L=1.
\tag{69}
\]

The positive incremental weight is

\[
W_\ell(S)
=
\exp\{
(\beta_\ell-\beta_{\ell-1})V_\Phi(S)
\}.
\tag{70}
\]

After weighting and resampling, use a blocked Gibbs mutation. Let

\[
m(S)=\sum_{j\in S}\phi_j.
\tag{71}
\]

At bridge level $\beta_\ell$, draw

\[
z\mid S,x,\beta_\ell
\sim
\mathcal N\left(
\sqrt{\beta_\ell}\,m(S),I_r
\right),
\tag{72}
\]

and then draw a complete basket conditional on $z$ using Stages 2--4 with weights

\[
w_j(z,\beta_\ell,x)
=
\exp\left\{
b_j(x)
-\frac{\beta_\ell}{2}\|\phi_j\|^2
+\sqrt{\beta_\ell}\,z^\top\phi_j
\right\}.
\tag{73}
\]

### Proposition 6 — the blocked mutation respects the bridge law

The transition $S\rightarrow z\rightarrow S'$ defined by Equations (72)--(73) leaves
$p_{\beta_\ell}(S\mid x)$ invariant.

#### Proof

At fixed $\beta$, define the positive augmented density proportional to

\[
\exp\left\{
E_0(S;x)
-\frac12\|z\|^2
+\sqrt\beta\,z^\top m(S)
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2
\right\}.
\tag{74}
\]

Completing the square gives Equation (72) as the full conditional of $z$. Holding $z$
fixed gives Equation (73) and the exact conditional basket law from Stages 2--4.
Alternating exact full conditionals is a Gibbs kernel and therefore preserves the
augmented law. Integrating out $z$ gives

\[
\exp\{E_0(S;x)+\beta V_\Phi(S)\},
\tag{75}
\]

which is the bridge target. $\square$

### 8.2 What the bridge is, and what it is not

The bridge is an algorithm for reaching the first-stage interaction distribution. It is
not:

- an extra fitted model;
- a modification of the Version-4 energy;
- an additional product-selection stage;
- a correction applied after Equation (62); or
- a way of adding interactions that were absent from training.

Its endpoint $\beta=1$ is the original Version-4 law. During SMC, basket and $z$ particles
are evolved jointly. The final $\beta=1$ basket is already the output basket. An optional
final Gibbs rejuvenation changes the particle realization but preserves the same target
law.

At finite particle count, SMC particles are not independent exact draws because
resampling creates shared ancestry. Under standard positivity, finite-moment, and
ergodicity conditions, their empirical distribution is consistent as particle count
increases.

The normalized bridge ESS is

\[
\frac{\operatorname{ESS}_\ell}{P}
=
\frac{1}{
P\sum_{p=1}^P(\overline W_\ell^{(p)})^2
}.
\tag{76}
\]

A small value proves concentration among represented particles. A large value does not
prove that an unvisited mode is absent. Independent replicates and held-out generation
checks remain necessary.

### 8.3 Direct factorization and the selected bridge implementation

If a reliable direct sampler for Equation (65) is available, use:

1. sample $z$ from Equation (65);
2. sample $N$ from Equation (53);
3. sample category counts from Equations (54)--(56);
4. sample category constituents from Equations (57)--(61);
5. return their union.

If positive SMC is used instead, use:

1. sample exact $\beta=0$ base baskets;
2. traverse the schedule in Equation (69);
3. at each level, weight by Equation (70), resample, and apply the invariant blocked
   mutation;
4. return the final $\beta=1$ baskets.

The second description is a numerical realization of the same target, not a sequence
performed after the first description.

---

## 9. Counterfactual generation

Let action $a$ change prices, promotions, or another declared contextual input. The
counterfactual context is $x_a$, and the counterfactual basket law remains

\[
p_\Theta(S\mid x_a)
=
\frac{\exp\{E_\Theta(S;x_a)\}}{Z_+(x_a)}.
\tag{77}
\]

The direct procedure is:

1. recompute every affected $b_j(x_a)$;
2. keep fitted structural parameters fixed unless the intervention explicitly changes
   their meaning;
3. run the same four-stage generator under $x_a$; and
4. compare factual and counterfactual summaries over the same context population.

For factual-particle reuse, define

\[
\Delta_a(S;x)
=
E_\Theta(S;x_a)-E_\Theta(S;x).
\tag{78}
\]

If only item utilities change,

\[
\Delta_a(S;x)
=
\sum_{j\in S}
\left[
b_j(x_a)-b_j(x)
\right].
\tag{79}
\]

For any basket statistic $f$,

\[
\mathbb E_{x_a}[f(S)]
=
\frac{
\mathbb E_x[
f(S)e^{\Delta_a(S;x)}
]
}{
\mathbb E_x[
e^{\Delta_a(S;x)}
]
}.
\tag{80}
\]

This density-ratio identity permits efficient reweighting when the factual and action
laws overlap. If action-weight ESS is poor, baskets should be generated again under
$x_a$.

Equation (80) is a probability identity, not causal identification. A causal price or
promotion claim additionally requires randomization, exogenous variation, or another
credible causal design.

---

## 10. Extending the simulator beyond non-empty incidence baskets

### 10.1 No-purchase occasions

Version-4 is fitted to observed non-empty trips. Suppose a separate opportunity model,
trained with valid exposure denominators, estimates

\[
q(x)=P(B=1\mid x),
\tag{81}
\]

where $B=1$ denotes that a purchase occurs. Then

\[
P(S=\varnothing\mid x)=1-q(x),
\tag{82}
\]

and, for $T\in\Omega_x^+$,

\[
P(S=T\mid x)
=
q(x)\,
p_\Theta(T\mid x,S\neq\varnothing).
\tag{83}
\]

Simulation first draws $B$. If $B=0$, it returns the empty basket. If $B=1$, it runs the
Version-4 generator.

The occurrence probability cannot be learned from purchase-only records because those
records do not contain the number of opportunities on which no purchase occurred.

### 10.2 Unit quantities

Equation (4) models product incidence. It does not determine units of each selected SKU.
To simulate quantities, fit a separate conditional law

\[
p(q_j\mid j\in S,x,S)
\tag{84}
\]

and draw quantities only after the incidence basket has been generated. Revenue and
profit simulation additionally require valid prices, margins, inventory rules, and
substitution behavior.

---

## 11. Computational cost

For recommendation, compute

\[
m_R=\sum_{k\in R}\phi_k
\tag{85}
\]

once and score every product using $\phi_j^\top m_R$. Interaction scoring costs
$O(J_xr)$ and needs no outer normalizer.

At fixed $z$, a conservative bound for one full conditional generation table is

\[
O(J_xn_{\max}+C_xn_{\max}^2),
\tag{86}
\]

where $C_x$ is the number of nonempty categories. Category-size caps, degree truncation,
and balanced polynomial multiplication reduce actual work.

With $P$ particles and $L$ bridge levels, interaction-statistic work costs

\[
O(PL\,\bar n r),
\tag{87}
\]

where $\bar n$ is average basket size. The larger expense is repeated conditional
dynamic-program evaluation. No step enumerates the $2^{J_x}$ possible baskets.

Vectorizing particles, reusing context-only terms, and caching unchanged polynomials may
reduce runtime without changing the target distribution.

---

## 12. What must be validated

### 12.1 Recommendation

- Use a locked hidden-item protocol.
- Keep the candidate assortment identical across models.
- Report MRR, MRR@$K$, Recall@$K$, and paired differences.
- Compare the full interaction score with its additive ablation.

### 12.2 Likelihood

- Score identical held-out baskets over identical full support.
- Audit the outer integral at increasing deterministic fidelity.
- Report paired uncertainty across baskets.
- Do not interpret signed quadrature nodes as samples.

### 12.3 Generation

Generated and observed baskets must use the same household, store, time, and assortment
mixture. Compare:

- mean, variance, quantiles, and tails of basket size;
- category-count distributions;
- product incidence probabilities;
- pair co-incidence and complement statistics;
- household- and segment-conditional moments;
- particle ESS and ancestry;
- independent replicate stability;
- explicit sampling failures and support violations; and
- counterfactual overlap for every candidate action.

A good likelihood does not automatically certify a finite-particle generator. A sampler
can also reproduce the fitted model accurately while the fitted model remains
miscalibrated against held-out data. Numerical sampling error and statistical model error
must therefore be reported separately.

---

## 13. Final interpretation

The complete argument is now:

1. Version-4 assigns probability to every supported non-empty basket through Equation
   (4).
2. Recommendation conditions that law on a revealed basket and exactly one missing
   product, causing the common normalizer and common basket energy to cancel.
3. Generation returns to the original joint law rather than using the recommendation
   formula.
4. The H--S identity introduces $z$ without changing the basket marginal.
5. Conditional on $z$, ESP/category dynamic programming gives exact probabilities for
   total size, category counts, and products.
6. A correct first-stage draw uses
   $p_\Theta(z\mid x)\propto\varphi_r(z)F_+(z;x)$, not an unweighted standard Gaussian.
7. Once $z$, size, category counts, and products have been drawn, the basket is complete.
8. The interaction bridge is the selected positive numerical method for realizing the
   difficult first-stage interaction distribution. Its endpoint is the original
   Version-4 law.
9. Counterfactual generation reruns the same law under changed contextual utilities.
10. No-purchase occasions and quantities require separately identified extensions.

The foundational energy, support, and probability theorem remain unchanged throughout.
