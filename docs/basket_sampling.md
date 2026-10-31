# How one basket is generated

This document goes through how the model generates one shopping basket for one trip, step by step.
Each step gives the exact probability distribution and the exact procedure that draws from it.
Every symbol is defined before it is used. The code is in three places:

- `v4/mixture.py`, function `generate`;
- `v4/sampling.py`, function `sample_baskets`;
- `v4/native/sampler.cpp`, function `sample_slots_cpp`.

Equations are written in TeX (`$...$` and `$$...$$`), which GitHub and most Markdown viewers render.

---

## 1. What is given, and what is produced

**Given (the context of the trip):**

- the household $h$ that makes the trip;
- the store $s$ and the week $w$ of the trip;
- the products the store offers on that trip, numbered $j = 1, \dots, J_t$;
- for each offered product:
  - its price and its regular price;
  - whether it is on display, and whether it is in the mailer;
  - the household's purchase history up to that trip;
- the fitted model parameters.

**Produced:** a basket $S$, a set of distinct offered products. It holds at least one product and at
most $N$ products: $N = 120$ on the Dunnhumby data and $N = 80$ in the synthetic world.

**Words used below:**

- *category*: the group a product belongs to. On Dunnhumby this is a commodity, such as "pasta sauce".
  Every product belongs to exactly one category, and $c(j)$ is the category of product $j$.
- $n_c$: the number of products of category $c$ in the basket.
- $n$: the number of products in the basket, $n = n_1 + n_2 + \dots + n_C$.
- $m$: the number of categories present in the basket, that is, the number of categories with $n_c \ge 1$.
- $R$: the largest number of products of one category that a basket may hold ($R = 40$).

---

## 2. The probability of a basket

The model gives every possible basket a probability, and generation draws one basket from exactly
that probability. This section builds the formula up in order. The later sections give the steps
that draw from it.

### 2.1 The taste of the trip

Each trip has a hidden *taste*, a list of $r = 16$ numbers:

$$
z = (z_1, z_2, \dots, z_{16}).
$$

The taste is never observed. Before anything is known about the basket, it follows the standard
normal distribution in 16 dimensions. That is, the 16 numbers are independent, and each has mean 0
and variance 1:

$$
\mathcal{N}(z) = (2\pi)^{-16/2} \exp\!\Big( -\tfrac{1}{2} \sum_{k=1}^{16} z_k^2 \Big).
$$

### 2.2 The base attractiveness of a product on this trip

For each offered product $j$ the model computes a number $b_j$. This is how attractive the product
is on this trip before the taste is taken into account. It is a sum of terms:

$$
\begin{aligned}
b_j \;=\;& \lambda_j && \text{popularity of the product} \\
&+ \theta_{h,\text{size}} + \textstyle\sum_k \theta_{h,k}\, a_{j,k} && \text{the household's lasting taste ($a_{j,k}$: product loadings)} \\
&- \text{price}_j && \text{see below} \\
&+ v_j\, \text{display}_j + q_j\, \text{mailer}_j && \text{promotions ($\text{display}_j, \text{mailer}_j \in \{0,1\}$)} \\
&+ \textstyle\sum_k \mu_{j,k}\, \delta_{w,k} && \text{season ($\delta_w$: numbers for week $w$)} \\
&+ \textstyle\sum_k \zeta_{j,k}\, \xi_{s,k} && \text{store ($\xi_s$: numbers for store $s$)} \\
&+ \text{history}_j - o_t && \text{purchase history, minus a trip-wide offset $o_t$} \\
&+ \textstyle\sum_k \theta_{h,k}\, e_{c(j),k} && \text{the household's taste for the product's category} \\
&+ \iota_{t,c(j)} && \text{competition inside the category (see below)}
\end{aligned}
$$

**Price term.** Two price numbers are used for each product:

- $d_j$ is the logarithm of product $j$'s price relative to its usual price, and $\bar d$ is the
  average of $d_j$ over all products offered on the trip;
- $x_j$ is the depth of the product's discount: the logarithm of price over regular price, centred.

Then

$$
\text{price}_j \;=\; g_{h,j}\big(\bar d + \kappa\,(d_j - \bar d)\big)
\;+\; c\,(d_j - \bar d)
\;+\; g_{h,j}\,(\kappa_{\text{disc}} - \kappa)\, x_j
\;+\; c_{\text{disc}}\, x_j .
$$

The quantities in this formula are:

- $g_{h,j} \ge 0$, the household's sensitivity to the price of product $j$. It is built from 8 numbers
  for the household and 8 for the product:

  $$
  g_{h,j} = \sum_{k=1}^{8} \operatorname{soft}(\gamma_{h,k})\, \operatorname{soft}(\beta_{j,k}),
  \qquad \operatorname{soft}(y) = \log(1 + e^{y}).
  $$

- $\kappa$, $\kappa_{\text{disc}}$, $c$ and $c_{\text{disc}}$, single numbers shared by everyone.
  The term $c$ exists only in models fitted with `--price-common`.

**Competition inside the category.** The term $\iota$ takes part of the effect of prices and
promotions away from the category as a whole, and puts it on the choice between the category's
products:

$$
\iota_{t,c} = (\lambda_c - 1)\Big[\log \sum_{i \in c} e^{b'_i} \;-\; \log \sum_{i \in c} e^{b''_i}\Big].
$$

Here:

- $b'_i$ is the attractiveness without $\iota$ and without the history offset;
- $b''_i$ is the same, but at regular prices and with no promotions;
- $0 < \lambda_c < 1$ is estimated for each category.

None of the terms in $b_j$ depends on the taste or on the rest of the basket.

### 2.3 The attractiveness of a product given the taste

Each product has a *taste profile* $\varphi_j = (\varphi_{j,1}, \dots, \varphi_{j,16})$. Given the taste $z$, the
attractiveness of product $j$ is

$$
u_j(z) = b_j - \tfrac{1}{2} \sum_{k=1}^{16} \varphi_{j,k}^2 + \sum_{k=1}^{16} z_k\, \varphi_{j,k},
$$

and its *weight* is

$$
w_j(z) = e^{u_j(z)}.
$$

A taste $z$ that points the same way as $\varphi_j$ raises the weight of product $j$. Products with similar
profiles are raised together, which is what makes some products tend to appear in the same basket.

### 2.4 The score of a basket given the taste

Each category $c$ has a profile $\psi_c$ of 16 numbers. It also has three kinds of penalty for buying
several products of the same category:

- $\rho_c$, charged once for every pair of products of the category in the basket;
- $f_c(k)$, a separate amount for each count $k$:
  - it takes its own value for $k = 2$, for $k = 3$, and one shared value for all $k \ge 4$;
  - it is zero for no products and for one product: $f_c(0) = f_c(1) = 0$;
- $p_c(z)$, charged once for every product of the category beyond the first:

$$
p_c(z) = \sum_{k=1}^{16} z_k\, \psi_{c,k} - \tfrac{1}{2} \sum_{k=1}^{16} \psi_{c,k}^2 .
$$

Finally, $\rho_0(m)$ is a size penalty that depends on the number of categories present, with
$\rho_0(1) = 0$.

The score of a basket $S$, given the taste $z$, is

$$
E(S \mid z) \;=\; \sum_{j \in S} u_j(z)
\;-\; \sum_{c} \Big[ \rho_c\, \frac{n_c (n_c - 1)}{2} + f_c(n_c) + \max(n_c - 1, 0)\, p_c(z) \Big]
\;-\; \rho_0(m).
$$

### 2.5 The probability of a basket

Given the taste, the probability of a basket is proportional to the exponential of its score:

$$
P(S \mid z) = \frac{e^{E(S \mid z)}}{Z(z)},
\qquad
Z(z) = \sum_{S'} e^{E(S' \mid z)},
$$

where the sum runs over every allowed basket $S'$. Dividing by $Z(z)$ makes the probabilities add up to 1.

The taste is not observed, so the probability of a basket is its average over all tastes:

$$
P(S) = \int \mathcal{N}(z)\, P(S \mid z)\, dz .
$$

**This is the distribution the generator draws from.** Draw $z$ from $\mathcal{N}(z)$, then draw
$S$ from $P(S \mid z)$. The basket that comes out has probability exactly $P(S)$.

The rest of this document explains how to draw $S$ from $P(S \mid z)$ without listing every possible
basket. There are far too many baskets to list.

---

## 3. A useful way to write $P(S \mid z)$

The score in section 2.4 has two useful properties:

- Apart from $\rho_0(m)$, it is a sum of separate parts, one per category.
- Inside one category, the score depends on which products are chosen only through the sum of their
  $u_j(z)$.

Together these let the draw be split into small steps.

**The category table.** For a category $c$ and a number $k$ of its products, define

$$
A_c(k) = \sum_{\substack{G \subseteq c \\ |G| = k}} \; \prod_{j \in G} w_j(z),
$$

the sum, over every group $G$ of $k$ different products of category $c$, of the product of their
weights. The first few values are:

- $A_c(0) = 1$;
- $A_c(1)$ is the sum of the weights of the category's products;
- $A_c(2)$ is the sum of $w_i w_j$ over all pairs of the category's products.

**The category factor.** Define

$$
F_c(k) = A_c(k)\, \exp\!\Big( -\rho_c\, \frac{k(k-1)}{2} - f_c(k) - \max(k-1, 0)\, p_c(z) \Big).
$$

$F_c(k)$ is the total weight of all the ways to put exactly $k$ products of category $c$ into the
basket. In particular $F_c(0) = 1$.

**The probability of the counts.** Add up $e^{E(S \mid z)}$ over all baskets that have the same
counts $(n_1, \dots, n_C)$:

- inside each category, adding over the choice of products gives $A_c(n_c)$;
- everything else depends only on the counts.

So

$$
P(n_1, \dots, n_C \mid z) = \frac{1}{Z(z)}\, \Big[\prod_{c=1}^{C} F_c(n_c)\Big]\, e^{-\rho_0(m)} .
$$

**The products given the counts.** Suppose category $c$ holds exactly $k$ products. Then a
particular group $G$ of $k$ of its products has probability

$$
P(G \mid n_c = k, z) = \frac{\prod_{j \in G} w_j(z)}{A_c(k)} ,
$$

and once their counts are fixed, the categories are independent of each other.

So a basket can be drawn in two parts: first the counts, then the products inside each category.
The counts themselves take two steps, because $\rho_0(m)$ ties the categories together.

---

## 4. The steps

The basket is drawn in six steps. Steps 2 and 3 only compute numbers; steps 1, 4, 5 and 6 draw at
random.

All the random numbers come from one seed:

- the torch library's random generator gives the seed;
- the C++ code turns the seed into a stream of uniform numbers between 0 and 1, using the SplitMix64
  generator.

The same seed always gives the same basket.

**One procedure used in several steps: drawing from a list of probabilities.** Suppose we want one
item from a list, with probabilities proportional to non-negative numbers $q_1, \dots, q_L$:

1. compute the total $Q = q_1 + \dots + q_L$;
2. draw a uniform number $U$ between 0 and 1;
3. go through the list adding up $q_1, q_2, \dots$, and choose the first item $i$ at which the
   running total reaches $U\,Q$:

$$
i = \min\Big\{ i' : \sum_{l=1}^{i'} q_l \ge U\, Q \Big\}.
$$

Item $i$ is then chosen with probability exactly $q_i / Q$.

### Step 1. Draw the taste

- **Distribution:** $z \sim \mathcal{N}(0, I_{16})$, the standard normal distribution in 16
  dimensions (section 2.1).
- **Procedure:** draw 16 independent standard normal numbers with the torch library's normal
  generator (`torch.randn`).

### Step 2. Compute the attractiveness and weight of every offered product

- **Distribution:** none. This step only computes numbers.
- **Procedure:** for every product offered on the trip, compute $b_j$ (section 2.2), then $u_j(z)$
  and $w_j(z) = e^{u_j(z)}$ (section 2.3).

### Step 3. Compute the category tables

- **Distribution:** none. This step only computes numbers.
- **Procedure:** for every category $c$ offered on the trip, compute $A_c(k)$ for $k = 0, 1, \dots, K_c$.
  Here $K_c$ is the smallest of three numbers: $R$, $N$, and the number of the category's offered
  products.

  The products are added one at a time. Write $A^{(p)}(k)$ for the table built from the first $p$
  products of the category. Start from

  $$
  A^{(0)}(0) = 1, \qquad A^{(0)}(k) = 0 \ \text{ for } k \ge 1 .
  $$

  Adding product $p$, which has weight $w_p$, changes the table to

  $$
  A^{(p)}(k) = A^{(p-1)}(k) + w_p\, A^{(p-1)}(k-1), \qquad k = 1, \dots, K_c .
  $$

  The code updates one array in place, running $k$ from $K_c$ down to 1.

  This rule is right because every group of $k$ products from the first $p$ either leaves out
  product $p$ (the first term) or contains it (the second term). So after the last product,
  $A_c(k) = A^{(P)}(k)$ counts each group exactly once. Then form $F_c(k)$ as in section 3.

  For numerical safety, the code:
  - works with logarithms;
  - divides all weights by the largest one before they are multiplied.

  This changes no probability.

### Step 4. Draw the basket size and the number of categories

- **Distribution:** the pair $(n, m)$ — the number of products and the number of categories present —
  has probability

  $$
  P(n, m \mid z) = \frac{T(n, m)\, e^{-\rho_0(m)}}{Z(z)},
  \qquad
  T(n, m) = \sum_{\substack{(n_1, \dots, n_C):\ \sum_c n_c = n \\ \#\{c\,:\, n_c \ge 1\} = m}} \;\prod_{c=1}^{C} F_c(n_c).
  $$

  The sum runs over all count lists with $n$ products in total, spread over exactly $m$ categories.
  The formula follows by grouping the formula of section 3 by $(n, m)$.

- **Procedure:**
  1. Build the table $T$ one category at a time. Let $T_i(n, m)$ be the same sum over the first $i$
     categories only. Start from $T_0(0, 0) = 1$, and 0 elsewhere.

     When category $i$ is added, it holds either none of its products or $k \ge 1$ of them. With
     none, the number of categories present does not change. With $k \ge 1$, one more category is
     present:

     $$
     T_i(n, m) = T_{i-1}(n, m)\, F_i(0) \;+\; \sum_{k=1}^{K_i} T_{i-1}(n - k,\, m - 1)\, F_i(k).
     $$

     After the last category, $T = T_C$. The tables $T_0, \dots, T_C$ along the way are kept for step 5.
  2. Draw one pair $(n, m)$ with $1 \le m \le n$, with probabilities proportional to
     $T(n, m)\, e^{-\rho_0(m)}$. Use the procedure for a list of probabilities.

- **The size limit.** Two facts make a limit worthwhile: building $T$ up to $n = N$ for every trip is
  slow, and very large baskets are usually almost impossible. So the code works as follows:
  1. It chooses a limit $L$ for the trip, with $32 \le L \le N$, and builds $T$ only up to $n = L$.
  2. It keeps this limit only if it can prove, from upper bounds computed for that trip, that

     $$
     P(n > L \mid z) \le 10^{-8}.
     $$

  3. If it cannot prove this, it builds $T$ all the way up to $N$.

  So the basket is drawn from one of two distributions: the exact one, or the exact one restricted
  to at most $L$ products. These two differ by at most $10^{-8}$ in the probability of any event.

### Step 5. Draw how many products come from each category

- **Distribution:** given the drawn $(n, m)$, a count list $(n_1, \dots, n_C)$ has probability

  $$
  P(n_1, \dots, n_C \mid n, m, z) = \frac{\prod_{c} F_c(n_c)}{T(n, m)}
  $$

  among the count lists that have $n$ products over $m$ categories.

- **Procedure:** draw the counts one category at a time, from the last category to the first.
  Keep two running numbers, $n_{\text{rest}}$ and $m_{\text{rest}}$: the products and categories still
  to be placed. At the start, $n_{\text{rest}} = n$ and $m_{\text{rest}} = m$. For category $i$, the
  probability that it holds $k$ products is

  $$
  P(n_i = k \mid n_{\text{rest}}, m_{\text{rest}}) \;\propto\;
  \begin{cases}
  T_{i-1}(n_{\text{rest}},\, m_{\text{rest}})\, F_i(0) & k = 0, \\[4pt]
  T_{i-1}(n_{\text{rest}} - k,\, m_{\text{rest}} - 1)\, F_i(k) & k \ge 1 .
  \end{cases}
  $$

  Draw $k$ with the procedure for a list of probabilities. Then update the running numbers:
  - $n_{\text{rest}} \leftarrow n_{\text{rest}} - k$;
  - if $k \ge 1$, also $m_{\text{rest}} \leftarrow m_{\text{rest}} - 1$.

  After the first category, both $n_{\text{rest}}$ and $m_{\text{rest}}$ are 0.

  **Why this is right.** $T_{i-1}(n', m')$ is the total weight of all the ways the first $i - 1$
  categories can hold $n'$ products over $m'$ categories. So the formula gives each choice of $k$
  its share of the total weight of the count lists that are still possible. Multiplying these
  probabilities over the categories, from $i = C$ down to $1$, gives the probability of the whole
  count list. Each $T$ in a denominator cancels against the $T$ in the next category's numerator,
  and what is left is $\prod_c F_c(n_c) / T(n, m)$, exactly the distribution above.

### Step 6. Draw which products are chosen inside each category

- **Distribution:** for each category $c$ with $n_c = k \ge 1$, a group $G$ of exactly $k$ different
  products of the category, with probability

  $$
  P(G) = \frac{\prod_{j \in G} w_j(z)}{A_c(k)} .
  $$

  The categories are drawn independently of each other.

- **Procedure:** number the category's products $1, \dots, P$, and use the tables $A^{(p)}$ of step 3
  (the code keeps all of them). Go through the products from the last, $p = P$, to the first. Keep
  $\text{need}$, the number of products still to choose; at the start $\text{need} = k$. At product $p$,
  while $\text{need} > 0$:
  - If $\text{need} = p$, all the remaining products must be chosen: choose products $1, \dots, p$ and stop.
  - Otherwise, choose product $p$ with probability

    $$
    \pi_p = \frac{w_p\, A^{(p-1)}(\text{need} - 1)}{A^{(p)}(\text{need})} .
    $$

    To do so, draw $U$ uniform between 0 and 1. If $U < \pi_p$, choose product $p$ and set
    $\text{need} \leftarrow \text{need} - 1$.

  If $k$ equals the number of products in the category, all of them are chosen directly.

  **Why this is right.** $A^{(p)}(\text{need})$ is the total weight of all groups of $\text{need}$
  products among the first $p$. Each such group either contains product $p$, or it does not:

  $$
  A^{(p)}(\text{need}) = \underbrace{w_p\, A^{(p-1)}(\text{need} - 1)}_{\text{groups with } p}
  + \underbrace{A^{(p-1)}(\text{need})}_{\text{groups without } p} .
  $$

  So $\pi_p$ is exactly product $p$'s share. When these choices are multiplied over the products,
  the denominators cancel, and what is left is exactly $P(G)$.

**The basket** is the set of all products chosen in step 6.

---

## 5. Summary of the steps

| step | what is drawn or computed | distribution | procedure |
|---|---|---|---|
| 1 | the taste $z$ (16 numbers) | $\mathcal{N}(0, I_{16})$ | 16 independent standard normal draws |
| 2 | $u_j(z)$ and $w_j(z)$ for every offered product | none (computed) | formulas of sections 2.2–2.3 |
| 3 | tables $A_c(k)$ and factors $F_c(k)$ | none (computed) | add the products one at a time |
| 4 | basket size $n$ and number of categories $m$ | $\propto T(n,m)\, e^{-\rho_0(m)}$ | build $T$ category by category; one draw from a list |
| 5 | count $n_c$ of each category | $\propto \prod_c F_c(n_c)$ given $(n, m)$ | one draw per category, last to first |
| 6 | the products inside each category | $\propto \prod_{j \in G} w_j(z)$ given $n_c$ | go through the products, choosing each with its exact share |

Steps 1 to 6 together give a basket with probability exactly

$$
P(S) = \int \mathcal{N}(z)\, P(S \mid z)\, dz ,
$$

up to the $10^{-8}$ size limit of step 4. No step is repeated until it settles, and no step
replaces the distribution with an approximation.

---

## 6. Generating with part of the basket known

Suppose some products of the basket are already known, for example when suggesting what else the
shopper will add. Knowing them changes what we know about the taste. Given the known products $K$,
the taste has density

$$
p(z \mid K) \;\propto\; \mathcal{N}(z)\, \frac{e^{E(K \mid z)}}{Z(z)} .
$$

The code handles this density as follows:

1. This density has a single peak. The code finds it by climbing toward it step by step.
2. It draws candidate tastes around the peak, from a mix of two normal distributions centred on it:
   one 1.2 times as wide as the peak, and one of standard width.
3. It weights each candidate $z^{(i)}$ by the density given $K$, divided by the density it was drawn
   from:

   $$
   \text{weight}_i \;\propto\; \frac{\mathcal{N}(z^{(i)})\, e^{E(K \mid z^{(i)})} / Z(z^{(i)})}{q(z^{(i)})} ,
   $$

   where $q$ is the density of the mix in step 2.

The repository uses these weighted tastes only to **score** the products that could be added
(`research/evaluate_mixture.py recommend`). It does not generate a full basket around the known
products. That would also change steps 4 to 6, which would then have to keep the known products in
the basket.

---

## 7. Cost

On the Dunnhumby model, which offers 5,455 products per trip, one basket takes about 5–7
milliseconds:

- about 2–3 ms to build the trip's context;
- about 3 ms for step 2;
- under 1 ms for steps 3–6.

In the synthetic world (522 products) one basket takes about 1 millisecond. These times were
measured on 200 single trips.
