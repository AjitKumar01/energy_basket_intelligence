# How one basket is generated

This document describes, step by step, how the model generates one shopping basket for one trip.
Each step gives three things:

- the exact probability distribution of what is drawn;
- the exact algorithm that draws from it, operation by operation, as the code performs it;
- a proof that the algorithm draws from that distribution.

Every symbol is defined before it is used. The code is in:

- `v4/mixture.py`, function `generate` (step 1);
- `v4/model.py`, functions `slot_utilities_given_z` and `presence_shift` (step 2);
- `v4/native/sampler.cpp`:
  - function `sample_slots_cpp`, for steps 3 and 6;
  - function `sample_trip_categories_2d_impl`, for steps 4 and 5;
  - functions `choose_trip_cap_m3` and `compute_log_z_inf_theta`, for the size limit in step 4.

Equations are written in TeX: `$...$` inline and `$$...$$` displayed.

---

## 1. What is given, and what is produced

**Given (the context of the trip):**

- the household $h$ that makes the trip;
- the store $s$ and the week $w$ of the trip;
- the products the store offers on that trip;
- for each offered product:
  - its price and its regular price;
  - whether it is on display, and whether it is in the mailer;
  - the household's purchase history up to that trip;
- the fitted model parameters.

**Produced:** a basket $S$, a set of distinct offered products. It holds at least one product and
at most $N$ products: $N = 120$ on the Dunnhumby data, and $N = 80$ in the synthetic world.

**Notation.**

- *Category*: the group a product belongs to. On Dunnhumby this is a commodity, such as
  "pasta sauce". Every product belongs to exactly one category, and $c(j)$ is the category of product $j$.
- The categories offered on the trip, i.e. those with at least one offered product, are numbered
  $1, \dots, C$ in increasing order of category id. Steps 4 and 5 go through them in this order.
- Category $c$ has $P_c$ offered products, numbered $1, \dots, P_c$ in the order the store's
  product list stores them. Steps 3 and 6 go through them in this order.
- $n_c$ is the number of products of category $c$ in the basket.
- $n = n_1 + \dots + n_C$ is the number of products in the basket.
- $m$ is the number of categories present in the basket, that is, with $n_c \ge 1$.
- $R = 40$ is the largest number of products one category may contribute.
- $K_c = \min(R, P_c, N)$ is the largest possible value of $n_c$.

---

## 2. The probability of a basket

### 2.1 The taste of the trip

Each trip has a hidden *taste*, a list of $r = 16$ numbers $z = (z_1, \dots, z_{16})$.
Before anything is known about the basket, its distribution is the standard normal distribution in
16 dimensions: the 16 numbers are independent, each with mean 0 and variance 1.

$$
\mathcal{N}(z) = (2\pi)^{-8} \exp\!\Big( -\tfrac{1}{2} \sum_{k=1}^{16} z_k^2 \Big).
$$

### 2.2 The base attractiveness of a product on this trip

For each offered product $j$ the model computes a number $b_j$, the product's attractiveness on
this trip before the taste is taken into account:

$$
\begin{aligned}
b_j \;=\;& \lambda_j && \text{popularity of the product} \\
&+ \theta_{h,\text{size}} + \textstyle\sum_k \theta_{h,k}\, a_{j,k} && \text{the household's lasting taste ($a_{j,k}$: product loadings)} \\
&- \text{price}_j && \text{see below} \\
&+ v_j\, \text{display}_j + q_j\, \text{mailer}_j && \text{promotions ($\text{display}_j, \text{mailer}_j \in \{0,1\}$)} \\
&+ \textstyle\sum_k \mu_{j,k}\, \delta_{w,k} && \text{season ($\delta_w$: numbers for week $w$)} \\
&+ \textstyle\sum_k \zeta_{j,k}\, \xi_{s,k} && \text{store ($\xi_s$: numbers for store $s$)} \\
&+ \text{avail}_j && \text{0 once the product is known to be sold at the store, $\log\varepsilon$ before} \\
&+ \text{history}_j - o_t && \text{purchase history, minus a trip-wide offset $o_t$} \\
&+ \textstyle\sum_k \theta_{h,k}\, e_{c(j),k} && \text{the household's taste for the product's category} \\
&+ \iota_{t,c(j)} && \text{competition inside the category (see below)}
\end{aligned}
$$

**History.** $\text{history}_j$ is a weighted sum of four numbers that describe the household's last
purchase in the product's sub-category (never bought; $e^{-\text{days}/7}$; $e^{-\text{days}/\text{typical gap}}$;
$\log(1+\text{days})/\log 100$), plus a weighted sum, with weights shared by the category, of three
numbers about the product itself (bought before; bought on the household's last purchase in the
category; $e^{-\text{days since last bought}/28}$). The offset $o_t$ is the
average of $\text{history}_j$ over the products offered on the trip, plus a fitted multiple of
$\log(1 + \text{number of offered products the household has bought before})$.

**Price term.** Two price numbers are used:

- $d_j$, the logarithm of product $j$'s price relative to its usual price, with $\bar d$ its average over
  the products offered on the trip;
- $x_j$, the depth of the product's discount: the logarithm of price over regular price, centred.

Then

$$
\text{price}_j \;=\; g_{h,j}\big(\bar d + \kappa\,(d_j - \bar d)\big)
\;+\; c\,(d_j - \bar d)
\;+\; g_{h,j}\,(\kappa_{\text{disc}} - \kappa)\, x_j
\;+\; c_{\text{disc}}\, x_j ,
$$

where

$$
g_{h,j} = \sum_{k=1}^{8} \operatorname{soft}(\gamma_{h,k})\, \operatorname{soft}(\beta_{j,k}) \ \ge 0,
\qquad \operatorname{soft}(y) = \log(1 + e^{y}),
$$

and $\kappa, \kappa_{\text{disc}}, c, c_{\text{disc}}$ are single numbers. The term $c$ exists only in
models fitted with `--price-common`.

**Competition inside the category.** Let $b'_i$ be the attractiveness without $\iota$ and without
$o_t$, and $b''_i$ the same at regular prices with no promotions. Then

$$
\iota_{t,c} = (\lambda_c - 1)\Big[\log \sum_{i \in c} e^{b'_i} \;-\; \log \sum_{i \in c} e^{b''_i}\Big],
\qquad 0 < \lambda_c < 1 .
$$

None of the terms in $b_j$ depends on the taste or on the rest of the basket.

### 2.3 The attractiveness and weight of a product given the taste

Each product has a *taste profile* $\varphi_j = (\varphi_{j,1}, \dots, \varphi_{j,16})$. Given the taste $z$,

$$
u_j(z) = b_j - \tfrac{1}{2} \sum_{k=1}^{16} \varphi_{j,k}^2 + \sum_{k=1}^{16} z_k\, \varphi_{j,k},
\qquad
w_j(z) = e^{u_j(z)} .
$$

### 2.4 The score of a basket given the taste

Each category $c$ has three penalties for holding several products:

- $\rho_c$, charged once for every pair of products of the category in the basket;
- $f_c(k)$, a fitted amount for $k = 2$, for $k = 3$, and one shared amount for all $k \ge 4$,
  with $f_c(0) = f_c(1) = 0$;
- $p_c(z)$, charged once for every product of the category beyond the first. It is computed from a
  profile $\psi_c$ of 16 numbers:

  $$
  p_c(z) = \sum_{k=1}^{16} z_k\, \psi_{c,k} - \tfrac{1}{2} \sum_{k=1}^{16} \psi_{c,k}^2 .
  $$

Finally, $\rho_0(m)$ is a penalty on the number of categories present, with $\rho_0(1) = 0$.

The score of a basket $S$ given $z$ is

$$
E(S \mid z) \;=\; \sum_{j \in S} u_j(z)
\;-\; \sum_{c=1}^{C} \Big[ \rho_c\, \frac{n_c (n_c - 1)}{2} + f_c(n_c) + \max(n_c - 1, 0)\, p_c(z) \Big]
\;-\; \rho_0(m).
$$

### 2.5 The probability of a basket

$$
P(S \mid z) = \frac{e^{E(S \mid z)}}{Z(z)},
\qquad
Z(z) = \sum_{S' :\ 1 \le |S'| \le N,\ n_c(S') \le K_c} e^{E(S' \mid z)},
$$

$$
P(S) = \int \mathcal{N}(z)\, P(S \mid z)\, dz .
$$

**This is the distribution the generator draws from.** If $z$ is drawn from $\mathcal{N}$ and then
$S$ is drawn from $P(S \mid z)$, then $S$ has probability exactly $P(S)$.

---

## 3. Splitting $P(S \mid z)$ into counts and products

**Category tables.** For category $c$ and $k = 0, \dots, K_c$,

$$
A_c(k) = \sum_{\substack{G \subseteq \{1..P_c\} \\ |G| = k}} \; \prod_{j \in G} w_j(z),
\qquad A_c(0) = 1 .
$$

**Category factors.**

$$
F_c(k) = A_c(k)\, \exp\!\Big( -\rho_c\, \frac{k(k-1)}{2} - f_c(k) - \max(k-1, 0)\, p_c(z) \Big),
\qquad F_c(0) = 1 .
$$

**Probability of the counts.** In $E(S \mid z)$, only $\sum_{j \in S} u_j(z)$ depends on *which*
products are in the basket. Every other term depends only on the counts $(n_1, \dots, n_C)$. So
adding $e^{E(S \mid z)}$ over all baskets with the same counts gives

$$
\sum_{S\,:\ \text{counts } (n_1..n_C)} e^{E(S \mid z)}
= \Big[\prod_{c=1}^{C} A_c(n_c)\Big] \exp\Big(-\sum_c \big[\rho_c\tfrac{n_c(n_c-1)}{2} + f_c(n_c) + \max(n_c-1,0)\,p_c(z)\big]\Big) e^{-\rho_0(m)}
= \Big[\prod_{c=1}^{C} F_c(n_c)\Big] e^{-\rho_0(m)} .
$$

$$
P(n_1, \dots, n_C \mid z) = \frac{1}{Z(z)} \Big[\prod_{c=1}^{C} F_c(n_c)\Big] e^{-\rho_0(m)},
\qquad 1 \le n \le N,\ 0 \le n_c \le K_c .
$$

**Probability of the products given the counts.** Divide $P(S \mid z)$ by the probability of its
counts. What is left factorises over categories: given the counts, the group $G_c$ chosen in each
category is independent of the other categories, and

$$
P(G_c \mid n_c = k, z) = \frac{\prod_{j \in G_c} w_j(z)}{A_c(k)} .
$$

**Size tables.** For $i = 0, \dots, C$, let $T_i(n, m)$ be the total of $\prod_{c \le i} F_c(n_c)$
over all count lists of the first $i$ categories with $n_1 + \dots + n_i = n$ and exactly $m$ of
$n_1, \dots, n_i$ at least 1. Every count list of the first $i$ categories gives category $i$ either
$0$ products (then $m$ is unchanged) or $k \ge 1$ products (then $m$ goes up by one), so

$$
T_0(0, 0) = 1,\quad T_0(n, m) = 0 \text{ otherwise};
\qquad
T_i(n, m) = T_{i-1}(n, m)\, F_i(0) + \sum_{k=1}^{K_i} T_{i-1}(n-k,\, m-1)\, F_i(k).
$$

Grouping the counts by $(n, m)$ then gives

$$
P(n, m \mid z) = \frac{T_C(n, m)\, e^{-\rho_0(m)}}{Z(z)},
\qquad
P(n_1, \dots, n_C \mid n, m, z) = \frac{\prod_c F_c(n_c)}{T_C(n, m)} .
$$

**The draw** follows this split:

1. draw $z$;
2. draw $(n, m)$ from $P(n, m \mid z)$;
3. draw the counts from $P(n_1, \dots, n_C \mid n, m, z)$;
4. draw the products of each category from $P(G_c \mid n_c, z)$.

The chain rule multiplies these back into $P(S \mid z)$.

---

## 4. Random numbers and the one draw procedure used throughout

**Uniform numbers.** The torch random generator gives one integer seed $\sigma$, drawn uniformly
from $\{1, \dots, 2^{62}-1\}$. The C++ code feeds $\sigma$ to the SplitMix64 generator, which
produces a stream of 64-bit integers $x_1, x_2, \dots$. Each uniform number is

$$
U = \lfloor x / 2^{11} \rfloor \,/\, 2^{53},
$$

which is uniform on the $2^{53}$ equally spaced points $0, 2^{-53}, \dots, 1 - 2^{-53}$ of $[0, 1)$.

From the main stream, the code first takes one 64-bit integer per trip in the batch. Each of these
seeds a separate SplitMix64 stream for that trip, which supplies the uniforms of steps 4 and 5.
Step 6 then takes its uniforms from the main stream, after those per-trip seeds.

**Drawing one item from a list.** Given non-negative numbers $q_1, \dots, q_L$ that are not all zero:

1. $Q \leftarrow q_1 + \dots + q_L$;
2. draw one uniform $U$;
3. $a \leftarrow 0$; for $i = 1, 2, \dots, L$: $a \leftarrow a + q_i$; if $a \ge U\, Q$, return $i$.
   If the loop ends without returning, which floating-point rounding can cause, the last item is returned.

Item $i$ is returned exactly when $\sum_{l<i} q_l < U Q \le \sum_{l \le i} q_l$, an interval of length
$q_i / Q$ for $U$. So $P(\text{item } i) = q_i / Q$, up to the $2^{-53}$ spacing of $U$. (The one
exception is $U = 0$, which has probability $2^{-53}$ and returns the first item even when $q_1 = 0$.)

Multiplying all the $q_i$ by the same positive number does not change $q_i / Q$. The algorithms
below use this repeatedly: they keep numbers only up to a common factor, to keep them inside the
range of a double.

---

## 5. The steps

### Step 1. Draw the taste

- **Distribution:** $z \sim \mathcal{N}(0, I_{16})$.
- **Algorithm:** `torch.randn(16, generator=gen)` returns 16 independent standard normal numbers.
  Then the seed $\sigma$ of section 4 is drawn from the same torch generator.

### Step 2. Compute every product's attractiveness and weight, and every category's $p_c(z)$

- **Distribution:** none. This step only computes numbers.
- **Algorithm:** for every offered product, compute $b_j$ (section 2.2) and $u_j(z)$ (section 2.3).
  For every offered category, compute $p_c(z)$ (section 2.4).

### Step 3. Compute the category factors $F_c(k)$

- **Distribution:** none. This step only computes numbers.
- **Algorithm**, for each offered category $c$, with products $1, \dots, P_c$:
  1. **Scale the weights.** Let $M_c = \max_{p} u_p(z)$, and set
     $\tilde w_p = e^{u_p(z) - M_c} \in (0, 1]$ for $p = 1, \dots, P_c$.
  2. **Build the scaled table**, adding the products one at a time:

     $$
     \tilde A(0) = 1,\ \tilde A(k) = 0 \ (k = 1..K_c);
     \quad\text{for } p = 1..P_c:\ \text{for } k = \min(p, K_c) \text{ down to } 1:\
     \tilde A(k) \leftarrow \tilde A(k) + \tilde A(k-1)\, \tilde w_p .
     $$

     The inner loop runs $k$ downwards. So $\tilde A(k-1)$ on the right still holds its value from
     before product $p$ was added. After product $p$, $\tilde A(k)$ is the sum of
     $\prod_{j \in G} \tilde w_j$ over the groups $G$ of $k$ products among the first $p$. The proof is
     by induction on $p$: each such group either leaves out $p$ (the old $\tilde A(k)$), or contains
     $p$ together with a group of $k-1$ products among the first $p-1$ (the term $\tilde A(k-1)\tilde w_p$).
  3. **Undo the scale.** Every group of $k$ products picks up $e^{-k M_c}$, so $A_c(k) = \tilde A(k)\, e^{k M_c}$.
  4. **Store the log of the factor**, $G_c(k) = \log F_c(k)$:

     $$
     G_c(0) = 0, \qquad
     G_c(k) = \log \tilde A(k) + k M_c - \Big[\rho_c \frac{k(k-1)}{2} + f_c(k) + (k-1)\, p_c(z)\Big],
     \quad k = 1..K_c .
     $$

     If $\tilde A(k) = 0$ in floating point, $G_c(k) = -\infty$, and that count is treated as impossible.
     This happens only when $\tilde A(k)$ is below the smallest positive double, about $5 \times 10^{-324}$.
  5. **Keep** $\tilde w_1, \dots, \tilde w_{P_c}$ for step 6.

### Step 4. Draw the basket size $n$ and the number of categories $m$

- **Distribution:**

  $$
  P(n, m \mid z, n \le L) = \frac{T_C(n, m)\, e^{-\rho_0(m)}}{\sum_{n' \le L} \sum_{m'} T_C(n', m')\, e^{-\rho_0(m')}},
  \qquad 1 \le m \le n \le L .
  $$

  Here $L$ is the size limit of the trip, chosen in 4a. The limit is kept only when baskets larger
  than $L$ are proven to carry at most $10^{-8}$ of $P(\cdot \mid z)$; otherwise $L = D$, the largest
  possible size, and the distribution is exactly $P(n, m \mid z)$.

- **Algorithm:**

  **4a. Choose the size limit $L$.**
  1. $D \leftarrow \min(N, \sum_c K_c)$. If $D \le 32$, set $L \leftarrow D$ and go to 4b.
  2. For $\vartheta \in \{0, 0.4, 0.8\}$ compute

     $$
     H(\vartheta) = \sum_{m=1}^{\min(C, N)} e^{-\rho_0(m)}\, B_\vartheta(m),
     $$

     where $B_\vartheta(m)$ is the coefficient of $y^m$ in
     $\prod_{c=1}^{C}\big(1 + y\, \ell_c(\vartheta)\big)$, with
     $\ell_c(\vartheta) = \sum_{k=1}^{K_c} F_c(k)\, e^{\vartheta k}$.

     It is computed with a one-dimensional table: start from $B(0) = 1$ and $B(m) = 0$ for $m \ge 1$;
     for each category $c$ and each $m$ from high to low, $B(m) \leftarrow B(m) + \ell_c(\vartheta)\, B(m-1)$.
     To stay inside the range of a double:
     - each $\ell_c$ is computed from logs as $\max_k(\cdot) + \log\sum_k e^{(\cdot) - \max}$;
     - each category's term is divided by $\max(1, \ell_c)$;
     - the table is divided by its largest entry each time the logs of these divisors add up to 60;
     - all the divisors are added back in logs at the end.

     **Why $H$ bounds the tail.** Let $W(n')$ be the total of $\prod_c F_c(n_c)\, e^{-\rho_0(m)}$
     over the count lists of size $n'$, with no upper limit on $n'$ other than $n_c \le K_c$. Expanding
     the product gives $H(\vartheta) = \sum_{n'} W(n')\, e^{\vartheta n'}$. Hence:
     - $H(0) \ge Z(z)$;
     - $\sum_{n' > L} W(n') \le e^{-\vartheta (L+1)} H(\vartheta)$ for every $\vartheta \ge 0$.
  3. Proposed limit, aiming at a tail of $5 \times 10^{-9}$ (note $-\log(5\times 10^{-9}) = 19.1138$):

     $$
     L \leftarrow \min_{\vartheta \in \{0.4,\, 0.8\}} \Big\lceil \frac{\log H(\vartheta) - \log H(0) + 19.1138}{\vartheta} \Big\rceil,
     \qquad\text{then clamp } L \text{ to } [32, D].
     $$

     If $L = D$, there is no limit.

  **4b. Build the scaled size tables up to $n = L$.** For each category $i = 1, \dots, C$ in order:
  1. $\mu_i \leftarrow \max\big(0, \max_{1 \le k \le \min(K_i, L)} G_i(k)\big)$;
     $\;q_i(0) \leftarrow e^{-\mu_i}$; $\;q_i(k) \leftarrow e^{G_i(k) - \mu_i}$ for $k = 1..\min(K_i, L)$.
     So $q_i(k) = F_i(k)\, e^{-\mu_i}$ for every $k$, and every $q_i(k) \le 1$.
  2. $K'_i \leftarrow$ the largest $k$ with $q_i(k) > 0$. If $K'_i = 0$, then $\tilde T_i \leftarrow \tilde T_{i-1}$
     and move to the next category.
  3. For $n = 0..d_i$ and $m = 0..\min(n, c_i)$, where $d_i = \min(L, d_{i-1} + K'_i)$ and
     $c_i = \min(L, c_{i-1} + 1)$ (starting from $d_0 = c_0 = 0$ and $\tilde T_0(0,0) = 1$):

     $$
     \tilde T_i(n, m) = \tilde T_{i-1}(n, m)\, q_i(0) + \sum_{k=1}^{K'_i} \tilde T_{i-1}(n-k,\, m-1)\, q_i(k),
     $$

     where entries of $\tilde T_{i-1}$ outside $n \le d_{i-1}$, $m \le \min(n, c_{i-1})$ count as 0.
  4. **Rescale when needed.** Keep two running quantities since the last rescale:
     - $a$, the running sum of $\mu_i$;
     - $\Pi$, the running product of $\sum_{k \ge 0} q_i(k)$.

     If $a \ge 60$, or $\Pi \ge 10^{60}$, or $q_i(0) = 0$, divide every entry of $\tilde T_i$ by its
     largest entry $v$. Then add $\log v$ to a running log-scale $\lambda$, and reset $a$ and $\Pi$.
     If $v = 0$, the table has underflowed: go to 4d.
  5. Add $\mu_i$ to $\lambda$.
  6. Store $\tilde T_i$, $d_{i-1}$ and $c_{i-1}$ for step 5.

  **Invariant.** For each $i$ there is one number $s_i$ such that
  $T_i(n, m) = \tilde T_i(n, m)\, e^{s_i}$ for all $(n, m)$ with $n \le L$. Indeed, multiplying all
  factors of category $i$ by $e^{-\mu_i}$, or dividing a whole table by $v$, multiplies every entry
  by the same number. Also $s_C = \lambda$.

  **4c. Certify the limit.** If $L < D$, compute

  $$
  \log Z_L = \lambda + \log \sum_{n=1}^{d_C} \sum_{m=1}^{\min(n, c_C)} \tilde T_C(n, m)\, e^{-\rho_0(m)} ,
  $$

  the log of the total weight of baskets with at most $L$ products. Keep $L$ if any one of these holds:

  $$
  \text{(i) } \log H(0) - \log Z_L \le 10^{-8};
  \qquad
  \text{(ii) } e^{\log H(0.4) - 0.4(L+1) - \log Z_L} \le 10^{-8};
  \qquad
  \text{(iii) } e^{\log H(0.8) - 0.8(L+1) - \log Z_L} \le 10^{-8}.
  $$

  Each condition implies $P(n > L \mid z) \le 10^{-8}$:
  - (i): $P(n > L \mid z) = 1 - Z_L / Z \le \log(Z / Z_L) \le \log H(0) - \log Z_L$.
  - (ii) and (iii): $P(n > L \mid z) = \sum_{n' > L} W(n') / Z \le e^{-\vartheta(L+1)} H(\vartheta) / Z_L$.

  **4d. Fall back.** If no condition holds, or the table underflowed, set $L \leftarrow D$ and redo 4b
  once. With $L = D$ no certificate is needed, because nothing is cut off.

  **4e. Draw $(n, m)$.** Let $\Lambda = \max_{n, m} \big[\log \tilde T_C(n, m) - \rho_0(m)\big]$, and set

  $$
  q(n, m) = \exp\big(\log \tilde T_C(n, m) - \rho_0(m) - \Lambda\big),
  \qquad n = 1..d_C,\ m = 1..\min(n, c_C).
  $$

  The code adds $\lambda$ to both terms; it cancels. Arrange the pairs in a list: $n$ ascending,
  and for each $n$, $m$ ascending. Draw one pair with the list procedure of section 4, using one
  uniform from the trip's stream.

- **Why this is right.** $q(n, m) = T_C(n, m)\, e^{-\rho_0(m)} \times e^{-s_C - \Lambda}$, and
  $e^{-s_C - \Lambda}$ is the same for every pair. So the list procedure returns $(n, m)$ with
  probability $T_C(n,m)e^{-\rho_0(m)} / Z_L$, which is the distribution above.

### Step 5. Draw how many products come from each category

Step 4 has drawn the basket size $n$ and the number of categories present, $m$. Step 5 draws the
**partition**: the list of counts $(n_1, n_2, \dots, n_C)$, one count per category.

#### 5.1 The probability of a whole partition

A partition is *allowed* when it satisfies three conditions:

- the counts add up to $n$: $n_1 + \dots + n_C = n$;
- exactly $m$ of the counts are at least 1;
- each count is within its limit: $0 \le n_c \le K_c$.

The probability of an allowed partition is

$$
\boxed{\;P(n_1, \dots, n_C \mid n, m, z) = \frac{F_1(n_1)\, F_2(n_2) \cdots F_C(n_C)}{T_C(n, m)}\;}
$$

and 0 for any other list. Here:

- $F_c(k)$ is the category factor of section 3, the total weight of all ways of putting $k$
  products of category $c$ in the basket;
- $T_C(n, m)$ is the sum of the numerator over all allowed partitions. That sum is what makes the
  probabilities add up to 1.

This formula comes from section 3. It is the probability of the counts,
$\prod_c F_c(n_c)\, e^{-\rho_0(m)} / Z(z)$, divided by the probability of $(n, m)$,
$T_C(n,m)\, e^{-\rho_0(m)} / Z(z)$.

The code never lists the partitions and never computes this formula for each of them, because there
are far too many. For example, 7 products over 4 of 187 categories already gives more than $10^{9}$
partitions. Instead it draws the counts **one category at a time**, starting from the last category,
$C$, and ending with the first.

#### 5.2 The partition probability as a product of conditional probabilities

By the chain rule of probability, the probability of a partition equals the probability of the
last count, times the probability of the next count given the last one, and so on:

$$
P(n_1, \dots, n_C \mid n, m)
= P(n_C \mid n, m)\;\cdot\; P(n_{C-1} \mid n_C, n, m)\;\cdot\; P(n_{C-2} \mid n_{C-1}, n_C, n, m)\;\cdots\; P(n_1 \mid n_2, \dots, n_C, n, m).
$$

(The taste $z$ is fixed throughout step 5, so it is left out of the notation.)

The algorithm draws $n_C$ from the first factor. It then draws $n_{C-1}$ from the second factor,
with the $n_C$ just drawn put in, and so on down to $n_1$. The partition it ends with therefore has
exactly the probability in 5.1. What remains is to write down each of these conditional
probabilities.

#### 5.3 Each conditional probability

When the algorithm reaches category $i$, the counts $n_{i+1}, \dots, n_C$ are already drawn. Two
numbers describe what is left for categories $1, \dots, i$:

$$
\hat n_i = n - (n_{i+1} + \dots + n_C) \quad\text{(products still to place)},
\qquad
\hat m_i = m - \#\{c > i : n_c \ge 1\} \quad\text{(categories still to fill)}.
$$

At the start, $\hat n_C = n$ and $\hat m_C = m$.

We also need the **size tables** of section 3. $T_i(a, b)$ is the total weight
$\prod_{c \le i} F_c(n_c)$ of all ways the first $i$ categories can hold $a$ products, with exactly
$b$ of those categories non-empty.

**The conditional probability of category $i$'s count is**

$$
\boxed{\;
P(n_i = k \mid n_{i+1}, \dots, n_C, n, m) =
\begin{cases}
\dfrac{F_i(0)\; T_{i-1}(\hat n_i,\ \hat m_i)}{T_i(\hat n_i,\ \hat m_i)} & k = 0, \\[14pt]
\dfrac{F_i(k)\; T_{i-1}(\hat n_i - k,\ \hat m_i - 1)}{T_i(\hat n_i,\ \hat m_i)} & k = 1, 2, \dots, \min(K_i, \hat n_i).
\end{cases}
\;}
$$

Read it like this:

- category $i$ takes $k$ products, which has weight $F_i(k)$;
- the earlier categories $1, \dots, i-1$ must then hold the remaining $\hat n_i - k$ products. They
  must fill $\hat m_i$ categories if $k = 0$, or $\hat m_i - 1$ if $k \ge 1$, because category $i$ is
  then one of the filled ones. The total weight of all the ways they can do this is the $T_{i-1}$ entry;
- the denominator is the sum of the numerators over $k$. It equals $T_i(\hat n_i, \hat m_i)$ by
  the recursion that builds the tables:

  $$
  T_i(a, b) = F_i(0)\, T_{i-1}(a, b) + \sum_{k \ge 1} F_i(k)\, T_{i-1}(a - k,\, b - 1).
  $$

**Where the formula comes from.** Add up the joint probability of 5.1 over the counts not yet drawn,
$n_1, \dots, n_{i-1}$. The factors $F_1(n_1) \cdots F_{i-1}(n_{i-1})$ add up to a $T_{i-1}$ entry,
by the definition of $T_{i-1}$. This gives the probability of the counts drawn so far:

$$
P(n_i, n_{i+1}, \dots, n_C \mid n, m) = \frac{F_i(n_i) \cdots F_C(n_C)\;\; T_{i-1}\big(\hat n_i - n_i,\ \hat m_i - [n_i \ge 1]\big)}{T_C(n, m)},
$$

where $[n_i \ge 1]$ is 1 if $n_i \ge 1$ and 0 otherwise. The same sum one category later gives

$$
P(n_{i+1}, \dots, n_C \mid n, m) = \frac{F_{i+1}(n_{i+1}) \cdots F_C(n_C)\;\; T_i(\hat n_i, \hat m_i)}{T_C(n, m)} .
$$

The conditional probability is the first divided by the second, which is the boxed formula.

**The product of the conditionals is the partition probability.** Multiply the boxed formula over
$i = C, C-1, \dots, 1$. The denominator at category $i$ is $T_i(\hat n_i, \hat m_i)$. The $T$ in
the numerator at category $i+1$ is $T_i(\hat n_{i+1} - n_{i+1}, \dots) = T_i(\hat n_i, \hat m_i)$,
the same number. So each $T$ cancels against the next one, and what is left is

$$
\prod_{i=1}^{C} P(n_i \mid n_{i+1}, \dots, n_C, n, m)
= \frac{F_1(n_1) \cdots F_C(n_C)\;\; T_0(0, 0)}{T_C(n, m)}
= \frac{F_1(n_1) \cdots F_C(n_C)}{T_C(n, m)} ,
$$

since $T_0(0,0) = 1$. This is exactly the boxed formula of 5.1.

#### 5.4 The sampling algorithm

```
n_left ← n ;  m_left ← m
for i = C, C−1, ..., 1:
    if n_left = 0:
        n_i = n_{i−1} = ... = n_1 = 0 ;  stop          # every remaining count must be 0
    for k = 0, 1, ..., min(K_i, n_left):
        if k = 0:  q_k ← F_i(0) · T_{i−1}(n_left, m_left)
        else:      q_k ← F_i(k) · T_{i−1}(n_left − k, m_left − 1)     (0 if m_left = 0)
    draw k from the list q_0, q_1, ..., in that order          # section 4, one uniform
    n_i ← k ;  n_left ← n_left − k ;  if k ≥ 1: m_left ← m_left − 1
```

Some entries count as $T_{i-1}(a, b) = 0$: those with $a < 0$, $b < 0$ or $b > a$, and those beyond
what the first $i-1$ categories can reach. The list procedure divides by $\sum_k q_k$, which equals
$T_i(n_{\text{left}}, m_{\text{left}})$, so the draw has exactly the probabilities of 5.3.

Stopping early at $n_{\text{left}} = 0$ agrees with the formula. With nothing left to place, $k = 0$
is the only possible value, and its probability is $F_i(0)\, T_{i-1}(0,0) / T_i(0,0) = 1$.

**The numbers actually stored.** The code does not store $F_i(k)$ and $T_{i-1}$ themselves. It stores
$q_i(k) = F_i(k)\, e^{-\mu_i}$ and $\tilde T_{i-1} = T_{i-1}\, e^{-s_{i-1}}$ (step 4), which keeps
them inside the range of a double. For a given category $i$, every $q_k$ in the list is multiplied
by the same number $e^{-\mu_i - s_{i-1}}$. That leaves $q_k / \sum q$ unchanged. If every product
in the list comes out as 0 in floating point, the code recomputes the list from the logarithms of
the stored numbers.

#### 5.5 A worked example

Take three categories, each holding at most 2 products, with these factors:

| $k$ | 0 | 1 | 2 |
|---|---|---|---|
| $F_1(k)$ | 1 | 2 | 1 |
| $F_2(k)$ | 1 | 3 | 1 |
| $F_3(k)$ | 1 | 1 | 2 |

Suppose step 4 drew $n = 3$ and $m = 2$. The allowed partitions and their weights $F_1 F_2 F_3$ are:

| $(n_1, n_2, n_3)$ | (2,1,0) | (1,2,0) | (2,0,1) | (1,0,2) | (0,2,1) | (0,1,2) |
|---|---|---|---|---|---|---|
| weight | 3 | 2 | 1 | 4 | 1 | 6 |

The total is $T_3(3, 2) = 17$, so, for example, $P(0,1,2) = 6/17$.

The tables needed are:

- $T_1(0,0) = 1$, $T_1(1,1) = 2$ and $T_1(2,1) = 1$;
- $T_2(1,1) = 5$, $T_2(2,1) = 2$ and $T_2(3,2) = 5$, from the recursion.

**Category 3** ($\hat n = 3$, $\hat m = 2$):

| $k$ | $q_k$ | probability |
|---|---|---|
| 0 | $F_3(0)\, T_2(3,2) = 1 \cdot 5 = 5$ | $5/17$ |
| 1 | $F_3(1)\, T_2(2,1) = 1 \cdot 2 = 2$ | $2/17$ |
| 2 | $F_3(2)\, T_2(1,1) = 2 \cdot 5 = 10$ | $10/17$ |

These match the table of partitions. For example, $P(n_3 = 2) = (4 + 6)/17$.

Say $k = 2$ is drawn. Then $\hat n = 1$ and $\hat m = 1$.

**Category 2** ($\hat n = 1$, $\hat m = 1$):

| $k$ | $q_k$ | probability |
|---|---|---|
| 0 | $F_2(0)\, T_1(1,1) = 2$ | $2/5$ |
| 1 | $F_2(1)\, T_1(0,0) = 3$ | $3/5$ |

The total is $5 = T_2(1,1)$. Say $k = 1$ is drawn. Then $\hat n = 0$, so the algorithm stops with
$n_1 = 0$.

**Result:** the partition $(0, 1, 2)$, drawn with probability $\tfrac{10}{17} \cdot \tfrac{3}{5} = \tfrac{6}{17}$,
which is its probability in the partition table.

---

### Step 6. Draw which products are chosen inside each category

Step 5 fixed each category's count. Now, separately for each category $c$ with $n_c = k \ge 1$,
step 6 draws **which** $k$ of its $P_c$ products are in the basket. The categories are independent
of each other given their counts, so each is drawn on its own.

#### 6.1 The probability of a whole group

For a group $G$ of exactly $k$ products of the category,

$$
\boxed{\;P(G \mid n_c = k) = \frac{\prod_{j \in G} \tilde w_j}{\tilde A^{(P_c)}(k)}\;}
$$

Here:

- $\tilde w_j = e^{u_j(z) - M_c}$ are the scaled weights from step 3. Scaling every weight by the
  same $e^{-M_c}$ multiplies every group of size $k$ by the same $e^{-k M_c}$, which does not change
  the probabilities;
- $\tilde A^{(p)}(\ell)$ is the sum of $\prod_{j\in G}\tilde w_j$ over all groups of $\ell$
  products among the first $p$ products of the category.

#### 6.2 The group probability as a product of conditional probabilities

Describe the group by one yes/no answer per product, $x_p = 1$ if product $p$ is in $G$ and
$x_p = 0$ if not. The code decides the products from the last ($p = P_c$) to the first. By the
chain rule,

$$
P(G) = P(x_{P_c})\; P(x_{P_c - 1} \mid x_{P_c})\; \cdots\; P(x_1 \mid x_2, \dots, x_{P_c}).
$$

When product $p$ is reached, let $\ell_p = k - (x_{p+1} + \dots + x_{P_c})$ be the number of
products still to choose among products $1, \dots, p$. **The conditional probability is**

$$
\boxed{\;
P(x_p = 1 \mid x_{p+1}, \dots, x_{P_c}) = \frac{\tilde w_p\; \tilde A^{(p-1)}(\ell_p - 1)}{\tilde A^{(p)}(\ell_p)},
\qquad
P(x_p = 0 \mid x_{p+1}, \dots, x_{P_c}) = \frac{\tilde A^{(p-1)}(\ell_p)}{\tilde A^{(p)}(\ell_p)} .
\;}
$$

- **Numerator:** the groups that contain product $p$ have total weight $\tilde w_p$ times the weight
  of all ways to pick the other $\ell_p - 1$ products from $1, \dots, p-1$.
- **The two cases:** the groups without $p$ have total weight $\tilde A^{(p-1)}(\ell_p)$.
- **Denominator:** the two cases add up to $\tilde A^{(p)}(\ell_p)$, because
  $\tilde A^{(p)}(\ell) = \tilde w_p \tilde A^{(p-1)}(\ell - 1) + \tilde A^{(p-1)}(\ell)$.

**The product of the conditionals is the group probability.** At product $p$ the denominator is
$\tilde A^{(p)}(\ell_p)$. The numerator always contains $\tilde A^{(p-1)}(\ell_{p-1})$, which is the
denominator at product $p-1$. Multiplying over $p = P_c, \dots, 1$, these cancel, and what is left is

$$
\frac{\prod_{p:\ x_p = 1} \tilde w_p \;\cdot\; \tilde A^{(0)}(0)}{\tilde A^{(P_c)}(k)}
= \frac{\prod_{j \in G}\tilde w_j}{\tilde A^{(P_c)}(k)} ,
$$

which is the formula of 6.1.

#### 6.3 The sampling algorithm

```
if k = P_c:  choose every product of the category ; done
build Ã^(p)(ℓ) for p = 0..P_c, ℓ = 0..k:
    Ã^(0)(0) = 1,  Ã^(0)(ℓ) = 0 for ℓ ≥ 1
    Ã^(p)(0) = 1,  Ã^(p)(ℓ) = Ã^(p−1)(ℓ) + w̃_p · Ã^(p−1)(ℓ−1)   for 1 ≤ ℓ ≤ min(p, k)
    Ã^(p)(ℓ) = 0 for ℓ > p
need ← k
for p = P_c, P_c−1, ..., 1, while need > 0:
    if p = need:  choose products 1..p ;  need ← 0 ;  stop      # probability 1, see below
    π ← w̃_p · Ã^(p−1)(need − 1) / Ã^(p)(need)                  # clipped to at most 1
    draw a uniform U ;  if U < π:  choose product p ;  need ← need − 1
```

The variable `need` is $\ell_p$. Since $P(U < \pi) = \pi$, each product is decided with exactly the
conditional probability of 6.2. When `need` $= p$, all remaining products must be chosen: the
formula gives $\pi = \tilde w_p \tilde A^{(p-1)}(p-1) / \tilde A^{(p)}(p) = 1$ for each of them, so
taking them all at once is the same draw.

#### 6.4 A worked example

Take a category with three products of weights $\tilde w = (1, 2, 3)$, and $k = 2$. The groups are:

| group | $\{1,2\}$ | $\{1,3\}$ | $\{2,3\}$ |
|---|---|---|---|
| weight | 2 | 3 | 6 |

The total is $\tilde A^{(3)}(2) = 11$. The other tables needed are $\tilde A^{(2)}(1) = 3$,
$\tilde A^{(2)}(2) = 2$, $\tilde A^{(1)}(0) = 1$ and $\tilde A^{(1)}(1) = 1$.

1. **Product 3** (need $= 2$): $\pi = 3 \cdot \tilde A^{(2)}(1) / 11 = 9/11$. Say it is chosen; need becomes 1.
2. **Product 2** (need $= 1$): $\pi = 2 \cdot \tilde A^{(1)}(0) / \tilde A^{(2)}(1) = 2/3$. Say it is chosen; need becomes 0.

**Result:** the group $\{2, 3\}$, drawn with probability $\tfrac{9}{11} \cdot \tfrac{2}{3} = \tfrac{6}{11}$,
which is its weight divided by the total.

**The basket** is the union of the groups chosen in step 6.

---

## 6. Summary

| step | drawn or computed | distribution | algorithm | random numbers used |
|---|---|---|---|---|
| 1 | taste $z$ | $\mathcal{N}(0, I_{16})$ | `torch.randn` | 16 normals (torch) |
| 2 | $u_j(z)$, $p_c(z)$ | none | formulas of section 2 | none |
| 3 | $\tilde w_p$, $G_c(k) = \log F_c(k)$ | none | scaled table, products added one at a time | none |
| 4 | $(n, m)$ | $\propto T_C(n,m)\, e^{-\rho_0(m)}$, $n \le L$ | certified limit $L$; scaled tables $\tilde T_i$; one list draw | 1 uniform |
| 5 | $n_C, \dots, n_1$ | $\prod_c F_c(n_c) / T_C(n,m)$ | backward, weights $\tilde T_{i-1}(\cdot)\,q_i(k)$; one list draw per category | 1 uniform per category reached |
| 6 | group $G_c$ in each category | $\prod_{j\in G}\tilde w_j / \tilde A_c(n_c)$ | prefix tables $\tilde A^{(p)}$; backward, keep $p$ with probability $\pi_p$ | 1 uniform per product examined |

Steps 1 to 6 give a basket with probability $P(S)$, with two exceptions:

- the size limit of step 4, which changes the probability of any event by at most $10^{-8}$;
- floating-point rounding: the $2^{-53}$ spacing of the uniforms, and values below the smallest
  double being stored as 0.

No step is repeated until it settles, and no distribution is replaced by an approximation.

---

## 7. Generating with part of the basket known

Given known products $K$, the taste has density

$$
p(z \mid K) \;\propto\; \mathcal{N}(z)\, \frac{e^{E(K \mid z)}}{Z(z)} .
$$

The code handles this density as follows:

1. The density has a single peak $\hat z$, which the code finds by gradient ascent with backtracking.
2. It draws candidate tastes from the mix $0.8\, \mathcal{N}(\hat z, 1.2^2\Sigma) + 0.2\, \mathcal{N}(\hat z, I)$,
   where $\Sigma$ is the spread of $p(z \mid K)$ at the peak.
3. It weights each candidate $z^{(i)}$ by
   $\mathcal{N}(z^{(i)}) e^{E(K \mid z^{(i)})} / \big(Z(z^{(i)})\, q(z^{(i)})\big)$, where $q$ is the
   density of that mix.

The repository uses these weighted tastes only to **score** each product that could be added
(`research/evaluate_mixture.py recommend`). It does not generate a full basket around known
products. Doing so would also change steps 4 to 6, which would have to keep the known products in
the basket.

---

## 8. Cost

On the Dunnhumby model, which offers 5,455 products per trip, one basket takes about 5–7 ms:

- about 2–3 ms to build the trip's context;
- about 3 ms for step 2;
- under 1 ms for steps 3–6.

In the synthetic world (522 products) it takes about 1 ms. These times were measured on 200
single trips.
