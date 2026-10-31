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

- **Distribution:** given the drawn $(n, m)$,

  $$
  P(n_1, \dots, n_C \mid n, m, z) = \frac{\prod_{c=1}^{C} F_c(n_c)}{T_C(n, m)},
  $$

  over the count lists with $\sum_c n_c = n$, exactly $m$ categories with $n_c \ge 1$, and $n_c \le K_c$.

- **Algorithm.** Keep two numbers: $\hat n$, the products still to place, and $\hat m$, the categories
  still to fill. Start with $\hat n \leftarrow n$ and $\hat m \leftarrow m$. Then for $i = C, C-1, \dots, 1$:
  1. If $\hat n = 0$: set $n_i = n_{i-1} = \dots = n_1 = 0$ and stop.
  2. For $k = 0, 1, \dots, \min(K_i, \hat n)$, compute the weight

     $$
     \omega(0) = \tilde T_{i-1}(\hat n,\, \hat m)\; q_i(0),
     \qquad
     \omega(k) = \tilde T_{i-1}(\hat n - k,\, \hat m - 1)\; q_i(k) \quad (k \ge 1),
     $$

     where:
     - $\tilde T_{i-1}$ and $q_i$ are the numbers stored in step 4;
     - an entry of $\tilde T_{i-1}$ with first index above $d_{i-1}$, or second index above
       $\min(\text{first index}, c_{i-1})$, is 0;
     - $\omega(k) = 0$ for $k \ge 1$ when $\hat m = 0$.
  3. If every $\omega(k)$ is 0 because the products underflowed, recompute the same weights in logs:

     $$
     \log\omega(0) = \log \tilde T_{i-1}(\hat n, \hat m),
     \qquad
     \log\omega(k) = \log \tilde T_{i-1}(\hat n - k, \hat m - 1) + G_i(k),
     $$

     and replace each $\omega(k)$ by $e^{\log\omega(k) - \max_{k'} \log\omega(k')}$. These are the
     earlier weights multiplied by the common factor $e^{\mu_i - \max}$.
  4. Draw $k$ from the list $\omega(0), \omega(1), \dots, \omega(\min(K_i, \hat n))$, in that order,
     with the list procedure, using one uniform from the trip's stream.
  5. Set $n_i \leftarrow k$, $\hat n \leftarrow \hat n - k$, and, if $k \ge 1$, $\hat m \leftarrow \hat m - 1$.

  At the end $\hat n = \hat m = 0$. The code checks this and reports an error otherwise.

- **Why this is right.**
  1. **The weights are proportional to the exact ones.** $\tilde T_{i-1} = T_{i-1} e^{-s_{i-1}}$ and
     $q_i(k) = F_i(k) e^{-\mu_i}$, so $\omega(k) = e^{-s_{i-1} - \mu_i} \times \Omega(k)$, where

     $$
     \Omega(0) = T_{i-1}(\hat n, \hat m)\, F_i(0), \qquad \Omega(k) = T_{i-1}(\hat n - k, \hat m - 1)\, F_i(k).
     $$

     The factor $e^{-s_{i-1} - \mu_i}$ is the same for every $k$, so the draw returns $k$ with
     probability $\Omega(k) / \sum_{k'} \Omega(k')$.
  2. **The total is the next table's entry.** By the recursion of section 3,
     $\sum_{k'} \Omega(k') = T_i(\hat n, \hat m)$. So

     $$
     P(n_i = k \mid \text{what was drawn for } i+1..C) = \frac{T_{i-1}(\hat n - k,\ \hat m - [k \ge 1])\, F_i(k)}{T_i(\hat n, \hat m)} .
     $$

  3. **This is the exact conditional.** Take the true distribution of the counts, conditioned on
     $n_C, \dots, n_{i+1}$ already being drawn. The first $i$ categories must hold $\hat n$ products in
     $\hat m$ categories, and their count list has probability $\prod_{c \le i} F_c(n_c) / T_i(\hat n, \hat m)$.
     Adding this over $n_1, \dots, n_{i-1}$ with $n_i = k$ fixed gives exactly the fraction in 2.
  4. **Multiplying the steps gives the target.** Write $(\hat n_i, \hat m_i)$ for the values when
     category $i$ is reached, so that $(\hat n_C, \hat m_C) = (n, m)$ and $(\hat n_0, \hat m_0) = (0, 0)$.
     Then

     $$
     \prod_{i=1}^{C} \frac{T_{i-1}(\hat n_{i-1}, \hat m_{i-1})\, F_i(n_i)}{T_i(\hat n_i, \hat m_i)}
     = \frac{T_0(0, 0)}{T_C(n, m)} \prod_{i=1}^{C} F_i(n_i)
     = \frac{\prod_i F_i(n_i)}{T_C(n, m)} ,
     $$

     because each $T_{i-1}(\hat n_{i-1}, \hat m_{i-1})$ in a numerator cancels the denominator of the
     next category down. When the loop stops early at $\hat n = 0$, the remaining categories must
     hold 0 products; their factors are $F(0) = 1$, and $T_{i}(0, 0) = 1$, so they contribute 1.

### Step 6. Draw which products are chosen inside each category

- **Distribution:** independently for each category $c$ with $n_c = k \ge 1$, a group $G$ of $k$
  of its products, with

  $$
  P(G) = \frac{\prod_{j \in G} w_j(z)}{A_c(k)} = \frac{\prod_{j \in G} \tilde w_j}{\tilde A_c(k)} .
  $$

  The second form holds because $\prod_{j \in G} w_j = e^{k M_c} \prod_{j \in G}\tilde w_j$ and
  $A_c(k) = e^{k M_c}\tilde A_c(k)$.

- **Algorithm**, for each category with $k = n_c \ge 1$. Categories are taken in the order they are
  stored (trip by trip, by increasing category id), and the uniforms come from the main stream.
  1. If $k = P_c$, choose every product of the category. No random number is used.
  2. Otherwise, **build the prefix tables** from the weights $\tilde w_p$ kept in step 3, only up to $k$.
     For $p = 0, 1, \dots, P_c$ and $\ell = 0, 1, \dots, k$:

     $$
     \tilde A^{(0)}(0) = 1,\quad \tilde A^{(0)}(\ell) = 0 \ (\ell \ge 1);
     \qquad
     \tilde A^{(p)}(0) = 1,\quad
     \tilde A^{(p)}(\ell) = \tilde A^{(p-1)}(\ell) + \tilde w_p\, \tilde A^{(p-1)}(\ell - 1)\ \ (1 \le \ell \le \min(p, k)),
     $$

     and $\tilde A^{(p)}(\ell) = 0$ for $\ell > p$. All $P_c + 1$ rows are kept. By the induction
     argument of step 3, $\tilde A^{(p)}(\ell)$ is the sum of $\prod_{j \in G}\tilde w_j$ over the groups
     of $\ell$ products among the first $p$.
  3. **Go through the products from last to first.** Set $\text{need} \leftarrow k$. For
     $p = P_c, P_c - 1, \dots, 1$, while $\text{need} > 0$:
     1. If $p = \text{need}$, choose products $1, \dots, p$, set $\text{need} \leftarrow 0$, and stop.
     2. Otherwise compute

        $$
        \pi_p = \min\!\Big(1,\ \frac{\tilde w_p\, \tilde A^{(p-1)}(\text{need} - 1)}{\tilde A^{(p)}(\text{need})}\Big),
        $$

        with $\pi_p = 0$ if the denominator is 0.
     3. Draw one uniform $U$. If $U < \pi_p$, choose product $p$ and set $\text{need} \leftarrow \text{need} - 1$.

     At the end $\text{need} = 0$. The code checks this and reports an error otherwise.

- **Why this is right.**
  1. **The marginal of the last product.** Every group of $\ell$ products among the first $p$
     either contains product $p$ or not, so

     $$
     \tilde A^{(p)}(\ell) = \underbrace{\tilde w_p\, \tilde A^{(p-1)}(\ell - 1)}_{\text{groups containing } p}
     + \underbrace{\tilde A^{(p-1)}(\ell)}_{\text{groups without } p} .
     $$

     Under $P(G) \propto \prod_{j\in G}\tilde w_j$ on groups of size $\ell$ among the first $p$, the
     probability that $p \in G$ is therefore $\tilde w_p \tilde A^{(p-1)}(\ell-1) / \tilde A^{(p)}(\ell)$.
     This is $\pi_p$ with $\ell = \text{need}$. The $\min(1, \cdot)$ only guards against rounding:
     the exact ratio is at most 1.
  2. **What is left is the same problem, one product smaller.**
     - Given $p \in G$, the rest of $G$ is a group of $\ell - 1$ products among the first $p - 1$, with
       probability $\prod \tilde w / \tilde A^{(p-1)}(\ell - 1)$.
     - Given $p \notin G$, $G$ is a group of $\ell$ products among the first $p - 1$, with probability
       $\prod \tilde w / \tilde A^{(p-1)}(\ell)$.

     So the loop, which carries on with $\text{need} - 1$ or $\text{need}$, draws each later product
     from its exact conditional.
  3. **The forced case.** When $p = \text{need}$, the only group of size $p$ among $p$ products is all of
     them. The formula agrees: it would give $\pi = \tilde w_p \tilde A^{(p-1)}(p-1) / \tilde A^{(p)}(p) = 1$.
  4. **Multiplying the steps gives the target.** For a group $G$, the probability that the loop
     outputs $G$ is a product over $p$ of $\pi_p$ (for $p \in G$) or $1 - \pi_p$ (for $p \notin G$).
     Here $1 - \pi_p = \tilde A^{(p-1)}(\ell)/\tilde A^{(p)}(\ell)$. Each $\tilde A^{(p-1)}(\cdot)$ in a
     numerator cancels the denominator of the next product down. What is left is
     $\prod_{j \in G}\tilde w_j \cdot \tilde A^{(0)}(0) / \tilde A^{(P_c)}(k) = P(G)$.

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
