# How one basket is generated

This document describes, step by step, how the model generates one shopping basket for one trip.
For every step it gives the exact probability distribution and the exact procedure used to draw from
it. Every symbol is defined before it is used. The code is in `v4/mixture.py` (function `generate`),
`v4/sampling.py` (function `sample_baskets`) and `v4/native/sampler.cpp` (function
`sample_slots_cpp`).

---

## 1. What is given, and what is produced

**Given (the context of the trip):**

- the household `h` that makes the trip;
- the store `s` and the week `w` of the trip;
- the products the store offers on that trip, numbered `j = 1, ..., J_t`;
- for each offered product: its price, its regular price, whether it is on display, whether it is in
  the mailer, and the household's purchase history up to that trip;
- the fitted model parameters.

**Produced:** a basket `S`, a set of distinct offered products. A basket contains at least one
product and at most `N` products (`N = 120` on the Dunnhumby data, `N = 80` in the synthetic world).

**Words used below:**

- *category*: the group a product belongs to (on Dunnhumby, a commodity such as "pasta sauce").
  Every product belongs to exactly one category.
- `n_c`: the number of products of category `c` in the basket.
- `n`: the number of products in the basket, `n = n_1 + n_2 + ... + n_C`.
- `m`: the number of categories present in the basket (the number of categories with `n_c >= 1`).
- `R`: the largest number of products of one category a basket may hold (`R = 40`).

---

## 2. The probability of a basket

The model gives every possible basket a probability. Generation draws one basket from exactly
this probability. It helps to see the whole formula first and then the steps that draw from it.

### 2.1 The taste of the trip

Each trip has a hidden *taste*, a list of `r = 16` numbers:

```
z = (z_1, z_2, ..., z_16)
```

The taste is not observed. Before anything is known about the basket, its distribution is the
standard normal distribution in 16 dimensions: the 16 numbers are independent, and each has mean 0
and variance 1.

```
density of z:   N(z) = (2π)^(-16/2) · exp( -(z_1² + z_2² + ... + z_16²) / 2 )
```

### 2.2 The base attractiveness of a product on this trip

For each offered product `j` the model computes a number `b_j`, the product's attractiveness on this
trip before the taste is taken into account. It is a sum of terms:

```
b_j =   λ_j                                   popularity of the product
      + θ_h,size + Σ_k θ_h,k · a_j,k          the household's lasting taste (a_j,k: product loadings)
      - price term_j                          see below
      + v_j · display_j + q_j · mailer_j       promotions (display_j, mailer_j are 0 or 1)
      + Σ_k μ_j,k · δ_w,k                     season (δ_w: numbers for week w)
      + Σ_k ζ_j,k · ξ_s,k                     store (ξ_s: numbers for store s)
      + history_j - o_t                       purchase history, minus a trip-wide offset o_t
      + Σ_k θ_h,k · e_c(j),k                  the household's taste for the product's category
      + ι_t,c(j)                              within-category competition (see below)
```

`c(j)` is the category of product `j`.

**Price term.** Let `d_j` be the logarithm of product `j`'s price relative to its usual price, and
`d̄` the average of `d_j` over all products offered on the trip. Let `x_j` be the depth of the
product's discount (the logarithm of price over regular price, centred). Then

```
price term_j =  g_h,j · ( d̄ + κ · (d_j - d̄) )
              + c · (d_j - d̄)
              + g_h,j · (κ_disc - κ) · x_j
              + c_disc · x_j
```

Here `g_h,j >= 0` is the household's sensitivity to the price of product `j` (it is built from 8
numbers for the household and 8 for the product: `g_h,j = Σ_k soft(γ_h,k) · soft(β_j,k)`, with
`soft(y) = log(1 + e^y)`), and `κ`, `κ_disc`, `c`, `c_disc` are single numbers shared by everyone.
The term `c` exists only in models fitted with `--price-common`.

**Within-category competition.** For the products of one category, `ι` moves part of the effect of
prices and promotions from the category as a whole to the choice between its products:

```
ι_t,c = (λ_c - 1) · [ log Σ_{i in c} exp(b'_i)  -  log Σ_{i in c} exp(b''_i) ]
```

where `b'_i` is the attractiveness without `ι` and the history offset, `b''_i` is the same with
regular prices and no promotions, and `0 < λ_c < 1` is estimated for each category.

None of the terms in `b_j` depends on the taste or on the rest of the basket.

### 2.3 The attractiveness of a product given the taste

Each product has a *taste profile* `φ_j`, a list of 16 numbers. Given the taste `z`, the
attractiveness of product `j` is

```
u_j(z) = b_j - (φ_j,1² + ... + φ_j,16²) / 2 + (z_1 φ_j,1 + ... + z_16 φ_j,16)
```

and its *weight* is `w_j(z) = exp(u_j(z))`. A taste `z` that lines up with `φ_j` raises the weight
of product `j`. Products with similar profiles are raised together, and this is what makes some
products tend to appear in the same basket.

### 2.4 The score of a basket given the taste

Each category `c` has a profile `ψ_c` (16 numbers), and three kinds of penalty for buying several
products of the same category:

- `ρ_c`, charged once for every pair of products of the category in the basket;
- `f_c(k)`, a free amount for `k = 2`, `k = 3` and `k >= 4` products of the category
  (`f_c(0) = f_c(1) = 0`);
- `p_c(z)`, charged once for every product of the category beyond the first:

```
p_c(z) = (z_1 ψ_c,1 + ... + z_16 ψ_c,16) - (ψ_c,1² + ... + ψ_c,16²) / 2
```

Finally, `ρ_0(m)` is a size penalty that depends on the number of categories present
(`ρ_0(1) = 0`).

The score of a basket `S`, given the taste `z`, is

```
E(S | z) =   Σ_{j in S} u_j(z)
           - Σ_c [ ρ_c · n_c (n_c - 1) / 2  +  f_c(n_c)  +  max(n_c - 1, 0) · p_c(z) ]
           - ρ_0(m)
```

### 2.5 The probability of a basket

Given the taste, the probability of a basket is proportional to the exponential of its score:

```
P(S | z) = exp( E(S | z) ) / Z(z)
```

where `Z(z)` is the sum of `exp(E(S' | z))` over every allowed basket `S'` (so that the
probabilities add up to 1).

Since the taste is not observed, the probability of a basket is the average over all tastes:

```
P(S) = ∫ N(z) · P(S | z) dz
```

**This is the distribution the generator draws from.** Drawing `z` from `N(z)` and then `S` from
`P(S | z)` gives a basket whose probability is exactly `P(S)`. The rest of this document is about how
to draw `S` from `P(S | z)` without listing all possible baskets, of which there are far too many.

---

## 3. A useful way to write P(S | z)

The score in 2.4 has a convenient shape. Apart from `ρ_0(m)`, it is a sum of separate parts, one per
category. Inside one category, it depends on which products are chosen only through the sum of their
`u_j(z)`. This lets us split the draw into small steps.

**The category table.** For a category `c` and a number `k` of its products, define

```
A_c(k) = sum, over every group of k different products of category c,
         of the product of their weights w_j(z)
```

So `A_c(0) = 1`, `A_c(1)` is the sum of the weights of the category's products, `A_c(2)` is the sum
over all pairs of `w_i · w_j`, and so on. Then define the *category factor*

```
F_c(k) = A_c(k) · exp( - ρ_c · k (k - 1) / 2  -  f_c(k)  -  max(k - 1, 0) · p_c(z) )
```

`F_c(k)` is the total score weight of all ways to put exactly `k` products of category `c` into the
basket. In particular `F_c(0) = 1`.

**The probability of the counts.** Sum `exp(E(S | z))` over all baskets with the same counts
`(n_1, ..., n_C)`. Within each category the sum over the choice of products gives `A_c(n_c)`, and
everything else depends only on the counts. So

```
P(n_1, ..., n_C | z) = F_1(n_1) · F_2(n_2) · ... · F_C(n_C) · exp( -ρ_0(m) ) / Z(z)
```

**The products given the counts.** Given that category `c` holds exactly `k` products, the
probability of a particular group `G` of `k` products is

```
P(G | n_c = k, z) = (product of w_j(z) over j in G) / A_c(k)
```

and the categories are independent of each other once their counts are fixed.

So a basket can be drawn in three parts: first the counts, then the products inside each category.
The counts are drawn in two steps, because `ρ_0(m)` ties the categories together.

---

## 4. The steps

The basket is drawn in six steps. Steps 2 and 3 only compute numbers. Steps 1, 4, 5 and 6 draw at
random.

All random numbers come from one seed. The torch library's random generator gives the seed, and the
C++ code turns it into a stream of uniform numbers between 0 and 1 (with the SplitMix64 generator).
The same seed always gives the same basket.

**One procedure used in several steps (drawing from a list of probabilities).** To draw one item
from a list of items with probabilities proportional to non-negative numbers `q_1, ..., q_L`:

1. compute the total `Q = q_1 + ... + q_L`;
2. draw a uniform number `U` between 0 and 1;
3. go through the list, adding up `q_1, q_2, ...`, and stop at the first item `i` where the running
   sum reaches `U · Q`.

Item `i` is then chosen with probability exactly `q_i / Q`.

### Step 1. Draw the taste

- **Distribution:** `z ~ N(0, I)`, the standard normal distribution in 16 dimensions (section 2.1).
- **Procedure:** draw 16 independent standard normal numbers with the torch library's normal
  generator (`torch.randn`).

### Step 2. Compute the attractiveness and weight of every offered product

- **Distribution:** none. This step only computes numbers.
- **Procedure:** compute `b_j` (section 2.2), then `u_j(z)` and `w_j(z) = exp(u_j(z))`
  (section 2.3), for every product offered on the trip.

### Step 3. Compute the category tables

- **Distribution:** none. This step only computes numbers.
- **Procedure:** for every category `c` offered on the trip, compute `A_c(k)` for
  `k = 0, 1, ..., K_c`, where `K_c` is the smaller of `R`, `N` and the number of the category's
  offered products. The products are added one at a time:

  ```
  start:  A(0) = 1,  A(k) = 0 for k >= 1
  for each product j of the category, with weight w_j:
      for k from K_c down to 1:
          A(k) = A(k) + A(k - 1) · w_j
  ```

  After all products, `A(k)` is the sum over all groups of `k` products of the product of their
  weights (each group is counted exactly once, because each product is either in it or not). Then
  form `F_c(k)` as in section 3.

  For numerical safety the code works with logarithms and divides all weights by the largest one
  before the products are multiplied. This changes no probability.

### Step 4. Draw the basket size and the number of categories

- **Distribution:** the pair `(n, m)` (products in the basket, categories present) has
  probability

  ```
  P(n, m | z) = T(n, m) · exp( -ρ_0(m) ) / Z(z)
  ```

  where `T(n, m)` is the sum of `F_1(n_1) · ... · F_C(n_C)` over all count lists with `n` products
  in total spread over exactly `m` categories. This follows by grouping the formula of section 3 by
  `(n, m)`.

- **Procedure:**
  1. Build the table `T` one category at a time. Let `T_i(n, m)` be the same sum over the first `i`
     categories only. Start from `T_0(0, 0) = 1` (and 0 elsewhere). Adding category `i` either adds
     none of its products (the count of categories present does not change) or `k >= 1` of them
     (one more category is present):

     ```
     T_i(n, m) = T_{i-1}(n, m) · F_i(0)  +  Σ_{k = 1..K_i} T_{i-1}(n - k, m - 1) · F_i(k)
     ```

     After the last category, `T = T_C`. The intermediate tables `T_0, ..., T_C` are kept for step 5.
  2. Draw one pair `(n, m)`, with `1 <= m <= n`, with probabilities proportional to
     `T(n, m) · exp(-ρ_0(m))`, using the procedure for a list of probabilities.

- **The size limit.** Building `T` up to `n = N` for every trip is slow, and very large baskets are
  usually almost impossible. So the code first chooses a limit `L <= N` for the trip (at least 32),
  and builds `T` only up to `n = L`. It keeps this limit only if it can prove, from upper bounds
  computed for that trip, that all baskets with more than `L` products together carry at most a
  fraction `10^-8` of the probability. If it cannot prove this, it builds `T` up to `N`. So the
  basket is drawn either from the exact distribution, or from the exact distribution restricted to at
  most `L` products, and the two differ by at most `10^-8` in the probability of any event.

### Step 5. Draw how many products come from each category

- **Distribution:** given the drawn `(n, m)`, the counts `(n_1, ..., n_C)` have probability
  proportional to `F_1(n_1) · ... · F_C(n_C)`, among the count lists with `n` products over `m`
  categories.

- **Procedure:** draw the counts one category at a time, from the last category to the first. Keep
  `n_rest` and `m_rest`, the products and categories still to be placed; at the start
  `n_rest = n` and `m_rest = m`. For category `i`, the probability that it holds `k` products is

  ```
  P(n_i = k | n_rest, m_rest) ∝  T_{i-1}(n_rest, m_rest) · F_i(0)        if k = 0
                                 T_{i-1}(n_rest - k, m_rest - 1) · F_i(k) if k >= 1
  ```

  Draw `k` with the procedure for a list of probabilities. Then set `n_rest = n_rest - k`, and, if
  `k >= 1`, `m_rest = m_rest - 1`. After the first category, `n_rest = 0` and `m_rest = 0`.

  **Why this is right.** `T_{i-1}(n', m')` is the total weight of all ways the first `i - 1`
  categories can hold `n'` products over `m'` categories. So the formula gives each choice of `k`
  its share of the total weight of all count lists that are still possible. Applying this category by
  category draws the whole count list with exactly the probability above.

### Step 6. Draw which products are chosen inside each category

- **Distribution:** for each category `c` with `n_c = k >= 1`, a group `G` of exactly `k` different
  products of the category, with probability

  ```
  P(G) = (product of w_j(z) over j in G) / A_c(k)
  ```

  The categories are drawn independently of each other.

- **Procedure:** number the category's products `1, ..., P`. Let `A_p(k)` be the table of step 3
  computed from the first `p` products only (the code keeps all of them). Go through the products
  from the last (`p = P`) to the first, with `need = k` products still to choose:

  ```
  for p from P down to 1, while need > 0:
      if need == p:                      # all remaining products must be chosen
          choose products 1, ..., p;  need = 0
      else:
          probability = w_p · A_{p-1}(need - 1) / A_p(need)
          draw U uniform between 0 and 1
          if U < probability:  choose product p;  need = need - 1
  ```

  If `k` equals the number of products of the category, all of them are chosen directly.

  **Why this is right.** `A_p(need)` is the total weight of all groups of `need` products among the
  first `p`. Those groups either contain product `p` (total weight `w_p · A_{p-1}(need - 1)`) or do
  not (total weight `A_{p-1}(need)`). So the line `probability = ...` gives product `p` its exact
  share, and multiplying these shares over the products gives exactly `P(G)`.

**The basket** is the set of all products chosen in step 6.

---

## 5. Summary of the steps

| step | what is drawn or computed | distribution | procedure |
|---|---|---|---|
| 1 | the taste `z` (16 numbers) | standard normal in 16 dimensions | 16 independent standard normal draws |
| 2 | attractiveness `u_j(z)` and weight `w_j(z)` of every offered product | none (computed) | formulas of sections 2.2–2.3 |
| 3 | category tables `A_c(k)` and factors `F_c(k)` | none (computed) | add the products one at a time |
| 4 | basket size `n` and number of categories `m` | `∝ T(n, m) · exp(-ρ_0(m))` | build `T` category by category; one draw from a list |
| 5 | count `n_c` of each category | `∝ F_1(n_1) ··· F_C(n_C)` given `(n, m)` | one draw per category, from the last to the first |
| 6 | which products inside each category | group `G` with probability `∝` product of its weights | go through the products, choosing each with its exact share |

Steps 1 to 6 together give a basket with probability exactly `P(S) = ∫ N(z) P(S | z) dz` (up to
the `10^-8` size limit of step 4). No step is repeated until it settles, and no step uses an
approximation of the distribution.

---

## 6. Generating with part of the basket known

When some products of the basket are already known (for example, to suggest what else the shopper
will add), what is known about the taste changes. Its distribution given the known products `K` is

```
density of z given K  ∝  N(z) · exp( E(K | z) ) / Z(z)
```

This distribution has a single peak. The code finds the peak by climbing it step by step. It then
draws candidate tastes around the peak (from a normal distribution 1.2 times as wide as the peak,
mixed with a standard-width normal around the peak), and weights each candidate by the density
given `K` divided by the density it was drawn from.

The repository uses these weighted tastes to **score** each product that could be added
(`research/evaluate_mixture.py recommend`). It does not generate a full basket around known
products. Doing so would also change steps 4 to 6, which would have to keep the known products in
the basket.

---

## 7. Cost

On the Dunnhumby model (5,455 products offered per trip), one basket takes about 5–7 milliseconds:
about 2–3 ms to build the trip's context, 3 ms for step 2, and under 1 ms for steps 3–6. In the
synthetic world (522 products) it takes about 1 millisecond. These times were measured on 200
single trips.
