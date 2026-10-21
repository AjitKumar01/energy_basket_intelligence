# The stress world

The stress world is a synthetic retailer whose baskets come from a procedural shopping story,
not from the Version-4 model. It has no energy function, pair matrix, category penalty or size
potential. Two uses:

1. **Known causal truth.** Because the generator is known, the true effect of a price change is
   computed by re-simulating the same shoppers with the changed price. A model's
   counterfactuals can be scored against it, which real data never allows.
2. **Misspecification.** The model has to fit mechanisms it does not contain (missions, nested
   choice, loyalty, stockpiling, loss aversion). A good fit is not an artefact of fitting
   the model to its own output.

Everything below is what `synthetic/generate_stress_world.py` does (seed 20260924). Section 9
explains how the world is scored.

## 1. Size and time

| | |
|---|---|
| households | 3,000 simulated; 2,802 with trips |
| stores | 15 |
| catalogue | 6 departments × 8 categories = 48 categories, 478 products (476 bought in training) |
| weeks | 52: training 1–32, validation 33–42, test 43–52 |
| trips | 42,231 non-empty baskets (26,140 training, 8,050 validation, 8,041 test) |
| basket size | mean 9.05, median 8, 90th percentile 15, 99th 22, largest 34 |

## 2. Catalogue

Each category $c$ has 1–4 subcategories, and each subcategory has $1 + \min(\max(Z, 1), 12)$
products with $Z \sim \mathrm{Zipf}(1.8)$, so sizes are uneven with a long tail. Product $j$ has:

$$
q_j = -1.1\,\log \pi_j + \epsilon_j,\quad \epsilon_j\sim N(0, 0.3^2)
\qquad \text{(quality; } \pi_j \text{ a random rank within the category)}
$$

$$
a_j \sim N(0, I_3) \quad (\text{attributes: premium, organic, value}), \qquad
\bar p_j = \operatorname{round}\big(e^{N(1,\,0.6^2)}, 2\big) \quad (\text{base price}).
$$

## 3. Households

- **Segment** $g_h \in \{1..4\}$. Mission preferences $\pi_h \sim \mathrm{Dirichlet}(20\,\pi_{g_h} + 0.1)$
  over the eight missions, with $\pi_g \sim \mathrm{Dirichlet}(0.7)$.
- **Shopping intensity** $\nu_h \sim \mathrm{Gamma}(2, 0.6)$ (mean number of extra missions per trip).
- **Visit rate** $v_h \sim \mathrm{Beta}(1.1, 3)$, a heavy tail of rare shoppers; a home store.
- **Price sensitivity** $\eta_h = e^{N(0.4,\,0.5^2)}$.
- **Taste** over the three attributes, $\tau_h^0 \sim N(0, 0.6^2 I_3)$.
- **Idiosyncratic liking** $\varepsilon_{hj} \sim N(0, 0.5^2)$ for a random 15% of (household, product)
  pairs, 0 otherwise.

## 4. Prices, promotions and availability

Prices are chain-wide and weekly:

$$
p_{jw} = \operatorname{round}\Big(\bar p_j\, e^{\,\omega_{jw}}\,(0.75)^{\mathbb 1[\mathrm{promo}_{jw}]},\ 2\Big),
\qquad \omega_{jw} \sim N(0, 0.08^2),\quad \mathrm{promo}_{jw} \sim \mathrm{Bernoulli}(0.07).
$$

A promotion is a 25% cut with a display and advertising, in every store. Products are stocked
in 85% of stores from week 1. A tenth of the products launch later in some stores, which a model
sees only through a retail sales feed. 3% of (product, store, week) cells are out of stock
without any record.

## 5. Trips

In each week household $h$ shops with probability $v_h$, at its home store with probability 0.85
(otherwise a random store), on a random day of the week. Trips are simulated in time order.
Each trip draws from its own random stream, seeded by the trip's id; section 7 depends on this.

**Weekly dynamics** (for everyone):

$$
\tau_{h,w+1} = 0.9\,\tau_{hw} + 0.1\,\tau_h^0 + \xi_{hw},\quad \xi_{hw}\sim N(0, 0.08^2 I_3),
\qquad \text{pantry}_{hc} \leftarrow \max(\text{pantry}_{hc} - 1, 0).
$$

**Utility** of product $j$ for household $h$ in week $w$, with loss-averse reference pricing
against the base price:

$$
U_{hjw} = q_j + a_j^\top \tau_{hw} + \varepsilon_{hj}
 - \eta_h\, \ell\!\left(\log\frac{p_{jw}}{\bar p_j}\right) + 0.5\,\mathrm{promo}_{jw},
\qquad
\ell(x) = \begin{cases} 2x & x > 0 \\ x & x \le 0 \end{cases}
$$

so a price rise above the reference hurts twice as much as an equal cut helps.

**Missions.** The trip has $1 + \mathrm{Poisson}(\nu_h)$ missions, each drawn from $\pi_h$:
breakfast, lunch, dinner recipe, snacks, cleaning, baby, party, or top-up. Each mission type
has a fixed template: 4–8 categories from one or two departments plus 2 random ones, each with
a need probability in $[0.2, 0.8]$, and 0–2 **recipes**, which are sets of 3 categories. Then:

- each template category becomes a **need** with its probability, unless the household's pantry
  for that category is still stocked;
- each recipe is added **all or none** with the template's recipe probability (0.2–0.5). This
  makes complementarity higher-order: three categories at once, not pairs.

**Choice for each need** (category $c$, products available in the store that week): a nested
logit over subcategories with nest parameter $\mu = 0.5$, after adding a loyalty bonus for the
product last bought in $c$:

$$
u_j = U_{hjw} + 1.2\,\mathbb 1[j = \text{last}_{hc}], \qquad
\mathrm{IV}_s = \mu \log \sum_{j \in s} e^{u_j/\mu}, \qquad
P(s) \propto e^{\mathrm{IV}_s}, \qquad P(j \mid s) \propto e^{u_j/\mu}.
$$

After the choice:

- **loyalty** moves to the chosen product;
- if it was on promotion, with probability 0.5 the household **stockpiles**: the category's
  pantry is set to 2–4 weeks, during which it is not needed;
- with probability 0.1 a **second product** of the category is added, chosen with probability
  $\propto e^{U}$ among the others (variety seeking).

Finally, $\mathrm{Poisson}(0.4)$ **impulse** products are added, chosen with probability $\propto e^{q_j}$
from all available products.

The basket is the set of chosen products; empty baskets are dropped. Quantities are
$1 + \mathrm{Poisson}(0.3)$ units (not used by the basket model).

## 6. What is recorded

The observed data are what a retailer would have:

- transactions;
- products with their category and subcategory;
- weekly prices and promotions;
- a retail sales feed per (product, store, week), which includes simulated non-panel units;
- the (household, store, week) shopping opportunities.

Stockouts, missions, pantry, loyalty and taste are not recorded. Products never bought in
training are dropped (2 of 478), as are households without a training trip.

## 7. The causal price truth

The 16 scenario products are chosen at random among products bought at least 30 times in
training. For each, the **test weeks are simulated again**:

- from the same household state at the start of the test weeks (loyalty, pantry, taste);
- with the same per-trip random streams (common random numbers);
- with only that product's test-week price multiplied by 0.8 or by 1.2.

That gives 32 scenarios. For each, over the non-empty test baskets, the generator records the
rate per basket of

$$
\text{own: } \tfrac{\#\{S \ni k\}}{\#S}, \qquad
\text{same subcategory: } \tfrac{\sum_{j \in \mathrm{sub}(k)\setminus k} \#\{S \ni j\}}{\#S}, \qquad
\text{rest of category: } \tfrac{\sum_{j \in c(k)\setminus \mathrm{sub}(k)} \#\{S \ni j\}}{\#S},
$$

and the mean basket size, in the factual and the changed simulation. A scenario's truth is the
relative change of each rate, and the change in mean size. Because both runs use the same random
numbers, the difference reflects the price change and very little else. All of this is written
before any model is fitted, to `counterfactual_truth.json`.

On average the truth is:

| | price × 0.8 | price × 1.2 |
|---|---|---|
| own purchase rate | +38% | −28% |
| rest of the subcategory | −2.9% | +8.0% |

The response is asymmetric (loss aversion): per unit of log price, a rise moves the product
about 1.25 times as much as a cut. Substitution is mostly to the same subcategory, the nest of
the nested logit.

## 8. The model-data bundle

The canonical output is converted into a model-data bundle (on `v4-training`,
`scripts/prepare_model_bundle.py --config configs/datasets/stress_world_category.json`):

- **categories:** the generator's 48 merchandise categories (largest 32 products), written as the
  model's partition;
- **prices:** chain weekly prices; the model's price deviation is relative to each product's
  training-mean log price;
- **promotions:** the `advertised` flag as the model's display, and no mailer;
- **availability:** `retail_first_sale`. A product is confirmed at a store from the first week the
  retail feed records a sale there by shoppers outside the modelled panel. A first sale within
  the store's first 13 feed weeks counts as available from the start, since slow sellers need
  not sell early. Before confirmation, the product stays in the support with its utility
  shifted by $\log\epsilon$; $\epsilon$ is the training purchase rate of unconfirmed cells
  relative to confirmed ones (about $10^{-4}$ here);
- **reference trips:** 4,096 test trips with the certified earlier model's per-trip
  log-likelihoods, for paired comparisons.

## 9. How a model is scored

1. **Held-out likelihood** on the 4,096 reference test trips, paired per trip with the earlier
   certified model, with a household-clustered 95% interval.
2. **Realism**: one generated basket per test trip per seed (3 seeds). The AUC of a
   gradient-boosted classifier telling real from generated baskets, on all products and on
   category presence.
3. **Price scenarios**: the model's relative change in each rate of section 7, on 400 random
   test trips, against the truth, summarised over the 32 scenarios as correlation, mean
   absolute error, sign agreement and means (`v4/scenarios.py`).

Pass rules, fixed in advance: paired likelihood interval above 0; category-presence AUC of at
most 0.80; own-price correlation of at least 0.85.

## 10. What the model can and cannot represent

| Generator mechanism | Closest Version-4 term | Exact? |
|---|---|---|
| missions: sets of categories bought together | interactions $\psi_c$ counted once per category present (M2); size potential on categories (M3) | approximately; the Gram term is pairwise, recipes are triplets |
| one product per need | within-category penalty $\rho_c > 0$, count potentials $f_c$ | approximately |
| nested choice: substitution inside a subcategory | none: categories are the partition and the Gram term only attracts | **no** |
| loyalty to the last product | product purchase history $r^{\text{item}}$ (bought before, was the last in its category) | approximately |
| pantry stockpiling after promotions | sub-commodity recency $r$ (days since the last purchase) | partly |
| taste drift (AR(1) weekly) | static household taste $\theta_h$ | no |
| household price sensitivity | $g_{hj} \ge 0$, household × product | yes in form |
| loss-averse, asymmetric price response | the model is smooth in log price | **no** |
| display effect | $w^{\text{dsp}}_j$ | yes in form |
| impulse purchases | popularity $\lambda_j$ | approximately |
| unrecorded stockouts | none (the offered set is the declared support) | no |

The two **no** rows in bold are why the stress world tests generality rather than recovery.
The model must come close on likelihood, realism and price direction without representing them.
The measured consequences (docs/PIPELINE.md §7) are:

- own-price correlation 0.88–0.89;
- substitution to the rest of the subcategory correlates 0.47–0.51 with the truth, but its
  average size is close to zero;
- the mean own response is 1.7× too strong.

These measured weaknesses show the model is not tuned to the generator. Building the
generator's loss aversion or nesting into the model would be fitting this world, not
improving the model.

## 11. Regenerating

```bash
python3 synthetic/generate_stress_world.py --output data/stress_world     # about 80 s
```

`--households`, `--stores`, `--departments`, `--categories-per-department`, `--weeks`,
`--train-weeks`, `--validation-weeks` and `--scenario-products` change the world. For larger
catalogues, `--mission-variants` gives each mission several category templates, roughly one
per 48 categories, so that every category is still bought. A different seed or size gives a
different world, and its results are not comparable with those above.
