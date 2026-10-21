#!/usr/bin/env python3
"""Exactness checks for shopper/shopper.py on small random models (float64).

1. The basket law sums to one over every nonempty subset of the offered items.
2. Gumbel-max sampling reproduces the exact subset probabilities.
3. Both ordering estimators of P(S) (uniform permutations, utility-guided importance sampling)
   are unbiased: over 200 seeds their mean reproduces exact enumeration of the orderings.
4. Autograd gradients of the sequence log-probability match central finite differences.
5. Padding (unoffered slots, a shorter basket) does not change any trip's probability.
"""
import itertools, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shopper import Shopper, Trips

torch.set_default_dtype(torch.float64)
res = {}


def trips_for(subsets, M, offered, price, user=0, week=0, valid=None, N=None):
    B = len(subsets)
    N = N or max(1, max(len(s) for s in subsets))
    basket = torch.full((B, N), -1, dtype=torch.long)
    for b, s in enumerate(subsets):
        basket[b, :len(s)] = torch.tensor(s, dtype=torch.long)
    return Trips(offered=offered.expand(B, M).clone(),
                 valid=(torch.ones(B, M, dtype=torch.bool) if valid is None else valid.expand(B, M).clone()),
                 price=price.expand(B, M).clone(), offset=torch.zeros(B, M),
                 user=torch.full((B,), user), week=torch.full((B,), week),
                 basket=basket, size=torch.tensor([len(s) for s in subsets]))


torch.manual_seed(0)
M = 5
model = Shopper(n_item=8, n_user=3, n_week=4, item_group=[0, 0, 1, 1, 2, 2, 3, 3], K=3, P=2, D=2, init_scale=0.8, seed=1)
offered = torch.tensor([[1, 2, 4, 6, 7]])
price = torch.randn(1, M) * 0.3
subsets = [list(c) for n in range(1, M + 1) for c in itertools.combinations(range(M), n)]
t = trips_for(subsets, M, offered, price, user=1, week=2)
lp = model.basket_logprob(t, exact_up_to=M)
res["sum_over_subsets"] = float(lp.exp().sum())

# 2. sampling
draws = 40000
ts = trips_for([[0]] * draws, M, offered, price, user=1, week=2)
X = model.sample(ts, seed=5)
emp = {}
for row in X.tolist():
    key = tuple(i for i, v in enumerate(row) if v)
    emp[key] = emp.get(key, 0) + 1
exact = {tuple(s): float(p) for s, p in zip(subsets, lp.exp())}
z = [abs(emp.get(k, 0) / draws - p) / math.sqrt(p * (1 - p) / draws) for k, p in exact.items() if p * draws > 20]
res["sampling_max_z"] = max(z)
res["sampling_cells"] = len(z)

# 3. permutation estimator vs exact for a 7-item basket in a 10-item store
M2 = 10
model2 = Shopper(n_item=12, n_user=2, n_week=1, item_group=list(range(12)), K=4, P=1, D=0, init_scale=0.6, seed=3)
off2 = torch.arange(10)[None]
t2 = trips_for([[0, 2, 3, 5, 6, 8, 9]], M2, off2, torch.zeros(1, M2))
ex = float(model2.basket_logprob(t2, exact_up_to=7))
res["perm_exact"] = ex
# Both estimators are unbiased for P(S): over many seeds the mean of P-hat reproduces P(S).
for name, fn in (("uniform", lambda g: model2.basket_logprob(t2, perms=64, exact_up_to=0, gen=g)),
                 ("is", lambda g: model2.basket_logprob_is(t2, perms=64, gen=g))):
    v = torch.tensor([float(fn(torch.Generator().manual_seed(k))) for k in range(200)])
    res[f"{name}_64_log_mean_P_minus_exact"] = float(torch.logsumexp(v, 0) - math.log(len(v)) - ex)
    res[f"{name}_64_sd_log"] = float(v.std())
res["is_sum_over_subsets"] = float(model.basket_logprob_is(t, perms=256, gen=torch.Generator().manual_seed(9)).exp().sum())

# 4. gradients
t3 = trips_for([[0, 3, 1], [2], [4, 0]], M, offered, price, user=2, week=1)
order = t3.basket
f = lambda: model.sequence_logprob(t3, order).sum()
model.zero_grad(); f().backward()
errs = []
for name, prm in model.named_parameters():
    g = prm.grad.reshape(-1)
    nz = torch.nonzero(g.abs() > 1e-8).squeeze(1)[:3]
    for k in nz.tolist():
        with torch.no_grad():
            flat = prm.view(-1); v = flat[k].item()
            flat[k] = v + 1e-6; fp = float(f()); flat[k] = v - 1e-6; fm = float(f()); flat[k] = v
        fd = (fp - fm) / 2e-6
        errs.append(abs(fd - float(g[k])) / max(abs(fd), 1e-6))
res["grad_max_rel_err"] = max(errs)
res["grad_entries_checked"] = len(errs)

# 5. padding invariance
valid = torch.tensor([[True, True, True, True, True, False, False]])
off_pad = torch.tensor([[1, 2, 4, 6, 7, 0, 3]])
price_pad = torch.cat([price, torch.zeros(1, 2)], 1)
tp = trips_for([[0, 3, 1]], 7, off_pad, price_pad, user=2, week=1, valid=valid, N=5)
t0 = trips_for([[0, 3, 1]], M, offered, price, user=2, week=1)
res["padding_abs_diff"] = abs(float(model.basket_logprob(tp, exact_up_to=5)) - float(model.basket_logprob(t0, exact_up_to=5)))

res["pass"] = bool(abs(res["sum_over_subsets"] - 1) < 1e-10 and res["sampling_max_z"] < 4.5
                   and abs(res["uniform_64_log_mean_P_minus_exact"]) < 0.02 and abs(res["is_64_log_mean_P_minus_exact"]) < 0.02
                   and abs(res["is_sum_over_subsets"] - 1) < 0.01 and res["grad_max_rel_err"] < 1e-5
                   and res["padding_abs_diff"] < 1e-10)
print(json.dumps(res, indent=1))
