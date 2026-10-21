#!/usr/bin/env python3
"""Does the SHOPPER fit recover the illustrative simulation's planted structure (paper Sec. 3)?

Items: 100 coffee, 101 diapers (new parents), 200 ramen, 201 candy (students), 300/301 hot dogs
and buns, 400/401 taco shells and seasoning (one complementary pair per trip, never both).
Prints the next-item probabilities of the paper's Table 1 trip, the complement/substitute
structure after the first pair item, and the own-price effect on preferred items."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shopper import Shopper, Trips
from shopper_data import TsvSource

torch.set_default_dtype(torch.float64)
d, ckpt = sys.argv[1], sys.argv[2]
src = TsvSource(d, norm_price=0)
b = torch.load(ckpt, weights_only=False)
c = b["config"]
m = Shopper(b["n_item"], b["n_user"], b["n_week"], b["item_group"], K=c["K"], P=c["P"], D=c["D"])
m.load_state_dict(b["model"]); m.eval()
I = {v: i for i, v in enumerate(src.items)}
names = {100: "coffee", 101: "diapers", 200: "ramen", 201: "candy", 300: "hotdogs", 301: "buns", 400: "tacoshells", 401: "seasoning"}
# who is a parent: buys diapers
parents = sorted({int(src.trip_user[t]) for t in range(len(src.basket)) if I[101] in src.basket[t]})
students = sorted({int(src.trip_user[t]) for t in range(len(src.basket)) if I[200] in src.basket[t]})
J = len(src.items)


def next_probs(user, prices, chosen):
    """Distribution of the next choice given items already chosen, in order."""
    t = Trips(offered=torch.arange(J)[None], valid=torch.ones(1, J, dtype=torch.bool),
              price=torch.log(torch.tensor([prices]).double()), offset=torch.zeros(1, J),
              user=torch.tensor([user]), week=torch.tensor([-1]), basket=torch.full((1, 1), -1), size=torch.tensor([0]))
    with torch.no_grad():
        psi = m.base(t)[0]; ck = m.checkout_base(t)[0]
        ctx = m.alpha[[I[x] for x in chosen]].mean(0) if chosen else torch.zeros(m.K)
        u = psi + m.rho @ ctx
        for x in chosen:
            u[I[x]] = -float("inf")
        ckv = ck + m.ck_rho @ ctx if chosen else torch.tensor(-float("inf"))
        p = torch.softmax(torch.cat([u, ckv[None]]), 0)
    return {**{names[v]: round(float(p[I[v]]), 3) for v in src.items}, "checkout": round(float(p[-1]), 3)}


low = {v: 1.0 for v in src.items}
table1 = {**low, 100: 5.0, 400: 5.0}   # coffee and taco shells marked up
pr = lambda d_: [d_[v] for v in src.items]
u = parents[0]
out = {"parents": len(parents), "students": len(students), "overlap": len(set(parents) & set(students))}
out["table1_parent"] = {"step1": next_probs(u, pr(table1), []),
                        "after_diapers": next_probs(u, pr(table1), [101]),
                        "after_diapers_hotdogs": next_probs(u, pr(table1), [101, 300]),
                        "after_diapers_hotdogs_buns": next_probs(u, pr(table1), [101, 300, 301])}
pp = [next_probs(x, pr(low), [300]) for x in parents[:20]]
out["after_hotdogs_mean_over_20_parents"] = {k: round(float(np.mean([q[k] for q in pp])), 3) for k in pp[0]}
eff = {}
for item, who in ((100, parents), (200, students)):
    lo = np.mean([next_probs(x, pr(low), [])[names[item]] for x in who[:20]])
    hi = np.mean([next_probs(x, pr({**low, item: 5.0}), [])[names[item]] for x in who[:20]])
    eff[names[item]] = {"first_choice_prob_low_price": round(float(lo), 3), "high_price": round(float(hi), 3)}
out["own_price_first_choice"] = eff
out["student_buys_coffee_first"] = round(float(np.mean([next_probs(x, pr(low), [])["coffee"] for x in students[:20]])), 4)
print(json.dumps(out, indent=1))
