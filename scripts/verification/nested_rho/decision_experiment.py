"""Decision experiment: is the nested substitution-group extension worth implementing?

Standalone prototype (no pipeline code). Two known-truth worlds with household taste,
randomized weekly prices and low-rank complements:

  A  "flat":   substitution follows merchandise categories only (null check);
  B  "nested": brand lines substitute inside categories, one category mixes two unrelated
               substitute sets, one category has no substitution, and several products are rare.

Models, each fitted by exact maximum likelihood (tree dynamic program, Gauss-Hermite over z):
  none       no substitution groups (phi only)
  flat       merchandise categories (today's category partition)
  candidate  model-free tree built from training data (paper/NESTED_SUBSTITUTION_GROUPS.md
             section 8), unpenalized
  selected   same tree, L1 grid chosen by validation, pruned, relaxed refit (section 7)
  oracle     the true substitution tree (upper reference)
  truth      true parameters (evaluation only)

Decision criteria, fixed before running:
  D1  world B: selected beats flat on test log-likelihood by >= 0.01 nats per basket, with a
      household-bootstrap 95% interval above zero;
  D2  world B: selected cuts the relative error of within-category cross-price effects or of
      bundle-promotion incremental units by >= 25% versus flat;
  D3  selected costs <= 2x flat (normalizer operations and fit time);
  D4  world A: selected is not worse than flat on test log-likelihood (95% interval includes
      or exceeds zero) and keeps no nested node with rho > 0.1 outside the categories.
  Worth implementing only if D1, D2, D3 and D4 all hold.

Run from the repository root:
  python scripts/verification/nested_rho/decision_experiment.py generate --world B
  python scripts/verification/nested_rho/decision_experiment.py fit --world B --model flat
  python scripts/verification/nested_rho/decision_experiment.py evaluate --world B
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "nested_rho_decision"

# ----------------------------------------------------------------------------- settings
NMAX = 6
C, PER = 6, 10
J = C * PER
H, W = 1000, 24
TRAIN_WEEKS, VAL_WEEKS = 16, 4          # weeks 0-15 train, 16-19 validation, 20-23 test
TRIPS_PER_WEEK = 2.0
KT, KZ = 2, 2                           # taste rank, complement rank
GH = 8                                  # Gauss-Hermite points per z dimension
L1_GRID = (0.0005, 0.002, 0.008)
PRUNE = 0.01
CHUNK = 3000


def C2(n):
    return n * (n - 1) / 2.0


# ----------------------------------------------------------------------------- trees
# A tree is {"nodes": [{"name", "items": [...], "parent": index or -1}]}; node 0 is the root.
def tree_from_sets(named_sets):
    """Laminar family (name, items) -> tree with parents by smallest strict superset."""
    sets = sorted(named_sets, key=lambda x: -len(x[1]))
    nodes = [{"name": "root", "items": list(range(J)), "parent": -1}]
    seen = set()
    for name, items in sets:
        key = tuple(sorted(items))
        if len(key) < 2 or key in seen or len(key) == J:
            continue
        seen.add(key)
        parent = 0
        for i, node in enumerate(nodes):
            if i and set(key) < set(node["items"]) and len(node["items"]) < len(nodes[parent]["items"]):
                parent = i
        nodes.append({"name": name, "items": list(key), "parent": parent})
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            s, t = set(a["items"]), set(b["items"])
            assert not (s & t) or s <= t or t <= s, "candidate family is not laminar"
    return {"nodes": nodes}


def children(tree):
    """Children of each node: sub-nodes, and products whose smallest containing node it is."""
    nodes = tree["nodes"]
    kids = {i: [] for i in range(len(nodes))}
    owner = {j: 0 for j in range(J)}
    for i, node in enumerate(nodes):
        if i:
            kids[node["parent"]].append(("node", i))
            for j in node["items"]:
                if len(node["items"]) < len(nodes[owner[j]]["items"]):
                    owner[j] = i
    for j, i in owner.items():
        kids[i].append(("item", j))
    return kids


def membership(tree):
    M = np.zeros((len(tree["nodes"]) - 1, J))
    for i, node in enumerate(tree["nodes"][1:]):
        M[i, node["items"]] = 1
    return M


def normalizer_ops(tree):
    """Polynomial multiplications per quadrature node: products plus child-node convolutions."""
    kids = children(tree)
    return int(sum(1 for i in kids for kind, _ in kids[i] if kind == "item")
               + sum(1 for i in kids for kind, _ in kids[i] if kind == "node"))


# ----------------------------------------------------------------------------- exact normalizer
def gh_rule():
    x, w = np.polynomial.hermite_e.hermegauss(GH)
    w = w / w.sum()
    grid = np.array(list(itertools.product(x, repeat=KZ)))
    weights = np.array([np.prod(c) for c in itertools.product(w, repeat=KZ)])
    return torch.as_tensor(grid), torch.as_tensor(np.log(weights))


def tree_log_total(logw, rho, rho0, tree, kids):
    """log sum_{1<=|S|<=nmax} prod_{j in S} w_j * node penalties * exp(-rho0(|S|)).

    logw [..., J] (may be -inf); rho [V] for non-root nodes; rho0 [nmax+1].
    """
    finite = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -1e4))
    scale = finite.max(-1, keepdim=True).values
    w = torch.exp(finite - scale)
    R = NMAX
    k = torch.arange(R + 1, dtype=w.dtype)

    def poly(i):
        p = torch.zeros(*w.shape[:-1], R + 1, dtype=w.dtype)
        p = p.index_fill(-1, torch.tensor([0]), 1.0)
        for kind, c in kids[i]:
            if kind == "item":
                p = torch.cat([p[..., :1], p[..., 1:] + w[..., c:c + 1] * p[..., :-1]], -1)
            else:
                q = poly(c)
                p = sum(torch.nn.functional.pad(p[..., s:s + 1] * q[..., :R + 1 - s], (s, 0))
                        for s in range(R + 1))
        if i:
            p = p * torch.exp(-rho[i - 1] * C2(k))
        return p

    root = poly(0)
    terms = torch.log(root[..., 1:].clamp_min(1e-300)) + k[1:] * scale - rho0[1:]
    return torch.logsumexp(terms, -1)


def log_normalizer(b, phi, rho, rho0, tree, kids, rule):
    """b [X, J] -> log Z_+ [X] by tensor Gauss-Hermite over z."""
    nodes, logq = rule
    logw = b[:, None, :] - 0.5 * (phi ** 2).sum(1)[None, None, :] + nodes @ phi.T  # [X, Q, J]
    return torch.logsumexp(logq[None, :] + tree_log_total(logw, rho, rho0, tree, kids), 1)


def observed_energy(Y, b_trip, phi, rho, rho0, Mnode):
    v = Y @ phi
    pair = 0.5 * ((v ** 2).sum(1) - Y @ (phi ** 2).sum(1))
    counts = Y @ Mnode.T
    pen = (C2(counts) * rho[None, :]).sum(1) if Mnode.shape[0] else 0.0
    size = Y.sum(1).long()
    return (Y * b_trip).sum(1) + pair - pen - rho0[size]


# ----------------------------------------------------------------------------- truth
def true_structure(world):
    cats = [list(range(c * PER, (c + 1) * PER)) for c in range(C)]
    sets, rho = [], {}
    if world == "A":
        for c, r in enumerate((0.8, 0.9, 0.7, 0.6, 0.0, 0.6)):
            sets.append((f"cat{c}", cats[c])); rho[f"cat{c}"] = r
    else:
        # cat0 two brand lines inside a mildly substitutable category
        sets += [("cat0", cats[0]), ("cat0_lineA", cats[0][:5]), ("cat0_lineB", cats[0][5:])]
        rho.update(cat0=0.3, cat0_lineA=1.0, cat0_lineB=0.9)
        sets.append(("cat1", cats[1])); rho["cat1"] = 0.8                     # flat
        sets += [("cat2", cats[2]), ("cat2_setA", cats[2][:5]), ("cat2_setB", cats[2][5:])]
        rho.update(cat2=0.0, cat2_setA=1.2, cat2_setB=1.1)                      # mixed category
        sets += [("cat3", cats[3]), ("cat3_l1", cats[3][:3]), ("cat3_l2", cats[3][3:6]),
                 ("cat3_l3", cats[3][6:])]
        rho.update(cat3=0.2, cat3_l1=1.0, cat3_l2=0.8, cat3_l3=1.1)            # three lines
        sets.append(("cat4", cats[4])); rho["cat4"] = 0.0                     # no substitution
        sets.append(("cat5", cats[5])); rho["cat5"] = 0.6                     # flat, rare items
    return sets, rho


def generate(world, seed):
    rng = np.random.default_rng(seed)
    sets, rho_named = true_structure(world)
    tree = tree_from_sets([(n, s) for n, s in sets])
    kids = children(tree)
    rho = np.array([rho_named[n["name"]] for n in tree["nodes"][1:]])
    lam = rng.normal(-3.3, 0.45, J)
    rare = [8, 17, 27, 36, 38, 52, 55, 59]
    lam[rare] = rng.normal(-5.6, 0.3, len(rare))
    alpha = rng.normal(0, 0.55, (J, KT))
    phi = np.zeros((J, KZ))
    phi[[3, 14, 41, 44], 0] = [0.8, 0.7, 0.75, 0.7]      # cross-category mission
    phi[[21, 33, 46, 47], 1] = [0.7, 0.8, 0.75, 0.8]     # includes a within-category complement pair
    g = np.array([1.2, 2.0, 0.8, 1.6, 1.0, 1.4])
    rho0 = np.concatenate([[0.0], 0.15 * (np.arange(1, NMAX + 1) - 1) ** 2])
    theta = rng.normal(0, 1, (H, KT))
    kappa = rng.normal(0, 0.3, H)
    dlp = rng.uniform(-0.25, 0.25, (W, J))
    trips = rng.poisson(TRIPS_PER_WEEK, (H, W))
    hh = np.repeat(np.arange(H), trips.sum(1))
    week = np.concatenate([np.repeat(np.arange(W), trips[h]) for h in range(H)])
    cat = np.repeat(np.arange(C), PER)
    b = lam[None] + kappa[hh, None] + theta[hh] @ alpha.T - g[cat][None] * dlp[week]
    Y = gibbs(b, phi, rho, rho0, tree, kids, rng)
    truth = dict(lam=lam, alpha=alpha, phi=phi, g=g, rho0=rho0, theta=theta, kappa=kappa, rho=rho)
    return dict(tree=tree, truth=truth, dlp=dlp, hh=hh, week=week, Y=Y)


def gibbs(b, phi, rho, rho0, tree, kids, rng, sweeps=40):
    """Exact Gibbs on (S, z): z | S ~ N(sum phi_S, I); S | z by top-down tree sampling."""
    N = b.shape[0]
    Y = np.zeros((N, J))
    R = NMAX
    k = np.arange(R + 1)
    for sweep in range(sweeps):
        z = Y @ phi + rng.normal(size=(N, KZ))
        logw = b - 0.5 * (phi ** 2).sum(1)[None] + z @ phi.T
        scale = logw.max(1, keepdims=True)
        w = np.exp(logw - scale)
        polys, prefixes = {}, {}

        def poly(i):
            parts = []
            p = np.zeros((N, R + 1)); p[:, 0] = 1
            for kind, c in kids[i]:
                if kind == "item":
                    q = np.zeros((N, R + 1)); q[:, 0] = 1; q[:, 1] = w[:, c]
                else:
                    q = poly(c)
                parts.append((kind, c, q))
            pre = [p]
            for _, _, q in parts:
                nxt = np.zeros((N, R + 1))
                for s in range(R + 1):
                    nxt[:, s:] += pre[-1][:, s:s + 1] * q[:, :R + 1 - s]
                pre.append(nxt)
            out = pre[-1] * (np.exp(-rho[i - 1] * C2(k))[None] if i else 1.0)
            polys[i], prefixes[i] = parts, pre
            return out

        root = poly(0)
        size_w = root[:, 1:] * np.exp(k[1:] * scale - rho0[1:])
        size_w = size_w / size_w.sum(1, keepdims=True)
        size = 1 + (rng.random((N, 1)) > np.cumsum(size_w, 1)).sum(1)
        Y = np.zeros((N, J))

        def descend(i, remaining):
            parts, pre = polys[i], prefixes[i]
            for idx in range(len(parts) - 1, -1, -1):
                kind, c, q = parts[idx]
                x = np.arange(R + 1)
                weight = np.stack([pre[idx][np.arange(N), np.clip(remaining - s, 0, R)] * q[:, s]
                                   * (s <= remaining) for s in x], 1)
                weight = weight / weight.sum(1, keepdims=True)
                take = (rng.random((N, 1)) > np.cumsum(weight, 1)).sum(1)
                take = np.minimum(take, remaining)
                if kind == "item":
                    Y[:, c] = take
                else:
                    descend(c, take)
                remaining = remaining - take

        descend(0, size)
    return Y


# ----------------------------------------------------------------------------- data access
def load(world):
    d = np.load(OUT / world / "world.npz", allow_pickle=True)
    tree = json.loads(str(d["tree"]))
    truth = json.loads(str(d["truth"]))
    truth = {k: np.array(v) for k, v in truth.items()}
    return dict(tree=tree, truth=truth, dlp=d["dlp"], hh=d["hh"], week=d["week"], Y=d["Y"])


def split_mask(week, split):
    return {"train": week < TRAIN_WEEKS,
            "validation": (week >= TRAIN_WEEKS) & (week < TRAIN_WEEKS + VAL_WEEKS),
            "test": week >= TRAIN_WEEKS + VAL_WEEKS}[split]


# ----------------------------------------------------------------------------- tree builder (model-free)
def build_candidate_tree(data):
    """Section 8 of NESTED_SUBSTITUTION_GROUPS.md, from training baskets only."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    m = split_mask(data["week"], "train")
    Y, hh = data["Y"][m], data["hh"][m]
    s = Y.sum(1)
    O = Y.T @ Y
    np.fill_diagonal(O, 0)
    # household-adjusted expected co-purchase: p_ht^j = min(1, a_h^j s_t / sum_t' s_t')
    A = np.zeros((H, J)); np.add.at(A, hh, Y)
    S = np.zeros(H); np.add.at(S, hh, s)
    P = np.minimum(1.0, A[hh] * (s / np.maximum(S[hh], 1))[:, None])
    E = P.T @ P
    np.fill_diagonal(E, 0)
    kappa = 1.0
    score = np.log((O + kappa) / (E + kappa))          # < 0: shortfall; > 0: excess
    # exchangeability: Jensen-Shannon between smoothed companion distributions, excluding the pair
    marginal = O.sum(0) + 1e-9
    prior = marginal / marginal.sum()
    Q = O + 5.0 * prior[None]
    js = np.zeros((J, J))
    for j in range(J):
        for k in range(j + 1, J):
            keep = np.ones(J, bool); keep[[j, k]] = False
            p, q = Q[j, keep] / Q[j, keep].sum(), Q[k, keep] / Q[k, keep].sum()
            mid = 0.5 * (p + q)
            js[j, k] = js[k, j] = 0.5 * (p * np.log(p / mid)).sum() + 0.5 * (q * np.log(q / mid)).sum()
    cat = np.repeat(np.arange(C), PER)
    named = []
    for c in range(C):
        items = np.flatnonzero(cat == c)
        named.append((f"cat{c}", items.tolist()))
        sc = score[np.ix_(items, items)]
        jc = js[np.ix_(items, items)]
        iu = np.triu_indices(len(items), 1)

        def rank(x):
            r = np.zeros_like(x)
            r[iu] = np.argsort(np.argsort(x[iu])) / max(len(iu[0]) - 1, 1)
            return r + r.T
        D = 0.5 * rank(sc) + 0.5 * rank(jc)
        cannot = (sc > 0.4) & (O[np.ix_(items, items)] >= 5)
        D = np.where(cannot, 5.0, D)
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), method="average")
        for level in (2, 3, 5):
            labels = fcluster(Z, level, criterion="maxclust")
            for lab in np.unique(labels):
                members = items[labels == lab].tolist()
                if 2 <= len(members) < len(items):
                    named.append((f"cat{c}_k{level}_{lab}", members))
    tree = tree_from_sets(named)
    evidence = {"score": score.tolist(), "js": js.tolist()}
    return tree, evidence


# ----------------------------------------------------------------------------- model
class Model(torch.nn.Module):
    def __init__(self, tree, init=None):
        super().__init__()
        V = len(tree["nodes"]) - 1
        self.tree, self.kids = tree, children(tree)
        self.Mnode = torch.as_tensor(membership(tree))
        g = torch.Generator().manual_seed(0)
        self.lam = torch.nn.Parameter(torch.full((J,), -3.5))
        self.alpha = torch.nn.Parameter(0.05 * torch.randn(J, KT, generator=g))
        self.theta = torch.nn.Parameter(0.05 * torch.randn(H, KT, generator=g))
        self.kappa = torch.nn.Parameter(torch.zeros(H))
        self.phi = torch.nn.Parameter(0.05 * torch.randn(J, KZ, generator=g))
        self.g = torch.nn.Parameter(torch.ones(C))
        self.rho0_free = torch.nn.Parameter(torch.zeros(NMAX - 1))
        self.rho_raw = torch.nn.Parameter(torch.full((V,), -3.0))
        if init is not None:
            for name in ("lam", "alpha", "theta", "kappa", "phi", "g", "rho0_free"):
                getattr(self, name).data.copy_(torch.as_tensor(init[name]))
            names = [n["name"] for n in tree["nodes"][1:]]
            for i, n in enumerate(names):
                if n in init.get("rho_named", {}):
                    r = max(init["rho_named"][n], 1e-6)
                    self.rho_raw.data[i] = math.log(math.expm1(r)) if r < 30 else r

    def rho(self):
        return torch.nn.functional.softplus(self.rho_raw)

    def rho0(self):
        return torch.cat([torch.zeros(2), self.rho0_free])

    def utilities(self, hh, week, dlp):
        cat = torch.as_tensor(np.repeat(np.arange(C), PER))
        return (self.lam[None] + self.kappa[hh, None] + self.theta[hh] @ self.alpha.T
                - self.g[cat][None] * dlp[week])

    def export(self):
        out = {k: getattr(self, k).detach().numpy().tolist()
               for k in ("lam", "alpha", "theta", "kappa", "phi", "g", "rho0_free")}
        out["rho_named"] = {n["name"]: float(r) for n, r in zip(self.tree["nodes"][1:], self.rho())}
        return out


def truth_model(data):
    t = data["truth"]
    model = Model(data["tree"])
    with torch.no_grad():
        for name in ("lam", "alpha", "theta", "kappa", "phi", "g"):
            getattr(model, name).copy_(torch.as_tensor(t[name]))
        model.rho0_free.copy_(torch.as_tensor(t["rho0"][2:]))
        model.rho_raw.copy_(torch.as_tensor(t["rho"]))
    model.rho = lambda: torch.as_tensor(t["rho"])
    return model


def contexts(data, mask):
    hh, week = data["hh"][mask], data["week"][mask]
    key = hh * W + week
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return uniq // W, uniq % W, inverse, counts


def per_trip_loglik(model, data, mask, rule):
    Y = torch.as_tensor(data["Y"][mask])
    hh = torch.as_tensor(data["hh"][mask]); week = torch.as_tensor(data["week"][mask])
    dlp = torch.as_tensor(data["dlp"])
    ch, cw, inverse, _ = contexts(data, mask)
    with torch.no_grad():
        e = observed_energy(Y, model.utilities(hh, week, dlp), model.phi, model.rho(), model.rho0(),
                            model.Mnode)
        logz = torch.cat([log_normalizer(model.utilities(torch.as_tensor(ch[i:i + CHUNK]),
                                                         torch.as_tensor(cw[i:i + CHUNK]), dlp),
                                         model.phi, model.rho(), model.rho0(), model.tree, model.kids, rule)
                          for i in range(0, len(ch), CHUNK)])
    return (e - logz[torch.as_tensor(inverse)]).numpy()


def fit_model(model, data, rule, l1=0.0, iterations=6, max_iter=60):
    mask = split_mask(data["week"], "train")
    Y = torch.as_tensor(data["Y"][mask])
    hh = torch.as_tensor(data["hh"][mask]); week = torch.as_tensor(data["week"][mask])
    dlp = torch.as_tensor(data["dlp"])
    ch, cw, _, counts = contexts(data, mask)
    ch_t, cw_t, counts_t = torch.as_tensor(ch), torch.as_tensor(cw), torch.as_tensor(counts, dtype=torch.float64)
    N = float(len(Y))
    params = [p for p in model.parameters()]
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=30)
    evaluations = [0]

    def closure():
        opt.zero_grad()
        evaluations[0] += 1
        e = observed_energy(Y, model.utilities(hh, week, dlp), model.phi, model.rho(), model.rho0(), model.Mnode)
        prior = 0.5 * ((model.theta ** 2).sum() + (model.kappa ** 2).sum() / 0.3 ** 2
                       + (model.alpha ** 2).sum() / 0.5 ** 2 + (model.phi ** 2).sum()) / N
        loss_obs = -e.sum() / N + prior + l1 * model.rho().sum()
        loss_obs.backward()
        total = float(loss_obs)
        for i in range(0, len(ch), CHUNK):
            b = model.utilities(ch_t[i:i + CHUNK], cw_t[i:i + CHUNK], dlp)
            lz = log_normalizer(b, model.phi, model.rho(), model.rho0(), model.tree, model.kids, rule)
            part = (counts_t[i:i + CHUNK] * lz).sum() / N
            part.backward()
            total += float(part)
        return total

    history = []
    for _ in range(iterations):
        loss = opt.step(closure)
        history.append(float(loss))
        if len(history) > 1 and abs(history[-2] - history[-1]) < 1e-7:
            break
    return {"train_objective": history, "evaluations": evaluations[0]}


# ----------------------------------------------------------------------------- expectations for decisions
def expected_incidence(model, data, ctx_h, ctx_w, price_cut, rule):
    """Mean over contexts of E[1_j] (exact derivative of log Z_+ with respect to b_j).

    price_cut [J] is -log(new price / old price); utilities rise by g_c * price_cut.
    """
    dlp = torch.as_tensor(data["dlp"])
    total = torch.zeros(J)
    for i in range(0, len(ctx_h), CHUNK):
        with torch.no_grad():
            b0 = model.utilities(torch.as_tensor(ctx_h[i:i + CHUNK]), torch.as_tensor(ctx_w[i:i + CHUNK]), dlp)
            cat = torch.as_tensor(np.repeat(np.arange(C), PER))
            b0 = b0 + model.g[cat][None] * torch.as_tensor(price_cut)[None]
            phi, rho, rho0 = model.phi.detach(), model.rho().detach(), model.rho0().detach()
        b = b0.clone().requires_grad_(True)
        lz = log_normalizer(b, phi, rho, rho0, model.tree, model.kids, rule).sum()
        (grad,) = torch.autograd.grad(lz, b)
        total += grad.sum(0)
    return (total / len(ctx_h)).numpy()


def decision_quantities(model, data, rule, n_contexts=2000, seed=5):
    mask = split_mask(data["week"], "test")
    ch, cw, _, _ = contexts(data, mask)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ch), size=min(n_contexts, len(ch)), replace=False)
    ch, cw = ch[pick], cw[pick]
    base = expected_incidence(model, data, ch, cw, np.zeros(J), rule)
    # utilities carry -g_c * dlp, so a price cut to fraction f adds g_c * (-log f)
    cross = np.zeros((J, J))
    for j in range(J):
        shift = np.zeros(J); shift[j] = -math.log(0.9)
        cross[j] = expected_incidence(model, data, ch, cw, shift, rule) - base
    actions = action_list(data)
    values = {}
    for name, items in actions.items():
        shift = np.zeros(J); shift[items] = -math.log(0.8)
        delta = expected_incidence(model, data, ch, cw, shift, rule) - base
        values[name] = {"incremental_units": float(delta.sum()),
                        "discounted_units_change": float(delta[items].sum()),
                        "cannibalized_units": float(-delta[[k for k in range(J) if k not in items
                                                            and k // PER in {i // PER for i in items}]].sum())}
    return {"base": base.tolist(), "cross": cross.tolist(), "actions": values}


def action_list(data):
    cats = [list(range(c * PER, (c + 1) * PER)) for c in range(C)]
    acts = {}
    for c in range(C):
        acts[f"all_cat{c}"] = cats[c]
        acts[f"half_cat{c}"] = cats[c][:5]
        acts[f"cross_half_cat{c}"] = cats[c][3:8]
        acts[f"pair_cat{c}"] = [cats[c][0], cats[c][1]]
        acts[f"split_pair_cat{c}"] = [cats[c][0], cats[c][9]]
    acts["mission_0"] = [3, 14, 41, 44]
    acts["mission_1"] = [21, 33, 46, 47]
    return acts


# ----------------------------------------------------------------------------- commands
def fit_command(world, model_name):
    data = load(world)
    rule = gh_rule()
    wdir = OUT / world
    torch.set_num_threads(3)
    cat_tree = tree_from_sets([(f"cat{c}", list(range(c * PER, (c + 1) * PER))) for c in range(C)])
    t0 = time.time()
    record = {"model": model_name}
    if model_name == "none":
        model = Model({"nodes": [{"name": "root", "items": list(range(J)), "parent": -1}]})
        record.update(fit_model(model, data, rule))
    elif model_name == "flat":
        model = Model(cat_tree)
        record.update(fit_model(model, data, rule))
    elif model_name in ("candidate", "oracle"):
        init = json.loads((wdir / "fit_flat.json").read_text())["parameters"]
        tree = (json.loads((wdir / "candidate_tree.json").read_text()) if model_name == "candidate"
                else data["tree"])
        model = Model(tree, init)
        record.update(fit_model(model, data, rule))
    elif model_name.startswith("l1_"):
        l1 = float(model_name[3:])
        init = json.loads((wdir / "fit_flat.json").read_text())["parameters"]
        tree = json.loads((wdir / "candidate_tree.json").read_text())
        penalized = Model(tree, init)
        record["penalized"] = fit_model(penalized, data, rule, l1=l1)
        keep = [n["name"] for n, r in zip(tree["nodes"][1:], penalized.rho()) if float(r) > PRUNE]
        record["penalized_rho"] = penalized.export()["rho_named"]
        pruned = tree_from_sets([(n["name"], n["items"]) for n in tree["nodes"][1:] if n["name"] in keep])
        model = Model(pruned, penalized.export())
        record["refit"] = fit_model(model, data, rule)
        record["pruned_tree"] = pruned
    else:
        raise SystemExit(f"unknown model {model_name}")
    record["fit_seconds"] = time.time() - t0
    record["normalizer_ops"] = normalizer_ops(model.tree)
    record["tree"] = model.tree
    record["parameters"] = model.export()
    record["validation_loglik"] = float(per_trip_loglik(model, data, split_mask(data["week"], "validation"), rule).mean())
    (wdir / f"fit_{model_name}.json").write_text(json.dumps(record))
    print(json.dumps({k: v for k, v in record.items() if k in ("model", "fit_seconds", "validation_loglik",
                                                               "normalizer_ops", "evaluations")}))


def generate_command(world):
    wdir = OUT / world
    wdir.mkdir(parents=True, exist_ok=True)
    data = generate(world, seed={"A": 101, "B": 202}[world])
    np.savez_compressed(wdir / "world.npz", tree=json.dumps(data["tree"]),
                        truth=json.dumps({k: np.asarray(v).tolist() for k, v in data["truth"].items()}),
                        dlp=data["dlp"], hh=data["hh"], week=data["week"], Y=data["Y"])
    data = load(world)
    tree, evidence = build_candidate_tree(data)
    (wdir / "candidate_tree.json").write_text(json.dumps(tree))
    # sampler check: sampled incidence vs exact expectations on the test split
    rule = gh_rule()
    model = truth_model(data)
    mask = split_mask(data["week"], "test")
    hh, week = data["hh"][mask], data["week"][mask]
    exact = expected_incidence(model, data, hh, week, np.zeros(J), rule)
    empirical = data["Y"][mask].mean(0)
    se = np.sqrt(exact * (1 - exact) / mask.sum())
    true_sets = {n["name"]: sorted(n["items"]) for n in data["tree"]["nodes"][1:]}
    cand_sets = [sorted(n["items"]) for n in tree["nodes"][1:]]
    global GH
    GH = 12
    fine_rule = gh_rule()
    GH = 8
    ll8 = per_trip_loglik(model, data, mask, rule).mean()
    ll12 = per_trip_loglik(model, data, mask, fine_rule).mean()
    summary = {
        "world": world, "trips": int(len(data["Y"])),
        "trips_by_split": {s: int(split_mask(data["week"], s).sum()) for s in ("train", "validation", "test")},
        "mean_basket_size": float(data["Y"].sum(1).mean()),
        "size_distribution": np.bincount(data["Y"].sum(1).astype(int), minlength=NMAX + 1)[1:].tolist(),
        "rare_product_train_purchases": data["Y"][split_mask(data["week"], "train")][:, [8, 17, 27, 36, 38, 52, 55, 59]].sum(0).tolist(),
        "sampler_check_max_abs_z": float(np.abs((empirical - exact) / se).max()),
        "sampler_check_mean_abs_z": float(np.abs((empirical - exact) / se).mean()),
        "quadrature_check_test_loglik_gh8_vs_gh12": [float(ll8), float(ll12)],
        "candidate_nodes": len(cand_sets),
        "true_sets_covered": {n: s in cand_sets for n, s in true_sets.items()},
    }
    (wdir / "world_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def bootstrap_diff(a, b, hh, reps=2000, seed=9):
    d = a - b
    sums = np.zeros(H); cnt = np.zeros(H)
    np.add.at(sums, hh, d); np.add.at(cnt, hh, 1)
    present = np.flatnonzero(cnt)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        s = rng.choice(present, len(present))
        draws.append(sums[s].sum() / cnt[s].sum())
    return [float(d.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def pair_penalty(model):
    rho = model.rho().detach().numpy()
    M = model.Mnode.numpy()
    return (M.T * rho) @ M if len(rho) else np.zeros((J, J))


def evaluate_command(world):
    data = load(world)
    rule = gh_rule()
    wdir = OUT / world
    torch.set_num_threads(8)
    fits = {}
    for name in ["none", "flat", "candidate", "oracle"] + [f"l1_{x}" for x in L1_GRID]:
        path = wdir / f"fit_{name}.json"
        if path.exists():
            fits[name] = json.loads(path.read_text())
    l1_names = [n for n in fits if n.startswith("l1_")]
    selected = max(l1_names, key=lambda n: fits[n]["validation_loglik"])
    fits["selected"] = fits[selected]
    models = {"truth": truth_model(data)}
    for name, rec in fits.items():
        models[name] = Model(rec["tree"], rec["parameters"])
    test = split_mask(data["week"], "test")
    hh = data["hh"][test]
    ll = {name: per_trip_loglik(m, data, test, rule) for name, m in models.items()}
    true_P = pair_penalty(models["truth"])
    cat = np.repeat(np.arange(C), PER)
    same = (cat[:, None] == cat[None]) & ~np.eye(J, dtype=bool)
    decisions = {name: decision_quantities(m, data, rule) for name, m in models.items()}
    truth_d = decisions["truth"]
    t_cross = np.array(truth_d["cross"])
    report = {"world": world, "selected_l1": selected, "models": {}}
    for name, m in models.items():
        d = decisions[name]
        cross = np.array(d["cross"])
        acts = list(truth_d["actions"])
        tv = np.array([truth_d["actions"][a]["incremental_units"] for a in acts])
        mv = np.array([d["actions"][a]["incremental_units"] for a in acts])
        tc = np.array([truth_d["actions"][a]["cannibalized_units"] for a in acts])
        mc = np.array([d["actions"][a]["cannibalized_units"] for a in acts])
        P = pair_penalty(m)
        entry = {
            "test_loglik": float(ll[name].mean()),
            "test_loglik_minus_flat": bootstrap_diff(ll[name], ll["flat"], hh) if name != "flat" else None,
            "gap_to_truth": float(ll["truth"].mean() - ll[name].mean()),
            "pair_penalty_corr_within_category": float(np.corrcoef(P[same], true_P[same])[0, 1]) if P[same].std() > 0 else None,
            "cross_price_within_category_rel_error": float(np.linalg.norm(cross[same] - t_cross[same]) / np.linalg.norm(t_cross[same])),
            "cross_price_within_category_corr": float(np.corrcoef(cross[same], t_cross[same])[0, 1]),
            "promotion_incremental_units_rel_error": float(np.linalg.norm(mv - tv) / np.linalg.norm(tv)),
            "promotion_incremental_units_corr": float(np.corrcoef(mv, tv)[0, 1]),
            "promotion_cannibalization_rel_error": float(np.linalg.norm(mc - tc) / np.linalg.norm(tc)),
            "best_action": acts[int(np.argmax(mv))], "true_best_action": acts[int(np.argmax(tv))],
            "incremental_units_by_action": dict(zip(acts, mv.round(5).tolist())),
        }
        if name in fits:
            rec = fits[name]
            entry["fit_seconds"] = rec["fit_seconds"]
            entry["normalizer_ops"] = rec["normalizer_ops"]
            entry["validation_loglik"] = rec["validation_loglik"]
            entry["active_nodes"] = {k: round(v, 3) for k, v in rec["parameters"]["rho_named"].items() if v > PRUNE}
        else:
            entry["active_nodes"] = {n["name"]: round(float(r), 3) for n, r in
                                     zip(data["tree"]["nodes"][1:], data["truth"]["rho"]) if r > 0}
        report["models"][name] = entry
    # the selection procedure costs every penalized fit plus its refit
    report["models"]["selected"]["selection_procedure_seconds"] = float(
        sum(fits[n]["fit_seconds"] for n in l1_names))
    report["l1_path"] = {n: {"validation_loglik": fits[n]["validation_loglik"],
                             "active_nodes": {k: round(v, 3) for k, v in fits[n]["parameters"]["rho_named"].items() if v > PRUNE}}
                         for n in l1_names}
    (wdir / "evaluation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({n: {k: v for k, v in e.items() if k != "incremental_units_by_action"}
                      for n, e in report["models"].items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "fit", "evaluate", "selfcheck"])
    parser.add_argument("--world", choices=["A", "B"], default="B")
    parser.add_argument("--model", default="flat")
    args = parser.parse_args()
    if args.command == "generate":
        generate_command(args.world)
    elif args.command == "fit":
        fit_command(args.world, args.model)
    elif args.command == "evaluate":
        evaluate_command(args.world)
    else:
        selfcheck()


def selfcheck():
    """Tree normalizer and Gibbs sampler against brute-force enumeration on a tiny world."""
    global J, C, PER, NMAX
    J, C, PER, NMAX = 8, 2, 4, 4
    rng = np.random.default_rng(1)
    tree = tree_from_sets([("cat0", [0, 1, 2, 3]), ("l0", [0, 1]), ("cat1", [4, 5, 6, 7]), ("l1", [5, 6, 7])])
    kids = children(tree)
    rho = np.array([0.4, 0.9, 0.3, 1.1])
    b = rng.normal(-0.8, 0.5, (3, J))
    phi = rng.normal(0, 0.4, (J, KZ))
    rho0 = np.concatenate([[0.0], 0.2 * (np.arange(1, NMAX + 1) - 1) ** 2])
    Mn = membership(tree)
    baskets = [s for n in range(1, NMAX + 1) for s in itertools.combinations(range(J), n)]
    Yall = np.zeros((len(baskets), J))
    for r, s in enumerate(baskets):
        Yall[r, list(s)] = 1
    global GH
    GH = 30
    rule = gh_rule()
    out = {}
    errs = []
    for x in range(3):
        e = observed_energy(torch.as_tensor(Yall), torch.as_tensor(np.repeat(b[x:x + 1], len(baskets), 0)),
                            torch.as_tensor(phi), torch.as_tensor(rho), torch.as_tensor(rho0), torch.as_tensor(Mn))
        exact = torch.logsumexp(e, 0)
        quad = log_normalizer(torch.as_tensor(b[x:x + 1]), torch.as_tensor(phi), torch.as_tensor(rho),
                              torch.as_tensor(rho0), tree, kids, rule)[0]
        errs.append(abs(float(exact - quad)))
    out["normalizer_max_abs_error"] = max(errs)
    N = 200000
    Y = gibbs(np.repeat(b[:1], N, 0), phi, rho, rho0, tree, kids, rng, sweeps=15)
    e = observed_energy(torch.as_tensor(Yall), torch.as_tensor(np.repeat(b[:1], len(baskets), 0)),
                        torch.as_tensor(phi), torch.as_tensor(rho), torch.as_tensor(rho0), torch.as_tensor(Mn)).numpy()
    p = np.exp(e - np.logaddexp.reduce(e))
    index = {s: r for r, s in enumerate(baskets)}
    counts = np.zeros(len(baskets))
    for row in Y:
        counts[index[tuple(np.flatnonzero(row))]] += 1
    out["gibbs_total_variation"] = float(0.5 * np.abs(counts / N - p).sum())
    out["multinomial_reference_total_variation"] = float(0.5 * np.abs(rng.multinomial(N, p) / N - p).sum())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
