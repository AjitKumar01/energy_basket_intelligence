"""Exact top-down basket sampling through the nested-rho tree dynamic program.

Run from the repository root: python scripts/verification/nested_rho/verify_sampler.py
"""
import itertools, json
import numpy as np

rng = np.random.default_rng(3)
J, nmax = 12, 5
tree = {"root": ["famA", "cat2", 11], "famA": ["cat0", "cat1"], "cat0": ["cl00", 3], "cl00": [0, 1, 2],
        "cat1": [4, 5, 6], "cat2": ["cl20", "cl21"], "cl20": [7, 8], "cl21": [9, 10]}
rho = {v: float(rng.uniform(0.2, 1.2)) for v in tree if v != "root"}
rho0 = np.concatenate([[0.0], rng.uniform(-0.5, 1.0, nmax)])
b = rng.normal(-0.3, 0.8, J)
w = np.exp(b)
C2 = lambda k: k * (k - 1) / 2.0


def leaves(n):
    return [n] if not isinstance(n, str) else sum((leaves(c) for c in tree[n]), [])


def mul(a, c):
    out = np.zeros(nmax + 1)
    for i, x in enumerate(a):
        out[i:] += x * c[:nmax + 1 - i]
    return out


polys = {}


def poly(n):
    if not isinstance(n, str):
        p = np.zeros(nmax + 1); p[0] = 1; p[1] = w[n]
        polys[n] = p
        return p
    p = np.zeros(nmax + 1); p[0] = 1
    for c in tree[n]:
        p = mul(p, poly(c))
    if n != "root":
        p = p * np.exp(-rho[n] * C2(np.arange(nmax + 1)))
    polys[n] = p
    return p


def sample_node(n, k, out):
    """Given k products under node n, split k among children proportional to prod child coefficients."""
    if not isinstance(n, str):
        if k == 1:
            out.append(n)
        return
    kids = tree[n]
    prefix = [np.zeros(nmax + 1)]; prefix[0][0] = 1
    for c in kids:
        prefix.append(mul(prefix[-1], polys[c]))
    remaining = k
    for i in range(len(kids) - 1, -1, -1):
        c = kids[i]
        choices = np.arange(remaining + 1)
        weight = np.array([prefix[i][remaining - x] * polys[c][x] for x in choices])
        x = rng.choice(choices, p=weight / weight.sum())
        sample_node(c, x, out)
        remaining -= x


root = poly("root")
total = root[1:] * np.exp(-rho0[1:])
size_probability = total / total.sum()
N = 200000
counts = {}
for _ in range(N):
    size = 1 + rng.choice(nmax, p=size_probability)
    basket = []
    sample_node("root", size, basket)
    key = tuple(sorted(basket))
    counts[key] = counts.get(key, 0) + 1
baskets = [s for n in range(1, nmax + 1) for s in itertools.combinations(range(J), n)]


def energy(S):
    e = sum(b[j] for j in S) - rho0[len(S)]
    for v in rho:
        e -= rho[v] * C2(len(set(S) & set(leaves(v))))
    return e


e = np.array([energy(s) for s in baskets])
p = np.exp(e - np.logaddexp.reduce(e))
empirical = np.array([counts.get(s, 0) / N for s in baskets])
result = {
    "baskets": len(baskets), "draws": N,
    "top_down_tree_sampler_total_variation": float(0.5 * np.abs(empirical - p).sum()),
    "same_size_exact_multinomial_reference_tv": float(0.5 * np.abs(rng.multinomial(N, p) / N - p).sum()),
}
print(json.dumps(result, indent=1))
json.dump(result, open("artifacts/nested_rho_verification/sampler.json", "w"), indent=1)
