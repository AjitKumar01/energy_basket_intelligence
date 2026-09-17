"""Exactness checks for the nested-rho (hierarchical substitution group) energy basket extension.

Run from the repository root: python scripts/verification/nested_rho/verify_exactness.py
"""
import itertools, json, math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)
report = {}

# ---------------------------------------------------------------- tree definition
# A tree is a dict node -> list of children; leaves are products (ints); internal nodes are strings.
def make_tree():
    return {
        "root": ["famA", "famB"],
        "famA": ["cat0", "cat1"],
        "famB": ["cat2", 11],                 # product 11 hangs directly under a family
        "cat0": ["cl00", 3],                  # sub-cluster + a loose product
        "cl00": [0, 1, 2],
        "cat1": [4, 5, 6],
        "cat2": ["cl20", "cl21"],
        "cl20": [7, 8],
        "cl21": [9, 10],
    }

def products_under(tree, node):
    if not isinstance(node, str):
        return [node]
    out = []
    for c in tree[node]:
        out += products_under(tree, c)
    return out

def internal_nodes(tree):
    return [n for n in tree]

C2 = lambda n: n * (n - 1) / 2.0

def energy_enum(S, b, phi, rho, rho0, tree, cap=None):
    """Direct energy of basket S (tuple of products)."""
    S = list(S)
    e = sum(b[j] for j in S)
    for j, k in itertools.combinations(S, 2):
        e += float(phi[j] @ phi[k])
    for v in internal_nodes(tree):
        if v == "root":
            continue
        n = len(set(S) & set(products_under(tree, v)))
        e -= rho[v] * C2(n)
    return e - rho0[len(S)]

def all_baskets(J, nmax, available):
    items = [j for j in range(J) if available[j]]
    return [s for n in range(1, nmax + 1) for s in itertools.combinations(items, n)]

# ---------------------------------------------------------------- tree dynamic program
def poly_mul(a, b, nmax):
    out = np.zeros(nmax + 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        m = min(len(b), nmax + 1 - i)
        out[i:i + m] += x * b[:m]
    return out

def node_poly(tree, node, w, rho, nmax, stats=None):
    """Polynomial sum_k [x^k] over subsets under node of prod w * node factors (truncated)."""
    if not isinstance(node, str):
        p = np.zeros(nmax + 1); p[0] = 1.0; p[1] = w[node]
        return p
    p = np.zeros(nmax + 1); p[0] = 1.0
    for child in tree[node]:
        p = poly_mul(p, node_poly(tree, child, w, rho, nmax), nmax)
    if node != "root":
        k = np.arange(nmax + 1)
        p = p * np.exp(-rho[node] * C2(k))
    return p

def logZ_given_z(tree, b, phi, z, rho, rho0, nmax, available):
    w = np.where(available, np.exp(b - 0.5 * (phi ** 2).sum(1) + phi @ z), 0.0)
    p = node_poly(tree, "root", w, rho, nmax)
    n = np.arange(nmax + 1)
    return math.log(float((p[1:] * np.exp(-rho0[1:])).sum()))

def enum_given_z(tree, b, phi, z, rho, rho0, nmax, available):
    """Same quantity by enumeration: sum_S prod w_j(z) * node factors * size."""
    total = 0.0
    for S in all_baskets(len(b), nmax, available):
        lw = sum(b[j] - 0.5 * float(phi[j] @ phi[j]) + float(phi[j] @ z) for j in S)
        for v in internal_nodes(tree):
            if v != "root":
                lw -= rho[v] * C2(len(set(S) & set(products_under(tree, v))))
        total += math.exp(lw - rho0[len(S)])
    return math.log(total)

J, nmax, Kz = 12, 5, 2
tree = make_tree()
rho = {v: float(rng.uniform(0.2, 1.2)) for v in tree if v != "root"}
rho0 = np.concatenate([[0.0], rng.uniform(-0.5, 1.0, nmax)])
b = rng.normal(-0.5, 0.8, J)
phi = rng.normal(0, 0.35, (J, Kz))
available = np.ones(J, bool); available[5] = False

# 1. DP given z == enumeration given z
z = rng.normal(size=Kz)
d1 = abs(logZ_given_z(tree, b, phi, z, rho, rho0, nmax, available)
         - enum_given_z(tree, b, phi, z, rho, rho0, nmax, available))
report["1_dp_given_z_vs_enumeration_abs_error"] = d1

# 1b. Hubbard-Stratonovich + Gauss-Hermite vs direct energy enumeration (full normalizer)
x, wq = np.polynomial.hermite_e.hermegauss(40); wq = wq / wq.sum()
vals = []
for (i, xi), (k, xk) in itertools.product(enumerate(x), enumerate(x)):
    vals.append(math.log(wq[i] * wq[k]) + logZ_given_z(tree, b, phi, np.array([xi, xk]), rho, rho0, nmax, available))
logZ_quad = float(np.logaddexp.reduce(vals))
E = [energy_enum(S, b, phi, rho, rho0, tree) for S in all_baskets(J, nmax, available)]
logZ_enum = float(np.logaddexp.reduce(E))
report["1b_full_normalizer_quadrature_vs_enumeration_abs_error"] = abs(logZ_quad - logZ_enum)

# 2. truncation commutes: truncating at nmax inside nodes equals computing untruncated then truncating
def node_poly_untrunc(tree, node, w, rho):
    big = 12
    return node_poly(tree, node, w, rho, big)
w0 = np.where(available, np.exp(b), 0.0)
p_trunc = node_poly(tree, "root", w0, rho, nmax)
p_full = node_poly_untrunc(tree, "root", w0, rho)[:nmax + 1]
report["2_truncation_commutes_max_abs"] = float(np.abs(p_trunc - p_full).max())

# 3. one-level tree equals the current flat model sum_c rho_c C(n_c,2)
flat = {"root": ["g0", "g1", "g2"], "g0": [0, 1, 2, 3], "g1": [4, 5, 6, 7], "g2": [8, 9, 10, 11]}
rho_flat = {"g0": 0.7, "g1": 0.3, "g2": 1.1}
cat = np.repeat([0, 1, 2], 4); rc = np.array([0.7, 0.3, 1.1])
def current_energy(S):
    S = list(S); e = sum(b[j] for j in S) + sum(float(phi[j] @ phi[k]) for j, k in itertools.combinations(S, 2))
    counts = np.bincount(cat[S], minlength=3)
    return e - float((rc * C2(counts)).sum()) - rho0[len(S)]
report["3_flat_tree_equals_current_energy_max_abs"] = max(
    abs(energy_enum(S, b, phi, rho_flat, rho0, flat) - current_energy(S))
    for S in all_baskets(J, nmax, np.ones(J, bool)))

# 4. gradient identity d logZ / d rho_v = -E[C(n_v,2)]
def logZ_enum_rho(r):
    return float(np.logaddexp.reduce([energy_enum(S, b, phi, r, rho0, tree) for S in all_baskets(J, nmax, available)]))
baskets = all_baskets(J, nmax, available)
p = np.exp(np.array(E) - logZ_enum)
worst = 0.0
for v in rho:
    h = 1e-6
    up = dict(rho); up[v] += h; dn = dict(rho); dn[v] -= h
    fd = (logZ_enum_rho(up) - logZ_enum_rho(dn)) / (2 * h)
    expected = -sum(pi * C2(len(set(S) & set(products_under(tree, v)))) for S, pi in zip(baskets, p))
    worst = max(worst, abs(fd - expected))
report["4_gradient_identity_max_abs"] = worst

# 5. conditional completion: P(T | R subset S) via node offsets rho_v [C2(k+f_v) - C2(f_v)]
R = (0, 7)
rest = [j for j in range(J) if available[j] and j not in R]
cond_E = {}
for n in range(0, nmax - len(R) + 1):
    for T in itertools.combinations(rest, n):
        cond_E[T] = energy_enum(R + T, b, phi, rho, rho0, tree)
logZc = float(np.logaddexp.reduce(list(cond_E.values())))
# offset DP given z=... check with phi=0 for a deterministic DP identity
phi0 = np.zeros_like(phi)
def cond_poly(node):
    if not isinstance(node, str):
        pp = np.zeros(nmax + 1); pp[0] = 1.0
        if node not in R and available[node]:
            pp[1] = math.exp(b[node])
        return pp
    pp = np.zeros(nmax + 1); pp[0] = 1.0
    for c in tree[node]:
        pp = poly_mul(pp, cond_poly(c), nmax)
    if node != "root":
        f = len(set(R) & set(products_under(tree, node)))
        k = np.arange(nmax + 1)
        pp = pp * np.exp(-rho[node] * (C2(k + f) - C2(f)))
    return pp
pc = cond_poly("root")
sizes = np.arange(nmax + 1)
valid = sizes + len(R) <= nmax
base_R = sum(b[j] for j in R) - sum(rho[v] * C2(len(set(R) & set(products_under(tree, v)))) for v in rho)
logZc_dp = math.log(float((pc[valid] * np.exp(-rho0[sizes[valid] + len(R)])).sum())) + base_R
cond_E0 = {T: energy_enum(R + T, b, phi0, rho, rho0, tree) for T in cond_E}
report["5_conditional_offsets_vs_enumeration_abs_error"] = abs(
    logZc_dp - float(np.logaddexp.reduce(list(cond_E0.values()))))

# 6. single-flip conditional and add-one score: Delta E(add j to R) = b_j + phi_j.sum phi_R - sum_{v ancestors of j} rho_v n_v(R) - [rho0(|R|+1)-rho0(|R|)]
worst = 0.0
for trial in range(50):
    Rset = tuple(sorted(rng.choice([j for j in range(J) if available[j]], size=rng.integers(1, nmax), replace=False)))
    j = int(rng.choice([k for k in range(J) if available[k] and k not in Rset]))
    direct = energy_enum(tuple(sorted(Rset + (j,))), b, phi, rho, rho0, tree) - energy_enum(Rset, b, phi, rho, rho0, tree)
    anc = [v for v in rho if j in products_under(tree, v)]
    formula = (b[j] + sum(float(phi[j] @ phi[k]) for k in Rset)
               - sum(rho[v] * len(set(Rset) & set(products_under(tree, v))) for v in anc)
               - (rho0[len(Rset) + 1] - rho0[len(Rset)]))
    worst = max(worst, abs(direct - formula))
report["6_add_one_increment_max_abs"] = worst

# 7. identifiability: pair penalty P(j,k) = sum of rho_v over nodes containing both; map rho -> P
nodes = [v for v in tree if v != "root"]
pairs = list(itertools.combinations(range(J), 2))
A = np.array([[1.0 if (j in products_under(tree, v) and k in products_under(tree, v)) else 0.0 for v in nodes] for j, k in pairs])
report["7_pair_map_rank_vs_nodes"] = [int(np.linalg.matrix_rank(A)), len(nodes)]
# root would add a column of ones -> equivalent to a C(n,2) size term, i.e. confounded with rho0
size_feature = np.array([C2(n) for n in range(nmax + 1)])
report["7b_root_penalty_equals_size_potential_shift"] = "rho_root*C(|S|,2) is a function of |S| only, absorbed exactly by rho0"
# unary chain: a node with a single child duplicates the child's column
chain = {"root": ["u"], "u": ["g"], "g": [0, 1, 2]}
Au = np.array([[1.0, 1.0] for _ in itertools.combinations(range(3), 2)])
report["7c_unary_chain_rank"] = int(np.linalg.matrix_rank(Au))

print(json.dumps(report, indent=1))
json.dump(report, open("artifacts/nested_rho_verification/exactness.json", "w"), indent=1)
