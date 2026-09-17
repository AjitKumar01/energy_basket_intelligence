"""Does exact maximum likelihood on a candidate tree select the true substitute groups?

Run from the repository root: python scripts/verification/nested_rho/verify_recovery.py [--per-context N]
"""
import itertools, json, math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
rng = np.random.default_rng(11)
import sys
J, nmax, contexts = 12, 5, 240
per_context = int(sys.argv[sys.argv.index("--per-context") + 1]) if "--per-context" in sys.argv else 150

def leaves(tree, node):
    return [node] if not isinstance(node, str) else sum((leaves(tree, c) for c in tree[node]), [])

# truth: substitute groups = three merchandise categories; complements via rank-1 phi across cat1/cat2
true_groups = {"cat0": [0, 1, 2, 3], "cat1": [4, 5, 6], "cat2": [7, 8, 9, 10]}
true_rho = {"cat0": 0.9, "cat1": 1.2, "cat2": 0.6}
lam = rng.normal(-1.0, 0.4, J)
phi_true = np.zeros((J, 2)); phi_true[[4, 5, 8, 9], 0] = 0.7          # cross-category complements
rho0_true = np.concatenate([[0.0], 0.1 * (np.arange(1, nmax + 1) - 1) ** 2])

baskets = [s for n in range(1, nmax + 1) for s in itertools.combinations(range(J), n)]
M = np.zeros((len(baskets), J))
for r, s in enumerate(baskets):
    M[r, list(s)] = 1
sizes = M.sum(1).astype(int)
offsets = rng.normal(0, 0.6, (contexts, J))                            # observed context covariates

def energies(lam_, phi_, rho_nodes, node_sets, rho0_, offs):
    bctx = lam_[None, :] + offs                                        # [X, J]
    lin = bctx @ M.T                                                   # [X, B]
    v = M @ phi_
    pair = 0.5 * ((v ** 2).sum(1) - M @ (phi_ ** 2).sum(1))
    pen = sum(r * (lambda n: n * (n - 1) / 2)(M[:, idx].sum(1)) for r, idx in zip(rho_nodes, node_sets))
    return lin + (pair - pen - rho0_[sizes])[None, :]

# simulate exact baskets
Et = energies(lam, phi_true, list(true_rho.values()), list(true_groups.values()), rho0_true, offsets)
P = np.exp(Et - np.logaddexp.reduce(Et, axis=1, keepdims=True))
counts = np.stack([rng.multinomial(per_context, p) for p in P])       # [X, B]
test_counts = np.stack([rng.multinomial(per_context // 3, p) for p in P])

def oracle_ll(c):
    return float((c * (Et - np.logaddexp.reduce(Et, axis=1, keepdims=True))).sum() / c.sum())

def fit(tree, l1=0.0, steps=400):
    nodes = [v for v in tree if v != "root"]
    node_sets = [leaves(tree, v) for v in nodes]
    Mt = torch.as_tensor(M); off = torch.as_tensor(offsets); cnt = torch.as_tensor(counts, dtype=torch.float64)
    lam_ = torch.zeros(J, requires_grad=True)
    phi_ = (0.05 * torch.randn(J, 2)).requires_grad_(True)
    rho_raw = torch.full((len(nodes),), -1.0, requires_grad=True)
    rho0_free = torch.zeros(nmax - 1, requires_grad=True)             # rho0(1)=0 fixes the linear gauge
    pair_counts = [torch.as_tensor(M[:, idx].sum(1)) for idx in node_sets]
    size_t = torch.as_tensor(sizes)
    def loss_fn():
        rho = torch.nn.functional.softplus(rho_raw)
        rho0 = torch.cat([torch.zeros(2), rho0_free])
        bctx = lam_[None, :] + off
        v = Mt @ phi_
        pair = 0.5 * ((v ** 2).sum(1) - Mt @ (phi_ ** 2).sum(1))
        pen = sum(rho[i] * n * (n - 1) / 2 for i, n in enumerate(pair_counts)) if nodes else 0.0
        E = bctx @ Mt.T + (pair - pen - rho0[size_t])[None, :]
        ll = (cnt * (E - torch.logsumexp(E, 1, keepdim=True))).sum() / cnt.sum()
        return -ll + l1 * rho.sum(), rho, E
    opt = torch.optim.LBFGS([lam_, phi_, rho_raw, rho0_free], max_iter=steps, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)
    def closure():
        opt.zero_grad(); loss, _, _ = loss_fn(); loss.backward(); return loss
    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        loss, rho, E = loss_fn()
        logp = E - torch.logsumexp(E, 1, keepdim=True)
        test_ll = float((torch.as_tensor(test_counts, dtype=torch.float64) * logp).sum() / test_counts.sum())
    return {"rho": {v: round(float(r), 3) for v, r in zip(nodes, rho)}, "test_loglik": round(test_ll, 5)}

# famB with a single child is a unary chain: collapse it (rule from the verification)
candidate = {"root": ["famA", "cat2", 11], "famA": ["cat0", "cat1"],
             "cat0": ["cl00", "cl01"], "cl00": [0, 1], "cl01": [2, 3],
             "cat1": [4, 5, 6], "cat2": ["cl20", 9, 10], "cl20": [7, 8]}
flat_truth = {"root": ["cat0", "cat1", "cat2", 11], "cat0": [0, 1, 2, 3], "cat1": [4, 5, 6], "cat2": [7, 8, 9, 10]}
missing = {"root": ["x", "y", 11], "x": [0, 1, 4, 5], "y": [2, 3, 6, 7, 8, 9, 10]}   # tree without the true groups
none = {"root": list(range(J))}

def prune(tree, keep):
    """Collapse internal nodes whose fitted rho is zero into their parent."""
    new = {}
    def walk(node):
        if not isinstance(node, str):
            return [node]
        kids = sum((walk(c) for c in tree[node]), [])
        if node == "root" or node in keep:
            new[node] = kids
            return [node]
        return kids
    walk("root")
    return new

report = {"per_context": per_context, "oracle_test_loglik": round(oracle_ll(test_counts), 5), "true_rho": true_rho}
report["candidate_tree_with_decoys_no_l1"] = fit(candidate)
for l1 in (0.0005, 0.002, 0.008):
    report[f"candidate_tree_with_decoys_l1_{l1}"] = fit(candidate, l1=l1)
selected = {v for v, r in report["candidate_tree_with_decoys_l1_0.002"]["rho"].items() if r > 1e-3}
report["relaxed_refit_after_l1_0.002"] = {"pruned_tree": prune(candidate, selected), **fit(prune(candidate, selected))}
report["flat_true_partition"] = fit(flat_truth)
report["tree_missing_true_groups"] = fit(missing)
report["no_groups_phi_only"] = fit(none)
print(json.dumps(report, indent=1))
json.dump(report, open(f"artifacts/nested_rho_verification/recovery_per_context_{per_context}.json", "w"), indent=1)
