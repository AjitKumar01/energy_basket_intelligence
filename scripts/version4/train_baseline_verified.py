"""Train one audited basket baseline with resumable, provenance-complete checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
import time

import numpy as np
import torch

import evalall as EA
from baselines import Batches, Bernoulli, DPP
from baselines2 import Multinomial, NDPP, Shopper, size_law
from data import build
from features import Features


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "out"))


def log(message):
    print(f"[blv] {message}", flush=True)


def acquire_training_lock(model, tag):
    """Ensure exactly one writer exists for a model/tag checkpoint lineage."""
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(OUT, f"baseline_verified_{model}{suffix}.lock")
    stream = open(path, "a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(f"another trainer holds {path}") from error
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return stream


def hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


class PlateauConvergence:
    """Fail-closed optimizer stopping rule evaluated only on a fixed validation panel."""

    def __init__(self, min_delta, patience, floor_patience, minimum_epochs):
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.floor_patience = int(floor_patience)
        self.minimum_epochs = float(minimum_epochs)
        self.significant_best = -float("inf")
        self.evals_since_improvement = 0
        self.floor_evals_since_improvement = 0
        self.converged = False

    def observe(self, score, iteration, batch, n_train, lr, minimum_lr):
        improved = score > self.significant_best + self.min_delta
        if improved:
            self.significant_best = float(score)
            self.evals_since_improvement = 0
        else:
            self.evals_since_improvement += 1
        at_floor = lr <= minimum_lr * (1.0 + 1e-9) + 1e-15
        if not at_floor or improved:
            self.floor_evals_since_improvement = 0
        else:
            self.floor_evals_since_improvement += 1
        epochs = iteration * batch / n_train
        self.converged = bool(
            epochs >= self.minimum_epochs
            and at_floor
            and self.evals_since_improvement >= self.patience
            and self.floor_evals_since_improvement >= self.floor_patience)
        return dict(improved=improved, at_floor=at_floor, epochs=epochs,
                    converged=self.converged)

    def state_dict(self):
        return dict(min_delta=self.min_delta, patience=self.patience,
                    floor_patience=self.floor_patience,
                    minimum_epochs=self.minimum_epochs,
                    significant_best=self.significant_best,
                    evals_since_improvement=self.evals_since_improvement,
                    floor_evals_since_improvement=self.floor_evals_since_improvement,
                    converged=self.converged)

    def load_state_dict(self, state):
        expected = (self.min_delta, self.patience, self.floor_patience,
                    self.minimum_epochs)
        received = (float(state["min_delta"]), int(state["patience"]),
                    int(state["floor_patience"]), float(state["minimum_epochs"]))
        if received != expected:
            raise RuntimeError("resume convergence settings differ from the checkpoint")
        self.significant_best = float(state["significant_best"])
        self.evals_since_improvement = int(state["evals_since_improvement"])
        self.floor_evals_since_improvement = int(
            state["floor_evals_since_improvement"])
        self.converged = bool(state["converged"])


def supported_training(D, nmax, R):
    ids = np.flatnonzero(D["trip_split"] == 0)
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in ids:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if 1 <= hi - lo <= nmax and np.bincount(lc[lo:hi]).max() <= R:
            keep.append(t)
    return np.asarray(keep, dtype=np.int64)


def absolute_popularity_logits(D, trips):
    """Training-only exposure-corrected log incidence, with its absolute level retained.

    RaggedModel centers this initializer because a common utility shift is a rho_0 gauge
    direction.  These external baselines have no rho_0, so that shift determines expected
    basket size and must not be removed.
    """
    J, C, S = (int(D[k]) for k in ("n_item", "n_cat", "n_store"))
    count = np.zeros(J, dtype=np.float64)
    for t in trips:
        lo, hi = int(D["line_ptr"][t]), int(D["line_ptr"][t + 1])
        count[D["line_item"][lo:hi]] += 1.0
    exposure = np.zeros(J, dtype=np.float64)
    store_n = np.bincount(D["trip_store"][trips], minlength=S)
    ptr, items = D["store_cat_ptr"], D["store_items"]
    for store in range(S):
        lo, hi = int(ptr[store * C]), int(ptr[(store + 1) * C])
        exposure[items[lo:hi]] += store_n[store]
    seen = exposure > 0
    value = np.empty(J, dtype=np.float64)
    value[seen] = np.log((count[seen] + 0.5) / (exposure[seen] + 1.0))
    value[~seen] = np.median(value[seen])
    return torch.as_tensor(value, dtype=torch.float64)


def make_model(name, D, a):
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    if name == "multinomial":
        return Multinomial(J, N, S, size_law(D, a.nmax, a.R), K=a.K, Kp=a.Kp,
                           seed=a.seed, taste_init=a.taste_init)
    if name == "bernoulli":
        return Bernoulli(J, N, S, K=a.K, Kp=a.Kp, seed=a.seed,
                         taste_init=a.taste_init)
    if name == "dpp":
        return DPP(J, N, S, rank=a.rank, K=a.K, Kp=a.Kp, seed=a.seed,
                   taste_init=a.taste_init, interaction_init=a.interaction_init)
    if name == "ndpp":
        return NDPP(J, N, S, rank=a.rank, srank=a.srank, K=a.K, Kp=a.Kp,
                    seed=a.seed, taste_init=a.taste_init,
                    interaction_init=a.interaction_init)
    if name == "shopper":
        return Shopper(J, N, S, K=a.K, Kp=a.Kp, Ki=a.interaction_rank,
                       seed=a.seed, taste_init=a.taste_init,
                       interaction_init=a.interaction_init)
    raise ValueError(name)


def call(model, name, batch, a, generator, evaluation=False):
    if name == "multinomial":
        return model.loglik(batch, category_cap=a.R)
    if name == "bernoulli":
        return model.loglik(batch, nmax=a.nmax, category_cap=a.R)
    if name == "shopper":
        return model.loglik(batch,
                            n_orders=a.eval_orders if evaluation else a.train_orders,
                            exact_max_n=a.exact_max_n if evaluation else 0,
                            gen=generator, max_size=a.nmax)
    return model.loglik(batch)


@torch.no_grad()
def evaluate(model, name, batcher, trips, a, seed):
    values, lines = [], 0
    generator = torch.Generator().manual_seed(seed)
    for k in range(0, len(trips), a.eval_chunk):
        sub = trips[k:k + a.eval_chunk]
        d = batcher.make(sub)
        ll = call(model, name, d, a, generator, evaluation=True)
        if not torch.isfinite(ll).all():
            raise RuntimeError(f"non-finite {name} validation score")
        values.extend(ll.tolist())
        lines += len(d["li"])
    x = np.asarray(values)
    return dict(per_basket=float(x.mean()),
                se=float(x.std(ddof=1) / np.sqrt(len(x))),
                per_line=float(x.sum() / lines), n=len(x))


def save(path, model, opt, sched, iteration, best, a, name, tr, va, te,
         rng, order_gen, convergence, lineage, history):
    blob = dict(format=3, kind="verified-basket-baseline", model_name=name,
                model=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=sched.state_dict(), iteration=iteration, best=best,
                config=vars(a), data=dict(partition=os.environ.get("V3_PARTITION", ""),
                                           affinity=os.environ.get("V3_AFFINITY", "0"),
                                           n_item=int(model.idx.lam.shape[0]),
                                           train_n=len(tr), train_hash=hash_ids(tr),
                                           valid_n=len(va), valid_hash=hash_ids(va),
                                           test_n=len(te), test_hash=hash_ids(te)),
                lineage=lineage, convergence_state=convergence.state_dict(),
                history=history,
                rng_np=rng.bit_generator.state, rng_torch=order_gen.get_state())
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def main(a):
    training_lock = acquire_training_lock(a.model, a.tag)
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    D = build()
    J, S = int(D["n_item"]), int(D["n_store"])
    batcher = Batches(D, Features(J, S, 712))
    tr = supported_training(D, a.nmax, a.R)
    # Exact fit.py manifest: support filter, fixed permutation, then the prefix.  A raw
    # prefix is about eight nats easier on these data; an independently sampled manifest
    # is legitimate for a separate experiment but not for the requested matched audit.
    def fit_manifest(split_code, seed):
        ids = np.flatnonzero(D["trip_split"] == split_code)
        lp, lc = D["line_ptr"], D["line_cat"]
        keep = []
        for t in ids:
            lo, hi = int(lp[t]), int(lp[t + 1])
            if 1 <= hi - lo <= a.nmax and np.bincount(lc[lo:hi]).max() <= a.R:
                keep.append(t)
        ids = np.asarray(keep, dtype=np.int64)
        ids = ids[np.random.default_rng(seed).permutation(len(ids))]
        return ids[:a.n_val]
    va = fit_manifest(1, a.manifest_seed)
    te = fit_manifest(2, a.manifest_seed)
    log(f"{a.model}: {len(tr):,} training trips ({hash_ids(tr)[:12]}); "
        f"valid {hash_ids(va)[:12]}, test {hash_ids(te)[:12]}")
    model = make_model(a.model, D, a).double()
    if a.init_popularity:
        with torch.no_grad():
            model.idx.lam.copy_(absolute_popularity_logits(D, tr))
        log(f"{a.model}: initialized product intercept from training incidence/exposure")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    minimum_lr = a.lr * a.lr_floor
    milestones = [int(x) for x in a.lr_milestones.split(",") if x.strip()]
    if a.scheduler == "milestone":
        if not milestones:
            raise ValueError("milestone scheduler requires --lr-milestones")
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=milestones, gamma=a.lr_gamma)
    elif a.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=a.iters, eta_min=minimum_lr)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=a.lr_gamma, patience=a.plateau_patience,
            threshold=a.convergence_delta, threshold_mode="abs",
            min_lr=minimum_lr)
    convergence = PlateauConvergence(
        a.convergence_delta, a.convergence_patience,
        a.floor_patience, a.minimum_epochs)
    rng = np.random.default_rng(a.seed)
    order_gen = torch.Generator().manual_seed(a.seed)
    suffix = f"_{a.tag}" if a.tag else ""
    path = os.path.join(OUT, f"baseline_verified_{a.model}{suffix}.pt")
    best_path = os.path.join(OUT, f"baseline_verified_{a.model}{suffix}_best.pt")
    start, best = 0, dict(per_basket=-float("inf"), iteration=0)
    history = []
    lineage = dict(fresh_initialization=True, started_unix=time.time(),
                   initial_seed=a.seed)
    if a.resume and os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") != 3 or blob.get("model_name") != a.model:
            raise RuntimeError("resume checkpoint is not the requested verified baseline")
        if not blob.get("lineage", {}).get("fresh_initialization"):
            raise RuntimeError("resume checkpoint lacks a fresh-initialization lineage")
        model.load_state_dict(blob["model"], strict=True)
        opt.load_state_dict(blob["optimizer"]); sched.load_state_dict(blob["scheduler"])
        start, best = int(blob["iteration"]), blob["best"]
        history = list(blob.get("history", []))
        lineage = blob["lineage"]
        convergence.load_state_dict(blob["convergence_state"])
        rng.bit_generator.state = blob["rng_np"]; order_gen.set_state(blob["rng_torch"])
        log(f"resumed iteration {start}, best {best['per_basket']:.4f}")
    if a.eval_initial:
        initial = evaluate(model.eval(), a.model, batcher, va, a, a.eval_seed)
        model.train()
        log(f"initial it {start:6d}: valid {initial['per_basket']:.3f} "
            f"+/- {initial['se']:.3f} per-line {initial['per_line']:.4f}")
    started = time.time()
    for iteration in range(start + 1, a.iters + 1):
        trips = tr[rng.choice(len(tr), size=a.batch, replace=False)]
        d = batcher.make(trips)
        loss = -call(model, a.model, d, a, order_gen).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {iteration}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if a.scheduler != "plateau":
            sched.step()
        if iteration % a.eval_every == 0 or iteration == a.iters:
            model.eval()
            val = evaluate(model, a.model, batcher, va, a, a.eval_seed)
            model.train()
            elapsed = time.time() - started
            if a.scheduler == "plateau":
                sched.step(val["per_basket"])
            status = convergence.observe(
                val["per_basket"], iteration, a.batch, len(tr),
                opt.param_groups[0]["lr"], minimum_lr)
            log(f"it {iteration:6d}/{a.iters} ep {status['epochs']:5.2f} "
                f"loss {float(loss.detach()):9.3f} "
                f"valid {val['per_basket']:9.3f} +/- {val['se']:.3f} "
                f"lr {opt.param_groups[0]['lr']:.2g} {elapsed/60:.1f} min "
                f"stale {convergence.evals_since_improvement} "
                f"floor-stale {convergence.floor_evals_since_improvement} "
                f"ETA {(elapsed/max(iteration-start,1))*(a.iters-iteration)/3600:.2f} h")
            history.append(dict(
                iteration=iteration, epochs=status["epochs"],
                train_loss=float(loss.detach()), validation=val,
                learning_rate=float(opt.param_groups[0]["lr"]),
                significant_improvement=bool(status["improved"]),
                at_floor=bool(status["at_floor"]),
                stale=convergence.evals_since_improvement,
                floor_stale=convergence.floor_evals_since_improvement))
            if val["per_basket"] > best["per_basket"]:
                best = dict(per_basket=val["per_basket"], iteration=iteration)
                save(best_path, model, opt, sched, iteration, best, a, a.model,
                     tr, va, te, rng, order_gen, convergence, lineage, history)
            save(path, model, opt, sched, iteration, best, a, a.model,
                 tr, va, te, rng, order_gen, convergence, lineage, history)
            if a.require_convergence and status["converged"]:
                log(f"convergence certified at iteration {iteration}: "
                    f"{status['epochs']:.2f} epochs, lr floor {minimum_lr:.3g}, "
                    f"best iteration {best['iteration']}")
                break
    terminal_iteration = iteration if a.iters > start else start
    certificate = dict(
        required=bool(a.require_convergence), passed=bool(convergence.converged),
        terminal_iteration=int(terminal_iteration), selected_iteration=int(best["iteration"]),
        rule=convergence.state_dict(), scheduler=a.scheduler,
        minimum_lr=minimum_lr, train_n=len(tr), batch=a.batch)
    if a.require_convergence and not convergence.converged:
        raise RuntimeError(
            f"{a.model} reached the {a.iters}-update safety ceiling without satisfying "
            "the convergence certificate; no test score was produced")
    best_blob = torch.load(best_path, map_location="cpu", weights_only=False)
    best_blob["convergence_certificate"] = certificate
    best_blob["history"] = history
    tmp = best_path + ".tmp"
    torch.save(best_blob, tmp)
    os.replace(tmp, best_path)
    model.load_state_dict(best_blob["model"], strict=True); model.eval()
    valid = evaluate(model, a.model, batcher, va, a, a.eval_seed)
    test = (evaluate(model, a.model, batcher, te, a, a.eval_seed + 1009)
            if a.score_test else None)
    result = dict(model=a.model, best=best_blob["best"], convergence=certificate,
                  history=history,
                  valid=valid, test=test,
                  checkpoint=os.path.basename(best_path), config=vars(a))
    with open(os.path.join(OUT, f"baseline_verified_{a.model}{suffix}.json"), "w") as stream:
        json.dump(result, stream, indent=2)
    suffix = (f", test {test['per_basket']:.4f}" if test is not None
              else "; test remains locked for the paired audit")
    log(f"best iteration {best['iteration']}: valid {valid['per_basket']:.4f}{suffix}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["multinomial", "bernoulli", "dpp", "ndpp", "shopper"], required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--iters", type=int, default=60000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-floor", type=float, default=0.02,
                   help="minimum learning rate as a fraction of --lr (not an absolute rate)")
    p.add_argument("--scheduler", choices=("milestone", "cosine", "plateau"),
                   default="milestone")
    p.add_argument("--lr-milestones", default="20000,26000")
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=3,
                   help="validation intervals before each plateau LR reduction")
    p.add_argument("--convergence-delta", type=float, default=0.002,
                   help="minimum deterministic validation gain in nats/basket")
    p.add_argument("--convergence-patience", type=int, default=8,
                   help="validation intervals without a material gain")
    p.add_argument("--floor-patience", type=int, default=4,
                   help="stale validation intervals required after reaching the LR floor")
    p.add_argument("--minimum-epochs", type=float, default=2.0)
    p.add_argument("--require-convergence", action="store_true")
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--eval-chunk", type=int, default=8)
    p.add_argument("--eval-initial", type=int, default=1,
                   help="score the raw/resumed checkpoint before any updates")
    p.add_argument("--threads", type=int, default=0,
                   help="PyTorch intra-op threads; zero keeps the runtime default")
    p.add_argument("--score-test", action="store_true",
                   help="diagnostic only; normal pipeline keeps test locked for final audit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taste-init", type=float, default=0.03)
    p.add_argument("--interaction-init", type=float, default=0.03)
    p.add_argument("--init-popularity", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=20260821)
    p.add_argument("--manifest-seed", type=int, default=12345)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--srank", type=int, default=8)
    p.add_argument("--interaction-rank", type=int, default=16)
    p.add_argument("--train-orders", type=int, default=4)
    p.add_argument("--eval-orders", type=int, default=128)
    p.add_argument("--exact-max-n", type=int, default=6)
    main(p.parse_args())
