"""Gaussian priors with estimated scales on the per-household and per-product context blocks
(docs/overfitting_theory.pdf, Section 4).

For a block X with one row per household (or product) and columns k, every row is a draw

    X[r, k] ~ N(m_k, s2_k)    independently,

so the block has 2 numbers per column, whatever the number of rows.  The fit maximises the
penalised likelihood (a MAP estimate of the rows for the current m, s2), and every few hundred steps
m and s2 are re-estimated by the EM update under the Laplace approximation

    m_k  <- mean_r X[r, k]
    s2_k <- mean_r [ (X[r, k] - m_k)^2 + 1 / (A[r, k] + 1 / s2_k) ],

where A[r, k] is the curvature of the row's own log-likelihood along X[r, k], estimated by the
empirical Fisher information: the sum, over the steps since the last update, of the squared
likelihood gradient of the batch.  (Near the optimum the gradients of different trips are
uncorrelated with mean zero, so the square of a batch sum estimates the sum of squares.)  The
second term is the posterior variance that the MAP estimate leaves out; without it a row with few
trips makes s2 too small.
"""
from __future__ import annotations

import torch

# Blocks and the columns that carry a prior (alpha's last column is fixed at 1: the size shift).
BLOCKS = {"theta": None, "gamma": None, "alpha": slice(0, -1), "beta": None, "mu": None, "zeta": None}


class HierPrior:
    def __init__(self, model, names=tuple(BLOCKS), floor=1e-4):
        self.model, self.floor = model, floor
        self.blocks = {}
        with torch.no_grad():
            for n in names:
                p = getattr(model, n)
                cols = BLOCKS[n] if BLOCKS[n] is not None else slice(None)
                x = p[:, cols]
                m = x.mean(0)
                s2 = (x - m).square().mean(0).clamp_min(floor)        # start: the spread of the start values
                self.blocks[n] = {"cols": cols, "m": m.clone(), "s2": s2.clone(), "A": torch.zeros_like(x)}

    def state(self):
        return {n: {"m": b["m"].clone(), "s2": b["s2"].clone()} for n, b in self.blocks.items()}

    def load(self, state):
        for n, b in self.blocks.items():
            b["m"].copy_(state[n]["m"])
            b["s2"].copy_(state[n]["s2"])

    def penalty(self, n_train):
        """sum over blocks of -log prior (up to a constant), divided by the number of training trips
        (the objective is the mean over trips)."""
        total = 0.0
        for n, b in self.blocks.items():
            x = getattr(self.model, n)[:, b["cols"]]
            total = total + 0.5 * ((x - b["m"]).square() / b["s2"]).sum()
        return total / n_train

    @torch.no_grad()
    def accumulate(self, batch_size):
        """Call after backward of -mean log-likelihood alone (before the prior's gradient is added):
        adds the squared batch-sum likelihood gradient of every row to its curvature estimate."""
        for n, b in self.blocks.items():
            g = getattr(self.model, n).grad
            if g is not None:
                b["A"].add_((batch_size * g[:, b["cols"]]).square())

    @torch.no_grad()
    def update(self):
        """The EM step for m and s2; resets the curvature estimates.  Returns a summary per block."""
        out = {}
        for n, b in self.blocks.items():
            x = getattr(self.model, n)[:, b["cols"]]
            m = x.mean(0)
            v = 1.0 / (b["A"] + 1.0 / b["s2"])                       # posterior variance of each entry
            s2 = ((x - m).square() + v).mean(0).clamp_min(self.floor)
            shrink = (b["A"] / (b["A"] + 1.0 / s2)).mean()            # fraction of each row kept from its data
            b["m"].copy_(m)
            b["s2"].copy_(s2)
            b["A"].zero_()
            out[n] = {"s2_mean": round(float(s2.mean()), 5), "s2_min": round(float(s2.min()), 5),
                      "s2_max": round(float(s2.max()), 5), "data_weight": round(float(shrink), 3)}
        return out
