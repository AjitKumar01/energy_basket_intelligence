"""Small-support exact checks for the headline basket baselines."""
import argparse
import math
import unittest

import torch

from baselines import Bernoulli
from baselines2 import Multinomial, NDPP, Shopper
from ragged import RaggedIndex
from train_baseline_verified import PlateauConvergence
from audit_other_baselines_fair import parse_models


torch.set_default_dtype(torch.float64)


def zero_index(index):
    with torch.no_grad():
        for p in index.parameters():
            p.zero_()


def batch(items, cats, chosen):
    items = torch.as_tensor(items, dtype=torch.long)
    cats = torch.as_tensor(cats, dtype=torch.long)
    order = torch.argsort(cats, stable=True)
    items, cats = items[order], cats[order]
    unique, counts = torch.unique_consecutive(cats, return_counts=True)
    row_of = torch.repeat_interleave(torch.arange(len(unique)), counts)
    rix = RaggedIndex(items, row_of, torch.zeros(len(unique), dtype=torch.long), unique, 1)
    pos = {int(j): i for i, j in enumerate(items)}
    li = torch.as_tensor(chosen, dtype=torch.long)
    zslot = torch.zeros(len(items), dtype=torch.long)
    zline = torch.zeros(len(li), dtype=torch.long)
    ctx = dict(dlp=torch.zeros(len(items)), disp=torch.zeros(len(items)),
               mail=torch.zeros(len(items)), week=zslot, store=zslot)
    lctx = dict(dlp=torch.zeros(len(li)), disp=torch.zeros(len(li)),
                mail=torch.zeros(len(li)), week=zline, store=zline)
    return dict(item=items, st=zslot, ctx=ctx, house=torch.zeros(1, dtype=torch.long), B=1,
                li=li, lt=zline, lctx=lctx,
                lslot=torch.as_tensor([pos[int(j)] for j in li]),
                off=torch.tensor([0, len(items)]), rix=rix)


class BaselineExactness(unittest.TestCase):
    def test_bernoulli_is_exactly_truncated_by_size_and_category(self):
        # Equal odds on four products.  With sizes 1..2 there are 4+6=10 sets.  With at
        # most one item per category (three products in cat 0, one in cat 1), only the
        # three cross-category pairs survive, hence 4+3=7 sets.
        m = Bernoulli(4, 1, 1, K=2, Kp=1)
        zero_index(m.idx)
        d = batch([0, 1, 2, 3], [0, 0, 0, 1], [0, 3])
        self.assertAlmostEqual(float(m.loglik(d, nmax=2, category_cap=2).detach()),
                               -math.log(10), places=12)
        self.assertAlmostEqual(float(m.loglik(d, nmax=2, category_cap=1).detach()),
                               -math.log(7), places=12)
        with torch.no_grad():
            m.idx.lam.copy_(torch.tensor([2.0, -1.0, 0.5, -2.0]))
        weights = torch.exp(m.idx.lam)
        normalizer = weights.sum()
        for i in range(4):
            for j in range(i + 1, 4):
                normalizer = normalizer + weights[i] * weights[j]
        want = m.idx.lam[0] + m.idx.lam[3] - torch.log(normalizer)
        self.assertAlmostEqual(float(m.loglik(d, nmax=2, category_cap=2).detach()),
                               float(want.detach()), places=12)

    def test_bernoulli_direct_and_category_factorizations_have_same_gradient(self):
        # One product per category makes category_cap=1 redundant.  The cap-1 call takes
        # the legacy category-factorized path while cap=2 takes the direct catalogue ESP.
        # Equality of values and every parameter derivative establishes that the fast
        # path is an algebraic refactor, not a baseline approximation.
        m = Bernoulli(4, 1, 1, K=2, Kp=1)
        zero_index(m.idx)
        with torch.no_grad():
            m.idx.lam.copy_(torch.tensor([0.3, -0.7, 1.1, -0.2]))
        d = batch([0, 1, 2, 3], [0, 1, 2, 3], [0, 3])
        category = m.loglik(d, nmax=2, category_cap=1)
        category_grad = torch.autograd.grad(category, tuple(m.parameters()))
        direct = m.loglik(d, nmax=2, category_cap=2)
        direct_grad = torch.autograd.grad(direct, tuple(m.parameters()))
        self.assertTrue(torch.allclose(category, direct, atol=1e-12, rtol=1e-12))
        for expected, got in zip(category_grad, direct_grad):
            self.assertTrue(torch.allclose(expected, got, atol=1e-11, rtol=1e-11))

    def test_multinomial_respects_category_cap(self):
        # Three products in category 0 and one in category 1.  At n=2 and cap=1 exactly
        # three of the six subsets survive; equal logits make each probability 1/3.
        p = torch.zeros(4); p[2] = 1.0
        m = Multinomial(4, 1, 1, p, K=2, Kp=1)
        zero_index(m.idx)
        d = batch([0, 1, 2, 3], [0, 0, 0, 1], [0, 3])
        self.assertAlmostEqual(float(m.loglik(d, category_cap=1).detach()),
                               -math.log(3), places=12)
        self.assertAlmostEqual(float(m.loglik(d, category_cap=None).detach()),
                               -math.log(6), places=12)

    def test_ndpp_is_exactly_conditioned_nonempty(self):
        # L=I on four products assigns weight one to every subset.  Conditional on nonempty,
        # every observed set therefore has probability 1/(2^4-1).
        m = NDPP(4, 1, 1, rank=2, srank=1, K=2, Kp=1)
        zero_index(m.idx)
        with torch.no_grad():
            m.V.zero_(); m.B.zero_(); m.lam.zero_()
        d = batch([0, 1, 2, 3], [0, 0, 1, 1], [0, 2])
        self.assertAlmostEqual(float(m.loglik(d).detach()), -math.log(15), places=12)

    def test_shopper_exact_set_sum_and_nonempty_conditioning(self):
        # With three equal item utilities and equal checkout utility, a two-item ordering
        # has probability 1/(4*3*2).  Two orders form the set, then divide by P(nonempty)=3/4.
        m = Shopper(3, 1, 1, K=2, Kp=1, Ki=2)
        zero_index(m.idx)
        with torch.no_grad():
            m.rho.zero_(); m.alpha_i.zero_(); m.checkout.zero_()
        d = batch([0, 1, 2], [0, 0, 1], [0, 1])
        got = m.loglik(d, exact_max_n=2)
        self.assertAlmostEqual(float(got.detach()), -math.log(9), places=12)

    def test_shopper_sampling_is_reproducible(self):
        m = Shopper(4, 1, 1, K=2, Kp=1, Ki=2)
        d = batch([0, 1, 2, 3], [0, 0, 1, 1], [0, 1, 2])
        a = m.loglik(d, n_orders=7, gen=torch.Generator().manual_seed(91))
        b = m.loglik(d, n_orders=7, gen=torch.Generator().manual_seed(91))
        self.assertTrue(torch.equal(a, b))

    def test_shopper_forces_checkout_at_maximum_size(self):
        # With three equal products, max size two, and equal checkout utility, one ordering
        # of a two-item set has probability 1/(4*3); checkout is forced after item two.
        # Summing its two orders and conditioning on nonempty (3/4) gives 2/9.
        m = Shopper(3, 1, 1, K=2, Kp=1, Ki=2)
        zero_index(m.idx)
        with torch.no_grad():
            m.rho.zero_(); m.alpha_i.zero_(); m.checkout.zero_()
        d = batch([0, 1, 2], [0, 0, 1], [0, 1])
        got = m.loglik(d, exact_max_n=2, max_size=2)
        self.assertAlmostEqual(float(got.detach()), math.log(2 / 9), places=12)


class BaselineConvergenceRule(unittest.TestCase):
    def test_requires_exposure_lr_floor_and_post_floor_patience(self):
        rule = PlateauConvergence(
            min_delta=0.01, patience=3, floor_patience=2, minimum_epochs=2.0)
        # A long plateau above the learning-rate floor cannot certify convergence.
        for iteration in (100, 200, 300, 400):
            status = rule.observe(-10.0, iteration, 10, 1000, 1e-3, 1e-4)
        self.assertFalse(status["converged"])
        # Reaching the floor is still insufficient before two epochs of exposure.
        status = rule.observe(-10.0, 100, 10, 1000, 1e-4, 1e-4)
        self.assertFalse(status["converged"])
        # Two stale observations at the floor after two epochs complete the contract.
        rule.observe(-10.0, 200, 10, 1000, 1e-4, 1e-4)
        status = rule.observe(-10.0, 210, 10, 1000, 1e-4, 1e-4)
        self.assertTrue(status["converged"])

    def test_material_improvement_resets_floor_patience(self):
        rule = PlateauConvergence(
            min_delta=0.01, patience=2, floor_patience=2, minimum_epochs=0.0)
        rule.observe(-10.0, 10, 10, 1000, 1e-4, 1e-4)
        rule.observe(-10.0, 20, 10, 1000, 1e-4, 1e-4)
        rule.observe(-9.98, 30, 10, 1000, 1e-4, 1e-4)
        self.assertEqual(rule.floor_evals_since_improvement, 0)
        self.assertFalse(rule.converged)


class BaselineSelection(unittest.TestCase):
    def test_external_subset_is_explicit_and_ordered(self):
        self.assertEqual(parse_models("bernoulli,dpp,ndpp"),
                         ("bernoulli", "dpp", "ndpp"))

    def test_rejects_unknown_and_duplicate_models(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_models("bernoulli,unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_models("dpp,dpp")


if __name__ == "__main__":
    unittest.main()
