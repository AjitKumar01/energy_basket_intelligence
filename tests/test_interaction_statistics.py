import numpy as np
import torch

from audit_particle_counterfactual_generation import selected_trip_panel
from fit_stratified_natural_interactions import pair_statistic


def test_pair_statistic_matches_gram_pair_energy():
    torch.manual_seed(17)
    basis, _ = torch.linalg.qr(torch.randn(13, 4, dtype=torch.float64))
    raw = torch.randn(4, 4, dtype=torch.float64)
    c_matrix = raw @ raw.T
    items = torch.tensor([1, 3, 4, 9], dtype=torch.long)
    statistic = pair_statistic(items, basis)
    phi = basis @ torch.linalg.cholesky(c_matrix)
    rows = phi[items]
    direct = 0.5 * ((rows.sum(0).square().sum()) - rows.square().sum())
    assert np.isclose(np.sum(c_matrix.numpy() * statistic), float(direct))



def test_generation_panel_uses_the_same_nonempty_support_as_the_model():
    data = {
        "trip_split": np.array([1, 1, 1, 0]),
        "trip_nlines": np.array([1, 2, 4, 1]),
    }
    selected = selected_trip_panel(data, 3, nmax=4, seed=8)
    assert set(selected.tolist()) == {0, 1, 2}
