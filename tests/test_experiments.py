from synthbench.config import load_yaml
from synthbench.experiments import build_grid

# ImageNet-100 classification design:
#   pilot: 1 dataset x 2 real_fractions x 1 seed x 2 baselines            = 4
#   full:  1 dataset x 7 real_fractions x 3 seeds
#          x (2 baselines + 2 generators x 5 ratios) = 21 x 12            = 252


def test_pilot_grid_is_unique():
    rows = build_grid(load_yaml("experiments/pilot.yaml"))
    assert len(rows) == 4
    assert len({row["run_id"] for row in rows}) == 4


def test_full_grid_is_unique():
    rows = build_grid(load_yaml("experiments/full.yaml"))
    assert len(rows) == 252
    assert len({row["run_id"] for row in rows}) == 252
