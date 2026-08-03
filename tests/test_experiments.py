from synthbench.config import load_yaml
from synthbench.experiments import build_grid


def test_pilot_has_18_unique_runs():
    rows = build_grid(load_yaml("experiments/pilot.yaml"))
    assert len(rows) == 18
    assert len({row["run_id"] for row in rows}) == 18


def test_full_grid_has_288_unique_runs():
    rows = build_grid(load_yaml("experiments/full.yaml"))
    assert len(rows) == 288
    assert len({row["run_id"] for row in rows}) == 288
