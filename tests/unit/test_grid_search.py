import pytest

from scripts.grid_search import apply_point, grid_points


def test_grid_points_is_the_full_cartesian_product():
    points = grid_points({"training.lr": [0.1, 0.2], "training.w_gain": [1, 2, 3]})
    assert len(points) == 6
    assert {"training.lr": 0.2, "training.w_gain": 3} in points


def test_apply_point_overrides_without_mutating_the_base():
    base = {"training": {"lr": 0.1, "w_gain": 1}}
    out = apply_point(base, {"training.lr": 0.5})
    assert out["training"] == {"lr": 0.5, "w_gain": 1}
    assert base["training"]["lr"] == 0.1


def test_apply_point_rejects_unknown_keys():
    with pytest.raises(SystemExit, match="unknown config key"):
        apply_point({"training": {"lr": 0.1}}, {"training.learning_rate": 0.5})
