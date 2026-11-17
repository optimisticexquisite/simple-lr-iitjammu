"""Tests: small programs that check our program.

Run from the project root:   pytest v3_package
"""
import numpy as np
import pytest

from linreg import LinearRegression, TrainConfig, load_data


def test_recovers_a_known_line():
    # If the data is EXACTLY y = 3x + 2, the model must find w = 3, b = 2.
    x = np.linspace(0, 5, 20)
    y = 3 * x + 2
    model = LinearRegression().fit(x, y, TrainConfig(lr=0.05, epochs=5000))
    assert model.w == pytest.approx(3, abs=1e-3)
    assert model.b == pytest.approx(2, abs=1e-3)


def test_loss_goes_down():
    # The bug from the first draft (walking uphill) would fail this test.
    x, y = load_data()
    model = LinearRegression().fit(x, y, TrainConfig(epochs=50))
    assert model.history[-1] < model.history[0]


def test_agrees_with_numpy():
    x, y = load_data()
    model = LinearRegression().fit(x, y)
    w_ref, b_ref = np.polyfit(x, y, 1)
    assert model.w == pytest.approx(w_ref, abs=0.01)
    assert model.b == pytest.approx(b_ref, abs=0.01)


def test_too_large_a_step_is_reported_clearly():
    x, y = load_data()
    with pytest.raises(FloatingPointError, match="smaller learning rate"):
        LinearRegression().fit(x, y, TrainConfig(lr=1.0))


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        TrainConfig(lr=-1)


def test_wrong_number_of_columns_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match="expected 2 columns"):
        load_data(bad)


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no data file"):
        load_data(tmp_path / "nope.csv")
