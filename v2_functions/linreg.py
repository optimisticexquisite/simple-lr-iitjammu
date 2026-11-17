"""linreg.py - v2: the same program, with every step given a name.

Nothing about the maths changed since v1. What changed is that each idea
(load, predict, measure, step, repeat) now lives in its own function, so it
can be read, reused and tested on its own.

Run from the project root:   python v2_functions/linreg.py
"""
import numpy as np


def load_data(path):
    """Read a CSV with two columns (hours, score). Return them as x, y."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def predict(x, w, b):
    """The model: a straight line."""
    return w * x + b


def mse(y_pred, y):
    """Mean squared error: one number that says how wrong we are."""
    return np.mean((y_pred - y) ** 2)


def gradients(x, y, y_pred):
    """Slope of the loss with respect to w and b (which way is uphill)."""
    n = len(x)
    error = y_pred - y
    grad_w = (2 / n) * np.sum(error * x)
    grad_b = (2 / n) * np.sum(error)
    return grad_w, grad_b


def train(x, y, lr=0.01, epochs=3000):
    """Gradient descent. Return the best w, b and the loss at every step."""
    w, b = 0.0, 0.0
    history = []
    for epoch in range(epochs):
        y_pred = predict(x, w, b)
        history.append(mse(y_pred, y))

        grad_w, grad_b = gradients(x, y, y_pred)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b, history


def main():
    x, y = load_data("data/study_hours.csv")
    print("loaded", len(x), "students")

    w, b, history = train(x, y, lr=0.01, epochs=3000)
    print(f"loss went from {history[0]:.2f} to {history[-1]:.2f}")
    print(f"ours : w = {w:.3f}   b = {b:.3f}")

    w_ref, b_ref = np.polyfit(x, y, 1)      # an answer we trust
    print(f"numpy: w = {w_ref:.3f}   b = {b_ref:.3f}")

    print(f"a student who studies 6 hours should score about "
          f"{predict(6, w, b):.1f}")


if __name__ == "__main__":      # only run main() when this file is executed,
    main()                      # not when another file imports it
