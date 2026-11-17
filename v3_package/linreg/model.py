"""The model: its numbers (w, b) and what it can do (fit, predict)."""
import numpy as np

from linreg.config import TrainConfig


class LinearRegression:
    """A straight line,  y = w * x + b,  learned by gradient descent.

    w and b always travel together - they ARE the model - so they live
    inside one object instead of floating around as loose variables.
    """

    def __init__(self):
        self.w = 0.0
        self.b = 0.0
        self.history = []       # the loss at every step of training

    def predict(self, x):
        return self.w * np.asarray(x, dtype=float) + self.b

    def loss(self, x, y):
        """Mean squared error of the current line on (x, y)."""
        return float(np.mean((self.predict(x) - y) ** 2))

    def fit(self, x, y, config=None):
        """Find the w and b that minimise the mean squared error."""
        config = config or TrainConfig()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )

        n = len(x)
        self.w, self.b, self.history = 0.0, 0.0, []

        for epoch in range(config.epochs):
            error = self.predict(x) - y
            loss = float(np.mean(error ** 2))

            # In v1 a run that "blew up" quietly printed inf. Here we stop
            # at the first sign of trouble and say what to do about it.
            if not np.isfinite(loss) or (self.history and loss > 1e6 * self.history[0]):
                raise FloatingPointError(
                    f"training diverged at epoch {epoch} (loss = {loss:.3g}). "
                    f"Try a smaller learning rate than {config.lr}."
                )
            self.history.append(loss)

            self.w -= config.lr * (2 / n) * np.sum(error * x)
            self.b -= config.lr * (2 / n) * np.sum(error)

        return self     # so you can write  model = LinearRegression().fit(x, y)
