"""train.py - the entry point: wire the pieces together and run them.

    python v3_package/train.py
    python v3_package/train.py --lr 0.02 --epochs 1000
    python v3_package/train.py --plot          # also saves fit.png
"""
import argparse

import numpy as np

from linreg import LinearRegression, TrainConfig, load_data
from linreg.data import DEFAULT_DATA


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a line to study data.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to a CSV file")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=3000, help="number of steps")
    parser.add_argument("--plot", action="store_true", help="save a picture of the fit")
    return parser.parse_args()


def save_plot(x, y, model, filename="fit.png"):
    import matplotlib
    matplotlib.use("Agg")                   # draw to a file, no window needed
    import matplotlib.pyplot as plt

    line_x = np.array([x.min(), x.max()])
    plt.scatter(x, y, label="students")
    plt.plot(line_x, model.predict(line_x), color="black", label="our line")
    plt.xlabel("hours studied")
    plt.ylabel("exam score")
    plt.legend()
    plt.savefig(filename, dpi=150)
    print("saved", filename)


def main():
    args = parse_args()
    config = TrainConfig(lr=args.lr, epochs=args.epochs)

    x, y = load_data(args.data)
    print(f"loaded {len(x)} students")

    model = LinearRegression().fit(x, y, config)
    print(f"loss went from {model.history[0]:.2f} to {model.history[-1]:.2f}")
    print(f"ours : w = {model.w:.3f}   b = {model.b:.3f}")

    w_ref, b_ref = np.polyfit(x, y, 1)      # an answer we trust
    print(f"numpy: w = {w_ref:.3f}   b = {b_ref:.3f}")

    if args.plot:
        save_plot(x, y, model)


if __name__ == "__main__":
    main()
