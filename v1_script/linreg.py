# linreg.py - v1: one file, read top to bottom. It works!
# Idea: predict a student's exam score from the hours they studied.
#       score = w * hours + b      (find the best w and b)
#
# Run from the project root:   python v1_script/linreg.py
import numpy as np

# 1. load the data
data = np.loadtxt("data/study_hours.csv", delimiter=",", skiprows=1)
x = data[:, 0]          # hours studied
y = data[:, 1]          # exam score
n = len(x)
print("loaded", n, "students")

# 2. start with a guess
w = 0.0
b = 0.0
lr = 0.01               # learning rate: the size of each step
epochs = 3000           # how many steps to take

# 3. gradient descent: repeat many small steps downhill
for epoch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y                      # how wrong is each prediction?
    loss = np.mean(error ** 2)              # one number: how wrong overall?

    grad_w = (2 / n) * np.sum(error * x)    # which way is uphill for w?
    grad_b = (2 / n) * np.sum(error)        # which way is uphill for b?

    w = w - lr * grad_w                     # step the other way: downhill
    b = b - lr * grad_b

    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}   loss {loss:8.2f}")

# 4. report - and check against an answer we trust
print(f"ours : w = {w:.3f}   b = {b:.3f}   loss = {loss:.3f}")

w_ref, b_ref = np.polyfit(x, y, 1)          # NumPy's built-in line fit
print(f"numpy: w = {w_ref:.3f}   b = {b_ref:.3f}")
