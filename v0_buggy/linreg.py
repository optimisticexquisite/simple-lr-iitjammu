# linreg.py - first draft
# Idea: predict a student's exam score from the hours they studied.
#       score = w * hours + b      (find the best w and b)
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
for epoch in range(epochs)
    y_pred = w * x + b
    error = y - y_pred
    loss = np.mean(error ** 2)

    grad_w = (2 / n) * np.sum(error * x)
    grad_b = (2 / n) * np.sum(error)

    w = w - learning_rate * grad_w
    b = b - lr * grad_b

# 4. report
print("w =", w, " b =", b, " loss =", loss)
