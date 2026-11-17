"""Create the toy dataset used in every version of the demo.

50 students: how many hours each one studied, and the exam score they got.
The "true" relationship is  score = 5 * hours + 40,  plus some random noise.

Run once from the project root:   python data/make_data.py
"""
import numpy as np

rng = np.random.default_rng(seed=7)        # fixed seed: same data every time

hours = rng.uniform(1, 10, size=50).round(1)
noise = rng.normal(0, 4, size=50)
score = (5 * hours + 40 + noise).round(1)

table = np.column_stack([hours, score])
np.savetxt("data/study_hours.csv", table, delimiter=",",
           header="hours,score", comments="", fmt="%.1f")
print("wrote data/study_hours.csv with", len(table), "rows")
