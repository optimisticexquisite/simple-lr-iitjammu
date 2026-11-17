# From an Idea to Working Code: lecture materials

A 20-minute lecture for first-year students: how a working programmer's
environment looks (VS Code, Git, venv, Linux, GPUs, containers, the cloud),
and the full cycle from a vague idea to tidy, tested code, shown on one tiny
machine-learning problem: fitting a straight line by gradient descent.

## What is in here

```
simple_lr/
main.pdf
├── v0_buggy/linreg.py        the first draft, with three planted bugs  ← live demo
├── v0_buggy/BUGS.md          spoilers and what to say for each bug (keep closed)
├── v1_script/linreg.py       the same file, fixed: one script, top to bottom
├── v2_functions/linreg.py    refactored into functions (the "verbs")
├── v3_package/               a class, a config, data checks, tests, a 
```

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt (TODO: Build the requirements file)
```

In VS Code: open this folder and accept the recommended Python extensions.
The interpreter in `.venv` is selected automatically.

## Running each version

Always run from the project root:

```bash
python v0_buggy/linreg.py          # SyntaxError, then NameError, then w = -inf
python v1_script/linreg.py         # w = 4.813, b = 41.120 (NumPy agrees)
python v2_functions/linreg.py
python v3_package/train.py --lr 0.02 --epochs 1000 --plot
pytest v3_package                  # 7 tests
```

## The live demo

1. Open `v0_buggy/linreg.py` and run it. Fix each error in turn. What to say
   and the exact output for each bug are in `v0_buggy/BUGS.md`.
2. For bug 3, set `epochs = 3` and add the print line, *or* set a breakpoint on
   line 23 and press F5 ("Debug: v0 buggy draft").

Slides 22 to 26 show the same steps with real output, so they double as
a backup if the live demo misbehaves.

## Timing plan for 20 minutes

| Minutes | Slides | Section | If you are running late |
|---|---|---|---|
| 0:00–1:00 | 1–2 | Opening, the loop | |
| 1:00–4:30 | 3–8 | The workshop: VS Code, Linux, Python, venv | slide 7 (Python vs Rust): one sentence |
| 4:30–8:00 | 9–13 | Git and GitHub, branches, secrets | |
| 8:00–11:00 | 14–18 | GPUs, no sudo, Docker, cloud, cost | slides 15 and 18: headline only |
| 11:00–13:00 | 19–21 | From idea to a crisp plan | |
| 13:00–18:00 | 22–30 | **Live demo**, then refactoring v1 → v2 → v3 | do the demo on 22, then flick through 23–26 |
| 18:00–20:00 | 31–33 | LLMs, and why to learn properly | |
