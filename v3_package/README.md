# linreg (v3)

Predict a student's exam score from the hours they studied, with a straight
line learned by gradient descent.

## Run

From the project root, with the virtual environment active:

```bash
python v3_package/train.py                        # default settings
python v3_package/train.py --lr 0.02 --epochs 1000
python v3_package/train.py --plot                 # also saves fit.png
pytest v3_package                                 # run the tests
```

## Layout

```
v3_package/
├── train.py            entry point: reads the command line, wires things up
├── linreg/
│   ├── config.py       TrainConfig: the settings (lr, epochs), validated
│   ├── data.py         load_data(): read the CSV and check it
│   └── model.py        LinearRegression: w, b, fit(), predict()
├── tests/
│   └── test_model.py   small programs that check our program
└── Dockerfile          run it anywhere, identically
```

## Use it from your own code

```python
from linreg import LinearRegression, load_data

x, y = load_data()
model = LinearRegression().fit(x, y)
print(model.predict(6))      # expected score after 6 hours of study
```
