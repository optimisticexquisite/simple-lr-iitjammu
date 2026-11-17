# A recipe for a small computer-in-a-box that can run this project anywhere.
#
# Build from the PROJECT ROOT (so the data folder is included):
#     docker build -t linreg -f v3_package/Dockerfile .
#     docker run --rm linreg
#     docker run --rm linreg python v3_package/train.py --lr 0.02
FROM python:3.12-slim

WORKDIR /app

# Install the libraries first: this layer is cached until requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY v3_package/ v3_package/

CMD ["python", "v3_package/train.py"]
