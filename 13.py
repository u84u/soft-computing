# Simple Univariate Time Series Prediction (Python – No TensorFlow)

import numpy as np

# Generate sine wave + noise
np.random.seed(1)
t = np.arange(0, 100, 0.1)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

# Prepare sequences
step = 10
X, y = [], []
for i in range(len(data) - step):
    X.append(data[i:i+step])
    y.append(data[i+step])

X = np.array(X)
y = np.array(y)

# Simple weight-based sequence prediction (LSTM concept demo)
W = np.random.randn(step)
predictions = X @ W / step

print("Actual next value:", y[-1])
print("Predicted next value:", predictions[-1])
