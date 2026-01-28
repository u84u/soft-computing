# Implementation of Single-Layer Perceptron

import numpy as np


class Perceptron:
    def __init__(self, n_inputs, lr=0.1, epochs=100):
        self.w = np.zeros(n_inputs + 1)
        self.lr = lr
        self.epochs = epochs

    def predict(self, x):
        s = np.dot(x, self.w[1:]) + self.w[0]
        return 1 if s.any(0) else 0

    def fit(self, x, y):
        for _ in range(self.epochs):
            for xi, target in zip(x, y):
                pred = self.predict(xi)
                error = target - pred
                self.w[1:] += self.lr * error * xi
                self.w[0] += self.lr * error


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    p = Perceptron(2)
    p.fit(X, y)
    print("Weights:", p.w)
    for x in X:
        print(x, "->", p.predict(x))
