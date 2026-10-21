"""Synthetic 3-class spiral: a sanity check that the network can learn non-linear boundaries."""
import numpy as np


def create_spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=int)

    c = 0
    while c < classes:
        ix_start = c * samples
        ix_end = (c + 1) * samples

        r = np.linspace(0.0, 1.0, samples)
        t = np.linspace(c * 4, (c + 1) * 4, samples) + np.random.randn(samples) * 0.2

        X[ix_start:ix_end, 0] = r * np.sin(t * 2.5)
        X[ix_start:ix_end, 1] = r * np.cos(t * 2.5)
        y[ix_start:ix_end] = c

        c = c + 1

    return X, y
