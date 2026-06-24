import numpy as np


def fit(L, H):
    X = np.vstack([L[:-1], H[:-1]])
    Y = np.vstack([L[1:],  H[1:]])
    A = Y @ np.linalg.pinv(X)
    return A, (A @ X).T, Y - A @ X


