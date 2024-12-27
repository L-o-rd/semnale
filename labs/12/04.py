import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

def roots(p):
    n = len(p)
    C = np.zeros((n, n))
    C[1:, : n - 1] = np.eye(n - 1, n - 1)
    C[:, n - 1] = -p
    return np.linalg.eig(C)[0]

if __name__ == '__main__':
    # p(x) = -24 + 6x - 4x^2 + x^3
    print(roots(np.array([-24, 6, -4])))