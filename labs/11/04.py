import matplotlib.pyplot as plt
import numpy as np

def hankelize(X_i):
    L, K = X_i.shape
    N = L + K - 1
    hankelized = np.zeros(N)
    for diag in range(N):
        elements = []
        for i in range(L):
            j = diag - i
            if 0 <= j < K:
                elements.append(X_i[i, j])
        hankelized[diag] = np.mean(elements)
    return hankelized

if __name__ == '__main__':
    np.random.seed(1)
    N = 6
    time = np.arange(N)

    trend = 0.0001 * time**2 - 0.005 * time + 5
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii

    L = 4
    K = N - L + 1
    X = np.zeros((L, K))
    for j in range(K):
        X[:, j] = y[j:j + L]

    U, S, Vt = np.linalg.svd(X, full_matrices = False)

    xhat = []
    sum = np.zeros_like(y)
    for i in range(len(S)):
        S_i = np.zeros_like(Vt)
        S_i[i, i] = S[i]
        X_i = U @ S_i @ Vt
        T_i = hankelize(X_i)
        xhat.append(T_i)
        sum += T_i

    xhat = np.array(xhat)
    print(xhat)
    print(np.linalg.norm(y - sum))