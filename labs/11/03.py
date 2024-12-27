import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # np.random.seed(1)
    N = 1000
    time = np.arange(N)

    trend = 0.0001 * time**2 - 0.005 * time + 5
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii

    L = 500
    K = N - L + 1
    X = np.zeros((L, K))
    for j in range(K):
        X[:, j] = y[j:j + L]

    S1 = X @ X.T
    S2 = X.T @ X
    eval1, evec1 = np.linalg.eigh(S1)
    A1 = np.diag(np.flip(eval1))
    # A1 = A1[np.argsort(eval1)[::-1][:], :]
    Ueig = np.flip(np.abs(evec1), axis = 1)

    # S1 = UA1U^(-1)

    eval2, evec2 = np.linalg.eigh(S2)
    A2 = np.diag(np.flip(eval2))
    # A2 = A2[np.argsort(eval2)[::-1][:], :]
    Veig = np.flip(np.abs(evec2), axis = 1)

    # S2 = VA2V^(-1)

    Usvd, Svs, Vhsvd = np.linalg.svd(X)
    Usvd = np.abs(Usvd)
    S = np.diag(Svs)

    # XTX = VhT @ ST @ S @ Vh
    # X.T @ X = U @ A @ U^(-1)
    # A = S.T @ S
    # U = Vh

    # (X.T @ X) @ S = Vh.T @ S.T @ S @ Vh @ S @ 

    # X = U @ S @ Vh

    # T = Vhsvd.T @ S.T @ S @ Vhsvd

    # XXT = U @ S @ ST @ UT

    # W = Usvd @ S @ S.T @ Usvd.T

    eps = 1e-6

    SS = np.zeros(A1.shape)
    SS[:S.shape[0], :S.shape[1]] = S
    S = SS

    print(np.linalg.norm(Ueig - Usvd))
    print(np.linalg.norm(Veig - np.abs(np.matrix(Vhsvd).H)))
    print(np.linalg.norm(A1 - (S.T @ S)))
    if np.linalg.norm(Ueig - Usvd) <= eps:
        print("U (eig) = U (svd)")
        
    if np.linalg.norm(Veig - np.abs(np.matrix(Vhsvd).H)) <= eps:
        print("V.H (eig) = V (svd)")

    if np.linalg.norm(A1 - (S.T @ S)) <= eps:
        print("A = S.T @ S")