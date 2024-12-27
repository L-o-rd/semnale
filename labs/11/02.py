import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
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
    
    plt.matshow(X)
    plt.savefig('./labs/11/02.pdf')
    plt.show()